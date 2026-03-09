# model/bert_parser.py
# INNOVATION 2: Replace the BiLSTM encoder with a pre-trained BERT / RoBERTa.
#
# Motivation:
#   Dozat & Manning (2017) uses random word embeddings + BiLSTM.
#   Pre-trained transformer models (BERT, RoBERTa, XLM-R) produce far richer
#   contextual representations. Replacing the BiLSTM with BERT typically yields
#   +2-4% UAS/LAS on standard benchmarks.
#
# Key implementation challenges:
#   1. BERT uses subword (WordPiece/BPE) tokenization, but dependency parsing
#      requires one vector per *word*. We handle this by averaging the subword
#      token representations that belong to each word.
#   2. BERT needs a much smaller learning rate (1e-5) than the MLP/biaffine
#      layers (1e-3). We expose separate parameter groups for this.
#   3. We keep the same MLP + Biaffine head as the baseline, so the two models
#      are directly comparable.
#
# References:
#   - Devlin et al. (2019) "BERT: Pre-training of Deep Bidirectional Transformers"
#   - Kondratyuk & Straka (2019) "75 Languages, 1 Model: Parsing Universal
#     Dependencies Universally"

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional, Tuple, List

from modules.mlp      import MLP
from modules.biaffine import Biaffine


class BERTBiaffineParser(nn.Module):
    """
    BERT-based biaffine dependency parser.

    Architecture:
        BERT encoder → MLP projections → Biaffine arc + label classifiers

    The BiLSTM is completely replaced by BERT. The MLP and Biaffine components
    are identical to the baseline, enabling fair comparison.
    """

    def __init__(
        self,
        n_rels:          int,
        bert_model:      str   = 'bert-base-uncased',
        arc_mlp_dim:     int   = 500,
        label_mlp_dim:   int   = 100,
        mlp_dropout:     float = 0.33,
        bert_dropout:    float = 0.1,
        freeze_bert:     bool  = False,   # set True to train MLP/biaffine only
    ):
        super().__init__()
        self.bert_model_name = bert_model

        # ------------------------------------------------------------------ #
        # 1. Pre-trained BERT encoder
        # ------------------------------------------------------------------ #
        self.bert    = AutoModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(p=bert_dropout)
        bert_dim     = self.bert.config.hidden_size  # 768 for base, 1024 for large

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # ------------------------------------------------------------------ #
        # 2. MLP projections (same design as baseline)
        # ------------------------------------------------------------------ #
        self.arc_dep_mlp    = MLP(bert_dim, arc_mlp_dim,   mlp_dropout)
        self.arc_head_mlp   = MLP(bert_dim, arc_mlp_dim,   mlp_dropout)
        self.label_dep_mlp  = MLP(bert_dim, label_mlp_dim, mlp_dropout)
        self.label_head_mlp = MLP(bert_dim, label_mlp_dim, mlp_dropout)

        # ------------------------------------------------------------------ #
        # 3. Biaffine classifiers (identical to baseline)
        # ------------------------------------------------------------------ #
        self.arc_biaffine = Biaffine(arc_mlp_dim,   out_features=1,
                                     bias_x=True,  bias_y=False)
        self.label_biaffine = Biaffine(label_mlp_dim, out_features=n_rels,
                                       bias_x=True,  bias_y=True)

    # ---------------------------------------------------------------------- #
    # Forward pass
    # ---------------------------------------------------------------------- #
    def forward(
        self,
        input_ids:      torch.Tensor,   # [batch, subword_len]
        attention_mask: torch.Tensor,   # [batch, subword_len]
        word_map:       torch.Tensor,   # [batch, seq_len, max_sub_per_word]
                                        # -1 for padding positions
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        word_map maps each word (including <root> at index 0) to its subword
        token positions in the BERT input. Subword representations for each
        word are averaged to produce one vector per word.

        Returns:
            arc_scores:   [batch, seq_len, seq_len]
            label_scores: [batch, seq_len, seq_len, n_rels]
        """
        # 1. BERT forward pass
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        subword_repr = self.dropout(bert_out.last_hidden_state)
        # subword_repr: [batch, subword_len, bert_dim]

        # 2. Subword → word pooling (average over subwords per word)
        r = self._pool_subwords(subword_repr, word_map)
        # r: [batch, seq_len, bert_dim]

        # 3. MLP projections
        h_arc_dep    = self.arc_dep_mlp(r)
        h_arc_head   = self.arc_head_mlp(r)
        h_label_dep  = self.label_dep_mlp(r)
        h_label_head = self.label_head_mlp(r)

        # 4. Biaffine scoring
        arc_scores   = self.arc_biaffine(h_arc_dep,   h_arc_head)
        label_scores = self.label_biaffine(h_label_dep, h_label_head)

        return arc_scores, label_scores

    def _pool_subwords(self, subword_repr: torch.Tensor,
                       word_map: torch.Tensor) -> torch.Tensor:
        """
        Averages subword token representations to produce word-level vectors.

        subword_repr: [batch, subword_len, dim]
        word_map:     [batch, seq_len, max_sub]   — subword indices per word
                      uses -1 as padding (no subword at that position)

        returns: [batch, seq_len, dim]
        """
        batch, seq_len, max_sub = word_map.shape
        dim = subword_repr.size(-1)

        # Clamp -1 to 0 for safe gather, then mask contributions of -1 positions
        valid_mask = word_map.ge(0)                          # [batch, seq, max_sub]
        safe_map   = word_map.clamp(min=0)                   # [batch, seq, max_sub]

        # Expand indices for gathering: [batch, seq, max_sub, dim]
        idx = safe_map.unsqueeze(-1).expand(batch, seq_len, max_sub, dim)

        # Expand subword_repr for gathering: [batch, 1, subword_len, dim]
        # → [batch, seq_len, subword_len, dim] via broadcast
        subword_exp = subword_repr.unsqueeze(1).expand(batch, seq_len, -1, dim)

        # Gather subword embeddings for each word
        gathered = subword_exp.gather(2, idx)
        # gathered: [batch, seq_len, max_sub, dim]

        # Zero out padded subword positions (-1 entries)
        gathered = gathered * valid_mask.unsqueeze(-1).float()

        # Average over subwords
        counts = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1).float()
        # counts: [batch, seq_len, 1]
        word_repr = gathered.sum(dim=2) / counts
        # word_repr: [batch, seq_len, dim]

        return word_repr

    # ---------------------------------------------------------------------- #
    # Inference helpers (same interface as base_parser.py)
    # ---------------------------------------------------------------------- #
    def predict_arcs(self, arc_scores: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
        arc_scores = arc_scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
        return arc_scores.argmax(dim=-1)

    def predict_labels(self, label_scores: torch.Tensor,
                       pred_heads: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _, n_rels = label_scores.shape
        head_idx = pred_heads.unsqueeze(-1).unsqueeze(-1).expand(
            batch, seq_len, 1, n_rels
        )
        selected = label_scores.gather(2, head_idx).squeeze(2)
        return selected.argmax(dim=-1)

    # ---------------------------------------------------------------------- #
    # Parameter groups for optimizer (BERT needs much smaller LR)
    # ---------------------------------------------------------------------- #
    def get_param_groups(self, bert_lr: float = 2e-5,
                         head_lr: float = 2e-3) -> List[dict]:
        """
        Returns separate parameter groups so the optimizer can apply
        a small LR to BERT and a larger LR to the task head.
        """
        bert_params = list(self.bert.parameters())
        head_params = (
            list(self.arc_dep_mlp.parameters()) +
            list(self.arc_head_mlp.parameters()) +
            list(self.label_dep_mlp.parameters()) +
            list(self.label_head_mlp.parameters()) +
            list(self.arc_biaffine.parameters()) +
            list(self.label_biaffine.parameters())
        )
        return [
            {'params': bert_params, 'lr': bert_lr},
            {'params': head_params, 'lr': head_lr},
        ]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# BERT Dataset / collate helpers
# --------------------------------------------------------------------------- #
class BERTDependencyDataset(torch.utils.data.Dataset):
    """
    Tokenizes sentences with a BERT tokenizer and builds the word_map
    that connects BERT subword tokens back to original words.
    """

    def __init__(self, sentences, vocab, tokenizer_name: str = 'bert-base-uncased'):
        from data.vocab import Vocab
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.samples   = []
        self.vocab     = vocab

        for sent in sentences:
            sample = self._encode(sent)
            self.samples.append(sample)

    def _encode(self, sent):
        words   = sent.words   # includes <root> at index 0
        heads   = sent.heads
        deprels = sent.deprels

        # Build BERT input: [CLS] + subwords of each word + [SEP]
        # Track which subword tokens belong to each word
        input_ids   = [self.tokenizer.cls_token_id]
        word_to_sub = [[]]  # index 0 = <root>, gets the [CLS] token

        # Handle <root> → map to [CLS] position (index 0)
        word_to_sub[0] = [0]

        for word in words[1:]:  # skip <root>
            sub_ids = self.tokenizer.encode(word, add_special_tokens=False)
            if not sub_ids:
                sub_ids = [self.tokenizer.unk_token_id]
            start = len(input_ids)
            input_ids.extend(sub_ids)
            word_to_sub.append(list(range(start, start + len(sub_ids))))

        input_ids.append(self.tokenizer.sep_token_id)

        attention_mask = [1] * len(input_ids)

        rel_ids = [self.vocab.rel_to_id(r) for r in deprels]

        return {
            'input_ids':      input_ids,
            'attention_mask': attention_mask,
            'word_to_sub':    word_to_sub,  # list of lists
            'heads':          heads,
            'rels':           rel_ids,
            'length':         len(words),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def bert_collate_fn(batch: List[dict]) -> dict:
    """Pads all tensors in a BERT batch to the same length."""
    max_sub_len  = max(len(s['input_ids']) for s in batch)
    max_word_len = max(s['length'] for s in batch)
    max_sub_per_word = max(
        max(len(sub) for sub in s['word_to_sub'])
        for s in batch
    )

    input_ids_batch   = []
    attn_mask_batch   = []
    word_map_batch    = []
    heads_batch       = []
    rels_batch        = []
    seq_mask_batch    = []

    for s in batch:
        # Pad subword sequence
        pad_sub = max_sub_len - len(s['input_ids'])
        input_ids_batch.append(s['input_ids']      + [0] * pad_sub)
        attn_mask_batch.append(s['attention_mask'] + [0] * pad_sub)

        # Build word_map: [max_word_len, max_sub_per_word] filled with -1
        wmap = []
        for sub_list in s['word_to_sub']:
            padded = sub_list + [-1] * (max_sub_per_word - len(sub_list))
            wmap.append(padded)
        # Pad missing words
        for _ in range(max_word_len - s['length']):
            wmap.append([-1] * max_sub_per_word)
        word_map_batch.append(wmap)

        # Pad heads and rels
        pad_word = max_word_len - s['length']
        heads_batch.append(s['heads'] + [0] * pad_word)
        rels_batch.append(s['rels']   + [0] * pad_word)
        seq_mask_batch.append([True]  * s['length'] + [False] * pad_word)

    return {
        'input_ids':      torch.tensor(input_ids_batch, dtype=torch.long),
        'attention_mask': torch.tensor(attn_mask_batch, dtype=torch.long),
        'word_map':       torch.tensor(word_map_batch,  dtype=torch.long),
        'heads':          torch.tensor(heads_batch,     dtype=torch.long),
        'rels':           torch.tensor(rels_batch,      dtype=torch.long),
        'mask':           torch.tensor(seq_mask_batch,  dtype=torch.bool),
    }
