# model/base_parser.py
# Full reproduction of Dozat & Manning (2017) "Deep Biaffine Attention for
# Neural Dependency Parsing" with optional CharLSTM extension (Innovation 1).
#
# Pipeline:
#   (word_embed [+ pos_embed] [+ char_embed])
#     → dropout
#     → 3-layer BiLSTM (400d each direction, variational dropout)
#     → 4 separate MLP projections (arc-dep, arc-head, lbl-dep, lbl-head)
#     → Biaffine arc scorer     → arc scores  [batch, dep, head]
#     → Biaffine label scorer   → label scores [batch, dep, head, n_rels]

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

from modules.bilstm   import BiLSTMEncoder
from modules.char_lstm import CharLSTM
from modules.mlp      import MLP
from modules.biaffine import Biaffine


class BiaffineParser(nn.Module):
    """
    Biaffine dependency parser.

    Innovations vs. the original paper:
      - use_char=True  : adds CharLSTM character embeddings (Innovation 1)
      - use_char=False : exact reproduction of the 2017 paper
    """

    def __init__(
        self,
        n_words:          int,
        n_tags:           int,
        n_rels:           int,
        n_chars:          int  = 0,
        # Embedding dimensions
        word_embed_dim:   int  = 100,
        tag_embed_dim:    int  = 100,
        char_embed_dim:   int  = 50,
        char_out_dim:     int  = 100,
        # BiLSTM
        lstm_hidden:      int  = 400,
        lstm_layers:      int  = 3,
        lstm_dropout:     float = 0.33,
        # MLPs
        arc_mlp_dim:      int  = 500,
        label_mlp_dim:    int  = 100,
        mlp_dropout:      float = 0.33,
        # Input dropout (Table 1 — paper drops words and tags independently)
        embed_dropout:    float = 0.33,
        # Pre-trained word embeddings
        pretrained_embeds: Optional[np.ndarray] = None,
        # Innovations
        use_char:         bool = False,
        use_pos:          bool = True,
    ):
        super().__init__()
        self.use_char = use_char
        self.use_pos  = use_pos

        # ------------------------------------------------------------------ #
        # 1. Embedding layers
        # ------------------------------------------------------------------ #
        self.word_embed = nn.Embedding(n_words, word_embed_dim, padding_idx=0)
        if use_pos:
            self.tag_embed = nn.Embedding(n_tags, tag_embed_dim, padding_idx=0)

        if pretrained_embeds is not None:
            self.word_embed.weight.data.copy_(
                torch.tensor(pretrained_embeds, dtype=torch.float)
            )

        encoder_in_dim = word_embed_dim + (tag_embed_dim if use_pos else 0)

        if use_char:
            assert n_chars > 0, "n_chars must be set when use_char=True"
            self.char_lstm  = CharLSTM(n_chars, char_embed_dim, char_out_dim)
            encoder_in_dim += char_out_dim

        # Separate dropout for word and tag embeddings (Section 4.2.4)
        # The paper drops words and tags independently at 33%
        self.embed_dropout = nn.Dropout(p=embed_dropout)

        # ------------------------------------------------------------------ #
        # 2. Stacked BiLSTM encoder
        # ------------------------------------------------------------------ #
        self.encoder = BiLSTMEncoder(
            in_features=encoder_in_dim,
            hidden_size=lstm_hidden,
            n_layers=lstm_layers,
            dropout=lstm_dropout,
        )
        lstm_out_dim = lstm_hidden * 2  # bidirectional

        # ------------------------------------------------------------------ #
        # 3. Dimension-reducing MLPs (Equations 4-5 of the paper)
        # ------------------------------------------------------------------ #
        # Arc: each word gets a "as dependent" and "as head" representation
        self.arc_dep_mlp  = MLP(lstm_out_dim, arc_mlp_dim,   mlp_dropout)
        self.arc_head_mlp = MLP(lstm_out_dim, arc_mlp_dim,   mlp_dropout)

        # Label: same but lower-dimensional (100 vs. 500)
        self.label_dep_mlp  = MLP(lstm_out_dim, label_mlp_dim, mlp_dropout)
        self.label_head_mlp = MLP(lstm_out_dim, label_mlp_dim, mlp_dropout)

        # ------------------------------------------------------------------ #
        # 4. Biaffine classifiers
        # ------------------------------------------------------------------ #
        # Arc scorer: one score per (dep, head) pair → [batch, dep, head]
        self.arc_biaffine = Biaffine(
            in_features=arc_mlp_dim,
            out_features=1,
            bias_x=True,   # linear-on-dep  term
            bias_y=False,  # no linear-on-head for arcs (see paper Eq. 2)
        )

        # Label scorer: n_rels scores per (dep, head) pair
        self.label_biaffine = Biaffine(
            in_features=label_mlp_dim,
            out_features=n_rels,
            bias_x=True,
            bias_y=True,
        )

    # ---------------------------------------------------------------------- #
    # Forward pass
    # ---------------------------------------------------------------------- #
    def forward(
        self,
        words: torch.Tensor,           # [batch, seq_len]
        tags:  Optional[torch.Tensor], # [batch, seq_len]
        chars: Optional[torch.Tensor] = None,  # [batch, seq_len, max_char_len]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            arc_scores:   [batch, seq_len, seq_len]
            label_scores: [batch, seq_len, seq_len, n_rels]
        """
        mask = words.ne(0)  # [batch, seq_len]

        # 1. Embed words
        word_emb = self.word_embed(words)  # [batch, seq, word_dim]

        # Independent word/tag dropout (Section 4.2.4 of the paper):
        # When training, randomly zero each word embedding independently,
        # and independently zero each tag embedding.
        if self.training:
            word_emb = self._independent_embed_dropout(word_emb)

        inputs = [word_emb]

        if self.use_pos and tags is not None:
            tag_emb = self.tag_embed(tags)
            if self.training:
                tag_emb = self._independent_embed_dropout(tag_emb)
            inputs.append(tag_emb)

        if self.use_char and chars is not None:
            char_emb = self.char_lstm(chars)  # [batch, seq, char_out_dim]
            inputs.append(char_emb)

        x = torch.cat(inputs, dim=-1)
        x = self.embed_dropout(x)

        # 2. BiLSTM encoder
        r = self.encoder(x, mask)   # [batch, seq_len, lstm_hidden*2]

        # 3. MLP projections
        h_arc_dep    = self.arc_dep_mlp(r)    # [batch, seq_len, arc_mlp_dim]
        h_arc_head   = self.arc_head_mlp(r)
        h_label_dep  = self.label_dep_mlp(r)  # [batch, seq_len, label_mlp_dim]
        h_label_head = self.label_head_mlp(r)

        # 4. Biaffine scoring
        # arc_scores[b, i, j] = score of arc j→i (head j, dependent i)
        arc_scores   = self.arc_biaffine(h_arc_dep,   h_arc_head)
        # [batch, seq_len, seq_len]

        label_scores = self.label_biaffine(h_label_dep, h_label_head)
        # [batch, seq_len, seq_len, n_rels]

        return arc_scores, label_scores

    def _independent_embed_dropout(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Applies the per-token embedding dropout described in Section 4.2.4.
        Each token's embedding is independently zeroed with probability p.
        (Different from standard dropout which zeros individual dimensions.)
        """
        p = self.embed_dropout.p
        # mask shape: [batch, seq_len, 1] — broadcast over embedding dim
        token_mask = emb.new_empty(emb.size(0), emb.size(1), 1).bernoulli_(1 - p)
        token_mask = token_mask / (1 - p)  # inverted dropout scaling
        return emb * token_mask

    # ---------------------------------------------------------------------- #
    # Inference helpers
    # ---------------------------------------------------------------------- #
    def predict_arcs(self, arc_scores: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
        """
        Greedy arc prediction: each word picks its highest-scoring head.
        arc_scores: [batch, dep, head]
        mask: [batch, seq_len]
        returns: pred_heads [batch, seq_len]
        """
        # Mask padding positions — set to -inf so they are never chosen
        arc_scores = arc_scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
        # Each dependent picks the max over all possible heads
        pred_heads = arc_scores.argmax(dim=-1)  # [batch, seq_len]
        return pred_heads

    def predict_labels(self, label_scores: torch.Tensor,
                       pred_heads: torch.Tensor) -> torch.Tensor:
        """
        Given predicted head for each word, select the label scores
        at that head position and take the argmax.

        label_scores: [batch, dep, head, n_rels]
        pred_heads:   [batch, seq_len]
        returns: pred_rels [batch, seq_len]
        """
        batch, seq_len, _, n_rels = label_scores.shape
        # Gather the label scores at the predicted head position
        head_idx = pred_heads.unsqueeze(-1).unsqueeze(-1).expand(
            batch, seq_len, 1, n_rels
        )
        # selected: [batch, seq_len, 1, n_rels]
        selected = label_scores.gather(2, head_idx).squeeze(2)
        # selected: [batch, seq_len, n_rels]
        pred_rels = selected.argmax(dim=-1)  # [batch, seq_len]
        return pred_rels

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
