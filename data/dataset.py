# data/dataset.py
# PyTorch Dataset and collate function for dependency parsing.
# Converts Sentence objects → padded tensors for batching.

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional

from data.conllu import Sentence
from data.vocab  import Vocab


class DependencyDataset(Dataset):
    """
    Converts a list of Sentence objects into indexed, tensorizable samples.
    Each sample is a dict of Python lists (not tensors yet — padding is done
    in the collate function so sequences of different lengths can be batched).
    """

    def __init__(self, sentences: List[Sentence], vocab: Vocab,
                 use_char: bool = False):
        self.samples  = []
        self.vocab    = vocab
        self.use_char = use_char

        for sent in sentences:
            n = len(sent)  # includes <root>

            word_ids = [vocab.word_to_id(w) for w in sent.words]
            tag_ids  = [vocab.tag_to_id(t)  for t in sent.upos]
            head_ids = sent.heads          # already int list
            rel_ids  = [vocab.rel_to_id(r) for r in sent.deprels]

            sample = {
                'words':  word_ids,
                'tags':   tag_ids,
                'heads':  head_ids,
                'rels':   rel_ids,
                'length': n,
            }

            if use_char:
                # chars[i] = list of char ids for word i
                char_ids = []
                for char_list in sent.chars:
                    char_ids.append([vocab.char_to_id(c) for c in char_list])
                sample['chars'] = char_ids  # list of lists

            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: List[dict], use_char: bool = False) -> dict:
    """
    Pads all sequences in a batch to the same length.
    Called automatically by DataLoader.

    Padding index is 0 for words/tags/rels (matches nn.Embedding padding_idx=0).
    Head padding is also 0 (root index — masked out during loss computation).
    """
    lengths = [sample['length'] for sample in batch]
    max_len = max(lengths)

    words_batch = []
    tags_batch  = []
    heads_batch = []
    rels_batch  = []
    mask_batch  = []

    for sample in batch:
        pad_len = max_len - sample['length']

        words_batch.append(sample['words'] + [0] * pad_len)
        tags_batch.append(sample['tags']   + [0] * pad_len)
        heads_batch.append(sample['heads'] + [0] * pad_len)
        rels_batch.append(sample['rels']   + [0] * pad_len)

        # mask: True for real tokens (including <root>), False for padding
        mask_batch.append([True] * sample['length'] + [False] * pad_len)

    result = {
        'words':  torch.tensor(words_batch, dtype=torch.long),
        'tags':   torch.tensor(tags_batch,  dtype=torch.long),
        'heads':  torch.tensor(heads_batch, dtype=torch.long),
        'rels':   torch.tensor(rels_batch,  dtype=torch.long),
        'mask':   torch.tensor(mask_batch,  dtype=torch.bool),
        'lengths': torch.tensor(lengths,    dtype=torch.long),
    }

    if use_char:
        # chars: [batch, seq_len, max_char_len]
        # First find the max character sequence length across the whole batch
        max_char_len = max(
            len(char_ids)
            for sample in batch
            for char_ids in sample['chars']
            if char_ids  # skip the <root> empty list
        )
        max_char_len = max(max_char_len, 1)

        chars_batch = []
        for sample in batch:
            sent_chars = []
            for i in range(max_len):
                if i < sample['length']:
                    cids = sample['chars'][i]  # may be [] for <root>
                    pad  = max_char_len - len(cids)
                    sent_chars.append(cids + [0] * pad)
                else:
                    sent_chars.append([0] * max_char_len)
            chars_batch.append(sent_chars)

        result['chars'] = torch.tensor(chars_batch, dtype=torch.long)

    return result


def build_dataloader(sentences: List[Sentence], vocab: Vocab,
                     batch_size: int = 32, shuffle: bool = True,
                     use_char: bool = False,
                     num_workers: int = 0) -> DataLoader:
    """
    Builds a DataLoader from a list of Sentence objects.
    Sorts sentences by length (longest first) within batches to minimize padding.
    """
    dataset = DependencyDataset(sentences, vocab, use_char=use_char)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_fn(b, use_char=use_char),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
