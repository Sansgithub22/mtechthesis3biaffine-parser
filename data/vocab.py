# data/vocab.py
# Builds and manages vocabularies for words, POS tags, dependency labels, and characters.

import os
import pickle
from collections import Counter
from typing import List, Optional
import numpy as np

from data.conllu import Sentence


class Vocab:
    """
    Manages four vocabularies:
      - words   : word forms (with min_freq threshold)
      - tags    : universal POS tags
      - rels    : dependency relation labels
      - chars   : individual characters (for CharLSTM innovation)

    Special tokens:
      PAD (0) : padding — always index 0 so nn.Embedding padding_idx=0 works
      UNK (1) : unknown word
      ROOT (2): the synthetic root token
    """

    PAD  = '<pad>'
    UNK  = '<unk>'
    ROOT = '<root>'

    def __init__(self):
        # Word vocabulary
        self.word2id: dict = {self.PAD: 0, self.UNK: 1, self.ROOT: 2}
        self.id2word: list = [self.PAD, self.UNK, self.ROOT]

        # POS tag vocabulary
        self.tag2id: dict = {self.PAD: 0, self.UNK: 1, self.ROOT: 2}
        self.id2tag: list = [self.PAD, self.UNK, self.ROOT]

        # Dependency relation vocabulary
        self.rel2id: dict = {self.PAD: 0, self.ROOT: 1}
        self.id2rel: list = [self.PAD, self.ROOT]

        # Character vocabulary (for CharLSTM)
        self.char2id: dict = {self.PAD: 0, self.UNK: 1}
        self.id2char: list = [self.PAD, self.UNK]

    # ------------------------------------------------------------------
    # Build vocabulary from training sentences
    # ------------------------------------------------------------------
    def build(self, sentences: List[Sentence], min_freq: int = 2):
        """
        Counts all tokens across sentences and adds those meeting min_freq
        to the word vocabulary. All tags, rels, and chars are added unconditionally.
        """
        word_counter = Counter()

        for sent in sentences:
            # Skip index 0 (<root>)
            for i in range(1, len(sent)):
                word_counter[sent.words[i]] += 1

                # POS tags — always add
                tag = sent.upos[i]
                if tag not in self.tag2id:
                    self.tag2id[tag] = len(self.id2tag)
                    self.id2tag.append(tag)

                # Dependency labels — always add
                rel = sent.deprels[i]
                if rel not in self.rel2id:
                    self.rel2id[rel] = len(self.id2rel)
                    self.id2rel.append(rel)

                # Characters — always add
                for ch in sent.words[i]:
                    if ch not in self.char2id:
                        self.char2id[ch] = len(self.id2char)
                        self.id2char.append(ch)

        # Add words that meet the frequency threshold
        for word, count in word_counter.items():
            if count >= min_freq and word not in self.word2id:
                self.word2id[word] = len(self.id2word)
                self.id2word.append(word)

        print(f"Vocabulary built: {self.n_words} words | {self.n_tags} tags | "
              f"{self.n_rels} rels | {self.n_chars} chars")

    # ------------------------------------------------------------------
    # Load pre-trained GloVe / fastText embeddings
    # ------------------------------------------------------------------
    def load_pretrained_embeds(self, embed_path: str,
                                embed_dim: int = 100) -> np.ndarray:
        """
        Loads a GloVe / fastText .txt file and returns an embedding matrix
        of shape [n_words, embed_dim] aligned to self.word2id.
        Words not found in the file are left as zeros.
        """
        print(f"Loading pre-trained embeddings from: {embed_path}")
        pretrained = {}
        with open(embed_path, encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip().split(' ')
                if len(parts) == embed_dim + 1:
                    pretrained[parts[0]] = np.array(parts[1:], dtype=np.float32)

        matrix = np.zeros((self.n_words, embed_dim), dtype=np.float32)
        found  = 0
        for word, idx in self.word2id.items():
            key = word.lower()   # case-insensitive lookup
            if key in pretrained:
                matrix[idx] = pretrained[key]
                found += 1

        coverage = 100.0 * found / max(self.n_words, 1)
        print(f"Pre-trained embedding coverage: {found}/{self.n_words} "
              f"({coverage:.1f}%)")
        return matrix

    # ------------------------------------------------------------------
    # Token → index helpers
    # ------------------------------------------------------------------
    def word_to_id(self, word: str) -> int:
        return self.word2id.get(word, self.word2id[self.UNK])

    def tag_to_id(self, tag: str) -> int:
        return self.tag2id.get(tag, self.tag2id[self.UNK])

    def rel_to_id(self, rel: str) -> int:
        return self.rel2id.get(rel, self.rel2id[self.PAD])

    def char_to_id(self, ch: str) -> int:
        return self.char2id.get(ch, self.char2id[self.UNK])

    def id_to_rel(self, idx: int) -> str:
        return self.id2rel[idx] if 0 <= idx < len(self.id2rel) else self.UNK

    # ------------------------------------------------------------------
    # Size properties
    # ------------------------------------------------------------------
    @property
    def n_words(self) -> int: return len(self.id2word)

    @property
    def n_tags(self) -> int:  return len(self.id2tag)

    @property
    def n_rels(self) -> int:  return len(self.id2rel)

    @property
    def n_chars(self) -> int: return len(self.id2char)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"Vocabulary saved to: {path}")

    @classmethod
    def load(cls, path: str) -> 'Vocab':
        vocab = cls.__new__(cls)
        with open(path, 'rb') as f:
            vocab.__dict__.update(pickle.load(f))
        print(f"Vocabulary loaded from: {path}")
        return vocab
