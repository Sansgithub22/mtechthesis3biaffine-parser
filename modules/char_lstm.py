# modules/char_lstm.py
# INNOVATION 1: Character-level BiLSTM encoder.
#
# Motivation (not in Dozat & Manning 2017):
#   - The original paper uses only word-level embeddings.
#   - Words not seen ≥2 times in training are mapped to <unk>.
#   - Character embeddings can reconstruct useful representations for:
#       * Rare and out-of-vocabulary words
#       * Morphologically rich languages (Czech, German, Turkish)
#       * Named entities and numbers
#
# Architecture:
#   For each word: embed each character → BiLSTM → take final hidden state
#   This produces a character-based word representation that is concatenated
#   with the word and POS embeddings before the BiLSTM encoder.
#
# References:
#   - Lample et al. (2016) "Neural Architectures for Named Entity Recognition"
#   - Ma & Hovy (2016) "End-to-end Sequence Labeling via BiLSTM-CRF"

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class CharLSTM(nn.Module):
    """
    Character-level BiLSTM that produces a fixed-size vector for each word.

    Parameters
    ----------
    n_chars    : vocabulary size (number of unique characters)
    embed_dim  : character embedding dimension (default 50)
    out_dim    : output dimension per word = BiLSTM hidden * 2 (default 100)
    dropout    : applied to character embeddings before LSTM

    Forward
    -------
    Input:  chars [batch, seq_len, max_char_len]  — padded character ids
    Output: [batch, seq_len, out_dim]             — one vector per word
    """

    def __init__(self, n_chars: int, embed_dim: int = 50,
                 out_dim: int = 100, dropout: float = 0.33):
        super().__init__()
        assert out_dim % 2 == 0, "out_dim must be even (split across two directions)"

        self.embed   = nn.Embedding(n_chars, embed_dim, padding_idx=0)
        self.lstm    = nn.LSTM(
            input_size=embed_dim,
            hidden_size=out_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(p=dropout)

        nn.init.uniform_(self.embed.weight, -0.5 / embed_dim, 0.5 / embed_dim)
        nn.init.zeros_(self.embed.weight[0])  # padding embedding = 0

    def forward(self, chars: torch.Tensor) -> torch.Tensor:
        """
        chars: [batch, seq_len, max_char_len]

        Strategy:
          1. Reshape to [batch * seq_len, max_char_len] to process all words together
          2. Embed characters
          3. Run BiLSTM, take the final hidden state (concatenate both directions)
          4. Reshape back to [batch, seq_len, out_dim]
        """
        batch, seq_len, max_char = chars.shape

        # Flatten: treat every word as an independent sequence
        chars_flat = chars.view(batch * seq_len, max_char)
        # chars_flat: [batch*seq_len, max_char_len]

        # Compute actual character lengths (non-padding positions)
        # clamp(min=1) prevents zero-length sequences from crashing pack
        lengths = chars_flat.ne(0).sum(dim=-1).clamp(min=1).cpu()
        # lengths: [batch*seq_len]

        # Character embeddings with dropout
        char_emb = self.dropout(self.embed(chars_flat))
        # char_emb: [batch*seq_len, max_char_len, embed_dim]

        # Pack → LSTM → final hidden state
        packed = pack_padded_sequence(
            char_emb, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n: [2, batch*seq_len, out_dim//2]
        #   h_n[0] = forward final state
        #   h_n[1] = backward final state

        # Concatenate both directions → one vector per word
        word_repr = torch.cat([h_n[0], h_n[1]], dim=-1)
        # word_repr: [batch*seq_len, out_dim]

        # Reshape back to sentence structure
        word_repr = word_repr.view(batch, seq_len, -1)
        # word_repr: [batch, seq_len, out_dim]

        return word_repr
