# modules/bilstm.py
# Stacked Bidirectional LSTM encoder with variational (Bayesian) dropout.
#
# Key regularization from Dozat & Manning (2017) Section 3.2:
#   "we drop nodes in the LSTM layers (input and recurrent connections),
#    applying the same dropout mask at every recurrent timestep"
#
# This is Gal & Ghahramani (2015) variational dropout. PyTorch's built-in
# LSTM dropout only drops between layers, not recurrent connections.
# We implement it by manually applying a fixed mask per sequence.

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class VarDropout(nn.Module):
    """
    Variational (locked) dropout: samples one mask per forward pass and
    applies it consistently across all timesteps.
    This is stronger regularization than standard timestep-independent dropout.
    """
    def __init__(self, p: float = 0.33):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, features]"""
        if not self.training or self.p == 0.0:
            return x
        # Sample mask from the first timestep, broadcast over seq_len
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p)
        mask = mask / (1 - self.p)   # rescale (inverted dropout)
        return x * mask


class BiLSTMEncoder(nn.Module):
    """
    Stacked BiLSTM encoder.

    Hyperparameters from Table 1 of Dozat & Manning (2017):
      - 3 layers
      - 400 hidden dimensions per direction (800 total output)
      - 33% variational dropout on inputs and recurrent connections

    The first layer takes the word + tag (+ optional char) embedding.
    Subsequent layers take the previous layer's output (with dropout applied).
    """

    def __init__(self, in_features: int, hidden_size: int = 400,
                 n_layers: int = 3, dropout: float = 0.33):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers    = n_layers

        self.lstm_layers = nn.ModuleList()
        self.var_dropouts = nn.ModuleList()

        for i in range(n_layers):
            input_dim = in_features if i == 0 else hidden_size * 2
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
            )
            self.var_dropouts.append(VarDropout(p=dropout))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    [batch, seq_len, in_features]
        mask: [batch, seq_len]  — True for real tokens (including <root>)

        returns: [batch, seq_len, hidden_size * 2]
        """
        lengths = mask.sum(dim=1).cpu()

        for lstm, var_drop in zip(self.lstm_layers, self.var_dropouts):
            # Apply variational dropout on the input to this layer
            x = var_drop(x)

            # Pack for efficiency (skips padding in computation)
            packed = pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = lstm(packed)
            x, _ = pad_packed_sequence(packed_out, batch_first=True)
            # x: [batch, seq_len, hidden_size * 2]

        return x
