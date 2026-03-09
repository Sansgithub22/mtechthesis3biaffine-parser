# modules/mlp.py
# Dimension-reducing MLP applied to BiLSTM outputs before biaffine scoring.
# Section 3.1 of Dozat & Manning (2017), Equations 4-5.
#
# Purpose: strips away information in the recurrent state that is irrelevant
# to the current decision (arc vs. label), reducing overfitting and speeding
# up training by operating in a lower-dimensional space.

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Single hidden-layer MLP with LeakyReLU activation and dropout.

    Architecture:
        Linear(in_features → out_features) → LeakyReLU → Dropout

    Used to project each LSTM hidden state into:
      - arc-dep  representation  (dim = arc_mlp_size,   default 500)
      - arc-head representation  (dim = arc_mlp_size,   default 500)
      - lbl-dep  representation  (dim = label_mlp_size, default 100)
      - lbl-head representation  (dim = label_mlp_size, default 100)
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.33):
        super().__init__()
        self.linear     = nn.Linear(in_features, out_features)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout    = nn.Dropout(p=dropout)

        # Xavier uniform initialization
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., in_features]
        returns: [..., out_features]
        """
        return self.dropout(self.activation(self.linear(x)))
