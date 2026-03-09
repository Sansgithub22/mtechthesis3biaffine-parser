# modules/biaffine.py
# Core biaffine attention layer from Dozat & Manning (2017).
#
# The biaffine classifier is the central contribution of the paper.
# It scores every (dependent, head) word pair using:
#
#   ARC SCORING (out_features=1):
#       s(i,j) = h_i^T W h_j + h_j^T u
#              = bilinear term + linear term on the head
#   Equation (2) and (6) in the paper.
#
#   LABEL SCORING (out_features=n_rels):
#       s(i,j,l) = h_i^T W_l h_j + (h_i ⊕ h_j)^T u_l + b_l
#   Equation (3) in the paper.
#
# The key insight over plain MLP attention (Kiperwasser & Goldberg 2016):
#   - Directly models p(word j receives any dependent) via h_j^T u
#   - Directly models p(i→j arc) via the bilinear h_i^T W h_j term
#   - No nonlinearity needed → faster and less prone to overfitting

import torch
import torch.nn as nn


class Biaffine(nn.Module):
    """
    Biaffine attention / classifier.

    Parameters
    ----------
    in_features : int
        Dimension of both input representations (dep and head).
    out_features : int
        1 for arc scoring, n_rels for label scoring.
    bias_x : bool
        Whether to append a bias term to the dependent representation.
        (Implements the linear-on-dep part of the biaffine.)
    bias_y : bool
        Whether to append a bias term to the head representation.
        (Implements the linear-on-head part — key for arc prior probability.)

    Forward inputs
    --------------
    x : [batch, seq_len, in_features]   dependent representations
    y : [batch, seq_len, in_features]   head representations

    Forward output
    --------------
    [batch, seq_len, seq_len]              if out_features == 1
    [batch, seq_len, seq_len, out_features] otherwise
    """

    def __init__(self, in_features: int, out_features: int = 1,
                 bias_x: bool = True, bias_y: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.bias_x       = bias_x
        self.bias_y       = bias_y

        # Weight tensor shape: [out, in_x + bias_x, in_y + bias_y]
        self.weight = nn.Parameter(
            torch.Tensor(out_features,
                         in_features + int(bias_x),
                         in_features + int(bias_y))
        )
        nn.init.zeros_(self.weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Optionally append ones for bias terms
        if self.bias_x:
            x = torch.cat([x, x.new_ones(*x.shape[:-1], 1)], dim=-1)
            # x: [batch, seq, in+1]
        if self.bias_y:
            y = torch.cat([y, y.new_ones(*y.shape[:-1], 1)], dim=-1)
            # y: [batch, seq, in+1]

        # Core bilinear computation:
        # x: [batch, dep_len, in_x]
        # W: [out, in_x, in_y]
        # y: [batch, head_len, in_y]
        # → scores: [batch, out, dep_len, head_len]
        #
        # Step 1: x @ W → [batch, dep_len, out, in_y]  (via einsum)
        # Step 2: contract with y
        scores = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # scores: [batch, out_features, dep_len, head_len]

        if self.out_features == 1:
            # Remove the out_features dimension for arc scoring
            scores = scores.squeeze(1)             # [batch, dep, head]
        else:
            # Permute to [batch, dep, head, out_features] for label scoring
            scores = scores.permute(0, 2, 3, 1)   # [batch, dep, head, n_rels]

        return scores

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'bias_x={self.bias_x}, bias_y={self.bias_y}')
