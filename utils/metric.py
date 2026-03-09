# utils/metric.py
# Evaluation metrics for dependency parsing.
#
# Standard metrics:
#   UAS (Unlabeled Attachment Score):
#       % of words whose predicted head is correct.
#   LAS (Labeled Attachment Score):
#       % of words whose predicted head AND relation label are both correct.
#
# Punctuation tokens (PUNCT, SYM) are typically excluded from evaluation
# on PTB datasets but included on UD datasets — controlled by `exclude_punct`.

import torch
from typing import Optional


class AttachmentMetric:
    """
    Accumulates UAS and LAS statistics across batches, then computes final
    scores. Call update() for each batch, then read .UAS and .LAS properties.
    """

    PUNCT_UPOS = {'PUNCT', 'SYM'}  # UD POS tags for punctuation

    def __init__(self):
        self.reset()

    def reset(self):
        self.n_total   = 0
        self.n_arc_ok  = 0   # correct arcs (for UAS)
        self.n_lbl_ok  = 0   # correct arcs + labels (for LAS)

    def update(
        self,
        pred_heads: torch.Tensor,    # [batch, seq_len]
        pred_rels:  torch.Tensor,    # [batch, seq_len]
        gold_heads: torch.Tensor,    # [batch, seq_len]
        gold_rels:  torch.Tensor,    # [batch, seq_len]
        mask:       torch.Tensor,    # [batch, seq_len] — True for real tokens
        punct_mask: Optional[torch.Tensor] = None,  # True for non-punct tokens
    ):
        """
        mask should be True for every real token including <root>, but <root>
        is excluded from scoring by zeroing out index 0 below.

        punct_mask (optional): True where token is NOT punctuation.
        If provided, punctuation tokens are excluded from scoring.
        """
        # Exclude the <root> token (index 0) — it has no head to predict
        eval_mask = mask.clone()
        eval_mask[:, 0] = False

        # Optionally exclude punctuation
        if punct_mask is not None:
            eval_mask = eval_mask & punct_mask

        # Arc correctness
        arc_correct   = (pred_heads == gold_heads) & eval_mask
        # Label correctness: arc must be correct first
        label_correct = arc_correct & (pred_rels == gold_rels)

        self.n_total  += eval_mask.sum().item()
        self.n_arc_ok += arc_correct.sum().item()
        self.n_lbl_ok += label_correct.sum().item()

    @property
    def UAS(self) -> float:
        if self.n_total == 0:
            return 0.0
        return 100.0 * self.n_arc_ok / self.n_total

    @property
    def LAS(self) -> float:
        if self.n_total == 0:
            return 0.0
        return 100.0 * self.n_lbl_ok / self.n_total

    def __repr__(self) -> str:
        return (f"UAS: {self.UAS:.2f}%  |  LAS: {self.LAS:.2f}%  "
                f"({self.n_arc_ok}/{self.n_total} arcs correct)")

    def result(self) -> dict:
        return {
            'UAS': self.UAS,
            'LAS': self.LAS,
            'n_correct_arc':   self.n_arc_ok,
            'n_correct_label': self.n_lbl_ok,
            'n_total':         self.n_total,
        }


class LossMetric:
    """
    Tracks average loss across batches.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.n_batches  = 0

    def update(self, loss: float):
        self.total_loss += loss
        self.n_batches  += 1

    @property
    def avg(self) -> float:
        return self.total_loss / max(self.n_batches, 1)

    def __repr__(self) -> str:
        return f"Loss: {self.avg:.4f}"
