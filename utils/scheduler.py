# utils/scheduler.py
# Annealed Adam learning rate scheduler from Dozat & Manning (2017).
#
# Section 3.2 / Table 1 of the paper:
#   "We optimize the network with annealed Adam for about 50,000 steps"
#   annealing factor: 0.75 every 5000 steps
#   i.e., lr_t = lr_0 * (0.75 ^ (t / 5000))
#
# Section 4.2.5:
#   "We find that setting β2 to .9 instead of .999 makes a large positive
#    impact on final performance."
#
# This module provides:
#   1. get_base_optimizer()  — Adam with paper's β1=0.9, β2=0.9
#   2. AnnealingScheduler   — applies the exponential decay every step

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR


def get_base_optimizer(model_params, lr: float = 2e-3) -> Adam:
    """
    Adam optimizer with the hyperparameters from Table 1:
      β1 = 0.9, β2 = 0.9 (NOT the default 0.999 — see Section 4.2.5)
      ε  = 1e-8
    """
    return Adam(model_params, lr=lr, betas=(0.9, 0.9), eps=1e-8)


class AnnealingScheduler:
    """
    Applies exponential decay to the optimizer's learning rate.
    Called once per training step (batch).

    lr at step t = lr_0 * (decay_factor ^ (t / decay_steps))

    Paper defaults: decay_factor=0.75, decay_steps=5000
    """

    def __init__(self, optimizer, decay_factor: float = 0.75,
                 decay_steps: int = 5000):
        self.optimizer    = optimizer
        self.decay_factor = decay_factor
        self.decay_steps  = decay_steps
        self.step_count   = 0
        # Store the initial learning rates per parameter group
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self):
        """Call after each optimizer.step()."""
        self.step_count += 1
        factor = self.decay_factor ** (self.step_count / self.decay_steps)
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * factor

    def get_lr(self):
        """Returns the current learning rate of the first parameter group."""
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        return {
            'step_count':   self.step_count,
            'decay_factor': self.decay_factor,
            'decay_steps':  self.decay_steps,
            'base_lrs':     self.base_lrs,
        }

    def load_state_dict(self, state: dict):
        self.step_count   = state['step_count']
        self.decay_factor = state['decay_factor']
        self.decay_steps  = state['decay_steps']
        self.base_lrs     = state['base_lrs']


def get_bert_optimizer(model, bert_lr: float = 2e-5,
                       head_lr: float = 2e-3) -> Adam:
    """
    For BERT models: use separate learning rates for the BERT encoder
    and the task-specific head (MLP + biaffine layers).

    BERT fine-tuning typically uses a much smaller LR (2e-5) to avoid
    catastrophic forgetting, while the new layers use a larger LR (2e-3).
    """
    param_groups = model.get_param_groups(bert_lr=bert_lr, head_lr=head_lr)
    return Adam(param_groups, betas=(0.9, 0.9), eps=1e-8)
