from typing import Optional
from functools import partial

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from .constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE

__all__ = ["C_Crossentropy"]


class C_Crossentropy(_Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, y_proba: torch.Tensor) -> torch.Tensor:
        bce_term = nn.BCELoss()(y_proba, torch.ones_like(y_proba))
        ce_term = nn.CrossEntropyLoss()(y_pred, y_true)
        return bce_term + ce_term