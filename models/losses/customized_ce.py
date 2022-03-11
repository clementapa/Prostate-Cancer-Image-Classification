from typing import Optional
from functools import partial

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from .segmentation.constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE

__all__ = ["C_Crossentropy"]


class C_Crossentropy(_Loss):
    def __init__(self, alpha_=0.4):
        super().__init__()
        self.alpha_ = alpha_

    def forward(self, y_pred_tuple: tuple([torch.Tensor, torch.Tensor]), y_true: torch.Tensor) -> torch.Tensor:
        y_pred, y_proba = y_pred_tuple
        bce_term = nn.BCELoss()(y_proba, torch.ones_like(y_proba))
        ce_term = nn.CrossEntropyLoss()(y_pred, y_true)
        return self.alpha_*bce_term + ce_term