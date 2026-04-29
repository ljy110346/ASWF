from __future__ import annotations

import torch
import torch.nn.functional as F


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)

