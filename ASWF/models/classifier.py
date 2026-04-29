from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class MultiScaleClassifier(nn.Module):
    def __init__(self, per_scale_dim: int, num_scales: int, hidden_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        input_dim = per_scale_dim * num_scales
        self.input_dim = input_dim
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_classes)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, scale_features: List[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        global_repr = torch.cat(scale_features, dim=-1)
        hidden = self.dropout(self.activation(self.hidden(global_repr)))
        logits = self.output(hidden)
        return logits, global_repr

