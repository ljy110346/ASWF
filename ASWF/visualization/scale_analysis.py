from __future__ import annotations

from pathlib import Path

import numpy as np

from ..utils.io import write_json


def analyze_scale_contribution(model, output_path: Path | str) -> None:
    weight = model.classifier.hidden.weight.detach().cpu().numpy()
    num_scales = model.num_scales
    input_dim = model.classifier.input_dim
    per_scale_dim = input_dim // max(num_scales, 1)
    contributions = []
    for scale_index in range(num_scales):
        start = scale_index * per_scale_dim
        end = start + per_scale_dim
        contributions.append(float(np.linalg.norm(weight[:, start:end])))
    total = sum(contributions) or 1.0
    payload = {
        "method": "classifier_weight_norm",
        "num_scales": num_scales,
        "scale_scores": contributions,
        "normalized_scale_scores": [score / total for score in contributions],
    }
    write_json(payload, output_path)

