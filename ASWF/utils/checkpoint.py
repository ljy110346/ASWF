from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(payload: Dict[str, Any], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path | str, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)

