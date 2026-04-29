from __future__ import annotations

import logging
from pathlib import Path

from .io import ensure_dir


def create_logger(name: str, log_dir: Path | str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        log_dir = ensure_dir(log_dir)
        file_handler = logging.FileHandler(Path(log_dir) / f"{name}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger

