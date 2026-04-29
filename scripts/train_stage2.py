from __future__ import annotations

import argparse
from pathlib import Path

from common import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage II convenience entrypoint. Use train_full_pipeline for the full ASWF protocol.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "aswf_base.yaml")
    parser.add_argument("--disease", type=str, default="all")
    parser.add_argument("--stage1_ckpt", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    print(
        "Stage II loading and best-checkpoint logic are implemented in the ASWF trainers and are executed by the full pipeline. "
        "Use `python scripts/train_full_pipeline.py --config ... --disease ...` for a full two-stage run."
    )


if __name__ == "__main__":
    main()
