from __future__ import annotations

import argparse
from pathlib import Path

from common import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage I convenience entrypoint. Use train_full_pipeline for the full ASWF protocol.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "aswf_base.yaml")
    parser.add_argument("--disease", type=str, default="all")
    return parser.parse_args()


def main() -> None:
    print(
        "Stage I is implemented in the ASWF trainers and executed by the full pipeline. "
        "Use `python scripts/train_full_pipeline.py --config ... --disease ...` to run the full two-stage protocol."
    )


if __name__ == "__main__":
    main()
