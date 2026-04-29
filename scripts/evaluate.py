from __future__ import annotations

import argparse
from pathlib import Path

from common import PROJECT_ROOT, aggregate_experiment_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate existing ASWF outputs into a compact JSON summary.")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir
    if not experiment_dir.is_absolute():
        experiment_dir = PROJECT_ROOT / experiment_dir
    payload = aggregate_experiment_results(experiment_dir)
    print(f"aggregated {len(payload['main_results'])} datasets into {experiment_dir / 'aggregate_results.json'}")


if __name__ == "__main__":
    main()
