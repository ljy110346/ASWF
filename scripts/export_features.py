from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect saved fold-level feature files into one NPZ archive.")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-name", type=str, default="exported_features.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir
    if not experiment_dir.is_absolute():
        experiment_dir = PROJECT_ROOT / experiment_dir

    features = []
    labels = []
    probs = []
    for npz_path in sorted(experiment_dir.glob("*/*/features/test_features.npz")):
        blob = np.load(npz_path)
        features.append(blob["global_repr"])
        labels.append(blob["labels"])
        probs.append(blob["positive_prob"])

    if not features:
        raise FileNotFoundError(f"No saved fold features found under {experiment_dir}")

    output_path = experiment_dir / args.output_name
    np.savez(
        output_path,
        global_repr=np.concatenate(features, axis=0),
        labels=np.concatenate(labels, axis=0),
        positive_prob=np.concatenate(probs, axis=0),
    )
    print(f"saved consolidated features to {output_path}")


if __name__ == "__main__":
    main()

