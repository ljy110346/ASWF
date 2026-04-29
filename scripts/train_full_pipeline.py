from __future__ import annotations

import argparse
from pathlib import Path

from common import PROJECT_ROOT, aggregate_experiment_results, load_experiment_config, resolve_diseases, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full ASWF repeated-CV pipeline.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "aswf_base.yaml")
    parser.add_argument("--disease", type=str, default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    experiment_dir = PROJECT_ROOT / config.output.root / config.output.experiment_name
    for disease in resolve_diseases(config, args.disease):
        result = run_experiment(config, disease)
        print(f"[{disease}] saved to {result['output_dir']}")
    aggregate_experiment_results(experiment_dir)
    print(f"aggregate summary saved to {experiment_dir / 'aggregate_results.json'}")


if __name__ == "__main__":
    main()
