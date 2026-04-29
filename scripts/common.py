from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ASWF.datasets import load_disease_bundle
from ASWF.trainers import run_cross_validation
from ASWF.utils.config import ExperimentConfig, load_config
from ASWF.utils.io import ensure_dir, read_json, write_json


def resolve_diseases(config: ExperimentConfig, disease: str) -> List[str]:
    if disease == "all":
        return list(config.data.diseases)
    return [disease]


def run_experiment(config: ExperimentConfig, disease: str) -> Dict[str, object]:
    bundle = load_disease_bundle(data_root=PROJECT_ROOT / config.data.root, disease=disease, config=config)
    output_dir = PROJECT_ROOT / config.output.root / config.output.experiment_name / disease
    summary = run_cross_validation(data_bundle=bundle, config=config, output_dir=output_dir)
    return {"disease": disease, "summary": summary, "output_dir": str(output_dir)}


def aggregate_experiment_results(experiment_dir: Path | str) -> Dict[str, object]:
    experiment_dir = Path(experiment_dir)
    disease_summaries = {}
    for path in sorted(experiment_dir.iterdir()):
        if not path.is_dir():
            continue
        summary_path = path / "summary.json"
        if summary_path.is_file():
            disease_summaries[path.name] = read_json(summary_path)

    main_results = {}
    total_samples = 0
    ir_dims = []
    raman_dims = []
    fold_counts = []
    model_params = None
    for disease, summary in disease_summaries.items():
        metrics = summary["metric_summary"]
        dataset = summary["dataset"]
        main_results[disease] = {
            "proposed_method": {
                "bacc": metrics.get("bacc", {}).get("mean"),
                "bacc_std": metrics.get("bacc", {}).get("std"),
                "auc": metrics.get("auc", {}).get("mean"),
                "auc_std": metrics.get("auc", {}).get("std"),
                "mcc": metrics.get("mcc", {}).get("mean"),
                "mcc_std": metrics.get("mcc", {}).get("std"),
                "sensitivity": metrics.get("sensitivity", {}).get("mean"),
                "specificity": metrics.get("specificity", {}).get("mean"),
            }
        }
        total_samples += int(dataset["num_samples"])
        ir_dims.append(int(dataset["modality_dims"]["ir"]))
        raman_dims.append(int(dataset["modality_dims"]["raman"]))
        fold_counts.append(int(summary["fold_count"]))
        model_params = int(summary["model_param_count"])

    macro_metrics = {"bacc": [], "auc": [], "mcc": [], "f1": []}
    for summary in disease_summaries.values():
        metrics = summary["metric_summary"]
        macro_metrics["bacc"].append(metrics.get("bacc", {}).get("mean"))
        macro_metrics["auc"].append(metrics.get("auc", {}).get("mean"))
        macro_metrics["mcc"].append(metrics.get("mcc", {}).get("mean"))
        macro_metrics["f1"].append(metrics.get("f1", {}).get("mean"))

    protocol = next(iter(disease_summaries.values()))["config"]["cross_validation"] if disease_summaries else {}
    payload = {
        "schema_version": "1.0",
        "main_results": main_results,
        "secondary_results": {
            "proposed_method_overall": {
                "macro_bacc": _safe_mean(macro_metrics["bacc"]),
                "macro_auc": _safe_mean(macro_metrics["auc"]),
                "macro_mcc": _safe_mean(macro_metrics["mcc"]),
                "macro_f1": _safe_mean(macro_metrics["f1"]),
                "model_params": model_params,
            },
            "evaluation_protocol": {
                "cv_n_splits": protocol.get("n_splits"),
                "cv_n_repeats": protocol.get("n_repeats"),
                "cv_total_folds": _resolve_total_folds(protocol),
                "val_ratio": protocol.get("val_ratio"),
                "train_subset_ratio": protocol.get("train_subset_ratio"),
                "batch_size": next(iter(disease_summaries.values()))["config"]["training"]["stage2"]["batch_size"] if disease_summaries else None,
                "max_epochs": next(iter(disease_summaries.values()))["config"]["training"]["stage2"]["max_epochs"] if disease_summaries else None,
            },
            "data_overview": {
                "num_datasets": len(disease_summaries),
                "total_samples": total_samples,
                "mean_ir_dim": _safe_mean(ir_dims),
                "mean_raman_dim": _safe_mean(raman_dims),
            },
        },
        "model_variant_results": {},
        "additional_findings": [],
        "runtime_stats": {},
    }
    write_json(payload, experiment_dir / "aggregate_results.json")
    return payload


def load_experiment_config(config_path: Path | str) -> ExperimentConfig:
    return load_config(config_path)


def _safe_mean(values: Iterable[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def _resolve_total_folds(protocol: Dict[str, object]) -> int | None:
    if not protocol:
        return None
    total = int(protocol.get("n_splits", 0)) * int(protocol.get("n_repeats", 0))
    max_folds = protocol.get("max_folds")
    if max_folds is None:
        return total
    return min(total, int(max_folds))
