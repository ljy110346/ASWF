from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from ..datasets import PairedSpectraDataset
from ..datasets.paired_spectra_dataset import DiseaseDataBundle
from ..models import build_model
from ..models.wavelet import WaveletLevelInfo, compute_common_wavelet_level
from ..utils.config import ExperimentConfig
from ..utils.io import ensure_dir, write_csv, write_json
from ..utils.logger import create_logger
from ..utils.metrics import summarize_metric_series
from ..utils.seed import seed_everything
from ..visualization.confusion_matrix import plot_confusion_matrix
from ..visualization.curves import plot_stage_history
from ..visualization.embedding_vis import plot_embeddings
from ..visualization.roc import plot_pr_curve, plot_roc_curve
from ..visualization.scale_analysis import analyze_scale_contribution
from ..visualization.shared_private_analysis import analyze_shared_private_ratio
from ..visualization.wavenumber_backmap import build_wavenumber_backmap
from .stage1_trainer import Stage1Trainer
from .stage2_trainer import Stage2Trainer
from .utils import apply_preprocessing, build_fold_split, class_counts, resolve_device, subsample_indices_stratified


def _fold_name(repeat_id: int, fold_id: int) -> str:
    return f"repeat_{repeat_id:02d}_fold_{fold_id:02d}"


def run_cross_validation(data_bundle: DiseaseDataBundle, config: ExperimentConfig, output_dir: Path | str) -> Dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    logger = create_logger(f"aswf_{data_bundle.disease}", output_dir / "logs")
    device = resolve_device(config.device)
    labels = data_bundle.labels
    splitter = RepeatedStratifiedKFold(
        n_splits=config.cross_validation.n_splits,
        n_repeats=config.cross_validation.n_repeats,
        random_state=config.seed,
    )

    fold_results: List[Dict[str, Any]] = []
    model_param_count = None
    for split_index, (train_val_indices, test_indices) in enumerate(splitter.split(np.zeros_like(labels), labels)):
        if config.cross_validation.max_folds is not None and split_index >= config.cross_validation.max_folds:
            break

        repeat_id = split_index // config.cross_validation.n_splits
        fold_id = split_index % config.cross_validation.n_splits
        fold_seed = config.seed + split_index
        seed_everything(fold_seed)
        fold_dir = ensure_dir(output_dir / _fold_name(repeat_id, fold_id))

        train_indices, val_indices = build_fold_split(
            labels=labels,
            train_val_indices=train_val_indices,
            val_ratio=config.cross_validation.val_ratio,
            seed=fold_seed,
        )
        original_train_indices = train_indices
        train_indices = subsample_indices_stratified(
            labels=labels,
            indices=train_indices,
            ratio=config.cross_validation.train_subset_ratio,
            seed=fold_seed,
        )

        preprocessing = apply_preprocessing(
            ir=data_bundle.ir,
            raman=data_bundle.raman,
            train_indices=train_indices,
            config=config.preprocessing,
        )
        wavelet_info = _resolve_wavelet_info(config=config, data_bundle=data_bundle)
        if config.model.ablation != "no_wavelet" and wavelet_info.J_common < config.model.J_config:
            logger.info(
                "Wavelet depth adjusted | disease=%s repeat=%d fold=%d J_config=%d -> J_common=%d (J_ir_max=%d, J_raman_max=%d)",
                data_bundle.disease,
                repeat_id,
                fold_id,
                config.model.J_config,
                wavelet_info.J_common,
                wavelet_info.J_ir_max,
                wavelet_info.J_raman_max,
            )
        fold_model_config = replace(
            config.model,
            J_config=wavelet_info.J_common if config.model.ablation != "no_wavelet" else 0,
        )
        num_scales = 1 if config.model.ablation == "no_wavelet" else wavelet_info.J_common + 1
        model = build_model(config=fold_model_config, num_scales=num_scales)
        if model_param_count is None:
            model_param_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

        train_dataset = PairedSpectraDataset(preprocessing.ir, preprocessing.raman, labels, train_indices)
        val_dataset = PairedSpectraDataset(preprocessing.ir, preprocessing.raman, labels, val_indices)
        test_dataset = PairedSpectraDataset(preprocessing.ir, preprocessing.raman, labels, test_indices)

        checkpoint_dir = ensure_dir(fold_dir / "checkpoints")
        stage1_artifacts = None
        if config.model.ablation not in {"no_decomposition", "single_stage"}:
            logger.info("Stage I | disease=%s repeat=%d fold=%d", data_bundle.disease, repeat_id, fold_id)
            stage1_trainer = Stage1Trainer(model=model, config=config.training.stage1, loss_config=config.loss, device=device)
            stage1_artifacts = stage1_trainer.fit(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                checkpoint_path=checkpoint_dir / "stage1_best.pth",
            )
            plot_stage_history(stage1_artifacts.history, fold_dir / "curves" / "stage1_history.png", stage_name="stage1")

        logger.info("Stage II | disease=%s repeat=%d fold=%d", data_bundle.disease, repeat_id, fold_id)
        stage2_trainer = Stage2Trainer(
            model=model,
            config=config.training.stage2,
            loss_config=config.loss,
            evaluation_config=config.evaluation,
            device=device,
        )
        stage2_artifacts = stage2_trainer.fit(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            checkpoint_path=checkpoint_dir / "stage2_best.pth",
            stage1_checkpoint_path=Path(stage1_artifacts.best_checkpoint_path) if stage1_artifacts is not None else None,
        )
        plot_stage_history(stage2_artifacts.history, fold_dir / "curves" / "stage2_history.png", stage_name="stage2")

        test_eval = stage2_trainer.predict(test_dataset)
        fold_payload = {
            **_build_fold_metadata(
                data_bundle=data_bundle,
                repeat_id=repeat_id,
                fold_id=fold_id,
                train_indices=train_indices,
                original_train_indices=original_train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
                wavelet_info=wavelet_info,
                config=config,
                stage1_artifacts=stage1_artifacts,
                stage2_artifacts=stage2_artifacts,
                fold_dir=fold_dir,
            ),
            "val_metrics": stage2_artifacts.best_val_metrics,
            "test_metrics": test_eval["metrics"].to_dict(),
            "prediction_summary": {
                "positive_prob_mean": float(np.mean(test_eval["positive_prob"])),
                "positive_prob_std": float(np.std(test_eval["positive_prob"])),
            },
            "warnings": list(data_bundle.warnings),
        }
        write_json(fold_payload, fold_dir / "fold_metadata.json")
        fold_results.append(fold_payload)
        _save_fold_visuals(
            fold_dir=fold_dir,
            test_eval=test_eval,
            data_bundle=data_bundle,
            model=model,
            wavelet_info=wavelet_info,
        )

    metrics_frame = _build_metrics_frame(fold_results)
    write_csv(metrics_frame, output_dir / "fold_metrics.csv")
    summary = _build_summary(
        data_bundle=data_bundle,
        config=config,
        device=device,
        model_param_count=int(model_param_count or 0),
        fold_results=fold_results,
        metrics_frame=metrics_frame,
    )
    write_json(summary, output_dir / "summary.json")
    return summary


def _resolve_wavelet_info(config: ExperimentConfig, data_bundle: DiseaseDataBundle) -> WaveletLevelInfo:
    if config.model.ablation == "no_wavelet":
        return WaveletLevelInfo(
            wavelet_name=config.model.wavelet_name,
            wavelet_mode=config.model.wavelet_mode,
            J_config=config.model.J_config,
            J_ir_max=0,
            J_raman_max=0,
            J_common=0,
        )
    return compute_common_wavelet_level(
        ir_len=data_bundle.ir.shape[1],
        raman_len=data_bundle.raman.shape[1],
        wavelet_name=config.model.wavelet_name,
        mode=config.model.wavelet_mode,
        J_config=config.model.J_config,
    )


def _build_fold_metadata(
    data_bundle: DiseaseDataBundle,
    repeat_id: int,
    fold_id: int,
    train_indices: np.ndarray,
    original_train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    wavelet_info: WaveletLevelInfo,
    config: ExperimentConfig,
    stage1_artifacts,
    stage2_artifacts,
    fold_dir: Path,
) -> Dict[str, Any]:
    return {
        "disease_name": data_bundle.disease,
        "repeat_id": repeat_id,
        "fold_id": fold_id,
        "train_indices": train_indices.tolist(),
        "val_indices": val_indices.tolist(),
        "test_indices": test_indices.tolist(),
        "num_train": int(train_indices.shape[0]),
        "num_train_before_subsample": int(original_train_indices.shape[0]),
        "num_val": int(val_indices.shape[0]),
        "num_test": int(test_indices.shape[0]),
        "class_counts_train": class_counts(data_bundle.labels, train_indices),
        "class_counts_val": class_counts(data_bundle.labels, val_indices),
        "class_counts_test": class_counts(data_bundle.labels, test_indices),
        "chosen_sheet_ir": data_bundle.files.chosen_sheets.get("ir"),
        "chosen_sheet_raman": data_bundle.files.chosen_sheets.get("raman"),
        "wavenumber_path_ir": str(data_bundle.wave_numbers["ir"].path) if data_bundle.wave_numbers["ir"].path else None,
        "wavenumber_path_raman": str(data_bundle.wave_numbers["raman"].path) if data_bundle.wave_numbers["raman"].path else None,
        **wavelet_info.to_dict(),
        "stage1_best_metric": stage1_artifacts.best_metric if stage1_artifacts is not None else None,
        "stage2_best_metric": stage2_artifacts.best_metric,
        "threshold_policy": config.evaluation.threshold_policy,
        "tuned_threshold": stage2_artifacts.threshold,
        "train_subset_ratio": config.cross_validation.train_subset_ratio,
        "checkpoint_paths": {
            "stage1": stage1_artifacts.best_checkpoint_path if stage1_artifacts is not None else None,
            "stage2": stage2_artifacts.best_checkpoint_path,
        },
        "fold_dir": str(fold_dir),
    }


def _save_fold_visuals(fold_dir: Path, test_eval: Dict[str, Any], data_bundle: DiseaseDataBundle, model, wavelet_info: WaveletLevelInfo) -> None:
    feature_dir = ensure_dir(fold_dir / "features")
    np.savez(
        feature_dir / "test_features.npz",
        global_repr=test_eval["global_repr"],
        labels=test_eval["labels"],
        positive_prob=test_eval["positive_prob"],
        pred=test_eval["pred"],
    )
    plot_confusion_matrix(test_eval["labels"], test_eval["pred"], fold_dir / "confusion_matrix" / "test_confusion.png")
    plot_roc_curve(test_eval["labels"], test_eval["positive_prob"], fold_dir / "roc" / "test_roc.png")
    plot_pr_curve(test_eval["labels"], test_eval["positive_prob"], fold_dir / "roc" / "test_pr.png")
    plot_embeddings(test_eval["global_repr"], test_eval["labels"], fold_dir / "embeddings", prefix="test")
    analyze_scale_contribution(model, fold_dir / "interpretability" / "scale_contribution.json")
    analyze_shared_private_ratio(
        fused_shared=test_eval.get("fused_shared"),
        fused_private=test_eval.get("fused_private"),
        private_ir=test_eval.get("private_ir"),
        private_raman=test_eval.get("private_raman"),
        output_path=fold_dir / "interpretability" / "shared_private_ratio.json",
    )
    build_wavenumber_backmap(
        wave_numbers=data_bundle.wave_numbers,
        wavelet_info=wavelet_info,
        output_path=fold_dir / "interpretability" / "wavenumber_backmap.json",
    )


def _build_metrics_frame(fold_results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for fold in fold_results:
        rows.append(
            {
                "repeat": fold["repeat_id"],
                "fold": fold["fold_id"],
                **{key: value for key, value in fold["test_metrics"].items() if isinstance(value, (int, float)) or value is None},
            }
        )
    return pd.DataFrame(rows)


def _build_summary(
    data_bundle: DiseaseDataBundle,
    config: ExperimentConfig,
    device,
    model_param_count: int,
    fold_results: List[Dict[str, Any]],
    metrics_frame: pd.DataFrame,
) -> Dict[str, Any]:
    metric_summary = {
        column: summarize_metric_series(metrics_frame[column].tolist())
        for column in metrics_frame.columns
        if column not in {"repeat", "fold"} and metrics_frame[column].notna().any()
    }
    return {
        "dataset": data_bundle.summary(),
        "device": str(device),
        "model_param_count": model_param_count,
        "config": config.to_dict(),
        "fold_count": len(fold_results),
        "metric_summary": metric_summary,
        "warnings": list(data_bundle.warnings),
    }
