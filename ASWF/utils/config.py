from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments.
    yaml = None


ThresholdPolicy = Literal["argmax", "fixed_0.5"]
AblationMode = Literal[
    "none",
    "no_decomposition",
    "no_asymmetric",
    "single_stage",
    "no_stability",
    "no_wavelet",
    "symmetric_fusion",
    "private_average",
]
AxisMismatchPolicy = Literal["warn", "error"]
NormalizationMode = Literal["none", "feature_zscore"]


@dataclass(frozen=True)
class DiseaseFileMapping:
    ir_normal: Optional[str] = None
    raman_normal: Optional[str] = None
    ir_abnormal: Optional[str] = None
    raman_abnormal: Optional[str] = None
    wavenumber_ir: Optional[str] = None
    wavenumber_raman: Optional[str] = None


@dataclass(frozen=True)
class DiseaseSheetOverride:
    ir: Optional[str] = None
    raman: Optional[str] = None
    wavenumber_ir: Optional[str] = None
    wavenumber_raman: Optional[str] = None


@dataclass(frozen=True)
class DataConfig:
    root: str = "data"
    diseases: List[str] = field(default_factory=lambda: ["example_disease"])
    file_mappings: Dict[str, DiseaseFileMapping] = field(default_factory=dict)
    sheet_overrides: Dict[str, DiseaseSheetOverride] = field(default_factory=dict)
    discovery_keywords: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "ir": ["ir", "infrared"],
            "raman": ["raman"],
            "normal": ["normal", "healthy", "health"],
            "abnormal": ["abnormal", "disease", "patient", "cancer"],
        }
    )
    axis_mismatch_policy: AxisMismatchPolicy = "warn"


@dataclass(frozen=True)
class CrossValidationConfig:
    n_splits: int = 5
    n_repeats: int = 5
    val_ratio: float = 0.2
    max_folds: Optional[int] = None
    train_subset_ratio: Optional[float] = None


@dataclass(frozen=True)
class PreprocessingConfig:
    spectral_normalization: NormalizationMode = "none"
    normalization_eps: float = 1e-6


@dataclass(frozen=True)
class EncoderConfig:
    hidden_channels: int = 32
    num_layers: int = 2
    kernel_size: int = 5
    pooled_length: int = 8
    projection_dim: int = 128
    dropout: float = 0.1


@dataclass(frozen=True)
class ModelConfig:
    wavelet_name: str = "db1"
    wavelet_mode: str = "zero"
    J_config: int = 3
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    shared_dim: int = 64
    private_dim: int = 64
    fusion_hidden_dim: int = 64
    classifier_hidden_dim: int = 128
    num_classes: int = 2
    perturbed_noise_std: float = 0.05
    ablation: AblationMode = "none"


@dataclass(frozen=True)
class LossConfig:
    lambda_cons: float = 1.0
    lambda_orth: float = 0.5
    lambda_stab: float = 0.2


@dataclass(frozen=True)
class StageTrainingConfig:
    batch_size: int = 16
    max_epochs: int = 120
    patience: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    grad_clip: Optional[float] = None
    scheduler: Literal["cosine", "plateau", "none"] = "cosine"
    min_delta: float = 1e-6


@dataclass(frozen=True)
class TrainingConfig:
    stage1: StageTrainingConfig = field(default_factory=StageTrainingConfig)
    stage2: StageTrainingConfig = field(default_factory=StageTrainingConfig)


@dataclass(frozen=True)
class EvaluationConfig:
    threshold_policy: ThresholdPolicy = "argmax"
    tune_threshold: bool = False
    threshold_metric: Literal["bacc", "f1", "mcc"] = "bacc"


@dataclass(frozen=True)
class OutputConfig:
    root: str = "outputs"
    experiment_name: str = "aswf_main"


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    cross_validation: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_data_config(raw: Dict[str, Any]) -> DataConfig:
    file_mappings = {key: DiseaseFileMapping(**value) for key, value in raw.get("file_mappings", {}).items()}
    sheet_overrides = {key: DiseaseSheetOverride(**value) for key, value in raw.get("sheet_overrides", {}).items()}
    top_level = {key: value for key, value in raw.items() if key not in {"file_mappings", "sheet_overrides"}}
    return DataConfig(**top_level, file_mappings=file_mappings, sheet_overrides=sheet_overrides)


def _build_training_config(raw: Dict[str, Any]) -> TrainingConfig:
    return TrainingConfig(
        stage1=StageTrainingConfig(**raw.get("stage1", {})),
        stage2=StageTrainingConfig(**raw.get("stage2", {})),
    )


def _build_model_config(raw: Dict[str, Any]) -> ModelConfig:
    encoder = EncoderConfig(**raw.get("encoder", {}))
    top_level = {key: value for key, value in raw.items() if key != "encoder"}
    return ModelConfig(**top_level, encoder=encoder)


def _build_experiment_config(raw: Dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        seed=raw.get("seed", 42),
        device=raw.get("device", "auto"),
        data=_build_data_config(raw.get("data", {})),
        cross_validation=CrossValidationConfig(**raw.get("cross_validation", {})),
        preprocessing=PreprocessingConfig(**raw.get("preprocessing", {})),
        model=_build_model_config(raw.get("model", {})),
        loss=LossConfig(**raw.get("loss", {})),
        training=_build_training_config(raw.get("training", {})),
        evaluation=EvaluationConfig(**raw.get("evaluation", {})),
        output=OutputConfig(**raw.get("output", {})),
    )


def load_config(path: Path | str) -> ExperimentConfig:
    path = Path(path)
    raw_text = path.read_text(encoding="utf-8")
    if yaml is not None:
        raw = yaml.safe_load(raw_text) or {}
    else:
        if path.suffix.lower() in {".yaml", ".yml"}:
            raise RuntimeError("PyYAML is required to load YAML configuration files. Install dependencies with `pip install -r requirements.txt`.")
        raw = json.loads(raw_text or "{}")
    if "extends" in raw:
        base_path = (path.parent / raw["extends"]).resolve()
        base_config = load_config(base_path).to_dict()
        raw = _merge_dict(base_config, {key: value for key, value in raw.items() if key != "extends"})
    merged = _merge_dict(ExperimentConfig().to_dict(), raw)
    return _build_experiment_config(merged)


def save_config(config: ExperimentConfig, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if yaml is not None:
            yaml.safe_dump(config.to_dict(), handle, sort_keys=False, allow_unicode=True)
        else:
            json.dump(config.to_dict(), handle, indent=2, ensure_ascii=False)
