# ASWF

ASWF is a PyTorch implementation of an asymmetric shared-private wavelet fusion framework for paired one-dimensional spectra, such as infrared and Raman measurements. The repository contains the model, training loop, evaluation utilities, and visualization helpers needed to run the method on private or public paired spectral datasets.

This public version intentionally does not include any dataset, trained checkpoint, generated output, or manuscript-specific experiment artifact.

## Method Overview

ASWF is designed for paired multimodal spectral classification. The main components are:

- Multi-scale wavelet decomposition for each modality.
- Scale-wise subband encoders for infrared and Raman spectra.
- Shared-private evidence decomposition.
- Consistency, orthogonality, and stability objectives for evidence boundary learning.
- Asymmetric fusion of modality-private evidence.
- Two-stage optimization:
  - Stage I learns stable shared-private evidence boundaries.
  - Stage II performs task-driven classification fine-tuning.

## Repository Layout

```text
ASWF/
├── ASWF/                  # Python package
│   ├── datasets/          # Paired spectral dataset loading and preprocessing
│   ├── losses/            # Classification and evidence decomposition losses
│   ├── models/            # ASWF model modules
│   ├── trainers/          # Stage I, Stage II, and cross-validation runners
│   ├── utils/             # Config, metrics, checkpoints, logging
│   └── visualization/     # Curves, ROC, embeddings, interpretation helpers
├── configs/               # Example configuration files
├── data/                  # Empty placeholder; users provide their own private data
├── docs/                  # Data format and method notes
├── scripts/               # Training, evaluation, and feature export entry points
├── requirements.txt
└── pyproject.toml
```

## Installation

Create a clean Python environment and install dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA version from the official PyTorch installation page before running the project.

## Data Privacy Notice

No data are included in this repository. Place your own paired spectra under `data/` following the format described in [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md).

At minimum, each dataset folder should contain four matrices:

- healthy infrared spectra
- healthy Raman spectra
- disease infrared spectra
- disease Raman spectra

Optional wavenumber axis files can also be provided.

## Quick Start

1. Prepare your data under `data/example_disease/`.
2. Edit `configs/aswf_base.yaml` so that `data.diseases` and `data.file_mappings` match your files.
3. Run the full two-stage cross-validation pipeline:

```bash
python scripts/train_full_pipeline.py --config configs/aswf_base.yaml --disease all
```

For a short CPU-only configuration:

```bash
python scripts/train_full_pipeline.py --config configs/aswf_smoke.yaml --disease example_disease
```

The smoke configuration still requires user-provided data. It only reduces folds, epochs, and model size.

## Useful Commands

Train all datasets listed in a config:

```bash
python scripts/train_full_pipeline.py --config configs/aswf_base.yaml --disease all
```

Train one dataset:

```bash
python scripts/train_full_pipeline.py --config configs/aswf_base.yaml --disease example_disease
```

Aggregate an existing run:

```bash
python scripts/evaluate.py --experiment-dir outputs/aswf_main
```

Export saved fold-level features:

```bash
python scripts/export_features.py --experiment-dir outputs/aswf_main
```

## Output Policy

Runtime outputs are written to `outputs/` and are ignored by Git. Generated files may include:

- fold metadata
- model checkpoints
- training curves
- ROC and PR curves
- confusion matrices
- embedding visualizations
- interpretation JSON files
- aggregate metric summaries

Do not commit private data, trained checkpoints, or generated outputs unless you explicitly intend to publish them.

## Citation

If you use this implementation in academic work, cite the corresponding ASWF manuscript or repository once a formal citation is available.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
