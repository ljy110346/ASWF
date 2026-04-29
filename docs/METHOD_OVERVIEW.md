# Method Overview

ASWF is a two-stage multimodal spectral classification framework.

## Stage I: Evidence Boundary Learning

Stage I trains the wavelet decomposition, subband encoders, and shared-private decomposition modules with auxiliary evidence objectives:

- consistency loss encourages modality-invariant shared evidence
- orthogonality loss separates shared and private evidence
- stability loss improves robustness under perturbation

This stage is intended to stabilize the shared-private evidence boundary before supervised classification is emphasized.

## Stage II: Task-Driven Fine-Tuning

Stage II initializes from the Stage I checkpoint and optimizes the classification objective together with the decomposition objectives. The classifier receives fused multi-scale evidence and predicts the binary class label.

## Main Model Modules

- `WaveletDecomposer1D`: multi-scale one-dimensional wavelet decomposition
- `ScaleSharedEncoder`: scale-wise subband feature encoder
- `SharedPrivateDecomposer`: shared-private evidence decomposition
- `DifferenceAwarePrivateFusion`: asymmetric private-evidence fusion
- `MultiScaleClassifier`: final classifier over fused evidence

## Interpretation Utilities

The repository includes helper functions for scale contribution, shared/private evidence ratio, and approximate wavenumber back-mapping. These utilities are designed for model analysis and should not be interpreted as definitive biochemical biomarker discovery without external validation.
