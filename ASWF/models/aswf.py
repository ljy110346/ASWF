from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from ..utils.config import ModelConfig
from .asymmetric_fusion import AverageFusion, ConservativeSharedFusion, DifferenceAwarePrivateFusion, UnifiedFusion
from .classifier import MultiScaleClassifier
from .decomposition import SharedPrivateDecomposer
from .subband_encoder import ScaleSharedEncoder
from .wavelet import WaveletDecomposer1D


class ASWF(nn.Module):
    def __init__(self, config: ModelConfig, num_scales: int) -> None:
        super().__init__()
        self.config = config
        self.num_scales = num_scales
        self.use_wavelet = config.ablation != "no_wavelet"
        self.use_decomposition = config.ablation != "no_decomposition"
        self.use_stability = config.ablation != "no_stability"
        self.fusion_mode = self._resolve_fusion_mode(config.ablation)

        self.wavelet = WaveletDecomposer1D(
            wavelet=config.wavelet_name,
            level=max(num_scales - 1, 1),
            mode=config.wavelet_mode,
            use_wavelet=self.use_wavelet,
        )

        self.encoders = nn.ModuleList(
            [
                ScaleSharedEncoder(
                    hidden_channels=config.encoder.hidden_channels,
                    projection_dim=config.encoder.projection_dim,
                    num_layers=config.encoder.num_layers,
                    kernel_size=config.encoder.kernel_size,
                    pooled_length=config.encoder.pooled_length,
                    dropout=config.encoder.dropout,
                )
                for _ in range(num_scales)
            ]
        )

        if self.use_decomposition:
            self.decomposers = nn.ModuleList(
                [
                    SharedPrivateDecomposer(
                        in_dim=config.encoder.projection_dim,
                        shared_dim=config.shared_dim,
                        private_dim=config.private_dim,
                        dropout=config.encoder.dropout,
                    )
                    for _ in range(num_scales)
                ]
            )
            if self.fusion_mode == "asymmetric":
                self.shared_fusions = nn.ModuleList(
                    [
                        ConservativeSharedFusion(feat_dim=config.shared_dim, hidden_dim=config.fusion_hidden_dim)
                        for _ in range(num_scales)
                    ]
                )
                self.private_fusions = nn.ModuleList(
                    [
                        DifferenceAwarePrivateFusion(feat_dim=config.private_dim, hidden_dim=config.fusion_hidden_dim)
                        for _ in range(num_scales)
                    ]
                )
            elif self.fusion_mode == "unified":
                self.shared_fusions = nn.ModuleList(
                    [
                        UnifiedFusion(feat_dim=config.shared_dim, hidden_dim=config.fusion_hidden_dim)
                        for _ in range(num_scales)
                    ]
                )
                self.private_fusions = nn.ModuleList(
                    [
                        UnifiedFusion(feat_dim=config.private_dim, hidden_dim=config.fusion_hidden_dim)
                        for _ in range(num_scales)
                    ]
                )
            elif self.fusion_mode == "shared_module":
                if config.shared_dim != config.private_dim:
                    raise ValueError("symmetric_fusion requires shared_dim == private_dim.")
                self.shared_fusions = nn.ModuleList(
                    [
                        UnifiedFusion(feat_dim=config.shared_dim, hidden_dim=config.fusion_hidden_dim)
                        for _ in range(num_scales)
                    ]
                )
                self.private_fusions = nn.ModuleList()
            elif self.fusion_mode == "private_average":
                self.shared_fusions = nn.ModuleList(
                    [
                        ConservativeSharedFusion(feat_dim=config.shared_dim, hidden_dim=config.fusion_hidden_dim)
                        for _ in range(num_scales)
                    ]
                )
                self.private_fusions = nn.ModuleList([AverageFusion() for _ in range(num_scales)])
            else:
                raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")
            per_scale_dim = config.shared_dim + config.private_dim
        else:
            self.decomposers = nn.ModuleList()
            self.shared_fusions = nn.ModuleList(
                [
                    UnifiedFusion(
                        feat_dim=config.encoder.projection_dim,
                        hidden_dim=config.fusion_hidden_dim,
                        out_dim=config.encoder.projection_dim,
                    )
                    for _ in range(num_scales)
                ]
            )
            self.private_fusions = nn.ModuleList()
            per_scale_dim = config.encoder.projection_dim

        self.classifier = MultiScaleClassifier(
            per_scale_dim=per_scale_dim,
            num_scales=num_scales,
            hidden_dim=config.classifier_hidden_dim,
            num_classes=config.num_classes,
        )

    def boundary_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        params.extend(list(self.encoders.parameters()))
        params.extend(list(self.decomposers.parameters()))
        return params

    def utility_parameters(self) -> List[nn.Parameter]:
        unique: dict[int, nn.Parameter] = {}
        for parameter in self.shared_fusions.parameters():
            unique[id(parameter)] = parameter
        for parameter in self.private_fusions.parameters():
            unique[id(parameter)] = parameter
        return list(unique.values())

    def classifier_parameters(self) -> List[nn.Parameter]:
        return list(self.classifier.parameters())

    def set_stage1_trainable(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.boundary_parameters():
            parameter.requires_grad = True

    def set_stage2_trainable(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = True

    def _decompose(self, x: torch.Tensor) -> List[torch.Tensor]:
        subbands = self.wavelet(x)
        if len(subbands) != self.num_scales:
            raise RuntimeError(f"Expected {self.num_scales} scales, got {len(subbands)}.")
        return subbands

    def forward(self, x_ir: torch.Tensor, x_raman: torch.Tensor, stage: str = "stage2", return_aux: bool = True) -> Dict[str, object]:
        subbands_ir = self._decompose(x_ir)
        subbands_raman = self._decompose(x_raman)

        encoded_ir: List[torch.Tensor] = []
        encoded_raman: List[torch.Tensor] = []
        for scale_index, encoder in enumerate(self.encoders):
            encoded_ir.append(encoder(subbands_ir[scale_index]))
            encoded_raman.append(encoder(subbands_raman[scale_index]))

        output: Dict[str, object] = {
            "subbands_ir": subbands_ir,
            "subbands_raman": subbands_raman,
            "encoded_ir": encoded_ir,
            "encoded_raman": encoded_raman,
        }

        if not self.use_decomposition:
            scale_fused = [
                self.shared_fusions[scale_index](encoded_ir[scale_index], encoded_raman[scale_index])
                for scale_index in range(self.num_scales)
            ]
            logits, global_repr = self.classifier(scale_fused)
            prob = torch.softmax(logits, dim=-1)
            output.update(
                {
                    "logits": logits,
                    "prob": prob,
                    "pred": prob.argmax(dim=-1),
                    "shared_ir": [],
                    "shared_raman": [],
                    "private_ir": [],
                    "private_raman": [],
                    "fused_shared": [],
                    "fused_private": [],
                    "scale_fused": scale_fused,
                    "global_repr": global_repr,
                    "perturbed_shared_ir": [],
                    "perturbed_shared_raman": [],
                    "perturbed_private_ir": [],
                    "perturbed_private_raman": [],
                }
            )
            return output

        shared_ir: List[torch.Tensor] = []
        shared_raman: List[torch.Tensor] = []
        private_ir: List[torch.Tensor] = []
        private_raman: List[torch.Tensor] = []
        perturbed_shared_ir: List[torch.Tensor] = []
        perturbed_shared_raman: List[torch.Tensor] = []
        perturbed_private_ir: List[torch.Tensor] = []
        perturbed_private_raman: List[torch.Tensor] = []

        compute_stability = stage in {"stage1", "stage2"} and self.use_stability
        for scale_index, decomposer in enumerate(self.decomposers):
            decomposed = decomposer(
                h_ir=encoded_ir[scale_index],
                h_raman=encoded_raman[scale_index],
                use_stability=compute_stability,
                noise_std=self.config.perturbed_noise_std,
            )
            shared_ir.append(decomposed["u_ir"])
            shared_raman.append(decomposed["u_ra"])
            private_ir.append(decomposed["z_ir"])
            private_raman.append(decomposed["z_ra"])
            if compute_stability:
                perturbed_shared_ir.append(decomposed["u_ir_tilde"])
                perturbed_shared_raman.append(decomposed["u_ra_tilde"])
                perturbed_private_ir.append(decomposed["z_ir_tilde"])
                perturbed_private_raman.append(decomposed["z_ra_tilde"])

        output.update(
            {
                "shared_ir": shared_ir,
                "shared_raman": shared_raman,
                "private_ir": private_ir,
                "private_raman": private_raman,
                "perturbed_shared_ir": perturbed_shared_ir,
                "perturbed_shared_raman": perturbed_shared_raman,
                "perturbed_private_ir": perturbed_private_ir,
                "perturbed_private_raman": perturbed_private_raman,
            }
        )

        if stage == "stage1":
            return output

        fused_shared = [
            self.shared_fusions[scale_index](shared_ir[scale_index], shared_raman[scale_index])
            for scale_index in range(self.num_scales)
        ]
        if self.fusion_mode == "shared_module":
            fused_private = [
                self.shared_fusions[scale_index](private_ir[scale_index], private_raman[scale_index])
                for scale_index in range(self.num_scales)
            ]
        else:
            fused_private = [
                self.private_fusions[scale_index](private_ir[scale_index], private_raman[scale_index])
                for scale_index in range(self.num_scales)
            ]
        scale_fused = [
            torch.cat([fused_shared[scale_index], fused_private[scale_index]], dim=-1)
            for scale_index in range(self.num_scales)
        ]

        logits, global_repr = self.classifier(scale_fused)
        prob = torch.softmax(logits, dim=-1)
        output.update(
            {
                "logits": logits,
                "prob": prob,
                "pred": prob.argmax(dim=-1),
                "fused_shared": fused_shared,
                "fused_private": fused_private,
                "scale_fused": scale_fused,
                "global_repr": global_repr,
            }
        )
        return output

    @staticmethod
    def _resolve_fusion_mode(ablation: str) -> str:
        if ablation == "no_asymmetric":
            return "unified"
        if ablation == "symmetric_fusion":
            return "shared_module"
        if ablation == "private_average":
            return "private_average"
        return "asymmetric"


def build_model(config: ModelConfig, num_scales: int) -> ASWF:
    return ASWF(config=config, num_scales=num_scales)
