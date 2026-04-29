from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..losses import classification_loss, compute_decomposition_loss
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.metrics import compute_binary_metrics, select_decision_threshold
from .utils import build_loader


@dataclass
class Stage2Artifacts:
    best_checkpoint_path: str
    best_epoch: int
    best_metric: float
    best_loss: float
    best_val_metrics: Dict[str, float]
    threshold: float | None
    history: List[Dict[str, float]]


class Stage2Trainer:
    def __init__(self, model, config, loss_config, evaluation_config, device: torch.device) -> None:
        self.model = model.to(device)
        self.config = config
        self.loss_config = loss_config
        self.evaluation_config = evaluation_config
        self.device = device

    def fit(self, train_dataset, val_dataset, checkpoint_path: Path, stage1_checkpoint_path: Optional[Path] = None) -> Stage2Artifacts:
        if stage1_checkpoint_path is not None:
            checkpoint = load_checkpoint(stage1_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)

        self.model.set_stage2_trainable()
        optimizer = torch.optim.AdamW(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = self._build_scheduler(optimizer)
        train_loader = build_loader(train_dataset, self.config.batch_size, shuffle=True, num_workers=self.config.num_workers)
        val_loader = build_loader(val_dataset, self.config.batch_size, shuffle=False, num_workers=self.config.num_workers)

        best_state = None
        best_epoch = 0
        best_metric = -float("inf")
        best_loss = float("inf")
        best_threshold = None
        best_val_metrics: Dict[str, float] | None = None
        history: List[Dict[str, float]] = []
        patience_steps = 0

        for epoch in range(1, self.config.max_epochs + 1):
            train_stats = self._run_epoch(train_loader, optimizer=optimizer)
            val_eval = self.evaluate(val_loader)
            self._scheduler_step(scheduler, val_eval["metrics"].loss if val_eval["metrics"].loss is not None else 0.0)

            history.append(
                {
                    "epoch": float(epoch),
                    "train_L_cls": train_stats["L_cls"],
                    "train_L_cons": train_stats["L_cons"],
                    "train_L_orth": train_stats["L_orth"],
                    "train_L_stab": train_stats["L_stab"],
                    "train_L_stage2": train_stats["L_stage2"],
                    "val_L_stage2": float(val_eval["metrics"].loss or 0.0),
                    "val_bacc": float(val_eval["metrics"].bacc),
                    "val_auc": float(val_eval["metrics"].auc),
                    "val_f1": float(val_eval["metrics"].f1),
                    "val_mcc": float(val_eval["metrics"].mcc),
                    "val_threshold": float(val_eval["threshold"]) if val_eval["threshold"] is not None else float("nan"),
                }
            )

            current_metric = float(val_eval["metrics"].bacc)
            current_loss = float(val_eval["metrics"].loss or 0.0)
            improved = current_metric > best_metric + self.config.min_delta
            tie_break = abs(current_metric - best_metric) <= self.config.min_delta and current_loss < best_loss

            if improved or tie_break:
                best_state = deepcopy(self.model.state_dict())
                best_epoch = epoch
                best_metric = current_metric
                best_loss = current_loss
                best_threshold = val_eval["threshold"]
                best_val_metrics = val_eval["metrics"].to_dict()
                patience_steps = 0
            else:
                patience_steps += 1

            if patience_steps >= self.config.patience:
                break

        if best_state is None or best_val_metrics is None:
            raise RuntimeError("Stage II did not produce a valid checkpoint.")

        self.model.load_state_dict(best_state)
        save_checkpoint(
            {
                "state_dict": best_state,
                "epoch": best_epoch,
                "best_metric": best_metric,
                "best_loss": best_loss,
                "best_val_metrics": best_val_metrics,
                "threshold": best_threshold,
                "history": history,
            },
            checkpoint_path,
        )
        return Stage2Artifacts(
            best_checkpoint_path=str(checkpoint_path),
            best_epoch=best_epoch,
            best_metric=best_metric,
            best_loss=best_loss,
            best_val_metrics=best_val_metrics,
            threshold=best_threshold,
            history=history,
        )

    def evaluate(self, loader, collect_outputs: bool = False) -> Dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        total_items = 0
        positive_probs: List[np.ndarray] = []
        argmax_preds: List[np.ndarray] = []
        labels_list: List[np.ndarray] = []
        global_repr_list: List[np.ndarray] = []
        scale_fused_list: List[List[np.ndarray]] = []
        aux_store: Dict[str, List[np.ndarray]] = {"fused_shared": [], "fused_private": [], "private_ir": [], "private_raman": []}

        stage_name = "inference" if collect_outputs else "stage2"
        with torch.no_grad():
            for batch in loader:
                ir = batch["ir"].to(self.device)
                raman = batch["raman"].to(self.device)
                labels = batch["label"].to(self.device)
                outputs = self.model(ir, raman, stage=stage_name, return_aux=True)
                batch_size = labels.shape[0]
                if stage_name != "inference":
                    decomp_losses = compute_decomposition_loss(outputs, self.loss_config)
                    cls = classification_loss(outputs["logits"], labels)
                    total = cls + decomp_losses["L_decomp"]
                    total_loss += float(total.item()) * batch_size
                    total_items += batch_size

                prob = outputs["prob"].detach().cpu().numpy()
                positive_probs.append(prob[:, 1])
                argmax_preds.append(prob.argmax(axis=1))
                labels_list.append(labels.detach().cpu().numpy())

                if collect_outputs:
                    global_repr_list.append(outputs["global_repr"].detach().cpu().numpy())
                    scale_fused_list.append([tensor.detach().cpu().numpy() for tensor in outputs["scale_fused"]])
                    if outputs["fused_shared"]:
                        aux_store["fused_shared"].append(
                            np.stack([tensor.detach().cpu().numpy() for tensor in outputs["fused_shared"]], axis=1)
                        )
                        aux_store["fused_private"].append(
                            np.stack([tensor.detach().cpu().numpy() for tensor in outputs["fused_private"]], axis=1)
                        )
                        aux_store["private_ir"].append(
                            np.stack([tensor.detach().cpu().numpy() for tensor in outputs["private_ir"]], axis=1)
                        )
                        aux_store["private_raman"].append(
                            np.stack([tensor.detach().cpu().numpy() for tensor in outputs["private_raman"]], axis=1)
                        )

        y_true = np.concatenate(labels_list, axis=0).astype(np.int64)
        positive_prob = np.concatenate(positive_probs, axis=0)
        argmax_pred = np.concatenate(argmax_preds, axis=0)
        threshold = self._resolve_threshold(y_true=y_true, positive_prob=positive_prob)
        pred = self._resolve_predictions(argmax_pred=argmax_pred, positive_prob=positive_prob, threshold=threshold)
        loss_value = total_loss / total_items if total_items > 0 else None
        metrics = compute_binary_metrics(
            y_true=y_true,
            positive_prob=positive_prob,
            pred=pred,
            threshold=threshold,
            loss=loss_value,
        )

        result: Dict[str, Any] = {
            "metrics": metrics,
            "threshold": threshold,
            "labels": y_true,
            "positive_prob": positive_prob,
            "pred": pred,
        }
        if collect_outputs:
            result["global_repr"] = np.concatenate(global_repr_list, axis=0) if global_repr_list else np.empty((0, 0))
            result["scale_fused"] = self._merge_scale_features(scale_fused_list)
            result["fused_shared"] = np.concatenate(aux_store["fused_shared"], axis=0) if aux_store["fused_shared"] else None
            result["fused_private"] = np.concatenate(aux_store["fused_private"], axis=0) if aux_store["fused_private"] else None
            result["private_ir"] = np.concatenate(aux_store["private_ir"], axis=0) if aux_store["private_ir"] else None
            result["private_raman"] = np.concatenate(aux_store["private_raman"], axis=0) if aux_store["private_raman"] else None
        return result

    def predict(self, dataset) -> Dict[str, Any]:
        loader = build_loader(dataset, self.config.batch_size, shuffle=False, num_workers=self.config.num_workers)
        return self.evaluate(loader, collect_outputs=True)

    def _run_epoch(self, loader, optimizer) -> Dict[str, float]:
        self.model.train(True)
        totals = {"L_cls": 0.0, "L_cons": 0.0, "L_orth": 0.0, "L_stab": 0.0, "L_stage2": 0.0}
        total_items = 0

        for batch in loader:
            ir = batch["ir"].to(self.device)
            raman = batch["raman"].to(self.device)
            labels = batch["label"].to(self.device)

            outputs = self.model(ir, raman, stage="stage2", return_aux=True)
            decomp_losses = compute_decomposition_loss(outputs, self.loss_config)
            cls_loss = classification_loss(outputs["logits"], labels)
            total_loss = cls_loss + decomp_losses["L_decomp"]

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            optimizer.step()

            batch_size = labels.shape[0]
            total_items += batch_size
            totals["L_cls"] += float(cls_loss.item()) * batch_size
            totals["L_cons"] += float(decomp_losses["L_cons"].item()) * batch_size
            totals["L_orth"] += float(decomp_losses["L_orth"].item()) * batch_size
            totals["L_stab"] += float(decomp_losses["L_stab"].item()) * batch_size
            totals["L_stage2"] += float(total_loss.item()) * batch_size

        if total_items == 0:
            raise RuntimeError("Encountered an empty loader during Stage II.")
        return {key: value / total_items for key, value in totals.items()}

    def _resolve_threshold(self, y_true: np.ndarray, positive_prob: np.ndarray) -> float | None:
        if self.evaluation_config.tune_threshold:
            return select_decision_threshold(y_true, positive_prob, metric=self.evaluation_config.threshold_metric)
        if self.evaluation_config.threshold_policy == "fixed_0.5":
            return 0.5
        return None

    @staticmethod
    def _resolve_predictions(argmax_pred: np.ndarray, positive_prob: np.ndarray, threshold: float | None) -> np.ndarray:
        if threshold is not None:
            return (positive_prob >= threshold).astype(np.int64)
        return argmax_pred.astype(np.int64)

    def _build_scheduler(self, optimizer):
        if self.config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.max_epochs)
        if self.config.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)
        return None

    @staticmethod
    def _scheduler_step(scheduler, monitored_value: float) -> None:
        if scheduler is None:
            return
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(monitored_value)
            return
        scheduler.step()

    @staticmethod
    def _merge_scale_features(scale_batches: List[List[np.ndarray]]) -> List[np.ndarray]:
        if not scale_batches:
            return []
        num_scales = len(scale_batches[0])
        return [np.concatenate([batch[scale_index] for batch in scale_batches], axis=0) for scale_index in range(num_scales)]
