from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch

from ..losses import compute_decomposition_loss
from ..utils.checkpoint import save_checkpoint
from .utils import build_loader


@dataclass
class Stage1Artifacts:
    best_checkpoint_path: str
    best_epoch: int
    best_metric: float
    best_metrics: Dict[str, float]
    history: List[Dict[str, float]]


class Stage1Trainer:
    def __init__(self, model, config, loss_config, device: torch.device) -> None:
        self.model = model.to(device)
        self.config = config
        self.loss_config = loss_config
        self.device = device

    def fit(self, train_dataset, val_dataset, checkpoint_path: Path) -> Stage1Artifacts:
        self.model.set_stage1_trainable()
        optimizer = torch.optim.AdamW(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = self._build_scheduler(optimizer)
        train_loader = build_loader(train_dataset, self.config.batch_size, shuffle=True, num_workers=self.config.num_workers)
        val_loader = build_loader(val_dataset, self.config.batch_size, shuffle=False, num_workers=self.config.num_workers)

        best_metric = float("inf")
        best_state = None
        best_epoch = 0
        best_metrics: Dict[str, float] | None = None
        patience_steps = 0
        history: List[Dict[str, float]] = []

        for epoch in range(1, self.config.max_epochs + 1):
            train_metrics = self._run_epoch(train_loader, optimizer=optimizer)
            val_metrics = self._run_epoch(val_loader, optimizer=None)
            self._scheduler_step(scheduler, val_metrics["L_decomp"])

            history.append(
                {
                    "epoch": float(epoch),
                    "train_L_cons": train_metrics["L_cons"],
                    "train_L_orth": train_metrics["L_orth"],
                    "train_L_stab": train_metrics["L_stab"],
                    "train_L_decomp": train_metrics["L_decomp"],
                    "val_L_cons": val_metrics["L_cons"],
                    "val_L_orth": val_metrics["L_orth"],
                    "val_L_stab": val_metrics["L_stab"],
                    "val_L_decomp": val_metrics["L_decomp"],
                }
            )

            if val_metrics["L_decomp"] < best_metric - self.config.min_delta:
                best_metric = val_metrics["L_decomp"]
                best_state = deepcopy(self.model.state_dict())
                best_epoch = epoch
                best_metrics = val_metrics
                patience_steps = 0
            else:
                patience_steps += 1

            if patience_steps >= self.config.patience:
                break

        if best_state is None or best_metrics is None:
            raise RuntimeError("Stage I did not produce a valid checkpoint.")

        self.model.load_state_dict(best_state)
        save_checkpoint(
            {
                "state_dict": best_state,
                "epoch": best_epoch,
                "best_metric": best_metric,
                "best_metrics": best_metrics,
                "history": history,
            },
            checkpoint_path,
        )
        return Stage1Artifacts(
            best_checkpoint_path=str(checkpoint_path),
            best_epoch=best_epoch,
            best_metric=best_metric,
            best_metrics=best_metrics,
            history=history,
        )

    def _run_epoch(self, loader, optimizer) -> Dict[str, float]:
        is_train = optimizer is not None
        self.model.train(is_train)
        totals = {"L_cons": 0.0, "L_orth": 0.0, "L_stab": 0.0, "L_decomp": 0.0}
        total_items = 0

        for batch in loader:
            ir = batch["ir"].to(self.device)
            raman = batch["raman"].to(self.device)
            outputs = self.model(ir, raman, stage="stage1", return_aux=True)
            losses = compute_decomposition_loss(outputs, self.loss_config)
            loss = losses["L_decomp"]

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                optimizer.step()

            batch_size = ir.shape[0]
            total_items += batch_size
            for key in totals:
                totals[key] += float(losses[key].item()) * batch_size

        if total_items == 0:
            raise RuntimeError("Encountered an empty loader during Stage I.")
        return {key: value / total_items for key, value in totals.items()}

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
