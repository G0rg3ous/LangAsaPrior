"""Loss helpers for 3D keypoint heatmap training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointHeatmapLoss(nn.Module):
    def __init__(self, loss_type: str = "mse", bce_weight: float = 0.1) -> None:
        super().__init__()
        self.loss_type = loss_type.lower()
        self.bce_weight = float(bce_weight)

    def forward(self, outputs, targets: torch.Tensor) -> torch.Tensor:
        logits = outputs["heatmaps"] if isinstance(outputs, dict) else outputs
        if logits.shape != targets.shape:
            raise ValueError(f"Prediction/target shape mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}")

        if self.loss_type == "mse":
            return F.mse_loss(torch.sigmoid(logits.float()), targets.float())
        if self.loss_type == "bce":
            return F.binary_cross_entropy_with_logits(logits.float(), targets.float())
        if self.loss_type == "combined":
            mse = F.mse_loss(torch.sigmoid(logits.float()), targets.float())
            bce = F.binary_cross_entropy_with_logits(logits.float(), targets.float())
            return mse + self.bce_weight * bce
        raise ValueError(f"Unsupported loss_type={self.loss_type}")


__all__ = ["KeypointHeatmapLoss"]
