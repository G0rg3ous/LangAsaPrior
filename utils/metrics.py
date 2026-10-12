"""Metrics for 3D keypoint detection."""

from __future__ import annotations

from typing import Dict, Iterable

import torch


def heatmaps_to_argmax_coords(heatmaps: torch.Tensor) -> torch.Tensor:
    if heatmaps.ndim != 5:
        raise ValueError(f"Expected heatmaps [B,K,D,H,W], got {tuple(heatmaps.shape)}")
    b, k, d, h, w = heatmaps.shape
    flat = heatmaps.flatten(2).argmax(dim=-1)
    z = flat // (h * w)
    y = (flat % (h * w)) // w
    x = flat % w
    return torch.stack([z, y, x], dim=-1).float()


def summarize_prediction_jsons(predictions, thresholds=(2, 3, 4, 5, 10)):
    """Summarize saved per-case prediction records.

    Args:
        predictions: List of dicts with ``case_id``, ``pred`` and ``gt``.
    """
    if not predictions:
        return {
            "MRE": 0.0,
            "SDR": {f"SDR@{int(t)}": 0.0 for t in thresholds},
            "MRE_by_landmark": {},
            "detailed_results": {},
        }

    num_keypoints = len(predictions[0]["pred"])
    errors_by_landmark = {f"landmark_{idx + 1}": [] for idx in range(num_keypoints)}
    detailed = {}
    all_errors = []
    for record in predictions:
        case_id = record["case_id"]
        detailed[case_id] = {}
        for idx, (pred_coord, gt_coord) in enumerate(zip(record["pred"], record["gt"]), start=1):
            error = sum((float(p) - float(g)) ** 2 for p, g in zip(pred_coord, gt_coord)) ** 0.5
            key = f"landmark_{idx}"
            detailed[case_id][key] = float(round(error, 5))
            errors_by_landmark[key].append(error)
            all_errors.append(error)

    mre_by_landmark = {
        key: float(round(sum(values) / max(len(values), 1), 5))
        for key, values in errors_by_landmark.items()
    }
    mre = float(sum(all_errors) / max(len(all_errors), 1))
    sdr = {
        f"SDR@{int(threshold)}": float(sum(e <= threshold for e in all_errors) / max(len(all_errors), 1))
        for threshold in thresholds
    }
    return {
        "MRE": mre,
        "SDR": sdr,
        "MRE_by_landmark": mre_by_landmark,
        "detailed_results": detailed,
    }


class MetricTracker:
    def __init__(self, thresholds: Iterable[float] = (2, 3, 4, 5, 10)) -> None:
        self.thresholds = tuple(float(t) for t in thresholds)
        self.distances = []
        self.losses = []

    def update(self, outputs, gt_coords: torch.Tensor, loss: float | None = None) -> None:
        heatmaps = outputs["heatmaps"] if isinstance(outputs, dict) else outputs
        pred = heatmaps_to_argmax_coords(heatmaps.detach().float())
        gt = gt_coords.detach().float().to(pred.device)
        distances = torch.linalg.norm(pred - gt, dim=-1).flatten().cpu()
        self.distances.append(distances)
        if loss is not None:
            self.losses.append(float(loss))

    def compute(self) -> Dict[str, float]:
        if not self.distances:
            out = {"loss": 0.0, "mre": 0.0, "num_points": 0.0, "distance_sum": 0.0}
            for threshold in self.thresholds:
                out[f"sdr_{int(threshold)}"] = 0.0
                out[f"sdr_{int(threshold)}_count"] = 0.0
            return out
        distances = torch.cat(self.distances)
        num_points = float(distances.numel())
        metrics = {
            "loss": sum(self.losses) / max(len(self.losses), 1),
            "mre": float(distances.mean().item()),
            "num_points": num_points,
            "distance_sum": float(distances.sum().item()),
        }
        for threshold in self.thresholds:
            count = float((distances <= threshold).float().sum().item())
            metrics[f"sdr_{int(threshold)}"] = count / max(num_points, 1.0)
            metrics[f"sdr_{int(threshold)}_count"] = count
        return metrics


__all__ = ["MetricTracker", "heatmaps_to_argmax_coords", "summarize_prediction_jsons"]
