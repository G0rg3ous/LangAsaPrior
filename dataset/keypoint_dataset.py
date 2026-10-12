"""PKL volume dataset for 3D keypoint heatmap training."""

from __future__ import annotations

import os
import pickle
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def coords_to_gaussian_heatmaps(
    coords: torch.Tensor,
    spatial_size: Sequence[int],
    sigma: float,
) -> torch.Tensor:
    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError(f"Expected coords [K,3], got {tuple(coords.shape)}")
    d, h, w = [int(v) for v in spatial_size]
    device = coords.device
    dtype = coords.dtype
    z = torch.arange(d, device=device, dtype=dtype).view(1, d, 1, 1)
    y = torch.arange(h, device=device, dtype=dtype).view(1, 1, h, 1)
    x = torch.arange(w, device=device, dtype=dtype).view(1, 1, 1, w)
    center = coords.view(coords.shape[0], 3, 1, 1, 1)
    dist2 = (z - center[:, 0]).square() + (y - center[:, 1]).square() + (x - center[:, 2]).square()
    return torch.exp(-dist2 / (2.0 * float(sigma) ** 2)).clamp_(0.0, 1.0)


class KeypointPKLDataset(Dataset):
    """Load ``volume`` and ``coords``/``keypoints`` from PKL files.

    Returns:
        volume: ``[C, D, H, W]`` float tensor normalized to ``[0, 1]``.
        heatmaps: ``[K, D, H, W]`` Gaussian target heatmaps.
        coords: ``[K, 3]`` float tensor in ``(z, y, x)`` order.
        filename: source file name.
    """

    def __init__(
        self,
        data_dir: str,
        file_list: Optional[List[str]] = None,
        num_keypoints: int = 12,
        sigma: float = 12.0,
    ) -> None:
        self.data_dir = data_dir
        self.file_list = file_list if file_list is not None else sorted(os.listdir(data_dir))
        self.file_list = [f for f in self.file_list if f.endswith(".pkl")]
        self.num_keypoints = int(num_keypoints)
        self.sigma = float(sigma)

    def __len__(self) -> int:
        return len(self.file_list)

    def _load_sample(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        path = os.path.join(self.data_dir, filename)
        with open(path, "rb") as f:
            data = pickle.load(f)
        volume = data["volume"]
        coords = data["keypoints"] if "keypoints" in data else data.get("coords")
        if coords is None:
            raise KeyError(f"Neither 'keypoints' nor 'coords' found in {path}")

        if isinstance(volume, torch.Tensor):
            volume = volume.detach().cpu().numpy()
        if isinstance(coords, torch.Tensor):
            coords = coords.detach().cpu().numpy()
        elif isinstance(coords, list):
            coords = np.asarray(coords)
        return volume.astype(np.float32), coords.astype(np.float32)

    def __getitem__(self, idx: int):
        filename = self.file_list[idx]
        volume, coords = self._load_sample(filename)
        if coords.shape != (self.num_keypoints, 3):
            raise ValueError(f"Expected coords {(self.num_keypoints, 3)}, got {coords.shape} in {filename}")

        if volume.ndim == 3:
            volume = volume[None, ...]
        elif volume.ndim != 4:
            raise ValueError(f"Expected volume [D,H,W] or [C,D,H,W], got {volume.shape} in {filename}")

        v_min = float(volume.min())
        v_max = float(volume.max())
        volume = (volume - v_min) / (v_max - v_min + 1e-8)

        volume_t = torch.from_numpy(volume).float()
        coords_t = torch.from_numpy(coords).float()
        heatmaps = coords_to_gaussian_heatmaps(coords_t, volume_t.shape[-3:], sigma=self.sigma)
        return volume_t, heatmaps.float(), coords_t, filename


__all__ = ["KeypointPKLDataset", "coords_to_gaussian_heatmaps"]
