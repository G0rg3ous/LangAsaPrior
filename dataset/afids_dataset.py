"""AFIDs NIfTI/FCSV dataset for 3D keypoint heatmap training."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .keypoint_dataset import coords_to_gaussian_heatmaps


class AFIDsDataset(Dataset):
    """Load AFIDs T1w NIfTI images and ground-truth FCSV annotations.

    Split entries use ``<dataset>/<subject>`` identifiers, for example
    ``AFIDs-HCP/sub-103111``. Returned coordinates follow the training
    framework's ``(z, y, x)`` voxel order.
    """

    def __init__(
        self,
        data_dir: str,
        case_list: Sequence[str],
        num_keypoints: int = 32,
        sigma: float = 3.0,
        target_size: Optional[Sequence[int]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.case_list = [str(case).strip().strip("/") for case in case_list]
        self.num_keypoints = int(num_keypoints)
        self.sigma = float(sigma)
        self.target_size = tuple(int(v) for v in target_size) if target_size is not None else None

        if self.target_size is not None and len(self.target_size) != 3:
            raise ValueError(f"target_size must be (D,H,W), got {self.target_size}")
        if not self.case_list:
            raise ValueError("AFIDs case_list is empty")

        self.samples = [self._resolve_case(case_id) for case_id in self.case_list]

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_case(self, case_id: str) -> tuple[str, Path, Path]:
        parts = Path(case_id).parts
        if len(parts) != 2:
            raise ValueError(f"AFIDs case ID must be '<dataset>/<subject>', got {case_id!r}")
        dataset_name, subject_id = parts
        dataset_dir = self.data_dir / dataset_name
        image_candidates = sorted((dataset_dir / subject_id / "anat").glob("*T1w.nii.gz"))
        fcsv_candidates = sorted(
            (dataset_dir / "derivatives" / "afids_groundtruth" / subject_id / "anat").glob(
                "*groundtruth_afids.fcsv"
            )
        )
        if not image_candidates:
            raise FileNotFoundError(f"No T1w NIfTI path found for {case_id} under {dataset_dir}")
        if not fcsv_candidates:
            raise FileNotFoundError(f"No ground-truth AFIDs FCSV found for {case_id} under {dataset_dir}")

        image_path = image_candidates[0]
        fcsv_path = fcsv_candidates[0]
        if not image_path.exists():
            raise FileNotFoundError(
                f"NIfTI content is not available for {case_id}: {image_path}. "
                "Retrieve the git-annex/DataLad file before training."
            )
        return case_id, image_path, fcsv_path

    @staticmethod
    def _import_nibabel():
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError(
                "AFIDsDataset requires nibabel. Install it in the training environment with "
                "`pip install nibabel`."
            ) from exc
        return nib

    def _load_fcsv_world_coords(self, path: Path) -> np.ndarray:
        coordinate_system = "RAS"
        columns = None
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("# CoordinateSystem"):
                    value = line.split("=", 1)[1].strip().upper()
                    coordinate_system = "LPS" if value in {"1", "LPS"} else "RAS"
                elif line.startswith("# columns"):
                    columns = [item.strip() for item in line.split("=", 1)[1].split(",")]
                elif line.strip() and not line.startswith("#"):
                    rows.append(line)

        if columns is None:
            raise ValueError(f"Missing '# columns' header in {path}")
        records = list(csv.DictReader(rows, fieldnames=columns))
        coords_by_label = {}
        for record in records:
            label = int(record["label"])
            coords_by_label[label] = [float(record["x"]), float(record["y"]), float(record["z"])]

        expected = set(range(1, self.num_keypoints + 1))
        missing = sorted(expected.difference(coords_by_label))
        if missing:
            raise ValueError(f"Missing AFIDs labels {missing} in {path}")
        coords = np.asarray([coords_by_label[idx] for idx in range(1, self.num_keypoints + 1)], dtype=np.float32)
        if coordinate_system == "LPS":
            coords[:, :2] *= -1.0
        return coords

    def __getitem__(self, idx: int):
        case_id, image_path, fcsv_path = self.samples[idx]
        nib = self._import_nibabel()
        image = nib.load(str(image_path))
        volume_xyz = np.asarray(image.get_fdata(dtype=np.float32))
        if volume_xyz.ndim != 3:
            raise ValueError(f"Expected 3D NIfTI image, got {volume_xyz.shape} in {image_path}")

        world_xyz = self._load_fcsv_world_coords(fcsv_path)
        voxel_xyz = nib.affines.apply_affine(np.linalg.inv(image.affine), world_xyz).astype(np.float32)
        coords_zyx = voxel_xyz[:, [2, 1, 0]]
        volume_zyx = np.transpose(volume_xyz, (2, 1, 0)).copy()

        v_min = float(volume_zyx.min())
        v_max = float(volume_zyx.max())
        volume_t = torch.from_numpy((volume_zyx - v_min) / (v_max - v_min + 1e-8)).float()[None]
        coords_t = torch.from_numpy(coords_zyx).float()

        if self.target_size is not None and tuple(volume_t.shape[-3:]) != self.target_size:
            original_size = torch.tensor(volume_t.shape[-3:], dtype=coords_t.dtype)
            target_size = torch.tensor(self.target_size, dtype=coords_t.dtype)
            coords_t = coords_t * (target_size - 1.0) / (original_size - 1.0).clamp_min(1.0)
            volume_t = F.interpolate(
                volume_t[None],
                size=self.target_size,
                mode="trilinear",
                align_corners=True,
            )[0]

        spatial_size = torch.tensor(volume_t.shape[-3:], dtype=coords_t.dtype)
        if torch.any(coords_t < 0) or torch.any(coords_t > spatial_size - 1):
            raise ValueError(f"At least one AFIDs coordinate is outside the image for {case_id}")

        heatmaps = coords_to_gaussian_heatmaps(coords_t, volume_t.shape[-3:], sigma=self.sigma)
        return volume_t, heatmaps.float(), coords_t, case_id


__all__ = ["AFIDsDataset"]
