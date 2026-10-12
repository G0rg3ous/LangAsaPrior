"""Dataset adapter for preprocessed AFIDs 1 mm, normalized 192^3 volumes."""

from __future__ import annotations

import numpy as np
import torch

from .afids_dataset import AFIDsDataset
from .keypoint_dataset import coords_to_gaussian_heatmaps


class AFIDs1mm192Dataset(AFIDsDataset):
    """Load AFIDs volumes prepared by ``prepare_afids_1mm_192.py``.

    The adapter validates that every input is already ``192 x 192 x 192``,
    has 1 mm isotropic spacing, and is normalized to ``[0, 1]``. No resizing
    or intensity transformation is applied.
    """

    EXPECTED_SHAPE_XYZ = (192, 192, 192)
    EXPECTED_SPACING_XYZ = (1.0, 1.0, 1.0)

    def __init__(
        self,
        data_dir: str,
        case_list,
        num_keypoints: int = 32,
        sigma: float = 3.0,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            case_list=case_list,
            num_keypoints=num_keypoints,
            sigma=sigma,
            target_size=None,
        )

    def __getitem__(self, idx: int):
        case_id, image_path, fcsv_path = self.samples[idx]
        nib = self._import_nibabel()
        image = nib.load(str(image_path))
        volume_xyz = np.asarray(image.get_fdata(dtype=np.float32))

        if tuple(volume_xyz.shape) != self.EXPECTED_SHAPE_XYZ:
            raise ValueError(
                f"Expected preprocessed AFIDs shape {self.EXPECTED_SHAPE_XYZ}, "
                f"got {volume_xyz.shape} for {case_id}"
            )
        spacing = tuple(float(v) for v in image.header.get_zooms()[:3])
        if not np.allclose(spacing, self.EXPECTED_SPACING_XYZ, atol=1e-5):
            raise ValueError(
                f"Expected preprocessed AFIDs spacing {self.EXPECTED_SPACING_XYZ}, "
                f"got {spacing} for {case_id}"
            )
        v_min = float(volume_xyz.min())
        v_max = float(volume_xyz.max())
        if v_min < -1e-6 or v_max > 1.0 + 1e-6:
            raise ValueError(f"Expected normalized AFIDs volume in [0,1], got [{v_min}, {v_max}] for {case_id}")

        world_xyz = self._load_fcsv_world_coords(fcsv_path)
        voxel_xyz = nib.affines.apply_affine(np.linalg.inv(image.affine), world_xyz).astype(np.float32)
        coords_t = torch.from_numpy(voxel_xyz[:, [2, 1, 0]]).float()
        volume_t = torch.from_numpy(np.transpose(volume_xyz, (2, 1, 0)).copy()).float()[None]

        spatial_size = torch.tensor(volume_t.shape[-3:], dtype=coords_t.dtype)
        if torch.any(coords_t < 0) or torch.any(coords_t > spatial_size - 1):
            raise ValueError(f"At least one AFIDs coordinate is outside the image for {case_id}")

        heatmaps = coords_to_gaussian_heatmaps(coords_t, volume_t.shape[-3:], sigma=self.sigma)
        return volume_t, heatmaps.float(), coords_t, case_id


__all__ = ["AFIDs1mm192Dataset"]
