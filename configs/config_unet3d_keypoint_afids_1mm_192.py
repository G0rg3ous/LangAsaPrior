"""Training config for the preprocessed AFIDs 1 mm, normalized 192^3 dataset."""

from __future__ import annotations

import os
from pathlib import Path

from .config_unet3d_keypoint import Config as BaseConfig


class Config(BaseConfig):
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    DATASET_TYPE = "afids_1mm_192"
    DATA_DIR = os.environ.get("AFIDS_DATA_DIR", str(_REPO_ROOT / "data" / "datasets_1mm_192"))
    SPLIT_FILE = str(_REPO_ROOT / "splits" / "split_afids_7_1_2.json")
    _OUTPUT_ROOT = Path(os.environ.get("AFIDS_OUTPUT_DIR", str(_REPO_ROOT / "outputs")))
    SAVE_DIR = str(_OUTPUT_ROOT / "checkpoints" / "unet3d_keypoint_afids_1mm_192")
    LOG_DIR = str(_OUTPUT_ROOT / "logs" / "unet3d_keypoint_afids_1mm_192")

    NUM_KEYPOINTS = 32
    HEATMAP_SIGMA = 12.0

    BATCH_SIZE = 1
    NUM_WORKERS = 2
