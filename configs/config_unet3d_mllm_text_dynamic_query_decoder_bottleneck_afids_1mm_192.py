"""Config for AFIDs 1 mm 192^3 with MLLM dynamic landmark decoder."""

from __future__ import annotations

import os
from pathlib import Path

from .config_unet3d_mllm_text_dynamic_query_decoder_bottleneck import Config as BaseConfig


class Config(BaseConfig):
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    DATASET_TYPE = "afids_1mm_192"
    DATA_DIR = os.environ.get("AFIDS_DATA_DIR", str(_REPO_ROOT / "data" / "datasets_1mm_192"))
    SPLIT_FILE = str(_REPO_ROOT / "splits" / "split_afids_7_1_2.json")
    _OUTPUT_ROOT = Path(os.environ.get("AFIDS_OUTPUT_DIR", str(_REPO_ROOT / "outputs")))
    SAVE_DIR = str(_OUTPUT_ROOT / "checkpoints" / "unet3d_mllm_text_dynamic_query_decoder_bottleneck_afids_1mm_192")
    LOG_DIR = str(_OUTPUT_ROOT / "logs" / "unet3d_mllm_text_dynamic_query_decoder_bottleneck_afids_1mm_192")

    NUM_KEYPOINTS = 32
    HEATMAP_SIGMA = 12.0
    TEXT_EMBEDDING_PATH = str(_REPO_ROOT / "fixed_text_embeddings_qwen3vl_afids32_names_only.pt")

    BATCH_SIZE = 1
    NUM_WORKERS = 2
