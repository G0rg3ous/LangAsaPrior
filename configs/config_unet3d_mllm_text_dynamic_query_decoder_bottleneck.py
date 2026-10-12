"""Config for MLLM-enhanced multi-scale dynamic landmark decoder."""

from __future__ import annotations

import os
from pathlib import Path


class Config:
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = os.environ.get("KEYPOINT_DATA_DIR", str(_REPO_ROOT / "data"))
    SPLIT_FILE = os.environ.get("KEYPOINT_SPLIT_FILE", str(_REPO_ROOT / "splits" / "split.json"))
    _OUTPUT_ROOT = Path(os.environ.get("KEYPOINT_OUTPUT_DIR", str(_REPO_ROOT / "outputs")))
    SAVE_DIR = str(_OUTPUT_ROOT / "checkpoints" / "unet3d_mllm_text_dynamic_query_decoder_bottleneck")
    LOG_DIR = str(_OUTPUT_ROOT / "logs" / "unet3d_mllm_text_dynamic_query_decoder_bottleneck")

    NUM_FOLDS = 5
    FOLD = 0
    SPLIT_SEED = 42

    IN_CHANNELS = 1
    NUM_KEYPOINTS = 12
    BASE_CHANNELS = 16
    CHANNEL_MULTIPLIERS = (1, 2, 4, 8)
    NORM = "instance"
    DROPOUT = 0.0
    COORD_TEMPERATURE = 1.0

    # Set QWEN3_VL_MODEL_DIR to a local, separately downloaded checkpoint.
    LLM_MODEL_NAME = os.environ.get("QWEN3_VL_MODEL_DIR", "")
    LLM_LAYER_IDX = 17
    LLM_DIM = None
    FREEZE_LLM = True
    SPATIAL_POOL_SIZE = (6, 6, 6)
    TEXT_EMBEDDING_PATH = str(_REPO_ROOT / "fixed_text_embeddings_qwen3vl_knee12_names_only.pt")
    TEXT_FIRST = False
    USE_3D_POSITION_ENCODING = True

    QUERY_DIM = 128
    QUERY_HEADS = 4
    QUERY_POOL_SIZES = ((6, 6, 6), (6, 6, 6), (8, 8, 8), (12, 12, 12))
    QUERY_DROPOUT = 0.0
    SEMANTIC_FUSION_INIT = 0.1

    HEATMAP_SIGMA = 12.0
    LOSS_TYPE = "mse"  # mse | bce | combined
    BCE_WEIGHT = 0.1

    BATCH_SIZE = 1
    NUM_EPOCHS = 300
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-5
    NUM_WORKERS = 2
    SCALE_LR_WITH_WORLD = False
    GRAD_CLIP = 1.0
    USE_AMP = True

    DIST_BACKEND = "nccl"
    OPTIMIZER = "adamw"
    SCHEDULER = "cosine"
    SCHEDULER_PARAMS = {"T_max": 300, "eta_min": 1e-5}
    SAVE_INTERVAL = 0
    VAL_INTERVAL = 1
    EXPORT_VALIDATION_JSON_AT_END = True

    SDR_THRESHOLDS = (2.0, 3.0, 4.0, 5.0, 10.0)

    @classmethod
    def create_dirs(cls) -> None:
        os.makedirs(cls.SAVE_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.SPLIT_FILE), exist_ok=True)
