"""Train AFIDs 1 mm 192^3 with MLLM-enhanced dynamic landmark decoder."""

from __future__ import annotations

import argparse
import os

import torch
import torch.multiprocessing as mp

from configs.config_unet3d_mllm_text_dynamic_query_decoder_bottleneck_afids_1mm_192 import Config
from models import get_unet3d_mllm_text_dynamic_query_decoder_bottleneck_keypoint_detector
import train_unet3d_keypoint as base_train


def build_model(cfg: Config) -> torch.nn.Module:
    return get_unet3d_mllm_text_dynamic_query_decoder_bottleneck_keypoint_detector(
        in_channels=cfg.IN_CHANNELS,
        num_keypoints=cfg.NUM_KEYPOINTS,
        base_channels=cfg.BASE_CHANNELS,
        channel_multipliers=cfg.CHANNEL_MULTIPLIERS,
        norm=cfg.NORM,
        dropout=cfg.DROPOUT,
        coord_temperature=cfg.COORD_TEMPERATURE,
        llm_model_name=cfg.LLM_MODEL_NAME,
        text_embedding_path=cfg.TEXT_EMBEDDING_PATH,
        llm_layer_idx=cfg.LLM_LAYER_IDX,
        llm_dim=cfg.LLM_DIM,
        spatial_pool_size=cfg.SPATIAL_POOL_SIZE,
        freeze_llm=cfg.FREEZE_LLM,
        text_first=cfg.TEXT_FIRST,
        use_3d_position_encoding=cfg.USE_3D_POSITION_ENCODING,
        query_dim=cfg.QUERY_DIM,
        query_heads=cfg.QUERY_HEADS,
        query_pool_sizes=cfg.QUERY_POOL_SIZES,
        query_dropout=cfg.QUERY_DROPOUT,
        semantic_fusion_init=cfg.SEMANTIC_FUSION_INIT,
    )


def train_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    base_train.Config = Config
    base_train.build_model = build_model
    base_train.train_worker(rank, world_size, args)


def run_split(args: argparse.Namespace) -> None:
    gpu_list = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    world_size = args.world_size if args.world_size is not None else len(gpu_list)
    if world_size != len(gpu_list):
        world_size = len(gpu_list)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_list))
    order = "text+visual" if Config.TEXT_FIRST else "visual+text"
    print(f"Starting AFIDs MLLM dynamic-query decoder on GPUs {gpu_list}, world_size={world_size}")
    print(f"Text embeddings: {Config.TEXT_EMBEDDING_PATH}")
    print(f"Token order: {order}")
    print(
        f"Dynamic query decoder: dim={Config.QUERY_DIM}, heads={Config.QUERY_HEADS}, "
        f"pools={Config.QUERY_POOL_SIZES}"
    )
    mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AFIDs MLLM-enhanced dynamic landmark decoder")
    parser.add_argument("--gpus", type=str, default="0", help="GPU IDs, e.g. 0 or 0,1")
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--master_port", type=str, default="12443")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint_last.pth")
    parser.add_argument("--resume_path", type=str, default=None, help="Optional explicit resume checkpoint")
    args = parser.parse_args()
    run_split(args)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
