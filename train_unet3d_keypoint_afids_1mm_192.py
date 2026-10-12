"""Train the 3D U-Net keypoint baseline on preprocessed AFIDs 1 mm 192^3 data."""

from __future__ import annotations

import argparse
import os

import torch.multiprocessing as mp

from configs.config_unet3d_keypoint_afids_1mm_192 import Config
import train_unet3d_keypoint as base_train


def train_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    base_train.Config = Config
    base_train.train_worker(rank, world_size, args)


def run_split(args: argparse.Namespace) -> None:
    gpu_list = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    world_size = args.world_size if args.world_size is not None else len(gpu_list)
    if world_size != len(gpu_list):
        world_size = len(gpu_list)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_list))
    print(f"Starting AFIDs 1 mm 192^3 train/val/test split on GPUs {gpu_list}, world_size={world_size}")
    mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 3D U-Net on preprocessed AFIDs 1 mm 192^3 data")
    parser.add_argument("--gpus", type=str, default="0", help="GPU IDs, e.g. 0 or 0,1")
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--master_port", type=str, default="12433")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint_last.pth")
    parser.add_argument("--resume_path", type=str, default=None, help="Optional explicit resume checkpoint")
    args = parser.parse_args()
    run_split(args)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
