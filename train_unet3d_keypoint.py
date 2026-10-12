"""Train 3D U-Net keypoint detector with a fixed train/val/test split."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from configs import Config
from dataset import AFIDs1mm192Dataset, AFIDsDataset, KeypointPKLDataset
from models import get_unet3d_keypoint_detector
from utils import KeypointHeatmapLoss, MetricTracker, heatmaps_to_argmax_coords, summarize_prediction_jsons

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None


def setup_distributed(rank: int, world_size: int, backend: str, master_port: str) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    init_kwargs = {
        "backend": backend,
        "init_method": "env://",
        "rank": rank,
        "world_size": world_size,
    }
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        if backend == "nccl":
            try:
                dist.init_process_group(**init_kwargs, device_id=torch.device("cuda", rank))
                return
            except TypeError:
                pass
    dist.init_process_group(**init_kwargs)


def distributed_barrier(rank: int) -> None:
    if not dist.is_initialized():
        return
    if torch.cuda.is_available():
        dist.barrier(device_ids=[rank])
    else:
        dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _canonical_case_name(name: str) -> str:
    stem = Path(str(name)).stem
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9]+", "_", stem)).strip("_").upper()


def _resolve_split_files(data_dir: str, files: Iterable[str], split_name: str) -> list[str]:
    available = {}
    for filename in os.listdir(data_dir):
        if filename.endswith(".pkl"):
            available.setdefault(_canonical_case_name(filename), []).append(filename)

    resolved = []
    missing = []
    for name in files:
        key = _canonical_case_name(str(name))
        candidates = available.get(key)
        prefer_side_case = False
        if candidates is None and key.endswith("_2"):
            candidates = available.get(key[:-2])
            prefer_side_case = candidates is not None
        filename = None
        if candidates:
            side_cases = [item for item in candidates if "左侧" in item or "右侧" in item]
            plain_cases = [item for item in candidates if item not in side_cases]
            if prefer_side_case and side_cases:
                filename = sorted(side_cases)[0]
            elif plain_cases:
                filename = sorted(plain_cases)[0]
            else:
                filename = sorted(candidates)[0]
        if filename is None:
            missing.append(str(name))
        else:
            resolved.append(filename)
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"{len(missing)} {split_name} cases in {data_dir} were not found after filename normalization: "
            f"{preview}{suffix}"
        )
    return resolved


def build_datasets(cfg: Config):
    with open(cfg.SPLIT_FILE, "r", encoding="utf-8") as f:
        split = json.load(f)
    required = {"train", "val", "test"}
    missing = sorted(required.difference(split))
    if missing:
        raise ValueError(f"Split file {cfg.SPLIT_FILE} is missing keys: {missing}")

    dataset_type = str(getattr(cfg, "DATASET_TYPE", "pkl")).lower()
    if dataset_type == "afids_1mm_192":
        kwargs = {
            "data_dir": cfg.DATA_DIR,
            "num_keypoints": cfg.NUM_KEYPOINTS,
            "sigma": cfg.HEATMAP_SIGMA,
        }
        train_ds = AFIDs1mm192Dataset(case_list=split["train"], **kwargs)
        val_ds = AFIDs1mm192Dataset(case_list=split["val"], **kwargs)
        test_ds = AFIDs1mm192Dataset(case_list=split["test"], **kwargs)
        return train_ds, val_ds, test_ds
    if dataset_type == "afids":
        kwargs = {
            "data_dir": cfg.DATA_DIR,
            "num_keypoints": cfg.NUM_KEYPOINTS,
            "sigma": cfg.HEATMAP_SIGMA,
            "target_size": getattr(cfg, "TARGET_SIZE", None),
        }
        train_ds = AFIDsDataset(case_list=split["train"], **kwargs)
        val_ds = AFIDsDataset(case_list=split["val"], **kwargs)
        test_ds = AFIDsDataset(case_list=split["test"], **kwargs)
        return train_ds, val_ds, test_ds
    if dataset_type != "pkl":
        raise ValueError(f"Unknown DATASET_TYPE: {dataset_type}")

    train_files = _resolve_split_files(cfg.DATA_DIR, split["train"], "train")
    val_files = _resolve_split_files(cfg.DATA_DIR, split["val"], "val")
    test_files = _resolve_split_files(cfg.DATA_DIR, split["test"], "test")
    train_ds = KeypointPKLDataset(
        cfg.DATA_DIR,
        file_list=train_files,
        num_keypoints=cfg.NUM_KEYPOINTS,
        sigma=cfg.HEATMAP_SIGMA,
    )
    val_ds = KeypointPKLDataset(
        cfg.DATA_DIR,
        file_list=val_files,
        num_keypoints=cfg.NUM_KEYPOINTS,
        sigma=cfg.HEATMAP_SIGMA,
    )
    test_ds = KeypointPKLDataset(
        cfg.DATA_DIR,
        file_list=test_files,
        num_keypoints=cfg.NUM_KEYPOINTS,
        sigma=cfg.HEATMAP_SIGMA,
    )
    return train_ds, val_ds, test_ds


def build_model(cfg: Config) -> torch.nn.Module:
    return get_unet3d_keypoint_detector(
        in_channels=cfg.IN_CHANNELS,
        num_keypoints=cfg.NUM_KEYPOINTS,
        base_channels=cfg.BASE_CHANNELS,
        channel_multipliers=cfg.CHANNEL_MULTIPLIERS,
        norm=cfg.NORM,
        dropout=cfg.DROPOUT,
        coord_temperature=cfg.COORD_TEMPERATURE,
    )


def build_optimizer(cfg: Config, model: torch.nn.Module, world_size: int) -> optim.Optimizer:
    lr = cfg.LEARNING_RATE * world_size if cfg.SCALE_LR_WITH_WORLD else cfg.LEARNING_RATE
    if cfg.OPTIMIZER.lower() == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.WEIGHT_DECAY)
    if cfg.OPTIMIZER.lower() == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.WEIGHT_DECAY)
    if cfg.OPTIMIZER.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=cfg.WEIGHT_DECAY)
    raise ValueError(f"Unknown optimizer: {cfg.OPTIMIZER}")


def build_scheduler(cfg: Config, optimizer: optim.Optimizer):
    if cfg.SCHEDULER.lower() == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.SCHEDULER_PARAMS.get("T_max", cfg.NUM_EPOCHS),
            eta_min=cfg.SCHEDULER_PARAMS.get("eta_min", 1e-5),
        )
    if cfg.SCHEDULER.lower() == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.SCHEDULER_PARAMS.get("step_size", 50),
            gamma=cfg.SCHEDULER_PARAMS.get("gamma", 0.1),
        )
    return None


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    cfg: Config,
    save_predictions: bool = False,
):
    model.eval()
    tracker = MetricTracker(thresholds=cfg.SDR_THRESHOLDS)
    prediction_records = []
    for volume, heatmaps, coords, filename in loader:
        volume = volume.to(device, non_blocking=True)
        heatmaps = heatmaps.to(device, non_blocking=True)
        coords = coords.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=cfg.USE_AMP and device.type == "cuda"):
            outputs = model(volume, return_dict=True)
            loss = criterion(outputs, heatmaps)
        tracker.update(outputs, coords, float(loss.item()))
        if save_predictions:
            pred_coords = heatmaps_to_argmax_coords(outputs["heatmaps"].detach().float()).cpu()
            gt_coords = coords.detach().float().cpu()
            for bi in range(pred_coords.shape[0]):
                case_name = filename[bi] if isinstance(filename, (list, tuple)) else str(filename)
                case_id = Path(case_name).stem
                prediction_records.append(
                    {
                        "case_id": case_id,
                        "filename": case_name,
                        "pred": pred_coords[bi].tolist(),
                        "gt": gt_coords[bi].tolist(),
                    }
                )
    metrics = tracker.compute()
    metrics["prediction_records"] = prediction_records
    return metrics


def train_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    cfg = Config()
    split_name = Path(cfg.SPLIT_FILE).stem
    cfg.SAVE_DIR = os.path.join(cfg.SAVE_DIR, split_name)
    cfg.LOG_DIR = os.path.join(cfg.LOG_DIR, split_name)

    setup_distributed(rank, world_size, cfg.DIST_BACKEND, args.master_port)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        os.makedirs(cfg.SAVE_DIR, exist_ok=True)
        os.makedirs(cfg.LOG_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cfg.SPLIT_FILE), exist_ok=True)
        writer = SummaryWriter(log_dir=cfg.LOG_DIR) if SummaryWriter is not None else None
        print("Training 3D U-Net keypoint detector | split=train/val/test")
        print(f"Dataset type: {getattr(cfg, 'DATASET_TYPE', 'pkl')}")
        print(f"Data: {cfg.DATA_DIR}")
        print(f"Split: {cfg.SPLIT_FILE}")
        print(f"Save: {cfg.SAVE_DIR}")
        print(f"World size: {world_size}")
    else:
        writer = None
    distributed_barrier(rank)

    train_ds, val_ds, test_ds = build_datasets(cfg)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_rank_ds = Subset(val_ds, list(range(rank, len(val_ds), world_size)))
    test_rank_ds = Subset(test_ds, list(range(rank, len(test_ds), world_size)))
    num_workers = max(0, min(cfg.NUM_WORKERS, max(1, cfg.NUM_WORKERS // max(world_size, 1))))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_rank_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_rank_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    model = build_model(cfg).to(device)
    model = DDP(model, device_ids=[rank] if device.type == "cuda" else None)
    criterion = KeypointHeatmapLoss(loss_type=cfg.LOSS_TYPE, bce_weight=cfg.BCE_WEIGHT).to(device)
    optimizer = build_optimizer(cfg, model, world_size)
    scheduler = build_scheduler(cfg, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.USE_AMP and device.type == "cuda")

    if rank == 0:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Test samples: {len(test_ds)}")
        print(f"Trainable params: {trainable / 1e6:.2f}M")

    best_val_mre = float("inf")
    start_epoch = 1
    last_checkpoint_path = os.path.join(cfg.SAVE_DIR, "checkpoint_last.pth")
    if getattr(args, "resume", False):
        resume_path = getattr(args, "resume_path", None) or last_checkpoint_path
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        best_val_mre = float(checkpoint.get("best_val_mre", best_val_mre))
        start_epoch = int(checkpoint["epoch"]) + 1
        if rank == 0:
            print(f"Resumed from {resume_path}: next_epoch={start_epoch}, best_val_mre={best_val_mre:.4f}")
    distributed_barrier(rank)

    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = torch.tensor(0.0, device=device)
        total_count = torch.tensor(0.0, device=device)
        iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.NUM_EPOCHS}") if rank == 0 else train_loader

        for volume, heatmaps, _coords, _filename in iterator:
            volume = volume.to(device, non_blocking=True)
            heatmaps = heatmaps.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=cfg.USE_AMP and device.type == "cuda"):
                outputs = model(volume, return_dict=True)
                loss = criterion(outputs, heatmaps)
            if not torch.isfinite(loss):
                if rank == 0:
                    print(f"Skipping non-finite loss at epoch {epoch}: {float(loss.detach().cpu())}")
                continue
            scaler.scale(loss).backward()
            if cfg.GRAD_CLIP:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.detach()
            total_count += 1
            if rank == 0 and isinstance(iterator, tqdm):
                iterator.set_postfix(loss=float(loss.item()), lr=optimizer.param_groups[0]["lr"])

        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
        avg_loss = float((total_loss / total_count.clamp_min(1)).item())
        if rank == 0:
            print(f"[Epoch {epoch}] train_loss={avg_loss:.6f}")
            if writer is not None:
                writer.add_scalar("Train/Loss", avg_loss, epoch)
                writer.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], epoch)

        if epoch % cfg.VAL_INTERVAL == 0:
            metrics = validate(model, val_loader, criterion, device, cfg, save_predictions=False)
            gathered = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, metrics)
            if rank == 0:
                merged = _merge_rank_metrics(gathered)
                print(
                    f"[Epoch {epoch}] val_loss={merged['loss']:.6f} "
                    f"MRE={merged['mre']:.4f} SDR@2={merged['sdr_2']:.4f} "
                    f"SDR@3={merged['sdr_3']:.4f} SDR@4={merged['sdr_4']:.4f} "
                    f"SDR@5={merged['sdr_5']:.4f} SDR@10={merged['sdr_10']:.4f}"
                )
                if writer is not None:
                    for key, value in merged.items():
                        writer.add_scalar(f"Val/{key}", value, epoch)
                if merged["mre"] < best_val_mre:
                    best_val_mre = merged["mre"]
                    torch.save(model.module.state_dict(), os.path.join(cfg.SAVE_DIR, "best_model.pth"))
                    print(f"Best model saved: val_mre={best_val_mre:.4f}")

        if cfg.SAVE_INTERVAL > 0 and epoch % cfg.SAVE_INTERVAL == 0 and rank == 0:
            torch.save(model.module.state_dict(), os.path.join(cfg.SAVE_DIR, f"epoch_{epoch}.pth"))
        if scheduler is not None:
            scheduler.step()
        if rank == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "scaler": scaler.state_dict(),
                    "best_val_mre": best_val_mre,
                    "split_file": cfg.SPLIT_FILE,
                },
                last_checkpoint_path,
            )
        distributed_barrier(rank)

    best_model_path = os.path.join(cfg.SAVE_DIR, "best_model.pth")
    if os.path.isfile(best_model_path):
        state_dict = torch.load(best_model_path, map_location=device)
        model.module.load_state_dict(state_dict)
        if rank == 0:
            print(f"Loaded best model for final export: {best_model_path}")

    if getattr(cfg, "EXPORT_VALIDATION_JSON_AT_END", True):
        metrics = validate(model, val_loader, criterion, device, cfg, save_predictions=True)
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, metrics)
        if rank == 0:
            merged = _merge_rank_metrics(gathered)
            _write_prediction_jsons(gathered, cfg.SAVE_DIR, split_name="validation")
            print(
                f"[Final validation export] val_loss={merged['loss']:.6f} "
                f"MRE={merged['mre']:.4f} SDR@2={merged['sdr_2']:.4f} "
                f"SDR@3={merged['sdr_3']:.4f} SDR@4={merged['sdr_4']:.4f} "
                f"SDR@5={merged['sdr_5']:.4f} SDR@10={merged['sdr_10']:.4f}"
            )
        distributed_barrier(rank)

        metrics = validate(model, test_loader, criterion, device, cfg, save_predictions=True)
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, metrics)
        if rank == 0:
            merged = _merge_rank_metrics(gathered)
            _write_prediction_jsons(gathered, cfg.SAVE_DIR, split_name="test")
            print(
                f"[Final test export] test_loss={merged['loss']:.6f} "
                f"MRE={merged['mre']:.4f} SDR@2={merged['sdr_2']:.4f} "
                f"SDR@3={merged['sdr_3']:.4f} SDR@4={merged['sdr_4']:.4f} "
                f"SDR@5={merged['sdr_5']:.4f} SDR@10={merged['sdr_10']:.4f}"
            )
    distributed_barrier(rank)

    if rank == 0 and writer is not None:
        writer.close()
    cleanup_distributed()


def _merge_rank_metrics(metrics_list):
    total_points = sum(float(m.get("num_points", 0.0)) for m in metrics_list)
    total_points = max(total_points, 1.0)
    merged = {
        "loss": sum(float(m["loss"]) for m in metrics_list) / max(len(metrics_list), 1),
        "mre": sum(float(m.get("distance_sum", 0.0)) for m in metrics_list) / total_points,
        "num_points": total_points,
    }
    for threshold in Config.SDR_THRESHOLDS:
        key = f"sdr_{int(threshold)}"
        count_key = f"{key}_count"
        merged[key] = sum(float(m.get(count_key, 0.0)) for m in metrics_list) / total_points
    return merged


def _write_prediction_jsons(metrics_list, output_dir: str, split_name: str) -> None:
    records = []
    for metrics in metrics_list:
        records.extend(metrics.get("prediction_records", []))
    records = sorted(records, key=lambda item: item["case_id"])

    validation_dir = Path(output_dir) / split_name
    validation_dir.mkdir(parents=True, exist_ok=True)

    prediction_all = {}
    gt_all = {}
    for record in records:
        case_id = record["case_id"]
        case_prediction = {}
        case_gt = {}
        for idx, pred_coord in enumerate(record["pred"], start=1):
            label = str(idx)
            lm_name = f"landmark_{idx}"
            coord = [int(round(float(v))) for v in pred_coord]
            gt_coord = [float(v) for v in record["gt"][idx - 1]]
            case_prediction[label] = {"coordinates": coord, "likelihood": 1.0}
            case_gt[lm_name] = gt_coord
        prediction_all[case_id] = {
            f"landmark_{idx}": [int(round(float(v))) for v in pred_coord]
            for idx, pred_coord in enumerate(record["pred"], start=1)
        }
        gt_all[case_id] = case_gt
        with open(validation_dir / f"{case_id}.json", "w", encoding="utf-8") as f:
            json.dump(case_prediction, f, indent=2, ensure_ascii=False)

    summary = summarize_prediction_jsons(records, thresholds=Config.SDR_THRESHOLDS)
    with open(validation_dir / "prediction_all_landmark_voxel.json", "w", encoding="utf-8") as f:
        json.dump(prediction_all, f, indent=2, ensure_ascii=False)
    with open(validation_dir / "gt_all_landmark_voxel.json", "w", encoding="utf-8") as f:
        json.dump(gt_all, f, indent=2, ensure_ascii=False)
    with open(validation_dir / "summary_voxel.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(validation_dir / "summary_mm.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def run_split(args: argparse.Namespace) -> None:
    gpu_list = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    world_size = args.world_size if args.world_size is not None else len(gpu_list)
    if world_size != len(gpu_list):
        world_size = len(gpu_list)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_list))
    print(f"Starting train/val/test split on GPUs {gpu_list}, world_size={world_size}")
    mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 3D U-Net keypoint detector")
    parser.add_argument("--gpus", type=str, default="0", help="GPU IDs, e.g. 0 or 0,1")
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--master_port", type=str, default="12412")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint_last.pth")
    parser.add_argument("--resume_path", type=str, default=None, help="Optional explicit resume checkpoint")
    args = parser.parse_args()

    run_split(args)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
