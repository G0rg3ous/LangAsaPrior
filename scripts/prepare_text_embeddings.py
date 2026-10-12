#!/usr/bin/env python3
"""Prepare fixed embeddings for text-ablation experiments."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def load_embedding_payload(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if "embeddings" not in obj:
            raise KeyError(f"{path} is a dict but does not contain an 'embeddings' key")
        return obj["embeddings"].detach().float(), dict(obj)
    if not torch.is_tensor(obj):
        raise TypeError(f"Expected tensor or dict payload in {path}, got {type(obj)}")
    return obj.detach().float(), {}


def save_random_fixed(reference: Path, output: Path, seed: int) -> None:
    ref_embeddings, ref_payload = load_embedding_payload(reference)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    embeddings = torch.randn(ref_embeddings.shape, generator=generator, dtype=torch.float32)
    embeddings = F.normalize(embeddings, p=2, dim=-1)
    payload = {
        "embeddings": embeddings,
        "mode": "random_fixed",
        "seed": int(seed),
        "reference_file": str(reference),
        "embedding_dim": int(embeddings.shape[-1]),
        "num_embeddings": int(embeddings.shape[0]),
        "reference_model_name": ref_payload.get("model_name"),
        "normalize": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


def load_prompts(path: Path) -> tuple[list[str], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("prompts", payload.get("landmarks", payload))
    if not isinstance(payload, list):
        raise TypeError("Prompt file must be a JSON list or dict containing prompts/landmarks")

    prompts = []
    names = []
    for idx, item in enumerate(payload, start=1):
        if isinstance(item, str):
            prompt = item
            name = re.sub(r"[^A-Za-z0-9]+", "_", item).strip("_") or f"landmark_{idx}"
        elif isinstance(item, dict):
            prompt = str(item.get("prompt", item.get("text", item.get("name", f"landmark {idx}"))))
            name = str(item.get("name", item.get("landmark", f"landmark_{idx}")))
        else:
            raise TypeError(f"Unsupported prompt item: {type(item)}")
        prompts.append(prompt)
        names.append(name)
    return prompts, names


@torch.no_grad()
def encode_prompts(
    model_name: str,
    prompts: list[str],
    device: str,
    dtype: str,
    batch_size: int,
    normalize: bool,
) -> torch.Tensor:
    try:
        import transformers
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required to generate landmark-ID-only embeddings") from exc

    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    candidates = [
        getattr(transformers, "AutoModelForImageTextToText", None),
        getattr(transformers, "AutoModelForVision2Seq", None),
        AutoModelForCausalLM,
        AutoModel,
    ]
    last_error: Exception | None = None
    for model_cls in [item for item in candidates if item is not None]:
        try:
            model = model_cls.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
            break
        except Exception as exc:
            last_error = exc
    else:
        raise RuntimeError(f"Could not load model from {model_name}") from last_error

    model.to(device)
    model.eval()
    chunks = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = model(**encoded, output_hidden_states=True, return_dict=True, use_cache=False)
        hidden = outputs.hidden_states[-1] if getattr(outputs, "hidden_states", None) is not None else outputs.last_hidden_state
        mask = encoded["attention_mask"].to(hidden.dtype)
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        chunks.append(pooled.float().cpu())

    embeddings = torch.cat(chunks, dim=0)
    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=-1)
    return embeddings


def save_landmark_id_only(args: argparse.Namespace) -> None:
    prompts, names = load_prompts(args.prompts)
    embeddings = encode_prompts(
        model_name=args.model_name,
        prompts=prompts,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        normalize=args.normalize,
    )
    payload = {
        "embeddings": embeddings,
        "mode": "landmark_id_only",
        "prompts": prompts,
        "landmark_names": names,
        "model_name": args.model_name,
        "embedding_dim": int(embeddings.shape[-1]),
        "num_embeddings": int(embeddings.shape[0]),
        "normalize": bool(args.normalize),
        "input_file": str(args.prompts),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    random_parser = subparsers.add_parser("random-fixed")
    random_parser.add_argument("--reference", type=Path, required=True)
    random_parser.add_argument("--output", type=Path, required=True)
    random_parser.add_argument("--seed", type=int, default=20260608)

    id_parser = subparsers.add_parser("landmark-id-only")
    id_parser.add_argument("--prompts", type=Path, required=True)
    id_parser.add_argument("--output", type=Path, required=True)
    id_parser.add_argument("--model-name", required=True)
    id_parser.add_argument("--device", default="cuda")
    id_parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    id_parser.add_argument("--batch-size", type=int, default=4)
    id_parser.add_argument("--normalize", action="store_true", default=True)

    args = parser.parse_args()
    if args.mode == "random-fixed":
        save_random_fixed(args.reference, args.output, args.seed)
    elif args.mode == "landmark-id-only":
        save_landmark_id_only(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
