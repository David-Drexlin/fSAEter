"""CLI entrypoints for fSAEter."""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fsaeter.backbones import LocalBackbone
from fsaeter.config_compat import normalize_build_h_config, normalize_extract_config, normalize_train_config
from fsaeter.data.cache import TokenCacheWriter, build_token_metadata, resolve_token_cache_info
from fsaeter.data.imagefolder import IndexedSubset, build_imagefolder_dataset, make_image_records, summarize_selection, write_jsonl
from fsaeter.h.build import run_build_h
from fsaeter.inspect.basic_qc import run_basic_qc
from fsaeter.train.runner import run_training
from fsaeter.utils.config import load_yaml_config, resolve_path, runtime_base_root, save_yaml_config
from fsaeter.utils.distributed import is_distributed, is_main_process


def get_device(device_value: str) -> torch.device:
    requested = str(device_value).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def autocast_context(device: torch.device, precision: str):
    precision = str(precision).lower()
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device.type == "cuda" and precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type=device.type, enabled=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fsaeter", description="Local SAE tooling for vision token caches.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract-tokens", help="Extract patch/global tokens into memmaps.")
    p_extract.add_argument("--config", required=True)
    p_extract.add_argument("--data-root", default=None)
    p_extract.add_argument("--split", default=None)
    p_extract.add_argument("--encoder", default=None)
    p_extract.add_argument("--batch-size", type=int, default=None)
    p_extract.add_argument("--out", default=None)
    p_extract.add_argument("--max-images", type=int, default=None)
    p_extract.add_argument("--dry-run", action="store_true")

    p_train = sub.add_parser("train-sae", help="Train a local SAE on patch-token memmaps.")
    p_train.add_argument("--config", required=True)
    p_train.add_argument("--tokens", default=None)
    p_train.add_argument("--out", default=None)
    p_train.add_argument("--device", default=None)
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--dry-run", action="store_true")

    p_build = sub.add_parser("build-h", help="Build H matrices from a local SAE checkpoint.")
    p_build.add_argument("--config", required=True)
    p_build.add_argument("--tokens", default=None)
    p_build.add_argument("--checkpoint", default=None)
    p_build.add_argument("--out", default=None)
    p_build.add_argument("--device", default=None)
    p_build.add_argument("--max-images", type=int, default=None)
    p_build.add_argument("--dry-run", action="store_true")

    p_inspect = sub.add_parser("inspect", help="Inspect a built concept space.")
    p_inspect.add_argument("--config", required=True)
    p_inspect.add_argument("--tokens", default=None)
    p_inspect.add_argument("--checkpoint", default=None)
    p_inspect.add_argument("--concept-dir", default=None)
    p_inspect.add_argument("--device", default=None)
    p_inspect.add_argument("--dry-run", action="store_true")

    p_mine = sub.add_parser("mine-concepts", help="Mine candidate concepts and previews from a built concept space.")
    p_mine.add_argument("--config", required=True)
    p_mine.add_argument("--tokens", default=None)
    p_mine.add_argument("--checkpoint", default=None)
    p_mine.add_argument("--concept-dir", default=None)
    p_mine.add_argument("--device", default=None)
    p_mine.add_argument("--dry-run", action="store_true")

    return parser


def _apply_extract_overrides(config: dict, args: argparse.Namespace) -> dict:
    config = normalize_extract_config(config)
    config.setdefault("run", {})
    config.setdefault("data", {})
    config.setdefault("encoder", {})
    config.setdefault("tokens", {})
    config["data"].setdefault("subset", {})
    if args.data_root is not None:
        config["data"]["root"] = args.data_root
    if args.split is not None:
        config["data"]["split"] = args.split
    if args.encoder is not None:
        config["encoder"]["name"] = args.encoder
    if args.batch_size is not None:
        config["tokens"]["batch_size"] = int(args.batch_size)
    if args.out is not None:
        config["run"]["out_dir"] = args.out
    if args.max_images is not None:
        config["data"]["subset"]["max_images"] = int(args.max_images)
    return config


def _apply_train_overrides(config: dict, args: argparse.Namespace) -> dict:
    config = normalize_train_config(config)
    config.setdefault("run", {})
    config.setdefault("tokens", {})
    config.setdefault("train", {})
    if args.tokens is not None:
        config["tokens"]["cache_dir"] = args.tokens
    if args.out is not None:
        config["run"]["out_dir"] = args.out
    if args.device is not None:
        config["train"]["device"] = args.device
    if args.epochs is not None:
        config["train"]["epochs"] = int(args.epochs)
    return config


def _apply_build_overrides(config: dict, args: argparse.Namespace) -> dict:
    config = normalize_build_h_config(config)
    config.setdefault("run", {})
    config.setdefault("tokens", {})
    config.setdefault("sae", {})
    config.setdefault("build_h", {})
    if args.tokens is not None:
        config["tokens"]["cache_dir"] = args.tokens
    if args.checkpoint is not None:
        config["sae"]["checkpoint"] = args.checkpoint
    if args.out is not None:
        config["run"]["out_dir"] = args.out
    if args.device is not None:
        config["build_h"]["device"] = args.device
    if args.max_images is not None:
        config["build_h"]["max_images"] = int(args.max_images)
    return config


def _apply_inspect_overrides(config: dict, args: argparse.Namespace) -> dict:
    config = normalize_build_h_config(config)
    config.setdefault("run", {})
    config.setdefault("tokens", {})
    config.setdefault("sae", {})
    config.setdefault("inspect", {})
    if args.tokens is not None:
        config["tokens"]["cache_dir"] = args.tokens
    if args.checkpoint is not None:
        config["sae"]["checkpoint"] = args.checkpoint
    if args.concept_dir is not None:
        config["run"]["out_dir"] = args.concept_dir
    if args.device is not None:
        config["inspect"]["device"] = args.device
    return config


@torch.no_grad()
def run_extract_tokens(config: dict, *, base_root: Path, dry_run: bool = False) -> dict:
    config = normalize_extract_config(config)
    run_cfg = dict(config.get("run") or {})
    data_cfg = dict(config.get("data") or {})
    token_cfg = dict(config.get("tokens") or {})
    encoder_cfg = dict(config.get("encoder") or {})

    out_dir = resolve_path(run_cfg.get("out_dir", "outputs/tokens"), base=base_root)
    tokens_dir = out_dir / "tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)

    dataset, selected_indices = build_imagefolder_dataset(config, base_root=base_root)
    summary = summarize_selection(dataset, selected_indices)
    if dry_run:
        return {"selection": summary}

    device = get_device(token_cfg.get("device", "auto"))
    backbone = LocalBackbone(
        encoder_cfg=encoder_cfg,
        device=device,
        base_root=base_root,
    ).eval().requires_grad_(False)

    subset = IndexedSubset(dataset, selected_indices)
    loader = DataLoader(
        subset,
        batch_size=int(token_cfg.get("batch_size", 16)),
        shuffle=False,
        num_workers=int(token_cfg.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    include_global = bool(encoder_cfg.get("include_global", True))
    normalize_tokens = bool(encoder_cfg.get("l2_normalize_tokens", True))
    save_dtype = str(token_cfg.get("save_dtype", "float16"))
    precision = str(token_cfg.get("precision", "fp32"))

    writer = None
    labels = np.empty((len(subset),), dtype=np.int64)
    records_by_row = make_image_records(dataset, selected_indices, data_root=resolve_path(data_cfg["root"], base=base_root))

    offset = 0
    for images, batch_labels, _dataset_indices, _paths in loader:
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        with autocast_context(device, precision):
            backbone_out = backbone.forward_tokens(images, include_global=include_global)
        patch_tokens = backbone_out.patch_tokens.float()
        global_tokens = None if backbone_out.global_tokens is None else backbone_out.global_tokens.float()
        if normalize_tokens:
            patch_tokens = torch.nn.functional.normalize(patch_tokens, dim=-1)
        if global_tokens is not None and bool(encoder_cfg.get("l2_normalize_global", False)):
            global_tokens = torch.nn.functional.normalize(global_tokens, dim=-1)
        if writer is None:
            writer = TokenCacheWriter(
                tokens_dir,
                num_images=len(subset),
                patch_shape=patch_tokens.shape[1:],
                global_shape=None if global_tokens is None else global_tokens.shape[1:],
                save_dtype=save_dtype,
            )
        batch_size = int(images.shape[0])
        writer.write(patch_tokens, global_tokens)
        labels[offset : offset + batch_size] = batch_labels.numpy().astype(np.int64, copy=False)
        offset += batch_size

    if writer is None:
        raise RuntimeError("No batches were produced by the dataloader.")
    writer.close()
    np.save(tokens_dir / "labels.npy", labels)
    write_jsonl(tokens_dir / "image_ids.jsonl", (dataclasses.asdict(record) for record in records_by_row))
    class_counts = Counter(int(label) for label in labels.tolist())
    metadata = build_token_metadata(
        config=config,
        encoder=backbone,
        num_images=len(subset),
        patch_shape=writer.patch_shape,
        global_shape=writer.global_shape,
        output_dir=tokens_dir,
        class_counts=class_counts,
        class_to_idx=dataset.class_to_idx,
    )
    with (tokens_dir / "token_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    save_yaml_config(config, tokens_dir / "config_resolved.yaml")
    return {"selection": summary, "wrote": str(tokens_dir), "metadata": metadata}


def run_inspect_command(config: dict, *, base_root: Path, dry_run: bool = False) -> dict:
    config = normalize_build_h_config(config)
    run_cfg = dict(config.get("run") or {})
    tokens_cfg = dict(config.get("tokens") or {})
    sae_cfg = dict(config.get("sae") or {})
    inspect_cfg = dict(config.get("inspect") or {})
    concept_dir = resolve_path(run_cfg.get("out_dir", "outputs/local_sae_h"), base=base_root)
    tokens_dir = resolve_path(tokens_cfg.get("cache_dir", ""), base=base_root)
    checkpoint_path = resolve_path(sae_cfg.get("checkpoint", ""), base=base_root)
    device = get_device(inspect_cfg.get("device", "auto"))
    if dry_run:
        return {"concept_dir": str(concept_dir), "tokens_dir": str(tokens_dir), "checkpoint": str(checkpoint_path)}
    qc = run_basic_qc(
        concept_dir=concept_dir,
        tokens_dir=tokens_dir,
        checkpoint_path=checkpoint_path,
        device=device,
        precision=str(inspect_cfg.get("precision", "fp32")),
        preview_score_mode=str(inspect_cfg.get("preview_score_mode", "max")),
        preview_concepts=int(inspect_cfg.get("preview_concepts", 12)),
        preview_images_per_concept=int(inspect_cfg.get("preview_images_per_concept", 12)),
        min_support=int(inspect_cfg.get("min_support", 64)),
        min_class_coverage=int(inspect_cfg.get("min_class_coverage", 8)),
        min_per_class=int(inspect_cfg.get("min_per_class", 4)),
        top_candidate_count=int(inspect_cfg.get("top_candidate_count", 50)),
    )
    return {"concept_dir": str(concept_dir), "qc": qc}


def preview_train_command(config: dict, *, base_root: Path) -> dict:
    config = normalize_train_config(config)
    token_info = resolve_token_cache_info(resolve_path(config["tokens"]["cache_dir"], base=base_root))
    train_cfg = dict(config.get("train") or {})
    sae_cfg = dict(config.get("sae") or {})
    return {
        "tokens_dir": str(resolve_path(config["tokens"]["cache_dir"], base=base_root)),
        "num_images": int(token_info.num_images),
        "tokens_per_image": int(token_info.tokens_per_image),
        "d_model": int(token_info.d_model),
        "variant": str(sae_cfg.get("variant", "batchtopk")),
        "target_k": int(sae_cfg.get("target_k", 0)),
        "d_sae": int(sae_cfg.get("d_sae", 0)),
        "epochs": int(train_cfg.get("epochs", 1)),
        "batch_size": int(train_cfg.get("batch_size", 1024)),
    }


def preview_build_command(config: dict, *, base_root: Path) -> dict:
    config = normalize_build_h_config(config)
    token_info = resolve_token_cache_info(resolve_path(config["tokens"]["cache_dir"], base=base_root))
    build_cfg = dict(config.get("build_h") or {})
    max_images_cfg = build_cfg.get("max_images")
    max_images = int(token_info.num_images) if max_images_cfg in (None, "", 0, "0") else min(int(max_images_cfg), int(token_info.num_images))
    return {
        "tokens_dir": str(resolve_path(config["tokens"]["cache_dir"], base=base_root)),
        "token_shape": [int(token_info.num_images), int(token_info.tokens_per_image), int(token_info.d_model)],
        "num_images_to_process": int(max_images),
        "checkpoint": str(resolve_path(config["sae"]["checkpoint"], base=base_root)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.config).expanduser()
    config = load_yaml_config(config_path)
    base_root = runtime_base_root(config_path)

    if args.command == "extract-tokens":
        payload = run_extract_tokens(_apply_extract_overrides(config, args), base_root=base_root, dry_run=bool(args.dry_run))
    elif args.command == "train-sae":
        normalized = _apply_train_overrides(config, args)
        payload = preview_train_command(normalized, base_root=base_root) if args.dry_run else run_training(normalized, base_root=base_root)
    elif args.command == "build-h":
        normalized = _apply_build_overrides(config, args)
        payload = preview_build_command(normalized, base_root=base_root) if args.dry_run else run_build_h(normalized, base_root=base_root)
    elif args.command in {"inspect", "mine-concepts"}:
        payload = run_inspect_command(_apply_inspect_overrides(config, args), base_root=base_root, dry_run=bool(args.dry_run))
    else:
        raise ValueError(f"Unknown command {args.command!r}")

    if not is_distributed() or is_main_process():
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
