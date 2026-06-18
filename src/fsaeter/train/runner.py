"""DDP-ready local SAE training runner."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from fsaeter.backends import TorchDenseBackend, TritonSparseBackend
from fsaeter.config_compat import normalize_train_config
from fsaeter.data.cache import resolve_token_cache_info
from fsaeter.data.datasets import PatchTokenMemmapDataset, split_image_rows
from fsaeter.h.helpers import autocast_context
from fsaeter.models.local_sae import (
    RunningFeatureStats,
    build_local_sae,
    save_local_sae_checkpoint,
)
from fsaeter.utils.config import resolve_path, save_yaml_config
from fsaeter.utils.distributed import (
    barrier,
    cleanup_distributed,
    init_distributed,
    is_distributed,
    is_main_process,
    local_rank,
    world_size,
)


def get_device(train_cfg: dict) -> torch.device:
    requested = str(train_cfg.get("device", "auto")).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            if is_distributed():
                return torch.device("cuda", local_rank())
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "cuda" and is_distributed():
        return torch.device("cuda", local_rank())
    return torch.device(requested)


def build_backend(name: str):
    normalized = str(name).lower()
    if normalized == "torch_dense":
        return TorchDenseBackend()
    if normalized == "triton_sparse":
        return TritonSparseBackend()
    raise ValueError(f"Unknown backend {name!r}")


def _unwrap_model(model: torch.nn.Module):
    if isinstance(model, DDP):
        return model.module
    return model


def run_epoch(
    *,
    model: torch.nn.Module,
    backend,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    optimizer: torch.optim.Optimizer | None,
    grad_clip_norm: float | None,
) -> tuple[dict, np.ndarray]:
    is_train = optimizer is not None
    base_model = _unwrap_model(model)
    stats = RunningFeatureStats(d_sae=int(base_model.d_sae))
    for batch in loader:
        x = batch[0].to(device=device, dtype=torch.float32, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            with autocast_context(device, precision):
                outputs = backend.forward_loss(model, x)
            for key in ("loss", "recon_mse", "aux_loss"):
                value = outputs[key]
                if torch.is_tensor(value) and value.ndim > 0:
                    outputs[key] = value.mean()
            loss = outputs["loss"]
            if is_train:
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                optimizer.step()
                if (
                    hasattr(base_model, "normalize_decoder_rows_")
                    and base_model.decoder_row_norm
                ):
                    base_model.normalize_decoder_rows_()
        stats.update(
            batch_size=int(x.shape[0]),
            loss=outputs["loss"].detach(),
            recon_mse=outputs["recon_mse"].detach(),
            aux_loss=outputs["aux_loss"].detach(),
            features=outputs["features"].detach(),
        )

    summary = stats.summary()
    return {
        "loss": float(summary.loss),
        "recon_mse": float(summary.recon_mse),
        "aux_loss": float(summary.aux_loss),
        "mean_l0": float(summary.mean_l0),
        "max_l0": int(summary.max_l0),
        "dead_fraction": float(summary.dead_fraction),
    }, stats.feature_frequency()


def run_training(config: dict, *, base_root: Path) -> dict:
    config = normalize_train_config(config)
    run_cfg = dict(config.get("run") or {})
    tokens_cfg = dict(config.get("tokens") or {})
    sae_cfg = dict(config.get("sae") or {})
    train_cfg = dict(config.get("train") or {})

    device = get_device(train_cfg)
    init_distributed(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    tokens_dir = resolve_path(tokens_cfg.get("cache_dir", ""), base=base_root)
    token_info = resolve_token_cache_info(tokens_dir)
    out_dir = resolve_path(run_cfg.get("out_dir", "outputs/local_sae_run"), base=base_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    sae_cfg.setdefault("d_model", int(token_info.d_model))
    if int(sae_cfg["d_model"]) != int(token_info.d_model):
        raise ValueError(
            f"Config d_model={sae_cfg['d_model']} does not match token dim "
            f"{token_info.d_model}"
        )
    config["sae"] = sae_cfg

    train_rows, val_rows = split_image_rows(
        token_info.num_images,
        val_fraction=float(train_cfg.get("val_fraction", 0.1)),
        seed=int(train_cfg.get("split_seed", run_cfg.get("seed", 0))),
    )
    train_set = PatchTokenMemmapDataset(
        tokens_dir,
        image_rows=train_rows,
        max_rows=train_cfg.get("max_train_rows"),
    )
    val_set = PatchTokenMemmapDataset(
        tokens_dir,
        image_rows=val_rows,
        max_rows=train_cfg.get("max_val_rows"),
    )

    if is_main_process():
        np.save(out_dir / "train_image_rows.npy", train_rows.astype(np.int64, copy=False))
        np.save(out_dir / "val_image_rows.npy", val_rows.astype(np.int64, copy=False))
        save_yaml_config(config, out_dir / "config_resolved.yaml")

    batch_size = max(1, int(train_cfg.get("batch_size", 1024)))
    num_workers = max(0, int(train_cfg.get("num_workers", 4)))
    epochs = max(1, int(train_cfg.get("epochs", 1)))
    grad_clip_norm = train_cfg.get("grad_clip_norm")
    if grad_clip_norm is not None:
        grad_clip_norm = float(grad_clip_norm)

    dist_rank = 0
    if is_distributed() and torch.distributed.is_initialized():
        dist_rank = int(torch.distributed.get_rank())
    train_sampler = (
        DistributedSampler(
            train_set,
            num_replicas=world_size(),
            rank=dist_rank,
            shuffle=True,
        )
        if is_distributed()
        else None
    )
    val_sampler = (
        DistributedSampler(
            val_set,
            num_replicas=world_size(),
            rank=dist_rank,
            shuffle=False,
        )
        if is_distributed() and len(val_set) > 0
        else None
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = build_local_sae(config).to(device)
    if bool(train_cfg.get("compile", False)) and hasattr(torch, "compile") and not is_distributed():
        model = torch.compile(model)  # type: ignore[assignment]
    if is_distributed():
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    optimizer = torch.optim.AdamW(
        _unwrap_model(model).parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        betas=(float(train_cfg.get("beta1", 0.9)), float(train_cfg.get("beta2", 0.95))),
    )
    backend = build_backend(train_cfg.get("backend", "torch_dense"))
    precision = str(train_cfg.get("precision", "fp32"))

    history: list[dict] = []
    best_val_loss: float | None = None
    best_epoch = -1
    metrics_jsonl = out_dir / "epoch_metrics.jsonl"

    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        epoch_start = time.time()
        model.train()
        train_metrics, train_feature_freq = run_epoch(
            model=model,
            backend=backend,
            loader=train_loader,
            device=device,
            precision=precision,
            optimizer=optimizer,
            grad_clip_norm=grad_clip_norm,
        )

        model.eval()
        if len(val_set) > 0:
            val_metrics, val_feature_freq = run_epoch(
                model=model,
                backend=backend,
                loader=val_loader,
                device=device,
                precision=precision,
                optimizer=None,
                grad_clip_norm=None,
            )
        else:
            val_metrics = {
                k: float("nan")
                for k in ("loss", "recon_mse", "aux_loss", "mean_l0", "dead_fraction")
            }
            val_metrics["max_l0"] = 0
            val_feature_freq = np.zeros((int(_unwrap_model(model).d_sae),), dtype=np.float32)

        epoch_seconds = float(time.time() - epoch_start)
        record = {
            "epoch": epoch,
            "epoch_seconds": epoch_seconds,
            "train_rows_per_second": float(len(train_set) / max(epoch_seconds, 1e-6)),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)

        if is_main_process():
            with metrics_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            np.save(
                out_dir / "feature_frequency_train_last.npy",
                train_feature_freq.astype(np.float32, copy=False),
            )
            np.save(
                out_dir / "feature_frequency_val_last.npy",
                val_feature_freq.astype(np.float32, copy=False),
            )

            checkpoints_dir = out_dir / "checkpoints"
            base_model = _unwrap_model(model)
            save_local_sae_checkpoint(
                checkpoints_dir / f"ep-{epoch:07d}.pt",
                model=base_model,
                config=config,
                epoch=epoch,
                step=epoch * len(train_loader),
                best_val_loss=best_val_loss,
                history=history,
            )
            if not np.isnan(val_metrics["loss"]):
                current_val = float(val_metrics["loss"])
            else:
                current_val = float(train_metrics["loss"])
            if best_val_loss is None or current_val < best_val_loss:
                best_val_loss = current_val
                best_epoch = epoch
                save_local_sae_checkpoint(
                    checkpoints_dir / "best.pt",
                    model=base_model,
                    config=config,
                    epoch=epoch,
                    step=epoch * len(train_loader),
                    best_val_loss=best_val_loss,
                    history=history,
                )
        barrier()

    summary = {
        "run_name": run_cfg.get("name"),
        "tokens_dir": str(tokens_dir),
        "out_dir": str(out_dir),
        "best_epoch": int(best_epoch),
        "best_val_loss": None if best_val_loss is None else float(best_val_loss),
        "num_epochs": int(epochs),
        "variant": str(sae_cfg.get("variant", "batchtopk")),
        "d_model": int(sae_cfg["d_model"]),
        "d_sae": int(sae_cfg["d_sae"]),
        "target_k": int(sae_cfg["target_k"]),
        "train_rows": int(len(train_set)),
        "val_rows": int(len(val_set)),
        "epoch_metrics_jsonl": str(metrics_jsonl),
        "best_checkpoint": str(out_dir / "checkpoints" / "best.pt"),
        "world_size": int(world_size()),
        "backend": backend.name,
    }
    if is_main_process():
        with (out_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2, sort_keys=True)
        with (out_dir / "train_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    cleanup_distributed()
    return summary
