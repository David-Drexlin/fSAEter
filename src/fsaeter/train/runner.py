"""DDP-ready local SAE training runner."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from fsaeter.backends import TorchDenseBackend, TorchSparseBackend, TritonSparseBackend
from fsaeter.config_compat import normalize_train_config
from fsaeter.data.cache import resolve_token_cache_info
from fsaeter.data.datasets import PatchTokenMemmapDataset, split_image_rows
from fsaeter.h.helpers import autocast_context
from fsaeter.models.local_sae import (
    RunningFeatureStats,
    StepMetrics,
    build_local_sae,
    save_local_sae_checkpoint,
)
from fsaeter.train.stats import load_activation_stats
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
from fsaeter.utils.repro import (
    build_dataloader_generator,
    build_worker_init_fn,
    resolve_run_seed,
    seed_everything,
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
    if normalized == "torch_sparse":
        return TorchSparseBackend()
    if normalized == "triton_sparse":
        return TritonSparseBackend()
    raise ValueError(f"Unknown backend {name!r}")


def _unwrap_model(model: torch.nn.Module):
    if isinstance(model, DDP):
        return model.module
    return model


def build_optimizer(model: torch.nn.Module, train_cfg: dict) -> torch.optim.Optimizer:
    base_model = _unwrap_model(model)
    decay_params = []
    no_decay_params = []
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            continue
        if name in {"b_enc", "b_dec"} or name.endswith("bias"):
            no_decay_params.append(param)
        elif name == "W_dec" and base_model.decoder_row_norm:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return torch.optim.AdamW(
        [
            {
                "params": decay_params,
                "weight_decay": float(train_cfg.get("weight_decay", 1e-4)),
            },
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=float(train_cfg.get("lr", 3e-4)),
        betas=(float(train_cfg.get("beta1", 0.9)), float(train_cfg.get("beta2", 0.95))),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    train_cfg: dict,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    if total_steps <= 0:
        return None
    warmup_steps = max(0, int(train_cfg.get("warmup_steps", 0)))
    decay = str(train_cfg.get("lr_decay", "cosine")).lower()
    min_lr_fraction = float(train_cfg.get("min_lr_fraction", 0.1))

    def lr_lambda(step: int) -> float:
        step = max(0, int(step))
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        if decay in {"none", "constant"}:
            return 1.0
        progress = 0.0
        if total_steps > warmup_steps:
            progress = min(
                1.0,
                max(0.0, float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))),
            )
        if decay == "linear":
            return max(min_lr_fraction, 1.0 - (1.0 - min_lr_fraction) * progress)
        if decay == "cosine":
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_fraction + (1.0 - min_lr_fraction) * cosine
        raise ValueError(f"Unsupported lr_decay {decay!r}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def capture_rng_state() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("python") is not None:
        random.setstate(state["python"])


def load_training_stats_into_model(model: torch.nn.Module, config: dict, *, base_root: Path) -> Path | None:
    base_model = _unwrap_model(model)
    tokens_cfg = dict(config.get("tokens") or {})
    stats_dir_value = tokens_cfg.get("stats_dir")
    if not stats_dir_value:
        return None
    stats_dir = resolve_path(stats_dir_value, base=base_root)
    stats = load_activation_stats(stats_dir)
    base_model.load_activation_stats_(mean=stats["mean"], scale=stats["scale"])
    if bool(dict(config.get("train") or {}).get("init_decoder_bias_from_stats", True)):
        base_model.initialize_decoder_bias_from_stats_()
    return stats_dir


def step_metrics_to_dict(metrics: StepMetrics) -> dict:
    return asdict(metrics)


def empty_step_metrics() -> dict:
    return {
        "loss": float("nan"),
        "recon_mse": float("nan"),
        "aux_loss": float("nan"),
        "mse": float("nan"),
        "normalized_mse": float("nan"),
        "zero_baseline_mse": float("nan"),
        "mean_baseline_mse": float("nan"),
        "variance_explained": float("nan"),
        "mean_l0": float("nan"),
        "p50_l0": float("nan"),
        "p90_l0": float("nan"),
        "p99_l0": float("nan"),
        "max_l0": 0,
        "alive_fraction": float("nan"),
        "dead_fraction": float("nan"),
        "dead_feature_count": 0,
    }


def run_epoch(
    *,
    model: torch.nn.Module,
    backend,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    scaler,
    grad_clip_norm: float | None,
    global_step: int,
) -> tuple[dict, np.ndarray, int]:
    is_train = optimizer is not None
    base_model = _unwrap_model(model)
    stats = RunningFeatureStats(
        d_sae=int(base_model.d_sae),
        d_model=int(base_model.d_model),
    )
    current_step = int(global_step)

    for batch in loader:
        x = batch[0].to(device=device, dtype=torch.float32, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            current_step += 1
        step_for_batch = current_step if is_train else None

        with torch.set_grad_enabled(is_train):
            with autocast_context(device, precision):
                outputs = backend.forward_loss(
                    model,
                    x,
                    global_step=step_for_batch,
                    update_state=is_train,
                )
            for key in ("loss", "recon_mse", "aux_loss"):
                value = outputs[key]
                if torch.is_tensor(value) and value.ndim > 0:
                    outputs[key] = value.mean()
            loss = outputs["loss"]
            if is_train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if base_model.decoder_row_norm:
                    base_model.project_decoder_grad_()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                if base_model.decoder_row_norm:
                    base_model.normalize_decoder_rows_()

        stats.update(
            batch_size=int(x.shape[0]),
            loss=outputs["loss"].detach(),
            recon_mse=outputs["recon_mse"].detach(),
            aux_loss=outputs["aux_loss"].detach(),
            target=outputs["target"].detach(),
            recon=outputs["recon"].detach(),
            features=outputs["features"],
        )

    reduced_summary, feature_frequency = stats.reduced_summary(device=device)
    return step_metrics_to_dict(reduced_summary), feature_frequency, current_step


def run_training(config: dict, *, base_root: Path) -> dict:
    config = normalize_train_config(config)
    run_cfg = dict(config.get("run") or {})
    tokens_cfg = dict(config.get("tokens") or {})
    sae_cfg = dict(config.get("sae") or {})
    train_cfg = dict(config.get("train") or {})
    seed = resolve_run_seed(config)

    device = get_device(train_cfg)
    init_distributed(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed_everything(seed)

    tokens_dir = resolve_path(tokens_cfg.get("cache_dir", ""), base=base_root)
    token_info = resolve_token_cache_info(tokens_dir)
    out_dir = resolve_path(run_cfg.get("out_dir", "outputs/local_sae_run"), base=base_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    sae_cfg.setdefault("d_model", int(token_info.d_model))
    if int(sae_cfg["d_model"]) != int(token_info.d_model):
        raise ValueError(
            f"Config d_model={sae_cfg['d_model']} does not match token dim {token_info.d_model}"
        )
    config["sae"] = sae_cfg
    train_cfg.setdefault("backend", "torch_sparse")
    train_cfg.setdefault("normalize_inputs", True)
    train_cfg.setdefault("init_decoder_bias_from_stats", True)
    train_cfg.setdefault("lr_decay", "cosine")
    train_cfg.setdefault("min_lr_fraction", 0.1)
    config["train"] = train_cfg

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
    max_steps_cfg = train_cfg.get("max_steps")
    max_steps = None
    if max_steps_cfg not in (None, "", 0, "0"):
        max_steps = max(1, int(max_steps_cfg))
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
        generator=build_dataloader_generator(seed, rank_offset=dist_rank),
        worker_init_fn=build_worker_init_fn(seed, rank_offset=dist_rank),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=build_dataloader_generator(seed + 1, rank_offset=dist_rank),
        worker_init_fn=build_worker_init_fn(seed + 1, rank_offset=dist_rank),
    )

    model = build_local_sae(config).to(device)
    stats_dir = load_training_stats_into_model(model, config, base_root=base_root)
    if bool(train_cfg.get("compile", False)) and hasattr(torch, "compile") and not is_distributed():
        model = torch.compile(model)  # type: ignore[assignment]
    if is_distributed():
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    optimizer = build_optimizer(model, train_cfg)
    total_steps = int(max_steps) if max_steps is not None else epochs * max(1, len(train_loader))
    scheduler = build_scheduler(optimizer, total_steps=total_steps, train_cfg=train_cfg)
    scaler = (
        torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and precision_is_fp16(train_cfg))
        if hasattr(torch.cuda, "amp")
        else None
    )
    backend = build_backend(train_cfg.get("backend", "torch_sparse"))
    precision = str(train_cfg.get("precision", "fp32"))

    history: list[dict] = []
    best_val_loss: float | None = None
    best_epoch = -1
    metrics_jsonl = out_dir / "metrics.jsonl"
    legacy_metrics_jsonl = out_dir / "epoch_metrics.jsonl"
    start_epoch = 1
    global_step = 0
    resume_from = train_cfg.get("resume_from")
    if resume_from:
        resume_path = resolve_path(resume_from, base=base_root)
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        _unwrap_model(model).load_state_dict(payload["state_dict"])
        if payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None and payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        restore_rng_state(payload.get("rng"))
        history = list(payload.get("history") or [])
        best_val_loss = payload.get("best_val_loss")
        start_epoch = int(payload.get("epoch", 0)) + 1
        global_step = int(payload.get("step", 0))

    if is_main_process():
        summary_seed = {
            "seed": int(seed),
            "backend": str(backend.name),
            "stats_dir": None if stats_dir is None else str(stats_dir),
            "resume_from": None if not resume_from else str(resolve_path(resume_from, base=base_root)),
            "training_sparsity_mode": "batchtopk_train_style",
        }
        (out_dir / "run_context.json").write_text(
            json.dumps(summary_seed, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    def write_metrics_record(record: dict, *, write_legacy: bool = False) -> None:
        if not is_main_process():
            return
        with metrics_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if write_legacy:
            with legacy_metrics_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    def current_scaler_state():
        if scaler is None or not scaler.is_enabled():
            return None
        return scaler.state_dict()

    def checkpoint_kwargs() -> dict:
        return {
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": None if scheduler is None else scheduler.state_dict(),
            "scaler_state": current_scaler_state(),
            "rng_state": capture_rng_state(),
        }

    def save_checkpoint_file(path: Path, *, epoch: int) -> None:
        save_local_sae_checkpoint(
            path,
            model=_unwrap_model(model),
            config=config,
            epoch=epoch,
            step=global_step,
            best_val_loss=best_val_loss,
            history=history,
            **checkpoint_kwargs(),
        )

    def evaluate_current_model() -> tuple[dict, np.ndarray]:
        model.eval()
        if len(val_set) > 0:
            val_metrics_local, val_feature_freq_local, _ = run_epoch(
                model=model,
                backend=backend,
                loader=val_loader,
                device=device,
                precision=precision,
                optimizer=None,
                scheduler=None,
                scaler=None,
                grad_clip_norm=None,
                global_step=global_step,
            )
        else:
            val_metrics_local = empty_step_metrics()
            val_feature_freq_local = np.zeros((int(_unwrap_model(model).d_sae),), dtype=np.float32)
        model.train()
        return val_metrics_local, val_feature_freq_local

    def write_summary(
        *,
        completed_epochs: int,
        current_lr: float,
        last_train: dict,
        last_val: dict,
    ) -> None:
        if not is_main_process():
            return
        train_summary = {
            "out_dir": str(out_dir),
            "backend": str(backend.name),
            "epochs": int(epochs),
            "completed_epochs": int(completed_epochs),
            "global_step": int(global_step),
            "seed": int(seed),
            "stats_dir": None if stats_dir is None else str(stats_dir),
            "best_epoch": int(best_epoch),
            "best_val_loss": None if best_val_loss is None else float(best_val_loss),
            "last_train": last_train,
            "last_val": last_val,
            "lr": current_lr,
            "training_sparsity_mode": "batchtopk_train_style",
            "max_steps": None if max_steps is None else int(max_steps),
        }
        (out_dir / "train_summary.json").write_text(
            json.dumps(train_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    latest_val_metrics = empty_step_metrics()
    latest_val_feature_freq = np.zeros((int(_unwrap_model(model).d_sae),), dtype=np.float32)
    completed_epochs = max(0, start_epoch - 1)
    checkpoints_dir = out_dir / "checkpoints"

    if max_steps is None:
        for epoch in range(start_epoch, epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)

            epoch_start = time.time()
            model.train()
            train_metrics, train_feature_freq, global_step = run_epoch(
                model=model,
                backend=backend,
                loader=train_loader,
                device=device,
                precision=precision,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                grad_clip_norm=grad_clip_norm,
                global_step=global_step,
            )
            latest_val_metrics, latest_val_feature_freq = evaluate_current_model()

            epoch_seconds = float(time.time() - epoch_start)
            current_lr = float(optimizer.param_groups[0]["lr"])
            record = {
                "epoch": epoch,
                "global_step": int(global_step),
                "epoch_seconds": epoch_seconds,
                "lr": current_lr,
                "train_rows_per_second": float(len(train_set) / max(epoch_seconds, 1e-6)),
                "train": train_metrics,
                "val": latest_val_metrics,
            }
            history.append(record)
            completed_epochs = int(epoch)

            if is_main_process():
                write_metrics_record(record, write_legacy=True)
                np.save(
                    out_dir / "feature_frequency_train_last.npy",
                    train_feature_freq.astype(np.float32, copy=False),
                )
                np.save(
                    out_dir / "feature_frequency_val_last.npy",
                    latest_val_feature_freq.astype(np.float32, copy=False),
                )
                save_checkpoint_file(checkpoints_dir / f"ep-{epoch:07d}.pt", epoch=epoch)
                current_val = (
                    float(latest_val_metrics["loss"])
                    if not np.isnan(latest_val_metrics["loss"])
                    else float(train_metrics["loss"])
                )
                if best_val_loss is None or current_val < best_val_loss:
                    best_val_loss = current_val
                    best_epoch = epoch
                    save_checkpoint_file(checkpoints_dir / "best.pt", epoch=epoch)
                (out_dir / "history.json").write_text(
                    json.dumps(history, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                write_summary(
                    completed_epochs=completed_epochs,
                    current_lr=current_lr,
                    last_train=train_metrics,
                    last_val=latest_val_metrics,
                )
            barrier()
    else:
        val_every_steps = max(1, int(train_cfg.get("val_every_steps", max(1, len(train_loader)))))
        checkpoint_every_steps = max(
            1,
            int(train_cfg.get("checkpoint_every_steps", val_every_steps)),
        )
        log_every_steps = max(1, int(train_cfg.get("log_every_steps", val_every_steps)))
        current_epoch = int(start_epoch)
        steps_in_epoch = 0
        epoch_start = time.time()
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(current_epoch)
        train_iter = iter(train_loader)
        epoch_stats = RunningFeatureStats(
            d_sae=int(_unwrap_model(model).d_sae),
            d_model=int(_unwrap_model(model).d_model),
        )
        last_train_metrics = empty_step_metrics()

        while global_step < int(max_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                if epoch_stats.total_rows > 0:
                    epoch_summary, train_feature_freq = epoch_stats.reduced_summary(device=device)
                    last_train_metrics = step_metrics_to_dict(epoch_summary)
                    epoch_seconds = float(time.time() - epoch_start)
                    current_lr = float(optimizer.param_groups[0]["lr"])
                    record = {
                        "epoch": int(current_epoch),
                        "global_step": int(global_step),
                        "epoch_seconds": epoch_seconds,
                        "lr": current_lr,
                        "train_rows_per_second": float(epoch_stats.total_rows / max(epoch_seconds, 1e-6)),
                        "train": last_train_metrics,
                        "val": latest_val_metrics,
                    }
                    history.append(record)
                    completed_epochs = int(current_epoch)
                    if is_main_process():
                        write_metrics_record(record, write_legacy=True)
                        np.save(
                            out_dir / "feature_frequency_train_last.npy",
                            train_feature_freq.astype(np.float32, copy=False),
                        )
                        (out_dir / "history.json").write_text(
                            json.dumps(history, indent=2, sort_keys=True),
                            encoding="utf-8",
                        )
                        write_summary(
                            completed_epochs=completed_epochs,
                            current_lr=current_lr,
                            last_train=last_train_metrics,
                            last_val=latest_val_metrics,
                        )
                    barrier()
                current_epoch += 1
                steps_in_epoch = 0
                epoch_start = time.time()
                epoch_stats = RunningFeatureStats(
                    d_sae=int(_unwrap_model(model).d_sae),
                    d_model=int(_unwrap_model(model).d_model),
                )
                if train_sampler is not None:
                    train_sampler.set_epoch(current_epoch)
                train_iter = iter(train_loader)
                continue

            x = batch[0].to(device=device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            with torch.set_grad_enabled(True):
                with autocast_context(device, precision):
                    outputs = backend.forward_loss(
                        model,
                        x,
                        global_step=global_step,
                        update_state=True,
                    )
                for key in ("loss", "recon_mse", "aux_loss"):
                    value = outputs[key]
                    if torch.is_tensor(value) and value.ndim > 0:
                        outputs[key] = value.mean()
                loss = outputs["loss"]
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if _unwrap_model(model).decoder_row_norm:
                    _unwrap_model(model).project_decoder_grad_()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                if _unwrap_model(model).decoder_row_norm:
                    _unwrap_model(model).normalize_decoder_rows_()

            epoch_stats.update(
                batch_size=int(x.shape[0]),
                loss=outputs["loss"].detach(),
                recon_mse=outputs["recon_mse"].detach(),
                aux_loss=outputs["aux_loss"].detach(),
                target=outputs["target"].detach(),
                recon=outputs["recon"].detach(),
                features=outputs["features"],
            )
            steps_in_epoch += 1

            should_validate = (global_step % val_every_steps == 0) or (global_step >= int(max_steps))
            should_checkpoint = (
                (global_step % checkpoint_every_steps == 0)
                or (global_step >= int(max_steps))
            )
            if global_step % log_every_steps == 0:
                step_summary, _ = epoch_stats.reduced_summary(device=device)
                last_train_metrics = step_metrics_to_dict(step_summary)
                if is_main_process():
                    write_metrics_record(
                        {
                            "kind": "train_step",
                            "epoch": int(current_epoch),
                            "global_step": int(global_step),
                            "lr": float(optimizer.param_groups[0]["lr"]),
                            "train": last_train_metrics,
                        }
                    )

            if should_validate:
                step_summary, train_feature_freq = epoch_stats.reduced_summary(device=device)
                last_train_metrics = step_metrics_to_dict(step_summary)
                latest_val_metrics, latest_val_feature_freq = evaluate_current_model()
                if is_main_process():
                    write_metrics_record(
                        {
                            "kind": "val_step",
                            "epoch": int(current_epoch),
                            "global_step": int(global_step),
                            "lr": float(optimizer.param_groups[0]["lr"]),
                            "train": last_train_metrics,
                            "val": latest_val_metrics,
                        }
                    )
                    np.save(
                        out_dir / "feature_frequency_train_last.npy",
                        train_feature_freq.astype(np.float32, copy=False),
                    )
                    np.save(
                        out_dir / "feature_frequency_val_last.npy",
                        latest_val_feature_freq.astype(np.float32, copy=False),
                    )
                    current_val = (
                        float(latest_val_metrics["loss"])
                        if not np.isnan(latest_val_metrics["loss"])
                        else float(last_train_metrics["loss"])
                    )
                    if best_val_loss is None or current_val < best_val_loss:
                        best_val_loss = current_val
                        best_epoch = int(current_epoch)
                        save_checkpoint_file(checkpoints_dir / "best.pt", epoch=current_epoch)
                    write_summary(
                        completed_epochs=max(completed_epochs, current_epoch - 1),
                        current_lr=float(optimizer.param_groups[0]["lr"]),
                        last_train=last_train_metrics,
                        last_val=latest_val_metrics,
                    )

            if should_checkpoint and is_main_process():
                save_checkpoint_file(
                    checkpoints_dir / f"step-{int(global_step):07d}.pt",
                    epoch=current_epoch,
                )
                (out_dir / "history.json").write_text(
                    json.dumps(history, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            if should_validate or should_checkpoint:
                barrier()

        if epoch_stats.total_rows > 0:
            epoch_summary, train_feature_freq = epoch_stats.reduced_summary(device=device)
            last_train_metrics = step_metrics_to_dict(epoch_summary)
            epoch_seconds = float(time.time() - epoch_start)
            current_lr = float(optimizer.param_groups[0]["lr"])
            record = {
                "epoch": int(current_epoch),
                "global_step": int(global_step),
                "epoch_seconds": epoch_seconds,
                "lr": current_lr,
                "train_rows_per_second": float(epoch_stats.total_rows / max(epoch_seconds, 1e-6)),
                "train": last_train_metrics,
                "val": latest_val_metrics,
            }
            history.append(record)
            completed_epochs = int(current_epoch)
            if is_main_process():
                write_metrics_record(record, write_legacy=True)
                np.save(
                    out_dir / "feature_frequency_train_last.npy",
                    train_feature_freq.astype(np.float32, copy=False),
                )
                (out_dir / "history.json").write_text(
                    json.dumps(history, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                write_summary(
                    completed_epochs=completed_epochs,
                    current_lr=current_lr,
                    last_train=last_train_metrics,
                    last_val=latest_val_metrics,
                )
            barrier()

    result = {
        "out_dir": str(out_dir),
        "epochs": int(epochs),
        "completed_epochs": int(completed_epochs),
        "global_step": int(global_step),
        "backend": str(backend.name),
        "stats_dir": None if stats_dir is None else str(stats_dir),
        "max_steps": None if max_steps is None else int(max_steps),
    }
    cleanup_distributed()
    return result


def precision_is_fp16(train_cfg: dict) -> bool:
    return str(train_cfg.get("precision", "fp32")).lower() == "fp16"
