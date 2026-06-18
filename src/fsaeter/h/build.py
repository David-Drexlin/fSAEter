"""Image-level H construction from local SAE checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from fsaeter.config_compat import normalize_build_h_config
from fsaeter.data.cache import resolve_token_cache_info
from fsaeter.h.helpers import pool_sae_image_batch
from fsaeter.inspect.basic_qc import select_sparse_topk_rows
from fsaeter.models.local_sae import load_local_sae_checkpoint, payload_to_local_sae_info
from fsaeter.utils.config import resolve_path, save_yaml_config


def run_build_h(config: dict, *, base_root: Path) -> dict:
    config = normalize_build_h_config(config)
    run_cfg = dict(config.get("run") or {})
    tokens_cfg = dict(config.get("tokens") or {})
    sae_cfg = dict(config.get("sae") or {})
    build_cfg = dict(config.get("build_h") or {})

    tokens_dir = resolve_path(tokens_cfg.get("cache_dir", ""), base=base_root)
    token_info = resolve_token_cache_info(tokens_dir)
    tokens = np.load(token_info.tokens_path, mmap_mode="r")
    max_images_cfg = build_cfg.get("max_images")
    max_images = int(tokens.shape[0]) if max_images_cfg in (None, "", 0, "0") else min(int(max_images_cfg), int(tokens.shape[0]))

    checkpoint_path = resolve_path(sae_cfg.get("checkpoint", ""), base=base_root)
    device_str = str(build_cfg.get("device", "auto")).lower()
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    model, payload = load_local_sae_checkpoint(checkpoint_path, device=device)
    model_info = payload_to_local_sae_info(payload, checkpoint_path)
    if int(model.d_model) != int(token_info.d_model):
        raise ValueError(f"Checkpoint d_model={model.d_model} but token dim is {token_info.d_model}")

    out_dir = resolve_path(run_cfg.get("out_dir", "outputs/local_sae_h"), base=base_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_yaml_config(config, out_dir / "config_resolved.yaml")

    save_dtype = np.dtype(build_cfg.get("save_dtype", "float16"))
    save_max = bool(build_cfg.get("save_max", True))
    image_top_k = min(int(build_cfg.get("image_top_k", 64)), int(model.d_sae))
    active_threshold = float(build_cfg.get("active_threshold", 1e-6))
    image_batch_size = max(1, int(build_cfg.get("image_batch_size", 16)))
    token_batch_size = max(1, int(build_cfg.get("token_batch_size", 2048)))
    precision = str(build_cfg.get("precision", "fp32"))

    h_mean = np.lib.format.open_memmap(out_dir / "H_mean.npy", mode="w+", dtype=save_dtype, shape=(int(max_images), int(model.d_sae)))
    h_max = None
    if save_max:
        h_max = np.lib.format.open_memmap(out_dir / "H_max.npy", mode="w+", dtype=save_dtype, shape=(int(max_images), int(model.d_sae)))
    top_indices = np.lib.format.open_memmap(out_dir / "H_top_indices.npy", mode="w+", dtype=np.int32, shape=(int(max_images), int(image_top_k)))
    top_values = np.lib.format.open_memmap(out_dir / "H_top_values.npy", mode="w+", dtype=save_dtype, shape=(int(max_images), int(image_top_k)))

    activation_sum = torch.zeros(int(model.d_sae), dtype=torch.float64)
    activation_max = torch.full((int(model.d_sae),), -torch.inf, dtype=torch.float32)
    image_active_counts = torch.zeros(int(model.d_sae), dtype=torch.float64)
    token_active_counts = torch.zeros(int(model.d_sae), dtype=torch.float64)

    for start in range(0, int(max_images), image_batch_size):
        end = min(start + image_batch_size, int(max_images))
        mean_rows, max_rows, active_tokens = pool_sae_image_batch(
            model,
            tokens[start:end],
            device=device,
            token_batch_size=token_batch_size,
            precision=precision,
            active_threshold=active_threshold,
        )
        mean_np = mean_rows.numpy().astype(save_dtype, copy=False)
        top_vals_np, top_ids_np = select_sparse_topk_rows(mean_rows.numpy(), k=image_top_k, active_threshold=active_threshold)
        h_mean[start:end] = mean_np
        top_indices[start:end] = top_ids_np.astype(np.int32, copy=False)
        top_values[start:end] = top_vals_np.astype(save_dtype, copy=False)
        if h_max is not None:
            h_max[start:end] = max_rows.numpy().astype(save_dtype, copy=False)

        activation_sum += mean_rows.double().sum(dim=0)
        activation_max = torch.maximum(activation_max, max_rows.max(dim=0).values)
        image_active_counts += (mean_rows > active_threshold).double().sum(dim=0)
        token_active_counts += active_tokens.double()

    h_mean.flush()
    if h_max is not None:
        h_max.flush()
    top_indices.flush()
    top_values.flush()

    total_tokens = float(max_images * int(token_info.tokens_per_image))
    image_frequency = (image_active_counts / float(max_images)).numpy().astype(np.float32, copy=False)
    token_frequency = (token_active_counts / total_tokens).numpy().astype(np.float32, copy=False)
    mean_activation = (activation_sum / float(max_images)).numpy().astype(np.float32, copy=False)
    max_activation = torch.where(torch.isfinite(activation_max), activation_max, torch.zeros_like(activation_max)).numpy().astype(np.float32, copy=False)
    np.savez_compressed(
        out_dir / "concept_stats.npz",
        mean_activation=mean_activation,
        max_activation=max_activation,
        image_frequency=image_frequency,
        token_frequency=token_frequency,
        active_threshold=np.asarray(active_threshold, dtype=np.float32),
    )

    concept_metadata = {
        "run_name": run_cfg.get("name"),
        "source": "local_sae",
        "checkpoint": str(checkpoint_path),
        "sae": {
            "checkpoint_path": model_info.checkpoint_path,
            "variant": model_info.variant,
            "d_model": int(model_info.d_model),
            "d_sae": int(model_info.d_sae),
            "target_k": int(model_info.target_k),
            "matryoshka_prefixes": [int(v) for v in model_info.matryoshka_prefixes],
            "matryoshka_weights": [float(v) for v in model_info.matryoshka_weights],
        },
        "token_cache": {
            "tokens": token_info.tokens_path,
            "metadata": token_info.metadata_path,
            "labels": token_info.labels_path,
            "encoder_name": token_info.encoder_name,
            "token_shape": [int(max_images), int(token_info.tokens_per_image), int(token_info.d_model)],
            "patch_grid": [int(v) for v in token_info.patch_grid],
            "encoder_input_size": int(token_info.encoder_input_size),
        },
        "build_h": {
            "H_mean": str(out_dir / "H_mean.npy"),
            "H_max": None if h_max is None else str(out_dir / "H_max.npy"),
            "H_top_indices": str(out_dir / "H_top_indices.npy"),
            "H_top_values": str(out_dir / "H_top_values.npy"),
            "concept_stats": str(out_dir / "concept_stats.npz"),
            "image_top_k": int(image_top_k),
            "active_threshold": float(active_threshold),
            "sparse_topk": True,
            "save_dtype": str(save_dtype),
        },
    }
    with (out_dir / "concept_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(concept_metadata, handle, indent=2, sort_keys=True)

    return {"out_dir": str(out_dir), "max_images": int(max_images), "image_top_k": int(image_top_k)}

