"""Image-level H construction from local SAE checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from fsaeter.config_compat import normalize_build_h_config
from fsaeter.data.cache import resolve_token_cache_info
from fsaeter.h.helpers import normalize_inference_mode, pool_sae_image_batch
from fsaeter.inspect.basic_qc import select_sparse_topk_rows
from fsaeter.models.local_sae import load_local_sae_checkpoint, payload_to_local_sae_info
from fsaeter.utils.config import resolve_path, save_yaml_config
from fsaeter.utils.repro import resolve_run_seed, seed_everything


def write_topk_csr(
    *,
    indices: np.ndarray,
    values: np.ndarray,
    out_prefix: Path,
    value_threshold: float,
) -> dict[str, str]:
    valid = (
        (indices >= 0)
        & np.isfinite(values)
        & (values > float(value_threshold))
    )
    row_counts = valid.sum(axis=1, dtype=np.int64)
    indptr = np.zeros((indices.shape[0] + 1,), dtype=np.int64)
    indptr[1:] = np.cumsum(row_counts, dtype=np.int64)
    csr_indices = indices[valid].astype(np.int32, copy=False)
    csr_values = values[valid].astype(values.dtype, copy=False)
    indptr_path = out_prefix.with_name(f"{out_prefix.name}_indptr.npy")
    indices_path = out_prefix.with_name(f"{out_prefix.name}_indices.npy")
    values_path = out_prefix.with_name(f"{out_prefix.name}_values.npy")
    np.save(indptr_path, indptr)
    np.save(indices_path, csr_indices)
    np.save(values_path, csr_values)
    return {
        "indptr": str(indptr_path),
        "indices": str(indices_path),
        "values": str(values_path),
    }


def run_build_h(config: dict, *, base_root: Path) -> dict:
    config = normalize_build_h_config(config)
    run_cfg = dict(config.get("run") or {})
    tokens_cfg = dict(config.get("tokens") or {})
    sae_cfg = dict(config.get("sae") or {})
    build_cfg = dict(config.get("build_h") or {})
    seed_everything(resolve_run_seed(config))

    tokens_dir = resolve_path(tokens_cfg.get("cache_dir", ""), base=base_root)
    token_info = resolve_token_cache_info(tokens_dir)
    tokens = np.load(token_info.tokens_path, mmap_mode="r")
    max_images_cfg = build_cfg.get("max_images")
    if max_images_cfg in (None, "", 0, "0"):
        max_images = int(tokens.shape[0])
    else:
        max_images = min(int(max_images_cfg), int(tokens.shape[0]))

    checkpoint_path = resolve_path(sae_cfg.get("checkpoint", ""), base=base_root)
    device_str = str(build_cfg.get("device", "auto")).lower()
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    model, payload = load_local_sae_checkpoint(checkpoint_path, device=device)
    model_info = payload_to_local_sae_info(payload, checkpoint_path)
    if int(model.d_model) != int(token_info.d_model):
        raise ValueError(
            f"Checkpoint d_model={model.d_model} but token dim is {token_info.d_model}"
        )

    out_dir = resolve_path(run_cfg.get("out_dir", "outputs/local_sae_h"), base=base_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_yaml_config(config, out_dir / "config_resolved.yaml")

    save_dtype = np.dtype(build_cfg.get("save_dtype", "float16"))
    save_dense_mean = bool(build_cfg.get("save_dense_mean", True))
    legacy_save_max = build_cfg.get("save_max", True)
    save_dense_max = bool(build_cfg.get("save_dense_max", legacy_save_max))
    save_topk_mean = bool(build_cfg.get("save_topk_mean", True))
    save_topk_max = bool(build_cfg.get("save_topk_max", True))
    save_sparse_csr_requested = bool(build_cfg.get("save_sparse_csr", False))
    if save_sparse_csr_requested:
        save_topk_mean = True
        save_topk_max = True
    inference_mode = normalize_inference_mode(
        build_cfg.get("inference_mode"),
        default="per_row_topk",
    )
    image_top_k = min(int(build_cfg.get("image_top_k", 64)), int(model.d_sae))
    active_threshold = float(build_cfg.get("active_threshold", 1e-6))
    image_batch_size = max(1, int(build_cfg.get("image_batch_size", 16)))
    token_batch_size = max(1, int(build_cfg.get("token_batch_size", 2048)))
    precision = str(build_cfg.get("precision", "fp32"))
    h_rows = np.arange(int(max_images), dtype=np.int64)

    dense_mean = None
    dense_max = None
    if save_dense_mean:
        dense_mean = np.lib.format.open_memmap(
            out_dir / "H_mean.npy",
            mode="w+",
            dtype=save_dtype,
            shape=(int(max_images), int(model.d_sae)),
        )
    if save_dense_max:
        dense_max = np.lib.format.open_memmap(
            out_dir / "H_max.npy",
            mode="w+",
            dtype=save_dtype,
            shape=(int(max_images), int(model.d_sae)),
        )

    mean_top_indices = mean_top_values = None
    max_top_indices = max_top_values = None
    compat_top_indices = compat_top_values = None
    if save_topk_mean:
        mean_top_indices = np.lib.format.open_memmap(
            out_dir / "H_mean_top_indices.npy",
            mode="w+",
            dtype=np.int32,
            shape=(int(max_images), int(image_top_k)),
        )
        mean_top_values = np.lib.format.open_memmap(
            out_dir / "H_mean_top_values.npy",
            mode="w+",
            dtype=save_dtype,
            shape=(int(max_images), int(image_top_k)),
        )
    if save_topk_max:
        max_top_indices = np.lib.format.open_memmap(
            out_dir / "H_max_top_indices.npy",
            mode="w+",
            dtype=np.int32,
            shape=(int(max_images), int(image_top_k)),
        )
        max_top_values = np.lib.format.open_memmap(
            out_dir / "H_max_top_values.npy",
            mode="w+",
            dtype=save_dtype,
            shape=(int(max_images), int(image_top_k)),
        )

    compat_top_indices = np.lib.format.open_memmap(
        out_dir / "H_top_indices.npy",
        mode="w+",
        dtype=np.int32,
        shape=(int(max_images), int(image_top_k)),
    )
    compat_top_values = np.lib.format.open_memmap(
        out_dir / "H_top_values.npy",
        mode="w+",
        dtype=save_dtype,
        shape=(int(max_images), int(image_top_k)),
    )

    activation_sum = torch.zeros(int(model.d_sae), dtype=torch.float64)
    activation_max = torch.full((int(model.d_sae),), -torch.inf, dtype=torch.float32)
    image_active_counts_mean = torch.zeros(int(model.d_sae), dtype=torch.float64)
    image_active_counts_max = torch.zeros(int(model.d_sae), dtype=torch.float64)
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
            inference_mode=inference_mode,
        )
        if dense_mean is not None:
            dense_mean[start:end] = mean_rows.numpy().astype(save_dtype, copy=False)
        if dense_max is not None:
            dense_max[start:end] = max_rows.numpy().astype(save_dtype, copy=False)

        mean_vals_np, mean_ids_np = select_sparse_topk_rows(
            mean_rows.numpy(),
            k=image_top_k,
            active_threshold=active_threshold,
        )
        max_vals_np, max_ids_np = select_sparse_topk_rows(
            max_rows.numpy(),
            k=image_top_k,
            active_threshold=active_threshold,
        )
        if mean_top_indices is not None and mean_top_values is not None:
            mean_top_indices[start:end] = mean_ids_np.astype(np.int32, copy=False)
            mean_top_values[start:end] = mean_vals_np.astype(save_dtype, copy=False)
        if max_top_indices is not None and max_top_values is not None:
            max_top_indices[start:end] = max_ids_np.astype(np.int32, copy=False)
            max_top_values[start:end] = max_vals_np.astype(save_dtype, copy=False)

        compat_ids = max_ids_np if save_topk_max else mean_ids_np
        compat_vals = max_vals_np if save_topk_max else mean_vals_np
        compat_top_indices[start:end] = compat_ids.astype(np.int32, copy=False)
        compat_top_values[start:end] = compat_vals.astype(save_dtype, copy=False)

        activation_sum += mean_rows.double().sum(dim=0)
        activation_max = torch.maximum(activation_max, max_rows.max(dim=0).values)
        image_active_counts_mean += (mean_rows > active_threshold).double().sum(dim=0)
        image_active_counts_max += (max_rows > active_threshold).double().sum(dim=0)
        token_active_counts += active_tokens.double()

    np.save(out_dir / "h_image_rows.npy", h_rows)
    if dense_mean is not None:
        dense_mean.flush()
    if dense_max is not None:
        dense_max.flush()
    if mean_top_indices is not None and mean_top_values is not None:
        mean_top_indices.flush()
        mean_top_values.flush()
    if max_top_indices is not None and max_top_values is not None:
        max_top_indices.flush()
        max_top_values.flush()
    compat_top_indices.flush()
    compat_top_values.flush()

    csr_paths_mean = None
    csr_paths_max = None
    if save_sparse_csr_requested:
        if mean_top_indices is None or mean_top_values is None:
            raise RuntimeError("Expected H_mean_top_* arrays to exist before CSR export.")
        if max_top_indices is None or max_top_values is None:
            raise RuntimeError("Expected H_max_top_* arrays to exist before CSR export.")
        csr_paths_mean = write_topk_csr(
            indices=np.asarray(mean_top_indices),
            values=np.asarray(mean_top_values),
            out_prefix=out_dir / "H_mean_csr",
            value_threshold=active_threshold,
        )
        csr_paths_max = write_topk_csr(
            indices=np.asarray(max_top_indices),
            values=np.asarray(max_top_values),
            out_prefix=out_dir / "H_max_csr",
            value_threshold=active_threshold,
        )

    total_tokens = float(max_images * int(token_info.tokens_per_image))
    image_frequency_mean = (
        image_active_counts_mean / float(max_images)
    ).numpy().astype(np.float32, copy=False)
    image_frequency_max = (
        image_active_counts_max / float(max_images)
    ).numpy().astype(np.float32, copy=False)
    token_frequency = (token_active_counts / total_tokens).numpy().astype(np.float32, copy=False)
    mean_activation = (activation_sum / float(max_images)).numpy().astype(np.float32, copy=False)
    max_activation = (
        torch.where(
            torch.isfinite(activation_max),
            activation_max,
            torch.zeros_like(activation_max),
        )
        .numpy()
        .astype(np.float32, copy=False)
    )
    np.savez_compressed(
        out_dir / "concept_stats.npz",
        mean_activation=mean_activation,
        max_activation=max_activation,
        image_frequency_mean=image_frequency_mean,
        image_frequency_max=image_frequency_max,
        token_frequency=token_frequency,
        active_threshold=np.asarray(active_threshold, dtype=np.float32),
    )

    build_summary = {
        "out_dir": str(out_dir),
        "max_images": int(max_images),
        "image_top_k": int(image_top_k),
        "inference_mode": inference_mode,
        "save_dense_mean": bool(save_dense_mean),
        "save_dense_max": bool(save_dense_max),
        "save_topk_mean": bool(save_topk_mean),
        "save_topk_max": bool(save_topk_max),
        "save_sparse_csr_requested": bool(save_sparse_csr_requested),
        "save_sparse_csr_written": bool(csr_paths_mean is not None and csr_paths_max is not None),
        "active_threshold": float(active_threshold),
        "H_mean_csr": csr_paths_mean,
        "H_max_csr": csr_paths_max,
    }
    (out_dir / "build_summary.json").write_text(
        json.dumps(build_summary, indent=2, sort_keys=True),
        encoding="utf-8",
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
            "normalize_inputs": bool(model_info.normalize_inputs),
            "aux_k": int(model_info.aux_k),
            "dead_steps_threshold": int(model_info.dead_steps_threshold),
            "aux_loss_weight": float(model_info.aux_loss_weight),
        },
        "token_cache": {
            "tokens": token_info.tokens_path,
            "metadata": token_info.metadata_path,
            "labels": token_info.labels_path,
            "encoder_name": token_info.encoder_name,
            "encoder_model": token_info.encoder_model,
            "encoder_factory_string": token_info.encoder_factory_string,
            "path_mode": token_info.path_mode,
            "token_shape": [int(max_images), int(token_info.tokens_per_image), int(token_info.d_model)],
            "patch_grid": [int(v) for v in token_info.patch_grid],
            "encoder_input_size": int(token_info.encoder_input_size),
            "h_image_rows": str(out_dir / "h_image_rows.npy"),
        },
        "build_h": {
            "H_mean": None if dense_mean is None else str(out_dir / "H_mean.npy"),
            "H_max": None if dense_max is None else str(out_dir / "H_max.npy"),
            "H_mean_top_indices": (
                None if mean_top_indices is None else str(out_dir / "H_mean_top_indices.npy")
            ),
            "H_mean_top_values": (
                None if mean_top_values is None else str(out_dir / "H_mean_top_values.npy")
            ),
            "H_max_top_indices": (
                None if max_top_indices is None else str(out_dir / "H_max_top_indices.npy")
            ),
            "H_max_top_values": (
                None if max_top_values is None else str(out_dir / "H_max_top_values.npy")
            ),
            "H_top_indices": str(out_dir / "H_top_indices.npy"),
            "H_top_values": str(out_dir / "H_top_values.npy"),
            "concept_stats": str(out_dir / "concept_stats.npz"),
            "build_summary": str(out_dir / "build_summary.json"),
            "H_mean_csr": csr_paths_mean,
            "H_max_csr": csr_paths_max,
            "image_top_k": int(image_top_k),
            "active_threshold": float(active_threshold),
            "inference_mode": inference_mode,
            "sparse_topk": True,
            "save_dtype": str(save_dtype),
            "save_sparse_csr_requested": bool(save_sparse_csr_requested),
            "save_sparse_csr_written": bool(csr_paths_mean is not None and csr_paths_max is not None),
        },
    }
    with (out_dir / "concept_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(concept_metadata, handle, indent=2, sort_keys=True)

    return build_summary
