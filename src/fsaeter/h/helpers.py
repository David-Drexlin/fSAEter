"""Shared SAE inference helpers."""

from __future__ import annotations

import contextlib

import numpy as np
import torch
from torch import Tensor

from fsaeter.train.sparse_ops import SparseActs, batch_topk_sparse, empty_sparse_acts

SUPPORTED_INFERENCE_MODES = ("per_row_topk", "batchtopk_train_style")


def normalize_inference_mode(
    inference_mode: str | None,
    *,
    default: str,
) -> str:
    normalized = default if inference_mode is None else str(inference_mode).strip().lower()
    if normalized not in SUPPORTED_INFERENCE_MODES:
        raise ValueError(
            f"Unsupported inference_mode {normalized!r}; expected one of {SUPPORTED_INFERENCE_MODES}"
        )
    return normalized


def per_row_topk_nonnegative(positive_acts: Tensor, target_k: int) -> Tensor:
    if positive_acts.ndim != 2:
        raise ValueError(f"Expected [B,F] activations, got {positive_acts.shape}")
    batch, features = positive_acts.shape
    if batch <= 0 or features <= 0:
        return torch.zeros_like(positive_acts)
    row_k = min(max(0, int(target_k)), int(features))
    if row_k <= 0:
        return torch.zeros_like(positive_acts)
    values, indices = torch.topk(positive_acts, k=row_k, dim=1, sorted=False)
    values = torch.where(values > 0, values, torch.zeros_like(values))
    dense = torch.zeros_like(positive_acts)
    dense.scatter_(1, indices, values)
    return dense


def per_row_topk_sparse(positive_acts: Tensor, target_k: int) -> SparseActs:
    if positive_acts.ndim != 2:
        raise ValueError(f"Expected [B,F] activations, got {positive_acts.shape}")
    batch, features = (int(v) for v in positive_acts.shape)
    if batch <= 0 or features <= 0:
        return empty_sparse_acts(
            batch_size=batch,
            d_sae=features,
            device=positive_acts.device,
            dtype=positive_acts.dtype,
        )
    row_k = min(max(0, int(target_k)), int(features))
    if row_k <= 0:
        return empty_sparse_acts(
            batch_size=batch,
            d_sae=features,
            device=positive_acts.device,
            dtype=positive_acts.dtype,
        )
    values, indices = torch.topk(positive_acts, k=row_k, dim=1, sorted=False)
    mask = values > 0
    if not torch.any(mask):
        return empty_sparse_acts(
            batch_size=batch,
            d_sae=features,
            device=positive_acts.device,
            dtype=positive_acts.dtype,
        )
    row_ids = (
        torch.arange(batch, device=positive_acts.device, dtype=torch.long)[:, None]
        .expand(batch, row_k)[mask]
    )
    feature_ids = indices[mask].long()
    sparse_values = values[mask]
    order = torch.argsort(row_ids * int(features) + feature_ids)
    return SparseActs(
        row_ids=row_ids[order],
        feature_ids=feature_ids[order],
        values=sparse_values[order],
        batch_size=batch,
        d_sae=features,
    )


def encode_sae(
    model: torch.nn.Module,
    x: Tensor,
    *,
    inference_mode: str | None = None,
) -> Tensor:
    if inference_mode is not None:
        normalized_mode = normalize_inference_mode(
            inference_mode,
            default="batchtopk_train_style",
        )
        if not hasattr(model, "prepare_target") or not hasattr(model, "preactivate_target"):
            raise TypeError(
                f"Model type {type(model).__name__} does not expose the local SAE inference surface"
            )
        target = model.prepare_target(x)
        h_x = model.preactivate_target(target)
        positive = torch.relu(h_x)
        if normalized_mode == "per_row_topk":
            return per_row_topk_nonnegative(positive, int(model.target_k))
        return model._batch_topk_nonnegative(positive, int(model.target_k))
    if hasattr(model, "encode"):
        encoded = model.encode(x)
        if hasattr(encoded, "f_x"):
            return encoded.f_x
        if isinstance(encoded, tuple) and len(encoded) >= 2:
            return encoded[1]
    output = model(x)
    if hasattr(output, "f_x"):
        return output.f_x
    if isinstance(output, tuple) and len(output) >= 2:
        return output[1]
    if torch.is_tensor(output):
        return output
    raise TypeError(f"Could not extract SAE activations from output type {type(output).__name__}")


def encode_sae_sparse(
    model: torch.nn.Module,
    x: Tensor,
    *,
    inference_mode: str,
) -> SparseActs:
    normalized_mode = normalize_inference_mode(
        inference_mode,
        default="batchtopk_train_style",
    )
    if not hasattr(model, "prepare_target") or not hasattr(model, "preactivate_target"):
        raise TypeError(
            f"Model type {type(model).__name__} does not expose the local SAE inference surface"
        )
    target = model.prepare_target(x)
    h_x = model.preactivate_target(target)
    positive = torch.relu(h_x)
    if normalized_mode == "per_row_topk":
        return per_row_topk_sparse(positive, int(model.target_k))
    return batch_topk_sparse(positive, int(model.target_k))


def autocast_context(device: torch.device, precision: str):
    precision = str(precision).lower()
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device.type == "cuda" and precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


@torch.no_grad()
def pool_sae_image_batch(
    model: torch.nn.Module,
    tokens: np.ndarray,
    *,
    device: torch.device,
    token_batch_size: int = 512,
    precision: str = "fp32",
    active_threshold: float = 0.0,
    inference_mode: str = "batchtopk_train_style",
) -> tuple[Tensor, Tensor, Tensor]:
    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens shaped [images, tokens, dim], got {tokens.shape}")
    num_images, tokens_per_image, d_model = (int(v) for v in tokens.shape)
    d_sae = int(model.d_sae) if hasattr(model, "d_sae") else 0
    if d_sae <= 0 and hasattr(model, "W_enc"):
        d_sae = int(model.W_enc.shape[1])
    if d_sae <= 0:
        probe = encode_sae(
            model,
            torch.from_numpy(np.asarray(tokens[:1], dtype=np.float32)).reshape(-1, d_model).to(device),
        ).float()
        d_sae = int(probe.shape[-1])

    h_mean = torch.zeros((num_images, d_sae), dtype=torch.float32)
    h_max = torch.zeros((num_images, d_sae), dtype=torch.float32)
    token_active_counts = torch.zeros((d_sae,), dtype=torch.float32)

    image_batch_size = max(1, int(token_batch_size) // max(1, tokens_per_image))
    for start in range(0, num_images, image_batch_size):
        end = min(start + image_batch_size, num_images)
        image_chunk = torch.from_numpy(
            np.asarray(tokens[start:end], dtype=np.float32)
        ).to(device=device, non_blocking=True)
        flat_chunk = image_chunk.reshape(-1, d_model)
        with autocast_context(device, precision):
            acts = encode_sae(
                model,
                flat_chunk,
                inference_mode=inference_mode,
            ).float()
        acts = acts.reshape(end - start, tokens_per_image, d_sae)
        h_mean[start:end] = acts.mean(dim=1).cpu()
        h_max[start:end] = acts.amax(dim=1).cpu()
        token_active_counts += (acts > active_threshold).sum(dim=(0, 1)).float().cpu()
    return h_mean, h_max, token_active_counts


@torch.no_grad()
def pool_sae_image_batch_sparse(
    model: torch.nn.Module,
    tokens: np.ndarray,
    *,
    device: torch.device,
    token_batch_size: int = 512,
    precision: str = "fp32",
    active_threshold: float = 0.0,
    inference_mode: str = "batchtopk_train_style",
    image_top_k: int | None = None,
    return_dense_mean: bool = False,
    return_dense_max: bool = False,
) -> tuple[Tensor | None, Tensor | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tensor]:
    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens shaped [images, tokens, dim], got {tokens.shape}")
    num_images, tokens_per_image, d_model = (int(v) for v in tokens.shape)
    d_sae = int(model.d_sae)
    image_top_k = min(
        int(image_top_k) if image_top_k is not None else int(model.target_k * max(1, tokens_per_image)),
        d_sae,
    )
    mean_top_indices = np.full((num_images, image_top_k), -1, dtype=np.int32)
    mean_top_values = np.zeros((num_images, image_top_k), dtype=np.float32)
    max_top_indices = np.full((num_images, image_top_k), -1, dtype=np.int32)
    max_top_values = np.zeros((num_images, image_top_k), dtype=np.float32)
    token_active_counts = torch.zeros((d_sae,), dtype=torch.float32)
    dense_mean = (
        torch.zeros((num_images, d_sae), dtype=torch.float32) if return_dense_mean else None
    )
    dense_max = (
        torch.zeros((num_images, d_sae), dtype=torch.float32) if return_dense_max else None
    )

    image_batch_size = max(1, int(token_batch_size) // max(1, tokens_per_image))
    for start in range(0, num_images, image_batch_size):
        end = min(start + image_batch_size, num_images)
        image_chunk = torch.from_numpy(
            np.asarray(tokens[start:end], dtype=np.float32)
        ).to(device=device, non_blocking=True)
        flat_chunk = image_chunk.reshape(-1, d_model)
        with autocast_context(device, precision):
            sparse_acts = encode_sae_sparse(
                model,
                flat_chunk,
                inference_mode=inference_mode,
            )
        image_count = end - start
        mean_accums: list[dict[int, float]] = [dict() for _ in range(image_count)]
        max_accums: list[dict[int, float]] = [dict() for _ in range(image_count)]
        if sparse_acts.nnz > 0:
            row_ids = sparse_acts.row_ids.detach().cpu().numpy().astype(np.int64, copy=False)
            feature_ids = sparse_acts.feature_ids.detach().cpu().numpy().astype(np.int64, copy=False)
            values = sparse_acts.values.detach().cpu().numpy().astype(np.float32, copy=False)
            token_active_counts += torch.bincount(
                sparse_acts.feature_ids.detach().cpu(),
                minlength=d_sae,
            ).float()
            for row_id, feature_id, value in zip(row_ids.tolist(), feature_ids.tolist(), values.tolist(), strict=True):
                image_idx = int(row_id) // int(tokens_per_image)
                mean_accums[image_idx][int(feature_id)] = mean_accums[image_idx].get(int(feature_id), 0.0) + float(value)
                max_accums[image_idx][int(feature_id)] = max(
                    max_accums[image_idx].get(int(feature_id), 0.0),
                    float(value),
                )
        for local_idx in range(image_count):
            mean_all_items = [
                (feature_id, score / float(tokens_per_image))
                for feature_id, score in mean_accums[local_idx].items()
            ]
            mean_items = [
                (feature_id, score)
                for feature_id, score in mean_all_items
                if score > float(active_threshold)
            ]
            mean_items.sort(key=lambda item: item[1], reverse=True)
            max_all_items = list(max_accums[local_idx].items())
            max_items = [(feature_id, score) for feature_id, score in max_all_items if score > float(active_threshold)]
            max_items.sort(key=lambda item: item[1], reverse=True)
            if dense_mean is not None:
                for feature_id, score in mean_all_items:
                    dense_mean[start + local_idx, int(feature_id)] = float(score)
            if dense_max is not None:
                for feature_id, score in max_all_items:
                    dense_max[start + local_idx, int(feature_id)] = float(score)
            for rank, (feature_id, score) in enumerate(mean_items[:image_top_k]):
                mean_top_indices[start + local_idx, rank] = int(feature_id)
                mean_top_values[start + local_idx, rank] = float(score)
            for rank, (feature_id, score) in enumerate(max_items[:image_top_k]):
                max_top_indices[start + local_idx, rank] = int(feature_id)
                max_top_values[start + local_idx, rank] = float(score)
    return (
        dense_mean,
        dense_max,
        mean_top_values,
        mean_top_indices,
        max_top_values,
        max_top_indices,
        token_active_counts,
    )
