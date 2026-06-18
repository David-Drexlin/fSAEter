"""Shared SAE inference helpers."""

from __future__ import annotations

import contextlib

import numpy as np
import torch
from torch import Tensor

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
