"""Shared SAE inference helpers."""

from __future__ import annotations

import contextlib

import numpy as np
import torch
from torch import Tensor


def encode_sae(model: torch.nn.Module, x: Tensor) -> Tensor:
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
) -> tuple[Tensor, Tensor, Tensor]:
    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens shaped [images, tokens, dim], got {tokens.shape}")
    num_images, tokens_per_image, d_model = (int(v) for v in tokens.shape)
    flat = torch.from_numpy(
        np.asarray(tokens, dtype=np.float32).reshape(
            num_images * tokens_per_image,
            d_model,
        )
    )

    d_sae = int(model.d_sae) if hasattr(model, "d_sae") else 0
    if d_sae <= 0 and hasattr(model, "W_enc"):
        d_sae = int(model.W_enc.shape[1])
    if d_sae <= 0:
        probe = encode_sae(model, flat[:1].to(device)).float()
        d_sae = int(probe.shape[-1])

    sums = torch.zeros((num_images, d_sae), dtype=torch.float32, device=device)
    maxes = torch.full((num_images, d_sae), -torch.inf, dtype=torch.float32, device=device)
    token_active_counts = torch.zeros((d_sae,), dtype=torch.float32, device=device)

    batch_size = max(1, int(token_batch_size))
    for start in range(0, flat.shape[0], batch_size):
        end = min(start + batch_size, flat.shape[0])
        x = flat[start:end].to(device=device, non_blocking=True)
        with autocast_context(device, precision):
            acts = encode_sae(model, x).float()
        local_images = torch.arange(start, end, device=device, dtype=torch.long) // tokens_per_image
        sums.index_add_(0, local_images, acts)
        token_active_counts += (acts > active_threshold).sum(dim=0).float()
        for image_idx in torch.unique(local_images):
            mask = local_images == image_idx
            maxes[int(image_idx.item())] = torch.maximum(
                maxes[int(image_idx.item())],
                acts[mask].max(dim=0).values,
            )

    h_mean = sums / float(tokens_per_image)
    h_max = torch.where(torch.isfinite(maxes), maxes, torch.zeros_like(maxes))
    return h_mean.cpu(), h_max.cpu(), token_active_counts.cpu()
