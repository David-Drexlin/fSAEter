"""Token cache metadata and memmap IO."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class TokenCacheInfo:
    tokens_path: str
    metadata_path: str
    labels_path: str | None
    num_images: int
    tokens_per_image: int
    d_model: int
    patch_grid: tuple[int, int]
    patch_size: int
    encoder_input_size: int
    data_image_size: int
    num_register_tokens: int
    encoder_name: str | None = None


def resolve_token_cache_info(tokens_dir: str | Path) -> TokenCacheInfo:
    root = Path(tokens_dir).expanduser().resolve()
    tokens_path = root / "tokens_patch.npy"
    metadata_path = root / "token_metadata.json"
    labels_path = root / "labels.npy"
    if not tokens_path.exists():
        raise FileNotFoundError(f"Token cache not found: {tokens_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Token metadata not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    tokens = np.load(tokens_path, mmap_mode="r")
    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens_patch.npy shaped [N,T,D], got {tokens.shape}")

    grid = metadata.get("patch_grid") or [int(round(tokens.shape[1] ** 0.5))] * 2
    return TokenCacheInfo(
        tokens_path=str(tokens_path),
        metadata_path=str(metadata_path),
        labels_path=str(labels_path) if labels_path.exists() else None,
        num_images=int(tokens.shape[0]),
        tokens_per_image=int(tokens.shape[1]),
        d_model=int(tokens.shape[2]),
        patch_grid=(int(grid[0]), int(grid[1])),
        patch_size=int(metadata.get("patch_size", 14)),
        encoder_input_size=int(metadata.get("encoder_input_size", metadata.get("data_image_size", 256))),
        data_image_size=int(metadata.get("data_image_size", 256)),
        num_register_tokens=int(metadata.get("num_register_tokens", 0)),
        encoder_name=metadata.get("encoder_name"),
    )


class TokenCacheWriter:
    """Append-style writer for `.npy` token caches with fixed total shape."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        num_images: int,
        patch_shape: Sequence[int],
        save_dtype: str = "float16",
        global_shape: Sequence[int] | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_images = int(num_images)
        self.patch_shape = tuple(int(v) for v in patch_shape)
        self.global_shape = None if global_shape is None else tuple(int(v) for v in global_shape)
        self.dtype = np.dtype(save_dtype)
        self.offset = 0
        self.patch_tokens = np.lib.format.open_memmap(
            self.output_dir / "tokens_patch.npy",
            mode="w+",
            dtype=self.dtype,
            shape=(self.num_images, *self.patch_shape),
        )
        self.global_tokens = None
        if self.global_shape is not None:
            self.global_tokens = np.lib.format.open_memmap(
                self.output_dir / "tokens_global.npy",
                mode="w+",
                dtype=self.dtype,
                shape=(self.num_images, *self.global_shape),
            )

    def write(self, patch_tokens: torch.Tensor, global_tokens: torch.Tensor | None = None) -> None:
        batch = int(patch_tokens.shape[0])
        end = self.offset + batch
        if end > self.num_images:
            raise ValueError(f"Token writer overflow: end={end}, num_images={self.num_images}")
        if tuple(patch_tokens.shape[1:]) != self.patch_shape:
            raise ValueError(f"Patch token shape changed from {self.patch_shape} to {tuple(patch_tokens.shape[1:])}")

        self.patch_tokens[self.offset : end] = patch_tokens.detach().cpu().numpy().astype(self.dtype, copy=False)
        if self.global_tokens is not None:
            if global_tokens is None:
                raise ValueError("Writer was initialized for global tokens, but this batch has none.")
            if tuple(global_tokens.shape[1:]) != self.global_shape:
                raise ValueError(f"Global token shape changed from {self.global_shape} to {tuple(global_tokens.shape[1:])}")
            self.global_tokens[self.offset : end] = global_tokens.detach().cpu().numpy().astype(self.dtype, copy=False)
        self.offset = end

    def close(self) -> None:
        if self.offset != self.num_images:
            raise ValueError(f"Token writer expected {self.num_images} rows, wrote {self.offset}")
        self.patch_tokens.flush()
        if self.global_tokens is not None:
            self.global_tokens.flush()


def build_token_metadata(
    *,
    config: dict[str, Any],
    encoder: Any,
    num_images: int,
    patch_shape: Sequence[int],
    global_shape: Sequence[int] | None,
    output_dir: Path,
    class_counts: Counter[int],
    class_to_idx: dict[str, int],
) -> dict[str, Any]:
    encoder_cfg = dict(config.get("encoder") or {})
    data_cfg = dict(config.get("data") or {})
    tokens_cfg = dict(config.get("tokens") or {})
    resolution = int(encoder_cfg.get("resolution", 256))
    patch_size = int(getattr(encoder, "patch_size", encoder_cfg.get("patch_size", 14)))
    patch_tokens_per_image = int(patch_shape[0])
    patch_grid = int(round(patch_tokens_per_image ** 0.5))
    num_register_tokens = int(getattr(encoder, "num_register_tokens", 0))
    global_token_order = None
    if global_shape is not None:
        global_token_order = ["cls"] + [f"reg_{idx}" for idx in range(num_register_tokens)]

    return {
        "encoder_name": encoder_cfg.get("name"),
        "token_shape": [int(num_images), *[int(v) for v in patch_shape]],
        "global_token_shape": None if global_shape is None else [int(num_images), *[int(v) for v in global_shape]],
        "num_images": int(num_images),
        "num_tokens_per_image": int(patch_tokens_per_image),
        "embedding_dim": int(patch_shape[1]),
        "data_image_size": int(data_cfg.get("image_size", resolution)),
        "encoder_resolution": int(resolution),
        "encoder_input_size": int(224 * (resolution // 256)),
        "patch_size": patch_size,
        "patch_grid": [patch_grid, patch_grid],
        "normalization": encoder_cfg.get("token_normalization", "encoder_norm_l2_optional"),
        "include_global": global_shape is not None,
        "global_token_order": global_token_order,
        "num_register_tokens": num_register_tokens,
        "save_dtype": str(np.dtype(tokens_cfg.get("save_dtype", "float16"))),
        "output_dir": str(output_dir),
        "class_counts": {str(k): int(v) for k, v in sorted(class_counts.items())},
        "class_to_idx": dict(sorted(class_to_idx.items(), key=lambda item: item[1])),
    }

