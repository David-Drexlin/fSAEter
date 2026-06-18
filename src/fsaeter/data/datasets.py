"""Dataset views over token caches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from fsaeter.data.cache import TokenCacheInfo, resolve_token_cache_info


def split_image_rows(
    num_images: int,
    *,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    num_images = int(num_images)
    if num_images <= 0:
        raise ValueError("num_images must be positive")
    if val_fraction <= 0:
        return np.arange(num_images, dtype=np.int64), np.empty((0,), dtype=np.int64)
    if val_fraction >= 1:
        return np.empty((0,), dtype=np.int64), np.arange(num_images, dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(num_images).astype(np.int64, copy=False)
    val_count = max(1, int(round(num_images * float(val_fraction))))
    val_rows = np.sort(perm[:val_count])
    train_rows = np.sort(perm[val_count:])
    return train_rows, val_rows


class PatchTokenMemmapDataset(Dataset[tuple[Tensor, int, int]]):
    """Expose patch-token rows from a token cache while preserving image/patch mapping."""

    def __init__(
        self,
        tokens_dir: str | None = None,
        *,
        token_info: TokenCacheInfo | None = None,
        image_rows: Sequence[int] | None = None,
        max_rows: int | None = None,
    ):
        if token_info is None and tokens_dir is None:
            raise ValueError("Provide either tokens_dir or token_info.")
        self.info = token_info or resolve_token_cache_info(tokens_dir)  # type: ignore[arg-type]
        self.tokens = np.load(self.info.tokens_path, mmap_mode="r")
        if image_rows is None:
            base_rows = np.arange(self.info.num_images, dtype=np.int64)
        else:
            base_rows = np.asarray(image_rows, dtype=np.int64)
        if base_rows.ndim != 1:
            raise ValueError(f"image_rows must be 1D, got {base_rows.shape}")
        if base_rows.size and (int(base_rows.min()) < 0 or int(base_rows.max()) >= self.info.num_images):
            raise IndexError("image_rows contain indices outside the token cache")

        self.image_rows = base_rows
        total_rows = int(self.image_rows.shape[0]) * int(self.info.tokens_per_image)
        if max_rows is None or max_rows <= 0 or max_rows >= total_rows:
            self.num_rows = total_rows
        else:
            self.num_rows = int(max_rows)

    def __len__(self) -> int:
        return self.num_rows

    def global_row_to_image_patch(self, row_idx: int) -> tuple[int, int]:
        row_idx = int(row_idx)
        if row_idx < 0 or row_idx >= self.num_rows:
            raise IndexError(f"row_idx {row_idx} out of range [0, {self.num_rows})")
        local_image_idx = row_idx // self.info.tokens_per_image
        patch_row = row_idx % self.info.tokens_per_image
        image_row = int(self.image_rows[local_image_idx])
        return image_row, patch_row

    def image_patch_to_global_row(self, image_row: int, patch_row: int) -> int:
        image_row = int(image_row)
        patch_row = int(patch_row)
        if patch_row < 0 or patch_row >= self.info.tokens_per_image:
            raise IndexError(f"patch_row {patch_row} outside [0, {self.info.tokens_per_image})")
        matches = np.where(self.image_rows == image_row)[0]
        if matches.size == 0:
            raise KeyError(f"image_row {image_row} is not present in this dataset split")
        return int(matches[0]) * self.info.tokens_per_image + patch_row

    def __getitem__(self, idx: int) -> tuple[Tensor, int, int]:
        image_row, patch_row = self.global_row_to_image_patch(idx)
        activation = np.asarray(self.tokens[image_row, patch_row], dtype=np.float32)
        return torch.from_numpy(activation), image_row, patch_row


@dataclass(frozen=True)
class DatasetPreview:
    num_images: int
    train_images: int
    val_images: int
    train_rows: int
    val_rows: int

