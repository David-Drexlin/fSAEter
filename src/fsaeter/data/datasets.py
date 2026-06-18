"""Dataset views over token caches."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset

from fsaeter.data.cache import TokenCacheInfo, open_patch_token_reader, resolve_token_cache_info


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
        if base_rows.size and (
            int(base_rows.min()) < 0 or int(base_rows.max()) >= self.info.num_images
        ):
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


@dataclass(frozen=True)
class LoaderDiagnostics:
    rows_yielded: int
    unique_images_seen: int
    image_entropy: float
    normalized_image_entropy: float
    bytes_read: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "rows_yielded": int(self.rows_yielded),
            "unique_images_seen": int(self.unique_images_seen),
            "image_entropy": float(self.image_entropy),
            "normalized_image_entropy": float(self.normalized_image_entropy),
            "bytes_read": int(self.bytes_read),
        }


class PatchTokenShardBatchIterable(IterableDataset[tuple[Tensor, Tensor, Tensor]]):
    """Block/shuffle-buffer token loader for shard-native caches."""

    def __init__(
        self,
        *,
        token_info: TokenCacheInfo,
        image_rows: Sequence[int] | None,
        batch_size: int,
        max_rows: int | None = None,
        image_block_size: int = 64,
        shuffle_buffer_rows: int = 8192,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        replicate_across_ranks: bool = False,
        shuffle: bool = True,
    ):
        if token_info.storage_format != "shard_v1":
            raise ValueError(
                "PatchTokenShardBatchIterable expects shard_v1 token caches. "
                f"Got {token_info.storage_format!r}."
            )
        self.info = token_info
        self.batch_size = max(1, int(batch_size))
        self.tokens_per_image = int(token_info.tokens_per_image)
        self.image_block_size = max(1, int(image_block_size))
        self.shuffle_buffer_rows = max(self.batch_size, int(shuffle_buffer_rows))
        self.seed = int(seed)
        self.epoch = 0
        self.rank = max(0, int(rank))
        self.world_size = max(1, int(world_size))
        self.replicate_across_ranks = bool(replicate_across_ranks)
        self.shuffle = bool(shuffle)
        if image_rows is None:
            base_rows = np.arange(int(token_info.num_images), dtype=np.int64)
        else:
            base_rows = np.asarray(image_rows, dtype=np.int64)
        if base_rows.ndim != 1:
            raise ValueError(f"image_rows must be 1D, got {base_rows.shape}")
        self.base_image_rows = base_rows.astype(np.int64, copy=False)
        if self.replicate_across_ranks or self.world_size <= 1:
            selected_rows = self.base_image_rows
        else:
            selected_rows = self.base_image_rows[self.rank :: self.world_size]
        self.image_rows = np.sort(selected_rows.astype(np.int64, copy=False))
        total_rows = int(self.image_rows.shape[0]) * int(self.tokens_per_image)
        if max_rows is None or int(max_rows) <= 0 or int(max_rows) >= total_rows:
            self.max_rows = total_rows
        else:
            self.max_rows = int(max_rows)
        self._last_diagnostics = LoaderDiagnostics(
            rows_yielded=0,
            unique_images_seen=0,
            image_entropy=0.0,
            normalized_image_entropy=0.0,
            bytes_read=0,
        )

    def __len__(self) -> int:
        return int(math.ceil(float(self.max_rows) / float(self.batch_size)))

    def last_diagnostics(self) -> LoaderDiagnostics:
        return self._last_diagnostics

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _entropy_from_counts(counts: np.ndarray) -> tuple[float, float]:
        if counts.size == 0:
            return 0.0, 0.0
        probs = counts.astype(np.float64, copy=False)
        probs_sum = float(probs.sum())
        if probs_sum <= 0:
            return 0.0, 0.0
        probs = probs / probs_sum
        nonzero = probs > 0
        entropy = float(-(probs[nonzero] * np.log(probs[nonzero])).sum())
        max_entropy = float(np.log(float(max(1, counts.size))))
        normalized = 0.0 if max_entropy <= 0 else float(entropy / max_entropy)
        return entropy, normalized

    def __iter__(self):
        reader = open_patch_token_reader(self.info)
        rng = np.random.default_rng(self.seed + self.epoch)
        row_limit = int(self.max_rows)
        rows_yielded = 0
        bytes_read = 0
        image_counts = np.zeros((int(self.info.num_images),), dtype=np.int64)

        buffer_tokens = np.empty((0, int(self.info.d_model)), dtype=np.float32)
        buffer_image_rows = np.empty((0,), dtype=np.int64)
        buffer_patch_rows = np.empty((0,), dtype=np.int32)

        def append_block(block_rows: np.ndarray, block_tokens: np.ndarray) -> None:
            nonlocal buffer_tokens, buffer_image_rows, buffer_patch_rows, rows_yielded, bytes_read
            if rows_yielded >= row_limit:
                return
            image_counts[block_rows] += int(self.tokens_per_image)
            bytes_read += int(block_tokens.nbytes)
            flat_tokens = block_tokens.reshape(-1, int(self.info.d_model)).astype(np.float32, copy=False)
            flat_image_rows = np.repeat(block_rows, int(self.tokens_per_image))
            flat_patch_rows = np.tile(
                np.arange(int(self.tokens_per_image), dtype=np.int32),
                int(block_rows.shape[0]),
            )
            remaining = row_limit - rows_yielded - int(buffer_tokens.shape[0])
            if remaining <= 0:
                return
            if flat_tokens.shape[0] > remaining:
                flat_tokens = flat_tokens[:remaining]
                flat_image_rows = flat_image_rows[:remaining]
                flat_patch_rows = flat_patch_rows[:remaining]
            buffer_tokens = np.concatenate([buffer_tokens, flat_tokens], axis=0)
            buffer_image_rows = np.concatenate([buffer_image_rows, flat_image_rows], axis=0)
            buffer_patch_rows = np.concatenate([buffer_patch_rows, flat_patch_rows], axis=0)

        read_rows = self.image_rows
        for start in range(0, int(read_rows.shape[0]), int(self.image_block_size)):
            block_rows = read_rows[start : start + int(self.image_block_size)]
            block_tokens = reader.load_image_rows(block_rows)
            append_block(block_rows, block_tokens)
            while buffer_tokens.shape[0] >= int(self.shuffle_buffer_rows):
                order = rng.permutation(buffer_tokens.shape[0])
                buffer_tokens = buffer_tokens[order]
                buffer_image_rows = buffer_image_rows[order]
                buffer_patch_rows = buffer_patch_rows[order]
                emit = (buffer_tokens.shape[0] // self.batch_size) * self.batch_size
                if emit <= 0:
                    break
                tokens_out = buffer_tokens[:emit]
                image_rows_out = buffer_image_rows[:emit]
                patch_rows_out = buffer_patch_rows[:emit]
                buffer_tokens = buffer_tokens[emit:]
                buffer_image_rows = buffer_image_rows[emit:]
                buffer_patch_rows = buffer_patch_rows[emit:]
                for batch_start in range(0, emit, self.batch_size):
                    batch_end = batch_start + self.batch_size
                    rows_yielded += self.batch_size
                    yield (
                        torch.from_numpy(tokens_out[batch_start:batch_end]),
                        torch.from_numpy(image_rows_out[batch_start:batch_end]),
                        torch.from_numpy(patch_rows_out[batch_start:batch_end]),
                    )

        if buffer_tokens.shape[0] > 0:
            order = rng.permutation(buffer_tokens.shape[0])
            buffer_tokens = buffer_tokens[order]
            buffer_image_rows = buffer_image_rows[order]
            buffer_patch_rows = buffer_patch_rows[order]
            for batch_start in range(0, buffer_tokens.shape[0], self.batch_size):
                batch_end = min(batch_start + self.batch_size, buffer_tokens.shape[0])
                rows_yielded += int(batch_end - batch_start)
                yield (
                    torch.from_numpy(buffer_tokens[batch_start:batch_end]),
                    torch.from_numpy(buffer_image_rows[batch_start:batch_end]),
                    torch.from_numpy(buffer_patch_rows[batch_start:batch_end]),
                )

        counts = image_counts[image_counts > 0]
        entropy, normalized_entropy = self._entropy_from_counts(counts)
        self._last_diagnostics = LoaderDiagnostics(
            rows_yielded=int(min(rows_yielded, row_limit)),
            unique_images_seen=int(counts.shape[0]),
            image_entropy=float(entropy),
            normalized_image_entropy=float(normalized_entropy),
            bytes_read=int(bytes_read),
        )
