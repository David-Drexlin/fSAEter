"""Token cache metadata and activation-cache IO."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    encoder_model: str | None = None
    encoder_factory_string: str | None = None
    path_mode: str = "relative_only"
    storage_format: str = "legacy_npy"
    save_dtype: str = "float16"
    global_tokens_path: str | None = None
    global_shape: tuple[int, int] | None = None
    patch_shard_paths: tuple[str, ...] = ()
    global_shard_paths: tuple[str, ...] = ()
    shard_num_images: tuple[int, ...] = ()


def _metadata_token_shape(metadata: dict[str, Any]) -> tuple[int, int, int] | None:
    token_shape = metadata.get("token_shape")
    if not isinstance(token_shape, list) or len(token_shape) != 3:
        return None
    return (int(token_shape[0]), int(token_shape[1]), int(token_shape[2]))


def _metadata_global_shape(metadata: dict[str, Any]) -> tuple[int, int] | None:
    token_shape = metadata.get("global_token_shape")
    if token_shape in (None, []):
        return None
    if not isinstance(token_shape, list) or len(token_shape) != 3:
        return None
    return (int(token_shape[1]), int(token_shape[2]))


def _resolve_relative_paths(root: Path, values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    resolved: list[str] = []
    for value in values:
        path = Path(value)
        resolved.append(str(path if path.is_absolute() else (root / path).resolve()))
    return tuple(resolved)


def load_token_metadata(tokens_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    root = Path(tokens_dir).expanduser().resolve()
    metadata_path = root / "token_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Token metadata not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return root, metadata


def resolve_token_cache_info(tokens_dir: str | Path) -> TokenCacheInfo:
    root, metadata = load_token_metadata(tokens_dir)
    metadata_path = root / "token_metadata.json"
    labels_path = root / "labels.npy"
    storage_format = str(metadata.get("storage_format", "legacy_npy")).lower()
    global_shape = _metadata_global_shape(metadata)
    save_dtype = str(metadata.get("save_dtype", "float16"))

    if storage_format == "shard_v1":
        patch_paths = _resolve_relative_paths(root, metadata.get("patch_shards"))
        if not patch_paths:
            patch_dir = root / "patch_shards"
            patch_paths = tuple(str(path.resolve()) for path in sorted(patch_dir.glob("patch_*.npy")))
        if not patch_paths:
            raise FileNotFoundError(f"Shard token cache not found under {root}")
        shard_arrays = [np.load(path, mmap_mode="r") for path in patch_paths]
        shard_num_images = tuple(int(array.shape[0]) for array in shard_arrays)
        token_shape = _metadata_token_shape(metadata)
        if token_shape is None:
            num_images = int(sum(shard_num_images))
            tokens_per_image = int(shard_arrays[0].shape[1])
            d_model = int(shard_arrays[0].shape[2])
        else:
            num_images, tokens_per_image, d_model = token_shape
        grid = metadata.get("patch_grid") or [int(round(tokens_per_image**0.5))] * 2
        global_paths = _resolve_relative_paths(root, metadata.get("global_shards"))
        if not global_paths:
            global_dir = root / "global_shards"
            if global_dir.is_dir():
                global_paths = tuple(
                    str(path.resolve()) for path in sorted(global_dir.glob("global_*.npy"))
                )
        return TokenCacheInfo(
            tokens_path=str((root / "patch_shards").resolve()),
            metadata_path=str(metadata_path),
            labels_path=str(labels_path) if labels_path.exists() else None,
            num_images=int(num_images),
            tokens_per_image=int(tokens_per_image),
            d_model=int(d_model),
            patch_grid=(int(grid[0]), int(grid[1])),
            patch_size=int(metadata.get("patch_size", 14)),
            encoder_input_size=int(
                metadata.get("encoder_input_size", metadata.get("data_image_size", 256))
            ),
            data_image_size=int(metadata.get("data_image_size", 256)),
            num_register_tokens=int(metadata.get("num_register_tokens", 0)),
            encoder_name=metadata.get("encoder_name"),
            encoder_model=metadata.get("encoder_model"),
            encoder_factory_string=metadata.get("encoder_factory_string"),
            path_mode=str(metadata.get("path_mode", "relative_only")),
            storage_format=storage_format,
            save_dtype=save_dtype,
            global_tokens_path=(
                None if not global_paths else str((root / "global_shards").resolve())
            ),
            global_shape=global_shape,
            patch_shard_paths=patch_paths,
            global_shard_paths=global_paths,
            shard_num_images=shard_num_images,
        )

    tokens_path = root / "tokens_patch.npy"
    if not tokens_path.exists():
        raise FileNotFoundError(f"Token cache not found: {tokens_path}")
    tokens = np.load(tokens_path, mmap_mode="r")
    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens_patch.npy shaped [N,T,D], got {tokens.shape}")
    grid = metadata.get("patch_grid") or [int(round(tokens.shape[1] ** 0.5))] * 2
    global_tokens_path = root / "tokens_global.npy"
    return TokenCacheInfo(
        tokens_path=str(tokens_path),
        metadata_path=str(metadata_path),
        labels_path=str(labels_path) if labels_path.exists() else None,
        num_images=int(tokens.shape[0]),
        tokens_per_image=int(tokens.shape[1]),
        d_model=int(tokens.shape[2]),
        patch_grid=(int(grid[0]), int(grid[1])),
        patch_size=int(metadata.get("patch_size", 14)),
        encoder_input_size=int(
            metadata.get("encoder_input_size", metadata.get("data_image_size", 256))
        ),
        data_image_size=int(metadata.get("data_image_size", 256)),
        num_register_tokens=int(metadata.get("num_register_tokens", 0)),
        encoder_name=metadata.get("encoder_name"),
        encoder_model=metadata.get("encoder_model"),
        encoder_factory_string=metadata.get("encoder_factory_string"),
        path_mode=str(metadata.get("path_mode", "relative_only")),
        storage_format="legacy_npy",
        save_dtype=save_dtype,
        global_tokens_path=(
            str(global_tokens_path) if global_tokens_path.exists() else None
        ),
        global_shape=global_shape,
    )


class PatchTokenReader:
    """Image-major reader for both legacy and shard-native token caches."""

    def __init__(self, info: TokenCacheInfo):
        self.info = info
        if info.storage_format == "shard_v1":
            self._arrays = [np.load(path, mmap_mode="r") for path in info.patch_shard_paths]
            if not self._arrays:
                raise FileNotFoundError("No patch shards were resolved for shard_v1 token cache.")
            self._starts: list[int] = []
            offset = 0
            for array in self._arrays:
                if array.ndim != 3:
                    raise ValueError(f"Expected shard array shaped [N,T,D], got {array.shape}")
                self._starts.append(offset)
                offset += int(array.shape[0])
        else:
            array = np.load(info.tokens_path, mmap_mode="r")
            if array.ndim != 3:
                raise ValueError(f"Expected tokens shaped [N,T,D], got {array.shape}")
            self._arrays = [array]
            self._starts = [0]
        sample = self._arrays[0]
        self.shape = (int(info.num_images), int(sample.shape[1]), int(sample.shape[2]))
        self.dtype = sample.dtype
        self.bytes_per_image = int(np.prod(sample.shape[1:]) * sample.dtype.itemsize)

    def __len__(self) -> int:
        return int(self.shape[0])

    def _shard_bounds(self, shard_index: int) -> tuple[int, int]:
        start = int(self._starts[shard_index])
        end = start + int(self._arrays[shard_index].shape[0])
        return start, end

    def load_image_rows(self, image_rows: Sequence[int] | np.ndarray) -> np.ndarray:
        rows = np.asarray(image_rows, dtype=np.int64)
        if rows.ndim != 1:
            raise ValueError(f"image_rows must be 1D, got {rows.shape}")
        if rows.size == 0:
            return np.empty((0, self.shape[1], self.shape[2]), dtype=self.dtype)
        if int(rows.min()) < 0 or int(rows.max()) >= int(self.shape[0]):
            raise IndexError("image_rows contain indices outside the token cache")
        out = np.empty((int(rows.shape[0]), self.shape[1], self.shape[2]), dtype=self.dtype)
        for shard_index, array in enumerate(self._arrays):
            start, end = self._shard_bounds(shard_index)
            mask = (rows >= start) & (rows < end)
            if not np.any(mask):
                continue
            out[mask] = np.asarray(array[rows[mask] - start], dtype=self.dtype)
        return out

    def load_image_slice(self, start: int, end: int) -> np.ndarray:
        start = int(start)
        end = int(end)
        if start < 0 or end < start or end > int(self.shape[0]):
            raise IndexError(f"Invalid image slice [{start}, {end}) for cache with {self.shape[0]} rows")
        return self.load_image_rows(np.arange(start, end, dtype=np.int64))


def open_patch_token_reader(token_info: TokenCacheInfo) -> PatchTokenReader:
    return PatchTokenReader(token_info)


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
            raise ValueError(
                f"Patch token shape changed from {self.patch_shape} to "
                f"{tuple(patch_tokens.shape[1:])}"
            )

        self.patch_tokens[self.offset : end] = (
            patch_tokens.detach().cpu().numpy().astype(self.dtype, copy=False)
        )
        if self.global_tokens is not None:
            if global_tokens is None:
                raise ValueError("Writer was initialized for global tokens, but this batch has none.")
            if tuple(global_tokens.shape[1:]) != self.global_shape:
                raise ValueError(
                    f"Global token shape changed from {self.global_shape} to "
                    f"{tuple(global_tokens.shape[1:])}"
                )
            self.global_tokens[self.offset : end] = (
                global_tokens.detach().cpu().numpy().astype(self.dtype, copy=False)
            )
        self.offset = end

    def close(self) -> None:
        if self.offset != self.num_images:
            raise ValueError(f"Token writer expected {self.num_images} rows, wrote {self.offset}")
        self.patch_tokens.flush()
        if self.global_tokens is not None:
            self.global_tokens.flush()


class ShardTokenCacheWriter:
    """Sequential writer for shard-native activation caches."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        num_images: int,
        patch_shape: Sequence[int],
        save_dtype: str = "float16",
        global_shape: Sequence[int] | None = None,
        shard_images: int = 256,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.patch_dir = self.output_dir / "patch_shards"
        self.patch_dir.mkdir(parents=True, exist_ok=True)
        self.global_dir = self.output_dir / "global_shards"
        self.num_images = int(num_images)
        self.patch_shape = tuple(int(v) for v in patch_shape)
        self.global_shape = None if global_shape is None else tuple(int(v) for v in global_shape)
        self.dtype = np.dtype(save_dtype)
        self.shard_images = max(1, int(shard_images))
        self.offset = 0
        self.shard_index = -1
        self.current_shard_start = 0
        self.current_shard_rows = 0
        self.current_patch = None
        self.current_global = None
        self.patch_shards: list[str] = []
        self.global_shards: list[str] = []
        self.shard_num_images: list[int] = []
        self._open_next_shard()

    @property
    def storage_format(self) -> str:
        return "shard_v1"

    def _open_next_shard(self) -> None:
        if self.current_patch is not None:
            self.current_patch.flush()
        if self.current_global is not None:
            self.current_global.flush()
        self.shard_index += 1
        self.current_shard_start = int(self.offset)
        remaining = self.num_images - int(self.current_shard_start)
        if remaining <= 0:
            self.current_patch = None
            self.current_global = None
            self.current_shard_rows = 0
            return
        shard_rows = min(int(self.shard_images), int(remaining))
        self.current_shard_rows = shard_rows
        patch_path = self.patch_dir / f"patch_{self.shard_index:05d}.npy"
        self.current_patch = np.lib.format.open_memmap(
            patch_path,
            mode="w+",
            dtype=self.dtype,
            shape=(shard_rows, *self.patch_shape),
        )
        self.patch_shards.append(str(patch_path.relative_to(self.output_dir)))
        if self.global_shape is not None:
            self.global_dir.mkdir(parents=True, exist_ok=True)
            global_path = self.global_dir / f"global_{self.shard_index:05d}.npy"
            self.current_global = np.lib.format.open_memmap(
                global_path,
                mode="w+",
                dtype=self.dtype,
                shape=(shard_rows, *self.global_shape),
            )
            self.global_shards.append(str(global_path.relative_to(self.output_dir)))
        else:
            self.current_global = None
        self.shard_num_images.append(int(shard_rows))

    def write(self, patch_tokens: torch.Tensor, global_tokens: torch.Tensor | None = None) -> None:
        batch_size = int(patch_tokens.shape[0])
        if tuple(patch_tokens.shape[1:]) != self.patch_shape:
            raise ValueError(
                f"Patch token shape changed from {self.patch_shape} to "
                f"{tuple(patch_tokens.shape[1:])}"
            )
        if self.global_shape is not None:
            if global_tokens is None:
                raise ValueError("Writer was initialized for global tokens, but this batch has none.")
            if tuple(global_tokens.shape[1:]) != self.global_shape:
                raise ValueError(
                    f"Global token shape changed from {self.global_shape} to "
                    f"{tuple(global_tokens.shape[1:])}"
                )

        patch_np = patch_tokens.detach().cpu().numpy().astype(self.dtype, copy=False)
        global_np = None
        if global_tokens is not None:
            global_np = global_tokens.detach().cpu().numpy().astype(self.dtype, copy=False)

        written = 0
        while written < batch_size:
            if self.current_patch is None:
                raise ValueError("Token writer overflow while opening the next shard.")
            shard_offset = int(self.offset - self.current_shard_start)
            room = int(self.current_shard_rows - shard_offset)
            if room <= 0:
                self._open_next_shard()
                continue
            take = min(room, batch_size - written)
            end = shard_offset + take
            self.current_patch[shard_offset:end] = patch_np[written : written + take]
            if self.current_global is not None and global_np is not None:
                self.current_global[shard_offset:end] = global_np[written : written + take]
            written += take
            self.offset += take
            if int(self.offset - self.current_shard_start) >= int(self.current_shard_rows):
                self._open_next_shard()

    def close(self) -> None:
        if self.offset != self.num_images:
            raise ValueError(f"Token writer expected {self.num_images} rows, wrote {self.offset}")
        if self.current_patch is not None:
            self.current_patch.flush()
        if self.current_global is not None:
            self.current_global.flush()


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
    storage_format: str = "legacy_npy",
    patch_shards: Sequence[str] | None = None,
    global_shards: Sequence[str] | None = None,
    shard_num_images: Sequence[int] | None = None,
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

    metadata = {
        "encoder_name": getattr(encoder, "encoder_name", None),
        "encoder_model": getattr(encoder, "encoder_model", None),
        "encoder_factory_string": getattr(encoder, "encoder_factory_string", None),
        "storage_format": str(storage_format),
        "token_shape": [int(num_images), *[int(v) for v in patch_shape]],
        "global_token_shape": (
            None
            if global_shape is None
            else [int(num_images), *[int(v) for v in global_shape]]
        ),
        "num_images": int(num_images),
        "num_tokens_per_image": int(patch_tokens_per_image),
        "embedding_dim": int(patch_shape[1]),
        "data_image_size": int(data_cfg.get("image_size", resolution)),
        "encoder_resolution": int(resolution),
        "encoder_input_size": int(data_cfg.get("image_size", resolution)),
        "patch_size": patch_size,
        "patch_grid": [patch_grid, patch_grid],
        "normalization": encoder_cfg.get("token_normalization", "encoder_norm_l2_optional"),
        "include_global": global_shape is not None,
        "global_token_order": global_token_order,
        "num_register_tokens": num_register_tokens,
        "path_mode": (
            "absolute_and_relative"
            if bool(data_cfg.get("write_absolute_paths", False))
            else "relative_only"
        ),
        "save_dtype": str(np.dtype(tokens_cfg.get("save_dtype", "float16"))),
        "output_dir": str(output_dir),
        "class_counts": {str(k): int(v) for k, v in sorted(class_counts.items())},
        "class_to_idx": dict(sorted(class_to_idx.items(), key=lambda item: item[1])),
    }
    if str(storage_format).lower() == "shard_v1":
        metadata["patch_shards"] = [str(value) for value in (patch_shards or ())]
        metadata["global_shards"] = [str(value) for value in (global_shards or ())]
        metadata["shard_num_images"] = [int(value) for value in (shard_num_images or ())]
    return metadata


def convert_token_cache(
    source_dir: str | Path,
    out_dir: str | Path,
    *,
    shard_images: int = 256,
) -> dict[str, Any]:
    source_root = Path(source_dir).expanduser().resolve()
    target_root = Path(out_dir).expanduser().resolve()
    info = resolve_token_cache_info(source_root)
    if info.storage_format != "legacy_npy":
        raise ValueError(
            "convert-token-cache currently migrates legacy_npy caches into shard_v1 output."
        )
    target_root.mkdir(parents=True, exist_ok=True)
    patch_tokens = np.load(info.tokens_path, mmap_mode="r")
    global_tokens = None
    if info.global_tokens_path is not None and Path(info.global_tokens_path).exists():
        global_tokens = np.load(info.global_tokens_path, mmap_mode="r")
    writer = ShardTokenCacheWriter(
        target_root,
        num_images=int(info.num_images),
        patch_shape=patch_tokens.shape[1:],
        global_shape=None if global_tokens is None else global_tokens.shape[1:],
        save_dtype=info.save_dtype,
        shard_images=shard_images,
    )
    for start in range(0, int(info.num_images), max(1, int(shard_images))):
        end = min(start + max(1, int(shard_images)), int(info.num_images))
        patch_batch = torch.from_numpy(np.asarray(patch_tokens[start:end], dtype=np.float32))
        global_batch = None
        if global_tokens is not None:
            global_batch = torch.from_numpy(np.asarray(global_tokens[start:end], dtype=np.float32))
        writer.write(patch_batch, global_batch)
    writer.close()

    _, metadata = load_token_metadata(source_root)
    metadata["storage_format"] = "shard_v1"
    metadata["patch_shards"] = list(writer.patch_shards)
    metadata["global_shards"] = list(writer.global_shards)
    metadata["shard_num_images"] = list(writer.shard_num_images)
    metadata["output_dir"] = str(target_root)
    (target_root / "token_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for filename in ("labels.npy", "image_ids.jsonl", "config_resolved.yaml"):
        source_path = source_root / filename
        if source_path.exists():
            shutil.copy2(source_path, target_root / filename)
    return {
        "source_dir": str(source_root),
        "out_dir": str(target_root),
        "storage_format": "shard_v1",
        "num_images": int(info.num_images),
        "patch_shards": list(writer.patch_shards),
        "global_shards": list(writer.global_shards),
        "shard_num_images": list(writer.shard_num_images),
    }
