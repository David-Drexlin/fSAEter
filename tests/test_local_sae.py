from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from fsaeter.data.cache import TokenCacheWriter, build_token_metadata
from fsaeter.data.datasets import PatchTokenMemmapDataset
from fsaeter.models.local_sae import LocalSparseAutoencoder


class DummyEncoder:
    patch_size = 14
    num_register_tokens = 4


def write_token_cache(tmp_path: Path) -> Path:
    writer = TokenCacheWriter(
        tmp_path,
        num_images=3,
        patch_shape=(4, 8),
        global_shape=(5, 8),
        save_dtype="float16",
    )
    patch = torch.arange(3 * 4 * 8, dtype=torch.float32).reshape(3, 4, 8)
    global_tokens = torch.zeros(3, 5, 8, dtype=torch.float32)
    writer.write(patch[:2], global_tokens[:2])
    writer.write(patch[2:], global_tokens[2:])
    writer.close()
    metadata = build_token_metadata(
        config={
            "encoder": {"name": "dinov2-vit-b[norm]", "resolution": 256},
            "data": {"image_size": 256},
            "tokens": {"save_dtype": "float16"},
        },
        encoder=DummyEncoder(),
        num_images=3,
        patch_shape=(4, 8),
        global_shape=(5, 8),
        output_dir=tmp_path,
        class_counts=Counter({0: 2, 1: 1}),
        class_to_idx={"a": 0, "b": 1},
    )
    (tmp_path / "token_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    np.save(tmp_path / "labels.npy", np.asarray([0, 1, 0], dtype=np.int64))
    return tmp_path


def test_patch_token_memmap_dataset_mapping_is_deterministic(tmp_path: Path):
    tokens_dir = write_token_cache(tmp_path)
    dataset = PatchTokenMemmapDataset(tokens_dir, image_rows=[2, 0], max_rows=6)
    assert len(dataset) == 6
    assert dataset.global_row_to_image_patch(0) == (2, 0)
    assert dataset.global_row_to_image_patch(5) == (0, 1)
    assert dataset.image_patch_to_global_row(0, 1) == 5


def test_batchtopk_keeps_exact_average_budget():
    model = LocalSparseAutoencoder(d_model=4, d_sae=10, target_k=3, variant="batchtopk")
    positive_acts = torch.arange(1, 41, dtype=torch.float32).reshape(4, 10)
    sparse = model._batch_topk_nonnegative(positive_acts, target_k=3)
    l0 = (sparse > 0).sum(dim=1)
    assert int(l0.sum().item()) == 12
    assert float(l0.float().mean().item()) == 3.0


def test_matryoshka_prefixes_are_monotonic():
    model = LocalSparseAutoencoder(
        d_model=4,
        d_sae=16,
        target_k=4,
        variant="matryoshka_batchtopk",
        matryoshka_prefixes=[4, 8, 16],
    )
    assert model.matryoshka_prefixes == (4, 8, 16)
