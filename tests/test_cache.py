from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from fsaeter.data.cache import TokenCacheWriter, build_token_metadata
from fsaeter.data.imagefolder import make_image_records, select_subset_indices


class DummyEncoder:
    patch_size = 14
    num_register_tokens = 4
    encoder_name = "resolved-dinov2-vit-b[norm]"
    encoder_model = "dinov2-b-reg"
    encoder_factory_string = "dinov2-vit-b[norm]"


def test_stratified_subset_round_robin_balances_classes():
    samples = [(f"class{label}/img{idx}.jpg", label) for label in range(3) for idx in range(5)]
    selected = select_subset_indices(samples, 7, strategy="stratified", seed=0)
    counts = Counter(samples[idx][1] for idx in selected)
    assert len(selected) == 7
    assert set(counts) == {0, 1, 2}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_token_cache_writer_shapes_and_metadata(tmp_path: Path):
    writer = TokenCacheWriter(
        tmp_path,
        num_images=3,
        patch_shape=(4, 8),
        global_shape=(5, 8),
        save_dtype="float16",
    )
    writer.write(torch.ones(2, 4, 8), torch.zeros(2, 5, 8))
    writer.write(torch.full((1, 4, 8), 2.0), torch.ones(1, 5, 8))
    writer.close()

    patch = np.load(tmp_path / "tokens_patch.npy", mmap_mode="r")
    global_tokens = np.load(tmp_path / "tokens_global.npy", mmap_mode="r")
    assert patch.shape == (3, 4, 8)
    assert global_tokens.shape == (3, 5, 8)

    metadata = build_token_metadata(
        config={
            "encoder": {"model": "dinov2-b-reg", "resolution": 256},
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
    assert metadata["global_token_order"] == ["cls", "reg_0", "reg_1", "reg_2", "reg_3"]
    assert metadata["encoder_name"] == "resolved-dinov2-vit-b[norm]"
    assert metadata["encoder_model"] == "dinov2-b-reg"
    assert metadata["encoder_factory_string"] == "dinov2-vit-b[norm]"
    assert metadata["encoder_input_size"] == 256
    assert metadata["path_mode"] == "relative_only"
    (tmp_path / "token_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    loaded = json.loads((tmp_path / "token_metadata.json").read_text(encoding="utf-8"))
    assert loaded["num_images"] == 3


def test_make_image_records_redacts_absolute_paths_by_default(tmp_path: Path):
    data_root = tmp_path / "data"
    sample_path = data_root / "train" / "class0" / "img0.png"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_bytes(b"stub")
    dataset = SimpleNamespace(
        samples=[(str(sample_path), 0)],
        class_to_idx={"class0": 0},
    )
    records = make_image_records(dataset, [0], data_root=data_root)
    assert records[0].path is None
    assert records[0].relative_path == "train/class0/img0.png"


def test_make_image_records_can_preserve_absolute_paths(tmp_path: Path):
    data_root = tmp_path / "data"
    sample_path = data_root / "train" / "class0" / "img0.png"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_bytes(b"stub")
    dataset = SimpleNamespace(
        samples=[(str(sample_path), 0)],
        class_to_idx={"class0": 0},
    )
    records = make_image_records(
        dataset,
        [0],
        data_root=data_root,
        write_absolute_paths=True,
    )
    assert records[0].path == str(sample_path)
    assert records[0].relative_path == "train/class0/img0.png"
