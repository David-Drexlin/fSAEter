from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fsaeter.cli import main
from fsaeter.data.cache import TokenCacheWriter, build_token_metadata


class DummyEncoder:
    patch_size = 14
    num_register_tokens = 0
    encoder_name = "dummy-factory"
    encoder_model = "dummy-model"
    encoder_factory_string = "dummy-factory"


def write_cache(tmp_path: Path) -> Path:
    tokens_dir = tmp_path / "tokens"
    writer = TokenCacheWriter(tokens_dir, num_images=4, patch_shape=(4, 8), save_dtype="float16")
    patch = torch.randn(4, 4, 8)
    writer.write(patch[:2])
    writer.write(patch[2:])
    writer.close()
    metadata = build_token_metadata(
        config={
            "encoder": {"model": "dummy-model", "resolution": 256},
            "data": {"image_size": 256},
            "tokens": {"save_dtype": "float16"},
        },
        encoder=DummyEncoder(),
        num_images=4,
        patch_shape=(4, 8),
        global_shape=None,
        output_dir=tokens_dir,
        class_counts=Counter({0: 2, 1: 2}),
        class_to_idx={"a": 0, "b": 1},
    )
    (tokens_dir / "token_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    np.save(tokens_dir / "labels.npy", np.asarray([0, 0, 1, 1], dtype=np.int64))
    return tokens_dir


def write_image_ids(tokens_dir: Path, *, relative_only: bool, with_files: bool) -> Path:
    data_root = tokens_dir.parent / "data"
    rows = []
    records = [
        ("train/a/img0.png", "a"),
        ("train/a/img1.png", "a"),
        ("train/b/img2.png", "b"),
        ("train/b/img3.png", "b"),
    ]
    for row_index, (relative_path, class_name) in enumerate(records):
        path_value = None
        image_path = data_root / relative_path
        if with_files:
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), color=(row_index * 20, 0, 0)).save(image_path)
            if not relative_only:
                path_value = str(image_path)
        rows.append(
            {
                "row_index": row_index,
                "dataset_index": row_index,
                "path": path_value,
                "relative_path": relative_path,
                "class_index": 0 if class_name == "a" else 1,
                "class_name": class_name,
            }
        )
    with (tokens_dir / "image_ids.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return data_root


def test_cli_smoke_cache_only_pipeline(tmp_path: Path):
    write_cache(tmp_path)
    train_cfg = tmp_path / "train.yaml"
    train_cfg.write_text(
        """
run:
  out_dir: out/train
tokens:
  cache_dir: tokens
sae:
  variant: batchtopk
  d_model: 8
  d_sae: 16
  target_k: 2
train:
  device: cpu
  precision: fp32
  batch_size: 4
  epochs: 1
  num_workers: 0
""",
        encoding="utf-8",
    )
    build_cfg = tmp_path / "build.yaml"
    build_cfg.write_text(
        """
run:
  out_dir: out/h
tokens:
  cache_dir: tokens
sae:
  checkpoint: out/train/checkpoints/best.pt
build_h:
  device: cpu
  precision: fp32
  image_batch_size: 2
  token_batch_size: 8
inspect:
  device: cpu
  preview_concepts: 2
  preview_images_per_concept: 2
  min_support: 1
  min_class_coverage: 1
  min_per_class: 1
""",
        encoding="utf-8",
    )
    cwd = Path.cwd()
    try:
        import os
        os.chdir(tmp_path)
        assert main(["train-sae", "--config", str(train_cfg)]) == 0
        assert main(["build-h", "--config", str(build_cfg)]) == 0
        assert main(["inspect", "--config", str(build_cfg)]) == 0
    finally:
        os.chdir(cwd)
    assert (tmp_path / "out" / "h" / "H_mean.npy").exists()
    assert (tmp_path / "out" / "h" / "qc_summary.json").exists()


def test_inspect_resolves_relative_only_records_with_data_root(tmp_path: Path):
    tokens_dir = write_cache(tmp_path)
    data_root = write_image_ids(tokens_dir, relative_only=True, with_files=True)
    train_cfg = tmp_path / "train.yaml"
    train_cfg.write_text(
        """
run:
  seed: 7
  out_dir: out/train
tokens:
  cache_dir: tokens
sae:
  variant: batchtopk
  d_model: 8
  d_sae: 16
  target_k: 2
train:
  device: cpu
  precision: fp32
  batch_size: 4
  epochs: 1
  num_workers: 0
""",
        encoding="utf-8",
    )
    build_cfg = tmp_path / "build.yaml"
    build_cfg.write_text(
        f"""
run:
  out_dir: out/h
tokens:
  cache_dir: tokens
data:
  root: {data_root}
sae:
  checkpoint: out/train/checkpoints/best.pt
build_h:
  device: cpu
  precision: fp32
  image_batch_size: 2
  token_batch_size: 8
inspect:
  device: cpu
  preview_concepts: 2
  preview_images_per_concept: 2
  min_support: 1
  min_class_coverage: 1
  min_per_class: 1
""",
        encoding="utf-8",
    )
    cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        assert main(["train-sae", "--config", str(train_cfg)]) == 0
        assert main(["build-h", "--config", str(build_cfg)]) == 0
        assert main(["inspect", "--config", str(build_cfg)]) == 0
    finally:
        os.chdir(cwd)
    qc_summary = json.loads((tmp_path / "out" / "h" / "qc_summary.json").read_text(encoding="utf-8"))
    assert qc_summary["previews"]["images_written"] > 0
    assert any((tmp_path / "out" / "h" / "top_images").glob("*.png"))


def test_inspect_skips_previews_cleanly_when_source_paths_are_unresolvable(tmp_path: Path):
    tokens_dir = write_cache(tmp_path)
    write_image_ids(tokens_dir, relative_only=True, with_files=False)
    train_cfg = tmp_path / "train.yaml"
    train_cfg.write_text(
        """
run:
  seed: 11
  out_dir: out/train
tokens:
  cache_dir: tokens
sae:
  variant: batchtopk
  d_model: 8
  d_sae: 16
  target_k: 2
train:
  device: cpu
  precision: fp32
  batch_size: 4
  epochs: 1
  num_workers: 0
""",
        encoding="utf-8",
    )
    build_cfg = tmp_path / "build.yaml"
    build_cfg.write_text(
        """
run:
  out_dir: out/h
tokens:
  cache_dir: tokens
sae:
  checkpoint: out/train/checkpoints/best.pt
build_h:
  device: cpu
  precision: fp32
  image_batch_size: 2
  token_batch_size: 8
inspect:
  device: cpu
  preview_concepts: 2
  preview_images_per_concept: 2
  min_support: 1
  min_class_coverage: 1
  min_per_class: 1
""",
        encoding="utf-8",
    )
    cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        assert main(["train-sae", "--config", str(train_cfg)]) == 0
        assert main(["build-h", "--config", str(build_cfg)]) == 0
        assert main(["inspect", "--config", str(build_cfg)]) == 0
    finally:
        os.chdir(cwd)
    qc_summary = json.loads((tmp_path / "out" / "h" / "qc_summary.json").read_text(encoding="utf-8"))
    assert qc_summary["previews"]["images_skipped_missing_source"] > 0
    assert not (tmp_path / "out" / "h" / "top_images").exists()


def test_train_seed_produces_stable_splits_and_metrics(tmp_path: Path):
    write_cache(tmp_path)
    base_cfg = tmp_path / "train.yaml"
    base_cfg.write_text(
        """
run:
  seed: 19
tokens:
  cache_dir: tokens
sae:
  variant: batchtopk
  d_model: 8
  d_sae: 16
  target_k: 2
train:
  device: cpu
  precision: fp32
  batch_size: 4
  epochs: 1
  num_workers: 0
""",
        encoding="utf-8",
    )
    cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        assert main(["train-sae", "--config", str(base_cfg), "--out", "out/train_a"]) == 0
        assert main(["train-sae", "--config", str(base_cfg), "--out", "out/train_b"]) == 0
    finally:
        os.chdir(cwd)
    train_rows_a = np.load(tmp_path / "out" / "train_a" / "train_image_rows.npy")
    train_rows_b = np.load(tmp_path / "out" / "train_b" / "train_image_rows.npy")
    val_rows_a = np.load(tmp_path / "out" / "train_a" / "val_image_rows.npy")
    val_rows_b = np.load(tmp_path / "out" / "train_b" / "val_image_rows.npy")
    history_a = json.loads((tmp_path / "out" / "train_a" / "history.json").read_text(encoding="utf-8"))
    history_b = json.loads((tmp_path / "out" / "train_b" / "history.json").read_text(encoding="utf-8"))
    np.testing.assert_array_equal(train_rows_a, train_rows_b)
    np.testing.assert_array_equal(val_rows_a, val_rows_b)
    assert [row["epoch"] for row in history_a] == [row["epoch"] for row in history_b]
    assert [row["train"] for row in history_a] == [row["train"] for row in history_b]
    assert [row["val"] for row in history_a] == [row["val"] for row in history_b]
