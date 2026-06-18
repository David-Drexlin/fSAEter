from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from fsaeter.cli import main
from fsaeter.data.cache import TokenCacheWriter, build_token_metadata


class DummyEncoder:
    patch_size = 14
    num_register_tokens = 0


def write_cache(tmp_path: Path) -> Path:
    tokens_dir = tmp_path / "tokens"
    writer = TokenCacheWriter(tokens_dir, num_images=4, patch_shape=(4, 8), save_dtype="float16")
    patch = torch.randn(4, 4, 8)
    writer.write(patch[:2])
    writer.write(patch[2:])
    writer.close()
    metadata = build_token_metadata(
        config={"encoder": {"name": "dummy", "resolution": 256}, "data": {"image_size": 256}, "tokens": {"save_dtype": "float16"}},
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


def test_cli_smoke_cache_only_pipeline(tmp_path: Path):
    tokens_dir = write_cache(tmp_path)
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
