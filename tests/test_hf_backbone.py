from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fsaeter.cli import main


def _write_tiny_imagefolder(root: Path) -> None:
    for class_name, color in (("a", (255, 0, 0)), ("b", (0, 255, 0))):
        class_dir = root / "train" / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(2):
            Image.new("RGB", (32, 32), color=color).save(class_dir / f"img{idx}.png")


def test_extract_tokens_hf_backbone_path_with_mocked_transformers(tmp_path: Path, monkeypatch):
    _write_tiny_imagefolder(tmp_path / "data")

    class DummyProcessor:
        @classmethod
        def from_pretrained(cls, repo_id: str):
            assert repo_id == "facebook/dinov2-base"
            return cls()

        def __call__(self, *, images, return_tensors: str, size=None, crop_size=None):
            assert return_tensors == "pt"
            height = int(size["height"]) if isinstance(size, dict) else 32
            width = int(size["width"]) if isinstance(size, dict) else 32
            batch = len(images)
            pixel_values = torch.linspace(
                0.0,
                1.0,
                steps=batch * 3 * height * width,
                dtype=torch.float32,
            ).reshape(batch, 3, height, width)
            return {"pixel_values": pixel_values}

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = types.SimpleNamespace(patch_size=16, num_register_tokens=1)
            self._anchor = torch.nn.Parameter(torch.zeros(()))

        @classmethod
        def from_pretrained(cls, repo_id: str):
            assert repo_id == "facebook/dinov2-base"
            return cls()

        def forward(self, *, pixel_values: torch.Tensor):
            batch = int(pixel_values.shape[0])
            hidden = torch.arange(batch * 6 * 8, dtype=torch.float32).reshape(batch, 6, 8)
            return types.SimpleNamespace(last_hidden_state=hidden)

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoImageProcessor = DummyProcessor
    transformers_stub.AutoModel = DummyModel
    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)

    config_path = tmp_path / "extract.yaml"
    config_path.write_text(
        """
run:
  out_dir: out
data:
  root: data
  split: train
  image_size: 32
encoder:
  resolution: 32
tokens:
  batch_size: 2
  num_workers: 0
  device: cpu
  precision: fp32
  save_dtype: float16
""",
        encoding="utf-8",
    )

    cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        assert (
            main(
                [
                    "extract-tokens",
                    "--config",
                    str(config_path),
                    "--encoder",
                    "hf:facebook/dinov2-base",
                ]
            )
            == 0
        )
    finally:
        os.chdir(cwd)

    tokens_dir = tmp_path / "out" / "tokens"
    patch_tokens = np.load(tokens_dir / "tokens_patch.npy")
    global_tokens = np.load(tokens_dir / "tokens_global.npy")
    metadata = json.loads((tokens_dir / "token_metadata.json").read_text(encoding="utf-8"))
    assert patch_tokens.shape == (4, 4, 8)
    assert global_tokens.shape == (4, 2, 8)
    assert metadata["encoder_model"] == "hf:facebook/dinov2-base"
    assert metadata["encoder_name"] == "facebook/dinov2-base"
    assert metadata["num_register_tokens"] == 1
