"""Preset-driven backbone loading through a generic local encoder factory."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


BACKBONE_PRESETS: dict[str, str] = {
    "dinov2-b-reg": "dinov2-vit-b[norm]",
    "dinov2-b": "dinov2-vit-b[norm,woreg]",
    "dinov3-b": "dinov3-vit-b",
    "siglip2-b": "siglip2-vit-b",
    "clip-b": "clip-vit-b",
    "uni2-h": "uni2-vit-h",
}


def add_encoder_factory_to_path(factory_src: str | Path | None = None, *, base_root: Path | None = None) -> Path:
    if factory_src is None:
        env = os.environ.get("FSAETER_ENCODER_FACTORY_SRC")
        if env:
            factory_src = env
        else:
            raise RuntimeError(
                "Encoder factory source path not provided and FSAETER_ENCODER_FACTORY_SRC is not set."
            )
    path = Path(factory_src).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Encoder factory source path not found: {path}")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def resolve_factory_string(encoder_cfg: dict) -> str:
    explicit = encoder_cfg.get("factory_string")
    if explicit:
        return str(explicit)
    preset = encoder_cfg.get("model") or encoder_cfg.get("preset")
    if preset:
        if str(preset) not in BACKBONE_PRESETS:
            raise ValueError(
                f"Unknown encoder preset {preset!r}. Available presets: {sorted(BACKBONE_PRESETS)}"
            )
        return BACKBONE_PRESETS[str(preset)]
    legacy = encoder_cfg.get("name")
    if legacy:
        return str(legacy)
    raise ValueError("Set encoder.model, encoder.factory_string, or legacy encoder.name.")


@dataclass(frozen=True)
class BackboneOutput:
    patch_tokens: torch.Tensor
    global_tokens: torch.Tensor | None


class LocalBackbone:
    def __init__(
        self,
        *,
        encoder_cfg: dict,
        device: torch.device,
        base_root: Path | None = None,
    ):
        add_encoder_factory_to_path(encoder_cfg.get("factory_src"), base_root=base_root)
        from encoders.vision_encoder import create_encoder

        encoder_name = resolve_factory_string(encoder_cfg)
        resolution = int(encoder_cfg.get("resolution", 256))
        self._encoder = create_encoder(encoder_name, device=device, resolution=resolution)
        self.patch_size = int(getattr(self._encoder, "patch_size", 14))
        self.num_register_tokens = int(getattr(self._encoder, "num_register_tokens", 0))
        self.encoder_name = encoder_name

    def eval(self) -> "LocalBackbone":
        self._encoder.eval()
        return self

    def requires_grad_(self, requires_grad: bool) -> "LocalBackbone":
        self._encoder.requires_grad_(requires_grad)
        return self

    def forward_tokens(self, images: torch.Tensor, *, include_global: bool) -> BackboneOutput:
        if include_global:
            patch_tokens, global_tokens = self._encoder.forward_with_global(images)
        else:
            patch_tokens = self._encoder(images)
            global_tokens = None
        return BackboneOutput(patch_tokens=patch_tokens, global_tokens=global_tokens)
