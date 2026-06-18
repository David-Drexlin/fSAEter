"""Preset-driven backbone loading through a generic local encoder factory or HF fallback."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import functional as tvf


BACKBONE_PRESETS: dict[str, str] = {
    "dinov2-b-reg": "dinov2-vit-b[norm]",
    "dinov2-b": "dinov2-vit-b[norm,woreg]",
    "dinov3-b": "dinov3-vit-b",
    "siglip2-b": "siglip2-vit-b",
    "clip-b": "clip-vit-b",
    "uni2-h": "uni2-vit-h",
}


def is_hf_backbone(encoder_cfg: dict[str, Any]) -> bool:
    model = encoder_cfg.get("model") or encoder_cfg.get("preset") or ""
    return str(model).startswith("hf:")


def add_encoder_factory_to_path(
    factory_src: str | Path | None = None,
    *,
    base_root: Path | None = None,
) -> Path:
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
        self.requires_raw_images = False
        if is_hf_backbone(encoder_cfg):
            self._init_hf_backbone(encoder_cfg=encoder_cfg, device=device)
        else:
            self._init_factory_backbone(
                encoder_cfg=encoder_cfg,
                device=device,
                base_root=base_root,
            )

    def _init_factory_backbone(
        self,
        *,
        encoder_cfg: dict,
        device: torch.device,
        base_root: Path | None = None,
    ) -> None:
        add_encoder_factory_to_path(encoder_cfg.get("factory_src"), base_root=base_root)
        from encoders.vision_encoder import create_encoder

        encoder_name = resolve_factory_string(encoder_cfg)
        resolution = int(encoder_cfg.get("resolution", 256))
        self._encoder = create_encoder(encoder_name, device=device, resolution=resolution)
        self.patch_size = int(getattr(self._encoder, "patch_size", 14))
        self.num_register_tokens = int(getattr(self._encoder, "num_register_tokens", 0))
        self.encoder_name = encoder_name
        self.encoder_model = encoder_cfg.get("model") or encoder_cfg.get("preset")
        self.encoder_factory_string = encoder_name
        self.encoder_resolution = resolution

    def _init_hf_backbone(self, *, encoder_cfg: dict, device: torch.device) -> None:
        model_name = str(encoder_cfg.get("model", "")).strip()
        if not model_name.startswith("hf:"):
            raise ValueError(f"Expected hf: model identifier, got {model_name!r}")
        repo_id = model_name[len("hf:") :]
        if not repo_id:
            raise ValueError("Set encoder.model to hf:<repo-id> for the Hugging Face path.")
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "The Hugging Face extraction path requires transformers. "
                "Install fSAEter with the [backbones] extra or install transformers manually."
            ) from exc

        self._processor = AutoImageProcessor.from_pretrained(repo_id)
        self._encoder = AutoModel.from_pretrained(repo_id).to(device)
        self._encoder.eval()
        self.requires_raw_images = True
        self.encoder_name = repo_id
        self.encoder_model = model_name
        self.encoder_factory_string = model_name
        self.encoder_resolution = int(encoder_cfg.get("resolution", 256))
        self.patch_size = int(
            getattr(self._encoder.config, "patch_size", encoder_cfg.get("patch_size", 14))
        )
        self.num_register_tokens = int(getattr(self._encoder.config, "num_register_tokens", 0))

    @staticmethod
    def _ensure_pil_images(images: list[Image.Image] | torch.Tensor) -> list[Image.Image]:
        if torch.is_tensor(images):
            return [tvf.to_pil_image(image.cpu()) for image in images]
        return [image if isinstance(image, Image.Image) else tvf.to_pil_image(image) for image in images]

    def eval(self) -> LocalBackbone:
        self._encoder.eval()
        return self

    def requires_grad_(self, requires_grad: bool) -> LocalBackbone:
        self._encoder.requires_grad_(requires_grad)
        return self

    def forward_tokens(
        self,
        images: list[Image.Image] | torch.Tensor,
        *,
        include_global: bool,
    ) -> BackboneOutput:
        if self.requires_raw_images:
            pil_images = self._ensure_pil_images(images)
            processor_kwargs: dict[str, Any] = {"return_tensors": "pt"}
            if self.encoder_resolution > 0:
                processor_kwargs["size"] = {
                    "height": int(self.encoder_resolution),
                    "width": int(self.encoder_resolution),
                }
                processor_kwargs["crop_size"] = {
                    "height": int(self.encoder_resolution),
                    "width": int(self.encoder_resolution),
                }
            inputs = self._processor(images=pil_images, **processor_kwargs)
            pixel_values = inputs["pixel_values"].to(
                next(self._encoder.parameters()).device,
                non_blocking=True,
            )
            outputs = self._encoder(pixel_values=pixel_values)
            hidden = outputs.last_hidden_state
            patch_grid_h = int(pixel_values.shape[-2]) // int(self.patch_size)
            patch_grid_w = int(pixel_values.shape[-1]) // int(self.patch_size)
            num_patch_tokens = int(patch_grid_h * patch_grid_w)
            global_token_count = int(hidden.shape[1]) - int(num_patch_tokens)
            if global_token_count < 1:
                raise ValueError(
                    f"HF backbone output has no global tokens: shape={tuple(hidden.shape)}"
                )
            inferred_registers = max(0, global_token_count - 1)
            if self.num_register_tokens <= 0:
                self.num_register_tokens = int(inferred_registers)
            patch_tokens = hidden[:, global_token_count:, :]
            global_tokens = hidden[:, :global_token_count, :] if include_global else None
            return BackboneOutput(
                patch_tokens=patch_tokens,
                global_tokens=global_tokens,
            )
        if include_global:
            patch_tokens, global_tokens = self._encoder.forward_with_global(images)
        else:
            patch_tokens = self._encoder(images)
            global_tokens = None
        return BackboneOutput(patch_tokens=patch_tokens, global_tokens=global_tokens)
