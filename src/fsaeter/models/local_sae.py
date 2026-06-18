"""Local sparse autoencoder models and checkpoints."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor


class SaeEncodeOutput(NamedTuple):
    h_x: Tensor
    f_x: Tensor


@dataclass(frozen=True)
class LocalSaeInfo:
    checkpoint_path: str
    variant: str
    d_model: int
    d_sae: int
    target_k: int
    matryoshka_prefixes: tuple[int, ...] = ()
    matryoshka_weights: tuple[float, ...] = ()


@dataclass(frozen=True)
class StepMetrics:
    loss: float
    recon_mse: float
    aux_loss: float
    mean_l0: float
    max_l0: int
    dead_fraction: float


class RunningFeatureStats:
    def __init__(self, d_sae: int):
        self.total_rows = 0
        self.total_loss = 0.0
        self.total_recon_mse = 0.0
        self.total_aux_loss = 0.0
        self.total_l0 = 0.0
        self.max_l0 = 0
        self.feature_counts = torch.zeros(int(d_sae), dtype=torch.float64)

    def update(
        self,
        *,
        batch_size: int,
        loss: Tensor,
        recon_mse: Tensor,
        aux_loss: Tensor,
        features: Tensor,
    ) -> None:
        batch_size = int(batch_size)
        l0 = (features > 0).sum(dim=1)
        self.total_rows += batch_size
        self.total_loss += float(loss.detach().item()) * batch_size
        self.total_recon_mse += float(recon_mse.detach().item()) * batch_size
        self.total_aux_loss += float(aux_loss.detach().item()) * batch_size
        self.total_l0 += float(l0.float().sum().item())
        self.max_l0 = max(self.max_l0, int(l0.max().item()) if l0.numel() else 0)
        self.feature_counts += (features > 0).sum(dim=0).detach().cpu().double()

    def summary(self) -> StepMetrics:
        if self.total_rows <= 0:
            raise ValueError("No rows were accumulated")
        dead_fraction = float((self.feature_counts == 0).double().mean().item())
        return StepMetrics(
            loss=self.total_loss / self.total_rows,
            recon_mse=self.total_recon_mse / self.total_rows,
            aux_loss=self.total_aux_loss / self.total_rows,
            mean_l0=self.total_l0 / self.total_rows,
            max_l0=int(self.max_l0),
            dead_fraction=dead_fraction,
        )

    def feature_frequency(self) -> np.ndarray:
        denom = max(1, self.total_rows)
        return (self.feature_counts.numpy() / float(denom)).astype(np.float32, copy=False)


class LocalSparseAutoencoder(torch.nn.Module):
    """A simple local SAE with nonnegative sparse features."""

    def __init__(
        self,
        *,
        d_model: int,
        d_sae: int,
        target_k: int,
        variant: str = "batchtopk",
        matryoshka_prefixes: Sequence[int] | None = None,
        matryoshka_weights: Sequence[float] | None = None,
        decoder_row_norm: bool = True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_sae = int(d_sae)
        self.target_k = int(target_k)
        self.variant = str(variant).lower()
        if self.variant not in {"batchtopk", "matryoshka_batchtopk"}:
            raise ValueError(f"Unsupported local SAE variant {variant!r}")
        self.decoder_row_norm = bool(decoder_row_norm)
        self.W_enc = torch.nn.Parameter(torch.empty(self.d_model, self.d_sae))
        self.W_dec = torch.nn.Parameter(torch.empty(self.d_sae, self.d_model))
        self.b_enc = torch.nn.Parameter(torch.zeros(self.d_sae))
        self.b_dec = torch.nn.Parameter(torch.zeros(self.d_model))

        self.matryoshka_prefixes = tuple(int(v) for v in (matryoshka_prefixes or ()))
        if matryoshka_weights is not None:
            if len(matryoshka_weights) != len(self.matryoshka_prefixes):
                raise ValueError("matryoshka_weights must match matryoshka_prefixes length")
            self.matryoshka_weights = tuple(float(v) for v in matryoshka_weights)
        elif self.matryoshka_prefixes:
            weights = [1.0 / len(self.matryoshka_prefixes)] * len(self.matryoshka_prefixes)
            self.matryoshka_weights = tuple(weights)
        else:
            self.matryoshka_weights = ()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.d_model)
        torch.nn.init.uniform_(self.W_enc, -bound, bound)
        torch.nn.init.uniform_(self.W_dec, -bound, bound)
        torch.nn.init.zeros_(self.b_enc)
        torch.nn.init.zeros_(self.b_dec)
        if self.decoder_row_norm:
            self.normalize_decoder_rows_()

    def normalize_decoder_rows_(self, eps: float = 1e-8) -> None:
        with torch.no_grad():
            norms = self.W_dec.norm(dim=1, keepdim=True).clamp_min(float(eps))
            self.W_dec.div_(norms)

    @staticmethod
    def _batch_topk_nonnegative(positive_acts: Tensor, target_k: int) -> Tensor:
        if positive_acts.ndim != 2:
            raise ValueError(f"Expected [B,F] activations, got {positive_acts.shape}")
        batch, features = positive_acts.shape
        if batch <= 0 or features <= 0:
            return torch.zeros_like(positive_acts)
        keep = min(int(target_k) * batch, positive_acts.numel())
        if keep <= 0:
            return torch.zeros_like(positive_acts)
        flat = positive_acts.reshape(-1)
        values, indices = torch.topk(flat, k=keep, sorted=False)
        sparse = torch.zeros_like(flat)
        sparse.scatter_(0, indices, values)
        return sparse.view_as(positive_acts)

    def encode(self, x: Tensor) -> SaeEncodeOutput:
        h_x = x.matmul(self.W_enc) + self.b_enc
        f_x = self._batch_topk_nonnegative(torch.relu(h_x), self.target_k)
        return SaeEncodeOutput(h_x=h_x, f_x=f_x)

    def decode(self, features: Tensor) -> Tensor:
        return features.matmul(self.W_dec) + self.b_dec

    def forward(self, x: Tensor) -> SaeEncodeOutput:
        return self.encode(x)

    def loss_dict(self, x: Tensor) -> dict[str, Tensor]:
        encoded = self.encode(x)
        recon = self.decode(encoded.f_x)
        recon_mse = torch.mean((recon - x) ** 2)
        aux_loss = torch.zeros((), device=x.device, dtype=x.dtype)

        if self.variant == "matryoshka_batchtopk" and self.matryoshka_prefixes:
            prefix_losses: list[Tensor] = []
            positive = torch.relu(encoded.h_x)
            for prefix, weight in zip(
                self.matryoshka_prefixes,
                self.matryoshka_weights,
                strict=True,
            ):
                prefix = min(int(prefix), self.d_sae)
                prefix_k = max(1, int(round(self.target_k * (prefix / float(self.d_sae)))))
                prefix_sparse = self._batch_topk_nonnegative(positive[:, :prefix], prefix_k)
                padded = torch.zeros_like(encoded.f_x)
                padded[:, :prefix] = prefix_sparse
                prefix_recon = self.decode(padded)
                prefix_losses.append(torch.mean((prefix_recon - x) ** 2) * float(weight))
            aux_loss = torch.stack(prefix_losses).sum() if prefix_losses else aux_loss
        loss = recon_mse + aux_loss
        return {
            "loss": loss,
            "recon_mse": recon_mse,
            "aux_loss": aux_loss,
            "features": encoded.f_x,
            "preactivations": encoded.h_x,
            "recon": recon,
        }


def build_local_sae(config: dict) -> LocalSparseAutoencoder:
    sae_cfg = dict(config.get("sae") or {})
    return LocalSparseAutoencoder(
        d_model=int(sae_cfg["d_model"]),
        d_sae=int(sae_cfg["d_sae"]),
        target_k=int(sae_cfg["target_k"]),
        variant=str(sae_cfg.get("variant", "batchtopk")).lower(),
        matryoshka_prefixes=sae_cfg.get("matryoshka_prefixes", []),
        matryoshka_weights=sae_cfg.get("matryoshka_weights", []),
        decoder_row_norm=bool(sae_cfg.get("decoder_row_norm", True)),
    )


def local_sae_info(model: LocalSparseAutoencoder, checkpoint_path: str | Path) -> LocalSaeInfo:
    return LocalSaeInfo(
        checkpoint_path=str(Path(checkpoint_path).expanduser().resolve()),
        variant=str(model.variant),
        d_model=int(model.d_model),
        d_sae=int(model.d_sae),
        target_k=int(model.target_k),
        matryoshka_prefixes=tuple(int(v) for v in model.matryoshka_prefixes),
        matryoshka_weights=tuple(float(v) for v in model.matryoshka_weights),
    )


def save_local_sae_checkpoint(
    path: str | Path,
    *,
    model: LocalSparseAutoencoder,
    config: dict,
    epoch: int,
    step: int,
    best_val_loss: float | None,
    history: list[dict],
) -> None:
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": config,
        "epoch": int(epoch),
        "step": int(step),
        "best_val_loss": None if best_val_loss is None else float(best_val_loss),
        "history": history,
        "model_info": asdict(local_sae_info(model, ckpt_path)),
    }
    torch.save(payload, ckpt_path)


def load_local_sae_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[LocalSparseAutoencoder, dict]:
    path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(path, map_location=device)
    config = dict(payload.get("config") or {})
    model = build_local_sae(config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def payload_to_local_sae_info(payload: dict, checkpoint_path: str | Path) -> LocalSaeInfo:
    info = dict(payload.get("model_info") or {})
    if info:
        return LocalSaeInfo(
            checkpoint_path=str(Path(checkpoint_path).expanduser().resolve()),
            variant=str(info["variant"]),
            d_model=int(info["d_model"]),
            d_sae=int(info["d_sae"]),
            target_k=int(info["target_k"]),
            matryoshka_prefixes=tuple(int(v) for v in info.get("matryoshka_prefixes", [])),
            matryoshka_weights=tuple(float(v) for v in info.get("matryoshka_weights", [])),
        )
    model = build_local_sae(dict(payload.get("config") or {}))
    return local_sae_info(model, checkpoint_path)
