"""Local sparse autoencoder models and checkpoints."""

from __future__ import annotations

import math
import pickle
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.distributed as dist
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
    normalize_inputs: bool = True
    aux_k: int = 0
    dead_steps_threshold: int = 0
    aux_loss_weight: float = 0.0


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
        features,
    ) -> None:
        batch_size = int(batch_size)
        self.total_rows += batch_size
        self.total_loss += float(loss.detach().item()) * batch_size
        self.total_recon_mse += float(recon_mse.detach().item()) * batch_size
        self.total_aux_loss += float(aux_loss.detach().item()) * batch_size

        if torch.is_tensor(features):
            l0 = (features > 0).sum(dim=1)
            self.total_l0 += float(l0.float().sum().item())
            self.max_l0 = max(self.max_l0, int(l0.max().item()) if l0.numel() else 0)
            self.feature_counts += (features > 0).sum(dim=0).detach().cpu().double()
            return

        if hasattr(features, "row_ids") and hasattr(features, "feature_ids"):
            from fsaeter.train.sparse_ops import sparse_feature_counts

            row_counts, feature_counts = sparse_feature_counts(features)
            self.total_l0 += float(row_counts.sum().item())
            self.max_l0 = max(
                self.max_l0,
                int(row_counts.max().item()) if row_counts.numel() else 0,
            )
            self.feature_counts += feature_counts.detach().cpu().double()
            return

        raise TypeError(f"Unsupported features payload {type(features).__name__}")

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

    def reduced_summary(self, *, device: torch.device) -> tuple[StepMetrics, np.ndarray]:
        if self.total_rows <= 0:
            raise ValueError("No rows were accumulated")

        sum_buffer = torch.tensor(
            [
                float(self.total_rows),
                float(self.total_loss),
                float(self.total_recon_mse),
                float(self.total_aux_loss),
                float(self.total_l0),
            ],
            dtype=torch.float64,
            device=device,
        )
        max_buffer = torch.tensor([int(self.max_l0)], dtype=torch.int64, device=device)
        feature_counts = self.feature_counts.to(device=device)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sum_buffer, op=dist.ReduceOp.SUM)
            dist.all_reduce(max_buffer, op=dist.ReduceOp.MAX)
            dist.all_reduce(feature_counts, op=dist.ReduceOp.SUM)

        total_rows = max(1, int(round(float(sum_buffer[0].item()))))
        reduced_counts = feature_counts.cpu()
        dead_fraction = float((reduced_counts == 0).double().mean().item())
        summary = StepMetrics(
            loss=float(sum_buffer[1].item() / total_rows),
            recon_mse=float(sum_buffer[2].item() / total_rows),
            aux_loss=float(sum_buffer[3].item() / total_rows),
            mean_l0=float(sum_buffer[4].item() / total_rows),
            max_l0=int(max_buffer.item()),
            dead_fraction=dead_fraction,
        )
        feature_frequency = (
            reduced_counts.numpy() / float(total_rows)
        ).astype(np.float32, copy=False)
        return summary, feature_frequency


class LocalSparseAutoencoder(torch.nn.Module):
    """A local SAE with optional input normalization and auxiliary dead-feature loss."""

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
        normalize_inputs: bool = True,
        aux_k: int = 0,
        dead_steps_threshold: int = 0,
        aux_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_sae = int(d_sae)
        self.target_k = int(target_k)
        self.variant = str(variant).lower()
        if self.variant not in {"batchtopk", "matryoshka_batchtopk"}:
            raise ValueError(f"Unsupported local SAE variant {variant!r}")
        self.decoder_row_norm = bool(decoder_row_norm)
        self.normalize_inputs = bool(normalize_inputs)
        self.aux_k = max(0, int(aux_k))
        self.dead_steps_threshold = max(0, int(dead_steps_threshold))
        self.aux_loss_weight = max(0.0, float(aux_loss_weight))

        self.W_enc = torch.nn.Parameter(torch.empty(self.d_model, self.d_sae))
        self.W_dec = torch.nn.Parameter(torch.empty(self.d_sae, self.d_model))
        self.b_enc = torch.nn.Parameter(torch.zeros(self.d_sae))
        self.b_dec = torch.nn.Parameter(torch.zeros(self.d_model))

        self.register_buffer("activation_mean", torch.zeros(self.d_model))
        self.register_buffer("activation_scale", torch.ones(()))
        self.register_buffer(
            "last_nonzero_step",
            torch.zeros(self.d_sae, dtype=torch.long),
        )

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
        self._validate_matryoshka()
        self.reset_parameters()

    def _validate_matryoshka(self) -> None:
        if not self.matryoshka_prefixes:
            return
        if any(prefix <= 0 for prefix in self.matryoshka_prefixes):
            raise ValueError("matryoshka_prefixes must be positive.")
        if any(prefix > self.d_sae for prefix in self.matryoshka_prefixes):
            raise ValueError("matryoshka_prefixes must be <= d_sae.")
        if tuple(sorted(set(self.matryoshka_prefixes))) != self.matryoshka_prefixes:
            raise ValueError("matryoshka_prefixes must be strictly increasing.")
        if any(weight < 0 for weight in self.matryoshka_weights):
            raise ValueError("matryoshka_weights must be nonnegative.")
        if self.matryoshka_prefixes[-1] != self.d_sae:
            warnings.warn(
                "Final Matryoshka prefix does not reach d_sae; the full dictionary "
                "will still train through the main reconstruction loss.",
                UserWarning,
                stacklevel=2,
            )

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.d_model)
        torch.nn.init.uniform_(self.W_enc, -bound, bound)
        torch.nn.init.uniform_(self.W_dec, -bound, bound)
        torch.nn.init.zeros_(self.b_enc)
        torch.nn.init.zeros_(self.b_dec)
        self.last_nonzero_step.zero_()
        if self.decoder_row_norm:
            self.normalize_decoder_rows_()

    def load_activation_stats_(
        self,
        *,
        mean: Tensor | np.ndarray | None = None,
        scale: float | Tensor | np.ndarray | None = None,
    ) -> None:
        if mean is not None:
            mean_tensor = torch.as_tensor(mean, dtype=self.activation_mean.dtype)
            if mean_tensor.shape != self.activation_mean.shape:
                raise ValueError(
                    f"Expected activation_mean shape {tuple(self.activation_mean.shape)}, "
                    f"got {tuple(mean_tensor.shape)}"
                )
            self.activation_mean.copy_(mean_tensor.to(device=self.activation_mean.device))
        if scale is not None:
            scale_tensor = torch.as_tensor(scale, dtype=self.activation_mean.dtype).reshape(())
            scale_value = float(scale_tensor.item())
            if not math.isfinite(scale_value) or scale_value <= 0:
                raise ValueError(f"activation_scale must be positive, got {scale_value}")
            self.activation_scale.copy_(scale_tensor.to(device=self.activation_scale.device))

    def initialize_decoder_bias_from_stats_(self) -> None:
        with torch.no_grad():
            if self.normalize_inputs:
                self.b_dec.zero_()
            else:
                self.b_dec.copy_(self.activation_mean.to(device=self.b_dec.device, dtype=self.b_dec.dtype))

    def normalize_decoder_rows_(self, eps: float = 1e-8) -> None:
        with torch.no_grad():
            norms = self.W_dec.norm(dim=1, keepdim=True).clamp_min(float(eps))
            self.W_dec.div_(norms)

    def project_decoder_grad_(self) -> None:
        if self.W_dec.grad is None:
            return
        with torch.no_grad():
            parallel = (self.W_dec.grad * self.W_dec).sum(dim=1, keepdim=True) * self.W_dec
            self.W_dec.grad.sub_(parallel)

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

    def prepare_target(self, x: Tensor) -> Tensor:
        if not self.normalize_inputs:
            return x
        scale = self.activation_scale.to(device=x.device, dtype=x.dtype).clamp_min(1e-8)
        mean = self.activation_mean.to(device=x.device, dtype=x.dtype)
        return (x - mean) / scale

    def denormalize_output(self, x: Tensor) -> Tensor:
        if not self.normalize_inputs:
            return x
        scale = self.activation_scale.to(device=x.device, dtype=x.dtype).clamp_min(1e-8)
        mean = self.activation_mean.to(device=x.device, dtype=x.dtype)
        return x * scale + mean

    def preactivate_target(self, target: Tensor) -> Tensor:
        return target.matmul(self.W_enc) + self.b_enc

    def preactivate(self, x: Tensor) -> Tensor:
        return self.preactivate_target(self.prepare_target(x))

    def encode_target(self, target: Tensor) -> SaeEncodeOutput:
        h_x = self.preactivate_target(target)
        f_x = self._batch_topk_nonnegative(torch.relu(h_x), self.target_k)
        return SaeEncodeOutput(h_x=h_x, f_x=f_x)

    def encode(self, x: Tensor) -> SaeEncodeOutput:
        return self.encode_target(self.prepare_target(x))

    def decode(self, features: Tensor) -> Tensor:
        return features.matmul(self.W_dec) + self.b_dec

    def decode_original(self, features: Tensor) -> Tensor:
        return self.denormalize_output(self.decode(features))

    def forward(self, x: Tensor) -> SaeEncodeOutput:
        return self.encode(x)

    def update_last_nonzero_from_dense_(self, features: Tensor, *, global_step: int) -> None:
        active = (features > 0).any(dim=0)
        if torch.any(active):
            self.last_nonzero_step[active] = int(global_step)

    def update_last_nonzero_from_ids_(self, feature_ids: Tensor, *, global_step: int) -> None:
        if feature_ids.numel() == 0:
            return
        active = torch.unique(feature_ids.detach())
        self.last_nonzero_step[active] = int(global_step)

    def dead_feature_mask(self, *, global_step: int | None) -> Tensor:
        if (
            self.aux_k <= 0
            or self.dead_steps_threshold <= 0
            or self.aux_loss_weight <= 0
            or global_step is None
            or int(global_step) < self.dead_steps_threshold
        ):
            return torch.zeros((self.d_sae,), device=self.last_nonzero_step.device, dtype=torch.bool)
        cutoff = int(global_step) - int(self.dead_steps_threshold)
        return self.last_nonzero_step <= cutoff

    def auxk_loss(
        self,
        *,
        target: Tensor,
        positive: Tensor,
        recon: Tensor,
        global_step: int | None,
        sparse_decode_fn=None,
    ) -> Tensor:
        zero = torch.zeros((), device=target.device, dtype=target.dtype)
        dead_mask = self.dead_feature_mask(global_step=global_step).to(device=target.device)
        if not torch.any(dead_mask):
            return zero

        dead_feature_ids = torch.nonzero(dead_mask, as_tuple=False).flatten()
        dead_positive = positive[:, dead_feature_ids]
        if dead_positive.numel() == 0:
            return zero

        residual = (target - recon).detach()
        if sparse_decode_fn is None:
            dead_features = self._batch_topk_nonnegative(dead_positive, self.aux_k)
            if torch.count_nonzero(dead_features) <= 0:
                return zero
            aux_recon = dead_features.matmul(self.W_dec[dead_feature_ids])
        else:
            from fsaeter.train.sparse_ops import batch_topk_sparse

            dead_acts = batch_topk_sparse(dead_positive, self.aux_k)
            if dead_acts.nnz <= 0:
                return zero
            aux_recon = sparse_decode_fn(
                dead_acts,
                self.W_dec[dead_feature_ids],
                torch.zeros_like(self.b_dec),
            )
        return torch.mean((aux_recon - residual) ** 2) * float(self.aux_loss_weight)

    def matryoshka_aux_loss(
        self,
        *,
        target: Tensor,
        positive: Tensor,
        sparse_decode_fn=None,
    ) -> Tensor:
        zero = torch.zeros((), device=target.device, dtype=target.dtype)
        if self.variant != "matryoshka_batchtopk" or not self.matryoshka_prefixes:
            return zero

        losses: list[Tensor] = []
        for prefix, weight in zip(self.matryoshka_prefixes, self.matryoshka_weights, strict=True):
            prefix_k = max(1, int(round(self.target_k * (int(prefix) / float(self.d_sae)))))
            prefix_positive = positive[:, : int(prefix)]
            if sparse_decode_fn is None:
                prefix_features = self._batch_topk_nonnegative(prefix_positive, prefix_k)
                prefix_recon = prefix_features.matmul(self.W_dec[: int(prefix)]) + self.b_dec
            else:
                from fsaeter.train.sparse_ops import batch_topk_sparse

                prefix_acts = batch_topk_sparse(prefix_positive, prefix_k)
                prefix_recon = sparse_decode_fn(
                    prefix_acts,
                    self.W_dec[: int(prefix)],
                    self.b_dec,
                )
            losses.append(torch.mean((prefix_recon - target) ** 2) * float(weight))
        return torch.stack(losses).sum() if losses else zero

    def loss_dict(
        self,
        x: Tensor,
        *,
        global_step: int | None = None,
        update_state: bool = False,
    ) -> dict[str, Tensor]:
        target = self.prepare_target(x)
        encoded = self.encode_target(target)
        positive = torch.relu(encoded.h_x)
        if update_state and global_step is not None:
            self.update_last_nonzero_from_dense_(encoded.f_x, global_step=global_step)
        recon = self.decode(encoded.f_x)
        recon_mse = torch.mean((recon - target) ** 2)
        aux_loss = self.auxk_loss(
            target=target,
            positive=positive,
            recon=recon,
            global_step=global_step,
        )
        aux_loss = aux_loss + self.matryoshka_aux_loss(
            target=target,
            positive=positive,
        )
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
    train_cfg = dict(config.get("train") or {})
    return LocalSparseAutoencoder(
        d_model=int(sae_cfg["d_model"]),
        d_sae=int(sae_cfg["d_sae"]),
        target_k=int(sae_cfg["target_k"]),
        variant=str(sae_cfg.get("variant", "batchtopk")).lower(),
        matryoshka_prefixes=sae_cfg.get("matryoshka_prefixes", []),
        matryoshka_weights=sae_cfg.get("matryoshka_weights", []),
        decoder_row_norm=bool(sae_cfg.get("decoder_row_norm", True)),
        normalize_inputs=bool(train_cfg.get("normalize_inputs", True)),
        aux_k=int(sae_cfg.get("aux_k", 0)),
        dead_steps_threshold=int(sae_cfg.get("dead_steps_threshold", 0)),
        aux_loss_weight=float(sae_cfg.get("aux_loss_weight", 0.0)),
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
        normalize_inputs=bool(model.normalize_inputs),
        aux_k=int(model.aux_k),
        dead_steps_threshold=int(model.dead_steps_threshold),
        aux_loss_weight=float(model.aux_loss_weight),
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
    optimizer_state: dict | None = None,
    scheduler_state: dict | None = None,
    scaler_state: dict | None = None,
    rng_state: dict | None = None,
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
        "optimizer": optimizer_state,
        "scheduler": scheduler_state,
        "scaler": scaler_state,
        "rng": rng_state,
    }
    torch.save(payload, ckpt_path)


def load_local_sae_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[LocalSparseAutoencoder, dict]:
    path = Path(checkpoint_path).expanduser().resolve()
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    except (pickle.UnpicklingError, RuntimeError, ValueError) as exc:
        warnings.warn(
            (
                "Falling back to legacy torch.load for checkpoint compatibility; "
                f"safe weights-only loading failed for {path}: {exc}"
            ),
            UserWarning,
            stacklevel=2,
        )
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
            normalize_inputs=bool(info.get("normalize_inputs", True)),
            aux_k=int(info.get("aux_k", 0)),
            dead_steps_threshold=int(info.get("dead_steps_threshold", 0)),
            aux_loss_weight=float(info.get("aux_loss_weight", 0.0)),
        )
    model = build_local_sae(dict(payload.get("config") or {}))
    return local_sae_info(model, checkpoint_path)
