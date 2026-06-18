"""Sparse activation helpers for BatchTopK training backends."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SparseActs:
    row_ids: Tensor
    feature_ids: Tensor
    values: Tensor
    batch_size: int
    d_sae: int

    @property
    def nnz(self) -> int:
        return int(self.values.numel())

    def to_dense(self) -> Tensor:
        dense = torch.zeros(
            (int(self.batch_size), int(self.d_sae)),
            device=self.values.device,
            dtype=self.values.dtype,
        )
        if self.nnz > 0:
            dense.index_put_((self.row_ids, self.feature_ids), self.values, accumulate=True)
        return dense


def empty_sparse_acts(
    *,
    batch_size: int,
    d_sae: int,
    device: torch.device,
    dtype: torch.dtype,
) -> SparseActs:
    empty_long = torch.empty((0,), device=device, dtype=torch.long)
    empty_values = torch.empty((0,), device=device, dtype=dtype)
    return SparseActs(
        row_ids=empty_long,
        feature_ids=empty_long.clone(),
        values=empty_values,
        batch_size=int(batch_size),
        d_sae=int(d_sae),
    )


def batch_topk_sparse(positive_acts: Tensor, k: int) -> SparseActs:
    if positive_acts.ndim != 2:
        raise ValueError(f"Expected [B,F] activations, got {positive_acts.shape}")
    batch_size, d_sae = (int(v) for v in positive_acts.shape)
    if batch_size <= 0 or d_sae <= 0:
        return empty_sparse_acts(
            batch_size=batch_size,
            d_sae=d_sae,
            device=positive_acts.device,
            dtype=positive_acts.dtype,
        )
    keep = min(max(0, int(k)) * batch_size, int(positive_acts.numel()))
    if keep <= 0:
        return empty_sparse_acts(
            batch_size=batch_size,
            d_sae=d_sae,
            device=positive_acts.device,
            dtype=positive_acts.dtype,
        )

    flat = positive_acts.reshape(-1)
    values, flat_ids = torch.topk(flat, k=keep, sorted=False)
    nonzero = values > 0
    values = values[nonzero]
    flat_ids = flat_ids[nonzero]
    if values.numel() == 0:
        return empty_sparse_acts(
            batch_size=batch_size,
            d_sae=d_sae,
            device=positive_acts.device,
            dtype=positive_acts.dtype,
        )

    row_ids = flat_ids.div(d_sae, rounding_mode="floor")
    feature_ids = torch.remainder(flat_ids, d_sae)
    order = torch.argsort(row_ids * int(d_sae) + feature_ids)
    return SparseActs(
        row_ids=row_ids[order].long(),
        feature_ids=feature_ids[order].long(),
        values=values[order],
        batch_size=batch_size,
        d_sae=d_sae,
    )


def sparse_decode_torch(acts: SparseActs, W_dec: Tensor, b_dec: Tensor) -> Tensor:
    out = torch.zeros(
        (int(acts.batch_size), int(W_dec.shape[1])),
        device=W_dec.device,
        dtype=W_dec.dtype,
    )
    if acts.nnz > 0:
        values = acts.values.to(device=W_dec.device, dtype=W_dec.dtype)
        row_ids = acts.row_ids.to(device=W_dec.device)
        feature_ids = acts.feature_ids.to(device=W_dec.device)
        contrib = values[:, None] * W_dec[feature_ids]
        out.index_add_(0, row_ids, contrib)
    return out + b_dec


def sparse_feature_counts(acts: SparseActs) -> tuple[Tensor, Tensor]:
    row_counts = torch.bincount(
        acts.row_ids,
        minlength=max(1, int(acts.batch_size)),
    ).to(dtype=torch.float32)
    feature_counts = torch.bincount(
        acts.feature_ids,
        minlength=max(1, int(acts.d_sae)),
    ).to(dtype=torch.float32)
    return row_counts, feature_counts
