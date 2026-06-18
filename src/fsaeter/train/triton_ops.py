"""Triton sparse decoder primitives."""

from __future__ import annotations

import torch
from torch import Tensor

from fsaeter.train.sparse_ops import SparseActs

try:  # pragma: no cover - optional dependency
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:  # pragma: no branch

    @triton.jit
    def _sparse_decode_kernel(
        row_ids_ptr,
        feature_ids_ptr,
        values_ptr,
        w_dec_ptr,
        out_ptr,
        nnz,
        d_model,
        stride_w0,
        stride_w1,
        stride_o0,
        stride_o1,
        BLOCK_NNZ: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_nnz = tl.program_id(0)
        pid_d = tl.program_id(1)
        nnz_offsets = pid_nnz * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
        d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        nnz_mask = nnz_offsets < nnz
        d_mask = d_offsets < d_model

        row_ids = tl.load(row_ids_ptr + nnz_offsets, mask=nnz_mask, other=0).to(tl.int32)
        feature_ids = tl.load(feature_ids_ptr + nnz_offsets, mask=nnz_mask, other=0).to(tl.int32)
        values = tl.load(values_ptr + nnz_offsets, mask=nnz_mask, other=0.0)

        weight_ptrs = w_dec_ptr + feature_ids[:, None] * stride_w0 + d_offsets[None, :] * stride_w1
        weights = tl.load(weight_ptrs, mask=nnz_mask[:, None] & d_mask[None, :], other=0.0)
        contrib = weights * values[:, None]

        out_ptrs = out_ptr + row_ids[:, None] * stride_o0 + d_offsets[None, :] * stride_o1
        tl.atomic_add(out_ptrs, contrib, mask=nnz_mask[:, None] & d_mask[None, :])


def require_triton() -> None:
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "Triton is not installed. Install Triton and run on CUDA to use backend=triton_sparse."
        )


def sparse_decode_triton(acts: SparseActs, W_dec: Tensor, b_dec: Tensor) -> Tensor:
    require_triton()
    if W_dec.device.type != "cuda" or acts.values.device.type != "cuda":
        raise RuntimeError("Triton sparse decode requires CUDA tensors.")

    out = torch.zeros(
        (int(acts.batch_size), int(W_dec.shape[1])),
        device=W_dec.device,
        dtype=W_dec.dtype,
    )
    if acts.nnz == 0:
        return out + b_dec

    row_ids = acts.row_ids.to(device=W_dec.device, dtype=torch.int32)
    feature_ids = acts.feature_ids.to(device=W_dec.device, dtype=torch.int32)
    values = acts.values.to(device=W_dec.device, dtype=W_dec.dtype)
    grid = (
        triton.cdiv(int(acts.nnz), 64),
        triton.cdiv(int(W_dec.shape[1]), 64),
    )
    _sparse_decode_kernel[grid](
        row_ids,
        feature_ids,
        values,
        W_dec,
        out,
        int(acts.nnz),
        int(W_dec.shape[1]),
        W_dec.stride(0),
        W_dec.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_NNZ=64,
        BLOCK_D=64,
    )
    return out + b_dec


class _SparseDecodeTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        row_ids: Tensor,
        feature_ids: Tensor,
        values: Tensor,
        W_dec: Tensor,
        b_dec: Tensor,
        batch_size: int,
        d_sae: int,
    ) -> Tensor:
        acts = SparseActs(
            row_ids=row_ids,
            feature_ids=feature_ids,
            values=values,
            batch_size=int(batch_size),
            d_sae=int(d_sae),
        )
        ctx.save_for_backward(row_ids, feature_ids, values, W_dec)
        ctx.batch_size = int(batch_size)
        ctx.d_sae = int(d_sae)
        return sparse_decode_triton(acts, W_dec, b_dec)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        row_ids, feature_ids, values, W_dec = ctx.saved_tensors
        grad_row_ids = None
        grad_feature_ids = None
        grad_batch_size = None
        grad_d_sae = None

        grad_values = None
        if ctx.needs_input_grad[2]:
            grad_values = (grad_output[row_ids] * W_dec[feature_ids]).sum(dim=1)

        grad_w_dec = None
        if ctx.needs_input_grad[3]:
            grad_w_dec = torch.zeros_like(W_dec)
            grad_w_dec.index_add_(0, feature_ids, values[:, None] * grad_output[row_ids])

        grad_b_dec = None
        if ctx.needs_input_grad[4]:
            grad_b_dec = grad_output.sum(dim=0)

        return (
            grad_row_ids,
            grad_feature_ids,
            grad_values,
            grad_w_dec,
            grad_b_dec,
            grad_batch_size,
            grad_d_sae,
        )


def sparse_decode_triton_autograd(acts: SparseActs, W_dec: Tensor, b_dec: Tensor) -> Tensor:
    require_triton()
    return _SparseDecodeTritonFn.apply(
        acts.row_ids,
        acts.feature_ids,
        acts.values,
        W_dec,
        b_dec,
        int(acts.batch_size),
        int(acts.d_sae),
    )
