from __future__ import annotations

import pytest
import torch

from fsaeter.train.sparse_ops import SparseActs, sparse_decode_torch
from fsaeter.train.triton_ops import TRITON_AVAILABLE, sparse_decode_triton_autograd


@pytest.mark.skipif(not TRITON_AVAILABLE or not torch.cuda.is_available(), reason="Triton CUDA backend unavailable")
def test_triton_sparse_decode_matches_torch_sparse_decode():
    acts = SparseActs(
        row_ids=torch.tensor([0, 0, 1, 1, 2], device="cuda", dtype=torch.long),
        feature_ids=torch.tensor([1, 1, 2, 3, 0], device="cuda", dtype=torch.long),
        values=torch.tensor(
            [1.0, 0.5, 2.0, 0.75, 1.5],
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        ),
        batch_size=3,
        d_sae=4,
    )
    w_dec = torch.randn(4, 8, device="cuda", requires_grad=True)
    b_dec = torch.randn(8, device="cuda", requires_grad=True)

    torch_acts = SparseActs(
        row_ids=acts.row_ids.clone(),
        feature_ids=acts.feature_ids.clone(),
        values=acts.values.detach().clone().requires_grad_(True),
        batch_size=acts.batch_size,
        d_sae=acts.d_sae,
    )
    triton_acts = SparseActs(
        row_ids=acts.row_ids.clone(),
        feature_ids=acts.feature_ids.clone(),
        values=acts.values.detach().clone().requires_grad_(True),
        batch_size=acts.batch_size,
        d_sae=acts.d_sae,
    )
    torch_w_dec = w_dec.detach().clone().requires_grad_(True)
    triton_w_dec = w_dec.detach().clone().requires_grad_(True)
    torch_b_dec = b_dec.detach().clone().requires_grad_(True)
    triton_b_dec = b_dec.detach().clone().requires_grad_(True)

    torch_out = sparse_decode_torch(torch_acts, torch_w_dec, torch_b_dec)
    triton_out = sparse_decode_triton_autograd(triton_acts, triton_w_dec, triton_b_dec)
    torch.testing.assert_close(triton_out, torch_out, atol=1e-4, rtol=1e-4)

    torch_loss = (torch_out.square().sum() + 0.5 * torch_out.sum())
    triton_loss = (triton_out.square().sum() + 0.5 * triton_out.sum())
    torch_loss.backward()
    triton_loss.backward()

    torch.testing.assert_close(triton_acts.values.grad, torch_acts.values.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(triton_w_dec.grad, torch_w_dec.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(triton_b_dec.grad, torch_b_dec.grad, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not TRITON_AVAILABLE or not torch.cuda.is_available(), reason="Triton CUDA backend unavailable")
def test_triton_sparse_decode_zero_nnz_matches_torch_sparse_decode():
    acts = SparseActs(
        row_ids=torch.empty((0,), device="cuda", dtype=torch.long),
        feature_ids=torch.empty((0,), device="cuda", dtype=torch.long),
        values=torch.empty((0,), device="cuda", dtype=torch.float32, requires_grad=True),
        batch_size=3,
        d_sae=4,
    )
    torch_acts = SparseActs(
        row_ids=acts.row_ids.clone(),
        feature_ids=acts.feature_ids.clone(),
        values=acts.values.detach().clone().requires_grad_(True),
        batch_size=acts.batch_size,
        d_sae=acts.d_sae,
    )
    triton_acts = SparseActs(
        row_ids=acts.row_ids.clone(),
        feature_ids=acts.feature_ids.clone(),
        values=acts.values.detach().clone().requires_grad_(True),
        batch_size=acts.batch_size,
        d_sae=acts.d_sae,
    )
    torch_w_dec = torch.randn(4, 6, device="cuda", requires_grad=True)
    triton_w_dec = torch_w_dec.detach().clone().requires_grad_(True)
    torch_b_dec = torch.randn(6, device="cuda", requires_grad=True)
    triton_b_dec = torch_b_dec.detach().clone().requires_grad_(True)

    torch_out = sparse_decode_torch(torch_acts, torch_w_dec, torch_b_dec)
    triton_out = sparse_decode_triton_autograd(triton_acts, triton_w_dec, triton_b_dec)
    torch.testing.assert_close(triton_out, torch_out, atol=1e-4, rtol=1e-4)

    torch_out.sum().backward()
    triton_out.sum().backward()
    torch.testing.assert_close(triton_w_dec.grad, torch_w_dec.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(triton_b_dec.grad, torch_b_dec.grad, atol=1e-4, rtol=1e-4)
    assert triton_acts.values.grad is None or triton_acts.values.grad.numel() == 0
    assert torch_acts.values.grad is None or torch_acts.values.grad.numel() == 0
