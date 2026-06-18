from __future__ import annotations

import pytest
import torch

from fsaeter.train.sparse_ops import SparseActs, sparse_decode_torch
from fsaeter.train.triton_ops import TRITON_AVAILABLE, sparse_decode_triton_autograd


@pytest.mark.skipif(not TRITON_AVAILABLE or not torch.cuda.is_available(), reason="Triton CUDA backend unavailable")
def test_triton_sparse_decode_matches_torch_sparse_decode():
    acts = SparseActs(
        row_ids=torch.tensor([0, 0, 1, 2], device="cuda", dtype=torch.long),
        feature_ids=torch.tensor([1, 3, 2, 0], device="cuda", dtype=torch.long),
        values=torch.tensor([1.0, 0.5, 2.0, 1.5], device="cuda", dtype=torch.float32, requires_grad=True),
        batch_size=3,
        d_sae=4,
    )
    w_dec = torch.randn(4, 8, device="cuda", requires_grad=True)
    b_dec = torch.randn(8, device="cuda", requires_grad=True)
    torch_out = sparse_decode_torch(acts, w_dec, b_dec)
    triton_out = sparse_decode_triton_autograd(acts, w_dec, b_dec)
    torch.testing.assert_close(triton_out, torch_out, atol=1e-4, rtol=1e-4)
