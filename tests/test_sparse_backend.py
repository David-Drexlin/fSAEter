from __future__ import annotations

import torch

from fsaeter.backends import TorchDenseBackend, TorchSparseBackend
from fsaeter.models.local_sae import LocalSparseAutoencoder
from fsaeter.train.sparse_ops import SparseActs, batch_topk_sparse, sparse_decode_torch


def test_batch_topk_sparse_matches_dense_budget():
    positive = torch.arange(1, 41, dtype=torch.float32).reshape(4, 10)
    dense = LocalSparseAutoencoder._batch_topk_nonnegative(positive, target_k=3)
    sparse = batch_topk_sparse(positive, k=3)
    assert sparse.nnz == int((dense > 0).sum().item())
    torch.testing.assert_close(sparse.to_dense(), dense)


def test_sparse_decode_torch_matches_dense_decode():
    acts = SparseActs(
        row_ids=torch.tensor([0, 0, 2], dtype=torch.long),
        feature_ids=torch.tensor([1, 3, 2], dtype=torch.long),
        values=torch.tensor([1.5, 0.25, 2.0], dtype=torch.float32),
        batch_size=3,
        d_sae=4,
    )
    W_dec = torch.randn(4, 6)
    b_dec = torch.randn(6)
    torch.testing.assert_close(
        sparse_decode_torch(acts, W_dec, b_dec),
        acts.to_dense().matmul(W_dec) + b_dec,
    )


def test_torch_sparse_backend_matches_dense_backend_losses():
    model = LocalSparseAutoencoder(
        d_model=8,
        d_sae=16,
        target_k=2,
        variant="batchtopk",
        normalize_inputs=True,
        aux_k=2,
        dead_steps_threshold=1,
        aux_loss_weight=0.1,
    )
    x = torch.randn(5, 8)
    dense = TorchDenseBackend().forward_loss(model, x, global_step=2, update_state=True)
    sparse = TorchSparseBackend().forward_loss(model, x, global_step=2, update_state=True)
    torch.testing.assert_close(sparse["features"].to_dense(), dense["features"])
    for key in ("loss", "recon_mse", "aux_loss", "recon"):
        torch.testing.assert_close(sparse[key], dense[key], atol=1e-5, rtol=1e-5)
