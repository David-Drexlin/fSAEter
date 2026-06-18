from __future__ import annotations

import numpy as np
import torch

from fsaeter.backends import TorchDenseBackend, TorchSparseBackend
from fsaeter.h.helpers import encode_sae, pool_sae_image_batch
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


def _make_model(**kwargs) -> LocalSparseAutoencoder:
    params = {
        "d_model": 8,
        "d_sae": 16,
        "target_k": 2,
        "variant": "batchtopk",
        "normalize_inputs": True,
    }
    params.update(kwargs)
    model = LocalSparseAutoencoder(**params)
    return model


def test_per_row_topk_pooling_is_invariant_to_chunking():
    model = _make_model(normalize_inputs=False)
    tokens = np.random.default_rng(0).standard_normal((3, 4, 8)).astype(np.float32)
    device = torch.device("cpu")

    mean_small, max_small, counts_small = pool_sae_image_batch(
        model,
        tokens,
        device=device,
        token_batch_size=4,
        inference_mode="per_row_topk",
    )
    mean_large, max_large, counts_large = pool_sae_image_batch(
        model,
        tokens,
        device=device,
        token_batch_size=12,
        inference_mode="per_row_topk",
    )

    torch.testing.assert_close(mean_small, mean_large)
    torch.testing.assert_close(max_small, max_large)
    torch.testing.assert_close(counts_small, counts_large)


def test_batchtopk_train_style_pooling_matches_direct_chunked_encoding():
    model = _make_model(normalize_inputs=False)
    tokens = np.random.default_rng(1).standard_normal((4, 3, 8)).astype(np.float32)
    device = torch.device("cpu")
    token_batch_size = 6
    image_batch_size = max(1, token_batch_size // tokens.shape[1])

    pooled_mean, pooled_max, pooled_counts = pool_sae_image_batch(
        model,
        tokens,
        device=device,
        token_batch_size=token_batch_size,
        inference_mode="batchtopk_train_style",
    )

    manual_mean = []
    manual_max = []
    manual_counts = torch.zeros((int(model.d_sae),), dtype=torch.float32)
    for start in range(0, tokens.shape[0], image_batch_size):
        end = min(start + image_batch_size, tokens.shape[0])
        flat = torch.from_numpy(tokens[start:end]).reshape(-1, tokens.shape[-1])
        acts = encode_sae(
            model,
            flat,
            inference_mode="batchtopk_train_style",
        ).reshape(end - start, tokens.shape[1], int(model.d_sae))
        manual_mean.append(acts.mean(dim=1))
        manual_max.append(acts.amax(dim=1))
        manual_counts += (acts > 0).sum(dim=(0, 1)).float()

    torch.testing.assert_close(pooled_mean, torch.cat(manual_mean, dim=0))
    torch.testing.assert_close(pooled_max, torch.cat(manual_max, dim=0))
    torch.testing.assert_close(pooled_counts, manual_counts)


@torch.no_grad()
def _copy_model(model: LocalSparseAutoencoder) -> LocalSparseAutoencoder:
    clone = _make_model(
        normalize_inputs=model.normalize_inputs,
        aux_k=model.aux_k,
        dead_steps_threshold=model.dead_steps_threshold,
        aux_loss_weight=model.aux_loss_weight,
    )
    clone.load_state_dict(model.state_dict())
    return clone


def test_torch_sparse_backend_matches_dense_backend_losses_without_auxk():
    base_model = _make_model(normalize_inputs=True)
    dense_model = _copy_model(base_model)
    sparse_model = _copy_model(base_model)
    x = torch.randn(5, 8)
    dense = TorchDenseBackend().forward_loss(dense_model, x, global_step=2, update_state=False)
    sparse = TorchSparseBackend().forward_loss(sparse_model, x, global_step=2, update_state=False)
    torch.testing.assert_close(sparse["features"].to_dense(), dense["features"])
    for key in ("loss", "recon_mse", "aux_loss", "recon"):
        torch.testing.assert_close(sparse[key], dense[key], atol=1e-5, rtol=1e-5)


def test_torch_sparse_backend_matches_dense_backend_losses_with_auxk():
    base_model = _make_model(
        normalize_inputs=True,
        aux_k=2,
        dead_steps_threshold=1,
        aux_loss_weight=0.1,
    )
    base_model.last_nonzero_step.zero_()
    dense_model = _copy_model(base_model)
    sparse_model = _copy_model(base_model)
    x = torch.randn(5, 8)
    dense = TorchDenseBackend().forward_loss(dense_model, x, global_step=2, update_state=False)
    sparse = TorchSparseBackend().forward_loss(sparse_model, x, global_step=2, update_state=False)
    assert float(dense["aux_loss"].item()) > 0.0
    assert float(sparse["aux_loss"].item()) > 0.0
    torch.testing.assert_close(sparse["features"].to_dense(), dense["features"])
    for key in ("loss", "recon_mse", "aux_loss", "recon"):
        torch.testing.assert_close(sparse[key], dense[key], atol=1e-5, rtol=1e-5)


def test_auxk_loss_is_zero_before_dead_step_threshold():
    base_model = _make_model(
        normalize_inputs=True,
        aux_k=2,
        dead_steps_threshold=4,
        aux_loss_weight=0.1,
    )
    dense_model = _copy_model(base_model)
    sparse_model = _copy_model(base_model)
    x = torch.randn(5, 8)

    dense = TorchDenseBackend().forward_loss(dense_model, x, global_step=1, update_state=False)
    sparse = TorchSparseBackend().forward_loss(sparse_model, x, global_step=1, update_state=False)
    assert float(dense["aux_loss"].item()) == 0.0
    assert float(sparse["aux_loss"].item()) == 0.0


def test_dense_and_sparse_backends_update_last_nonzero_step_consistently():
    base_model = _make_model(
        normalize_inputs=False,
        aux_k=1,
        dead_steps_threshold=2,
        aux_loss_weight=0.1,
    )
    with torch.no_grad():
        base_model.W_enc.zero_()
        base_model.W_enc[0, 0] = 1.0
        base_model.W_enc[1, 2] = 1.0
        base_model.b_enc.zero_()
        base_model.W_dec.zero_()
        base_model.b_dec.zero_()
        base_model.last_nonzero_step.zero_()
    dense_model = _copy_model(base_model)
    sparse_model = _copy_model(base_model)
    x = torch.tensor([[1.0, 0.0] + [0.0] * 6], dtype=torch.float32)

    TorchDenseBackend().forward_loss(dense_model, x, global_step=5, update_state=True)
    TorchSparseBackend().forward_loss(sparse_model, x, global_step=5, update_state=True)
    torch.testing.assert_close(dense_model.last_nonzero_step, sparse_model.last_nonzero_step)
    assert dense_model.last_nonzero_step[0].item() == 5
    assert dense_model.last_nonzero_step[2].item() == 0
