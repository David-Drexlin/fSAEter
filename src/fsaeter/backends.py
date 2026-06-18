"""Training backend shims."""

from __future__ import annotations

import torch

from fsaeter.train.sparse_ops import batch_topk_sparse, sparse_decode_torch
from fsaeter.train.triton_ops import sparse_decode_triton_autograd


def _unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    return model


class TorchDenseBackend:
    name = "torch_dense"

    def forward_loss(self, model, x, *, global_step: int | None = None, update_state: bool = False):
        if hasattr(model, "loss_dict"):
            try:
                return model.loss_dict(x, global_step=global_step, update_state=update_state)
            except TypeError:
                return model.loss_dict(x)
        output = model(x)
        if not isinstance(output, dict):
            raise TypeError(
                f"Expected dict output from training model, got {type(output).__name__}"
            )
        return output


class TorchSparseBackend:
    name = "torch_sparse"

    def forward_loss(self, model, x, *, global_step: int | None = None, update_state: bool = False):
        base_model = _unwrap_model(model)
        target = base_model.prepare_target(x)
        preactivations = base_model.preactivate_target(target)
        positive = torch.relu(preactivations)
        sparse_features = batch_topk_sparse(positive, base_model.target_k)
        if update_state and global_step is not None:
            base_model.update_last_nonzero_from_ids_(
                sparse_features.feature_ids,
                global_step=global_step,
            )
        recon = sparse_decode_torch(sparse_features, base_model.W_dec, base_model.b_dec)
        recon_mse = torch.mean((recon - target) ** 2)
        aux_loss = base_model.auxk_loss(
            target=target,
            positive=positive,
            recon=recon,
            global_step=global_step,
            sparse_decode_fn=sparse_decode_torch,
        )
        aux_loss = aux_loss + base_model.matryoshka_aux_loss(
            target=target,
            positive=positive,
            sparse_decode_fn=sparse_decode_torch,
        )
        loss = recon_mse + aux_loss
        return {
            "loss": loss,
            "recon_mse": recon_mse,
            "aux_loss": aux_loss,
            "features": sparse_features,
            "preactivations": preactivations,
            "recon": recon,
        }


class TritonSparseBackend:
    name = "triton_sparse"

    def forward_loss(self, model, x, *, global_step: int | None = None, update_state: bool = False):
        base_model = _unwrap_model(model)
        target = base_model.prepare_target(x)
        preactivations = base_model.preactivate_target(target)
        positive = torch.relu(preactivations)
        sparse_features = batch_topk_sparse(positive, base_model.target_k)
        if update_state and global_step is not None:
            base_model.update_last_nonzero_from_ids_(
                sparse_features.feature_ids,
                global_step=global_step,
            )
        recon = sparse_decode_triton_autograd(
            sparse_features,
            base_model.W_dec,
            base_model.b_dec,
        )
        recon_mse = torch.mean((recon - target) ** 2)
        aux_loss = base_model.auxk_loss(
            target=target,
            positive=positive,
            recon=recon,
            global_step=global_step,
            sparse_decode_fn=sparse_decode_torch,
        )
        aux_loss = aux_loss + base_model.matryoshka_aux_loss(
            target=target,
            positive=positive,
            sparse_decode_fn=sparse_decode_torch,
        )
        loss = recon_mse + aux_loss
        return {
            "loss": loss,
            "recon_mse": recon_mse,
            "aux_loss": aux_loss,
            "features": sparse_features,
            "preactivations": preactivations,
            "recon": recon,
        }
