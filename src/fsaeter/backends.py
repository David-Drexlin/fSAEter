"""Training backend shims."""

from __future__ import annotations

import torch
import torch.distributed as dist

from fsaeter.train.sparse_ops import batch_topk_sparse, sparse_decode_torch
from fsaeter.train.triton_ops import sparse_decode_triton_autograd


def _unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    return model


class TorchDenseBackend:
    name = "torch_dense"
    reduce_metrics_across_ranks = True
    requires_replicated_batches = False

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

    def synchronize_gradients(self, model) -> None:
        return None

    def set_epoch(self, epoch: int) -> None:
        return None


class TorchSparseBackend:
    name = "torch_sparse"
    reduce_metrics_across_ranks = True
    requires_replicated_batches = False

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
            "target": target,
        }

    def synchronize_gradients(self, model) -> None:
        return None

    def set_epoch(self, epoch: int) -> None:
        return None


class TritonSparseBackend:
    name = "triton_sparse"
    reduce_metrics_across_ranks = True
    requires_replicated_batches = False

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
            "target": target,
        }

    def synchronize_gradients(self, model) -> None:
        return None

    def set_epoch(self, epoch: int) -> None:
        return None


def _distributed_ready() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _feature_shard_bounds(d_sae: int) -> tuple[int, int]:
    if not _distributed_ready():
        return 0, int(d_sae)
    world = int(dist.get_world_size())
    rank = int(dist.get_rank())
    start = (int(d_sae) * rank) // world
    end = (int(d_sae) * (rank + 1)) // world
    return int(start), int(end)


def _all_gather_variable(values: torch.Tensor, *, fill_value: float = 0.0) -> list[torch.Tensor]:
    if not _distributed_ready():
        return [values]
    local_count = torch.tensor([int(values.numel())], device=values.device, dtype=torch.int64)
    gathered_counts = [torch.zeros_like(local_count) for _ in range(int(dist.get_world_size()))]
    dist.all_gather(gathered_counts, local_count)
    counts = [int(item.item()) for item in gathered_counts]
    max_count = max(counts, default=0)
    if max_count <= 0:
        return [values.new_empty((0,)) for _ in counts]
    padded = torch.full(
        (max_count,),
        fill_value=fill_value,
        dtype=values.dtype,
        device=values.device,
    )
    if values.numel() > 0:
        padded[: int(values.numel())] = values
    gathered = [torch.empty_like(padded) for _ in counts]
    dist.all_gather(gathered, padded)
    return [tensor[:count] for tensor, count in zip(gathered, counts, strict=True)]


class ShardedBatchTopKBackend:
    name = "sharded_batchtopk"
    reduce_metrics_across_ranks = False
    requires_replicated_batches = True

    def __init__(self) -> None:
        self._dense_fallback = TorchSparseBackend()

    def set_epoch(self, epoch: int) -> None:
        return None

    def _decode_local(self, acts, base_model):
        if acts.nnz <= 0:
            return torch.zeros(
                (int(acts.batch_size), int(base_model.d_model)),
                device=base_model.W_dec.device,
                dtype=base_model.W_dec.dtype,
            )
        zero_bias = torch.zeros_like(base_model.b_dec)
        if base_model.W_dec.device.type == "cuda":
            try:
                return sparse_decode_triton_autograd(acts, base_model.W_dec, zero_bias)
            except RuntimeError:
                pass
        return sparse_decode_torch(acts, base_model.W_dec, zero_bias)

    def _select_global_sparse(self, positive_local: torch.Tensor, *, feature_offset: int, d_sae: int, target_k: int):
        local_sparse = batch_topk_sparse(positive_local, target_k)
        if local_sparse.nnz > 0:
            local_sparse = type(local_sparse)(
                row_ids=local_sparse.row_ids,
                feature_ids=local_sparse.feature_ids + int(feature_offset),
                values=local_sparse.values,
                batch_size=local_sparse.batch_size,
                d_sae=int(d_sae),
            )
        if not _distributed_ready():
            return local_sparse

        gathered_values = _all_gather_variable(local_sparse.values, fill_value=float("-inf"))
        gathered_rows = _all_gather_variable(
            local_sparse.row_ids.to(dtype=torch.int64),
            fill_value=0.0,
        )
        gathered_features = _all_gather_variable(
            local_sparse.feature_ids.to(dtype=torch.int64),
            fill_value=0.0,
        )
        all_values = torch.cat(gathered_values, dim=0) if gathered_values else local_sparse.values
        if all_values.numel() <= 0:
            return local_sparse
        all_rows = torch.cat(gathered_rows, dim=0).to(dtype=torch.long)
        all_features = torch.cat(gathered_features, dim=0).to(dtype=torch.long)
        keep = min(max(0, int(target_k)) * int(local_sparse.batch_size), int(all_values.numel()))
        if keep <= 0:
            return type(local_sparse)(
                row_ids=all_rows[:0],
                feature_ids=all_features[:0],
                values=all_values[:0],
                batch_size=local_sparse.batch_size,
                d_sae=int(d_sae),
            )
        values, indices = torch.topk(all_values, k=keep, sorted=False)
        rows = all_rows[indices]
        features = all_features[indices]
        positive_mask = values > 0
        values = values[positive_mask]
        rows = rows[positive_mask]
        features = features[positive_mask]
        if values.numel() <= 0:
            return type(local_sparse)(
                row_ids=rows,
                feature_ids=features,
                values=values,
                batch_size=local_sparse.batch_size,
                d_sae=int(d_sae),
            )
        order = torch.argsort(rows * int(d_sae) + features)
        return type(local_sparse)(
            row_ids=rows[order].long(),
            feature_ids=features[order].long(),
            values=values[order],
            batch_size=local_sparse.batch_size,
            d_sae=int(d_sae),
        )

    def forward_loss(self, model, x, *, global_step: int | None = None, update_state: bool = False):
        base_model = _unwrap_model(model)
        if not _distributed_ready():
            return self._dense_fallback.forward_loss(
                model,
                x,
                global_step=global_step,
                update_state=update_state,
            )
        if int(base_model.aux_k) > 0:
            raise NotImplementedError(
                "backend=sharded_batchtopk does not yet support aux_k dead-feature loss."
            )
        if tuple(getattr(base_model, "matryoshka_prefixes", ())):
            raise NotImplementedError(
                "backend=sharded_batchtopk does not yet support Matryoshka auxiliary losses."
            )

        target = base_model.prepare_target(x)
        feature_start, feature_end = _feature_shard_bounds(int(base_model.d_sae))
        preactivations_local = target.matmul(base_model.W_enc[:, feature_start:feature_end])
        preactivations_local = preactivations_local + base_model.b_enc[feature_start:feature_end]
        positive_local = torch.relu(preactivations_local)
        sparse_features = self._select_global_sparse(
            positive_local,
            feature_offset=feature_start,
            d_sae=int(base_model.d_sae),
            target_k=int(base_model.target_k),
        )
        if update_state and global_step is not None:
            base_model.update_last_nonzero_from_ids_(
                sparse_features.feature_ids,
                global_step=global_step,
            )
            dist.all_reduce(base_model.last_nonzero_step, op=dist.ReduceOp.MAX)

        local_mask = (
            (sparse_features.feature_ids >= int(feature_start))
            & (sparse_features.feature_ids < int(feature_end))
        )
        local_sparse = type(sparse_features)(
            row_ids=sparse_features.row_ids[local_mask],
            feature_ids=sparse_features.feature_ids[local_mask],
            values=sparse_features.values[local_mask],
            batch_size=sparse_features.batch_size,
            d_sae=sparse_features.d_sae,
        )
        recon = self._decode_local(local_sparse, base_model)
        dist.all_reduce(recon, op=dist.ReduceOp.SUM)
        recon = recon + base_model.b_dec
        recon_mse = torch.mean((recon - target) ** 2)
        aux_loss = recon_mse.new_zeros(())
        loss = recon_mse
        return {
            "loss": loss,
            "recon_mse": recon_mse,
            "aux_loss": aux_loss,
            "features": sparse_features,
            "preactivations": preactivations_local,
            "recon": recon,
            "target": target,
        }

    def synchronize_gradients(self, model) -> None:
        if not _distributed_ready():
            return None
        base_model = _unwrap_model(model)
        world = float(dist.get_world_size())
        for name, parameter in base_model.named_parameters():
            grad = parameter.grad
            if grad is None:
                grad = torch.zeros_like(parameter)
                parameter.grad = grad
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            if name == "b_dec":
                grad.div_(world)
