"""Training backend shims."""

from __future__ import annotations


class TorchDenseBackend:
    name = "torch_dense"

    def forward_loss(self, model, x):
        if hasattr(model, "loss_dict"):
            return model.loss_dict(x)
        output = model(x)
        if not isinstance(output, dict):
            raise TypeError(f"Expected dict output from training model, got {type(output).__name__}")
        return output


class TritonSparseBackend:
    name = "triton_sparse"

    def forward_loss(self, model, x):
        raise NotImplementedError(
            "The Triton sparse backend is a documented seam only. "
            "Use backend=torch_dense for now."
        )
