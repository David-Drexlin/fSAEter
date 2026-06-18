from __future__ import annotations

import json
import os
import socket
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

from fsaeter.backends import ShardedBatchTopKBackend, TorchSparseBackend
from fsaeter.data.cache import (
    TokenCacheWriter,
    build_token_metadata,
    convert_token_cache,
    open_patch_token_reader,
    resolve_token_cache_info,
)
from fsaeter.h.build import run_build_h
from fsaeter.inspect.basic_qc import run_basic_qc
from fsaeter.models.local_sae import LocalSparseAutoencoder, save_local_sae_checkpoint
from fsaeter.train.runner import run_training
from fsaeter.train.stats import compute_token_stats
from fsaeter.utils.distributed import cleanup_distributed, init_distributed


class DummyEncoder:
    patch_size = 14
    num_register_tokens = 0
    encoder_name = "dummy-factory"
    encoder_model = "dummy-model"
    encoder_factory_string = "dummy-factory"


def write_legacy_cache(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    writer = TokenCacheWriter(root, num_images=8, patch_shape=(4, 8), save_dtype="float16")
    patch = torch.arange(8 * 4 * 8, dtype=torch.float32).reshape(8, 4, 8) / 100.0
    writer.write(patch[:4])
    writer.write(patch[4:])
    writer.close()
    metadata = build_token_metadata(
        config={
            "encoder": {"model": "dummy-model", "resolution": 256},
            "data": {"image_size": 256},
            "tokens": {"save_dtype": "float16"},
        },
        encoder=DummyEncoder(),
        num_images=8,
        patch_shape=(4, 8),
        global_shape=None,
        output_dir=root,
        class_counts=Counter({0: 4, 1: 4}),
        class_to_idx={"a": 0, "b": 1},
    )
    (root / "token_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    np.save(root / "labels.npy", np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64))
    rows = []
    for idx in range(8):
        rows.append(
            {
                "row_index": idx,
                "dataset_index": idx,
                "path": None,
                "relative_path": f"train/{'a' if idx < 4 else 'b'}/img{idx}.png",
                "class_index": 0 if idx < 4 else 1,
                "class_name": "a" if idx < 4 else "b",
            }
        )
    with (root / "image_ids.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return root


def build_train_config(cache_dir: Path, out_dir: Path, stats_dir: Path) -> dict:
    return {
        "run": {"seed": 7, "out_dir": str(out_dir)},
        "tokens": {"cache_dir": str(cache_dir), "stats_dir": str(stats_dir)},
        "sae": {"variant": "batchtopk", "d_model": 8, "d_sae": 16, "target_k": 2},
        "train": {
            "device": "cpu",
            "precision": "fp32",
            "batch_size": 4,
            "epochs": 1,
            "backend": "torch_sparse",
            "num_workers": 0,
            "max_steps": 3,
            "val_every_steps": 2,
            "checkpoint_every_steps": 2,
            "log_every_steps": 1,
            "loader": {"image_block_size": 2, "shuffle_buffer_rows": 8},
        },
    }


def build_h_config(cache_dir: Path, checkpoint_path: Path, out_dir: Path) -> dict:
    return {
        "run": {"seed": 7, "out_dir": str(out_dir)},
        "tokens": {"cache_dir": str(cache_dir)},
        "sae": {"checkpoint": str(checkpoint_path)},
        "build_h": {
            "device": "cpu",
            "precision": "fp32",
            "image_batch_size": 2,
            "token_batch_size": 8,
            "max_images": 4,
            "inference_mode": "per_row_topk",
            "activation_mode": "sparse_stream",
            "save_sparse_csr": True,
        },
        "inspect": {
            "device": "cpu",
            "preview_concepts": 2,
            "preview_images_per_concept": 2,
            "candidate_score_mode": "max",
            "miners": ["localized", "broad"],
            "min_support": 1,
            "min_class_coverage": 1,
            "min_per_class": 1,
            "scan_mode": "selected_only",
        },
    }


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _sharded_backend_worker(rank: int, world_size: int, port: int, out_dir: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    init_distributed(torch.device("cpu"))
    torch.manual_seed(0)
    model = LocalSparseAutoencoder(d_model=4, d_sae=8, target_k=2, variant="batchtopk")
    with torch.no_grad():
        model.W_enc.copy_(torch.arange(32, dtype=torch.float32).reshape(4, 8) / 50.0)
        model.W_dec.copy_(torch.arange(32, dtype=torch.float32).reshape(8, 4) / 40.0)
        model.b_enc.zero_()
        model.b_dec.zero_()
    x = torch.tensor(
        [
            [0.2, 0.1, 0.4, -0.2],
            [-0.3, 0.5, 0.7, 0.1],
        ],
        dtype=torch.float32,
    )
    backend = ShardedBatchTopKBackend()
    outputs = backend.forward_loss(model, x, global_step=1, update_state=True)
    outputs["loss"].backward()
    backend.synchronize_gradients(model)
    payload = {
        "loss": float(outputs["loss"].item()),
        "recon": outputs["recon"].detach().cpu().tolist(),
        "feature_ids": outputs["features"].feature_ids.detach().cpu().tolist(),
        "row_ids": outputs["features"].row_ids.detach().cpu().tolist(),
        "values": outputs["features"].values.detach().cpu().tolist(),
        "grad_w_enc": model.W_enc.grad.detach().cpu().tolist(),
        "grad_w_dec": model.W_dec.grad.detach().cpu().tolist(),
        "grad_b_dec": model.b_dec.grad.detach().cpu().tolist(),
    }
    Path(out_dir, f"rank{rank}.json").write_text(json.dumps(payload), encoding="utf-8")
    cleanup_distributed()


def validate_sharded_backend(tmp_root: Path) -> None:
    reference_model = LocalSparseAutoencoder(d_model=4, d_sae=8, target_k=2, variant="batchtopk")
    with torch.no_grad():
        reference_model.W_enc.copy_(torch.arange(32, dtype=torch.float32).reshape(4, 8) / 50.0)
        reference_model.W_dec.copy_(torch.arange(32, dtype=torch.float32).reshape(8, 4) / 40.0)
        reference_model.b_enc.zero_()
        reference_model.b_dec.zero_()
    x = torch.tensor(
        [
            [0.2, 0.1, 0.4, -0.2],
            [-0.3, 0.5, 0.7, 0.1],
        ],
        dtype=torch.float32,
    )
    reference = TorchSparseBackend().forward_loss(reference_model, x, global_step=1, update_state=True)
    reference["loss"].backward()
    out_dir = tmp_root / "sharded_backend"
    out_dir.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    mp.spawn(_sharded_backend_worker, args=(2, port, str(out_dir)), nprocs=2, join=True)
    rank0 = json.loads((out_dir / "rank0.json").read_text(encoding="utf-8"))
    rank1 = json.loads((out_dir / "rank1.json").read_text(encoding="utf-8"))
    assert rank0 == rank1
    assert np.allclose(rank0["recon"], reference["recon"].detach().cpu().numpy(), atol=1e-5)
    assert np.isclose(rank0["loss"], float(reference["loss"].item()), atol=1e-5)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="fsaeter_speed_tranche_") as tmpdir:
        root = Path(tmpdir)
        legacy_dir = write_legacy_cache(root / "legacy_tokens")
        converted = convert_token_cache(legacy_dir, root / "shard_tokens", shard_images=3)
        assert converted["storage_format"] == "shard_v1"
        info = resolve_token_cache_info(root / "shard_tokens")
        assert info.storage_format == "shard_v1"
        reader = open_patch_token_reader(info)
        shard_tokens = reader.load_image_slice(0, info.num_images)
        legacy_tokens = np.load(legacy_dir / "tokens_patch.npy", mmap_mode="r")
        assert np.allclose(shard_tokens.astype(np.float32), np.asarray(legacy_tokens, dtype=np.float32))

        stats_payload = compute_token_stats(
            {
                "run": {"out_dir": str(root / "stats")},
                "tokens": {"cache_dir": str(root / "shard_tokens"), "stats_dir": str(root / "stats")},
            },
            base_root=root,
        )
        assert Path(stats_payload["stats_dir"]).exists()

        train_cfg = build_train_config(root / "shard_tokens", root / "train", root / "stats")
        train_payload = run_training(train_cfg, base_root=root)
        checkpoint_path = root / "train" / "checkpoints" / "best.pt"
        assert Path(train_payload["out_dir"]).exists()
        assert checkpoint_path.exists()

        build_cfg = build_h_config(root / "shard_tokens", checkpoint_path, root / "h")
        build_payload = run_build_h(build_cfg, base_root=root)
        assert build_payload["activation_mode"] == "sparse_stream"
        assert build_payload["save_sparse_csr_written"] is True

        qc_payload = run_basic_qc(
            concept_dir=root / "h",
            tokens_dir=root / "shard_tokens",
            checkpoint_path=checkpoint_path,
            device=torch.device("cpu"),
            precision="fp32",
            preview_score_mode="max",
            candidate_score_mode="max",
            preview_concepts=2,
            preview_images_per_concept=2,
            min_support=1,
            min_class_coverage=1,
            min_per_class=1,
            top_candidate_count=4,
            miners=("localized", "broad"),
            data_root=None,
            scan_mode="selected_only",
        )
        assert qc_payload["inference_mode"] == "per_row_topk"
        validate_sharded_backend(root)
    print("speed tranche smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
