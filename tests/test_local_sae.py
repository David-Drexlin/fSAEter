from __future__ import annotations

import json
import socket
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from fsaeter.data.cache import (
    TokenCacheWriter,
    build_token_metadata,
    convert_token_cache,
    open_patch_token_reader,
    resolve_token_cache_info,
)
from fsaeter.data.datasets import PatchTokenMemmapDataset, PatchTokenShardBatchIterable
from fsaeter.models.local_sae import (
    LocalSparseAutoencoder,
    RunningFeatureStats,
    load_local_sae_checkpoint,
    save_local_sae_checkpoint,
)
from fsaeter.utils.distributed import cleanup_distributed, init_distributed


class DummyEncoder:
    patch_size = 14
    num_register_tokens = 4
    encoder_name = "resolved-dinov2-vit-b[norm]"
    encoder_model = "dinov2-b-reg"
    encoder_factory_string = "dinov2-vit-b[norm]"


def write_token_cache(tmp_path: Path) -> Path:
    writer = TokenCacheWriter(
        tmp_path,
        num_images=3,
        patch_shape=(4, 8),
        global_shape=(5, 8),
        save_dtype="float16",
    )
    patch = torch.arange(3 * 4 * 8, dtype=torch.float32).reshape(3, 4, 8)
    global_tokens = torch.zeros(3, 5, 8, dtype=torch.float32)
    writer.write(patch[:2], global_tokens[:2])
    writer.write(patch[2:], global_tokens[2:])
    writer.close()
    metadata = build_token_metadata(
        config={
            "encoder": {"name": "dinov2-vit-b[norm]", "resolution": 256},
            "data": {"image_size": 256},
            "tokens": {"save_dtype": "float16"},
        },
        encoder=DummyEncoder(),
        num_images=3,
        patch_shape=(4, 8),
        global_shape=(5, 8),
        output_dir=tmp_path,
        class_counts=Counter({0: 2, 1: 1}),
        class_to_idx={"a": 0, "b": 1},
    )
    (tmp_path / "token_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    np.save(tmp_path / "labels.npy", np.asarray([0, 1, 0], dtype=np.int64))
    return tmp_path


def test_patch_token_memmap_dataset_mapping_is_deterministic(tmp_path: Path):
    tokens_dir = write_token_cache(tmp_path)
    dataset = PatchTokenMemmapDataset(tokens_dir, image_rows=[2, 0], max_rows=6)
    assert len(dataset) == 6
    assert dataset.global_row_to_image_patch(0) == (2, 0)
    assert dataset.global_row_to_image_patch(5) == (0, 1)
    assert dataset.image_patch_to_global_row(0, 1) == 5


def test_convert_token_cache_preserves_patch_tokens(tmp_path: Path):
    legacy_dir = write_token_cache(tmp_path / "legacy")
    converted_dir = tmp_path / "converted"
    payload = convert_token_cache(legacy_dir, converted_dir, shard_images=2)
    info = resolve_token_cache_info(converted_dir)
    reader = open_patch_token_reader(info)
    converted = reader.load_image_slice(0, info.num_images)
    original = np.load(legacy_dir / "tokens_patch.npy", mmap_mode="r")
    assert payload["storage_format"] == "shard_v1"
    assert info.storage_format == "shard_v1"
    assert converted.shape == original.shape
    assert np.allclose(converted.astype(np.float32), np.asarray(original, dtype=np.float32))


def test_patch_token_shard_batch_iterable_reports_loader_diagnostics(tmp_path: Path):
    legacy_dir = write_token_cache(tmp_path / "legacy")
    converted_dir = tmp_path / "converted"
    convert_token_cache(legacy_dir, converted_dir, shard_images=2)
    info = resolve_token_cache_info(converted_dir)
    loader = PatchTokenShardBatchIterable(
        token_info=info,
        image_rows=[0, 1, 2],
        batch_size=3,
        max_rows=6,
        image_block_size=2,
        shuffle_buffer_rows=6,
        seed=7,
    )
    batches = list(loader)
    diagnostics = loader.last_diagnostics()
    assert sum(int(batch[0].shape[0]) for batch in batches) == 6
    assert diagnostics.rows_yielded == 6
    assert diagnostics.unique_images_seen == 3
    assert diagnostics.bytes_read > 0


def test_batchtopk_keeps_exact_average_budget():
    model = LocalSparseAutoencoder(d_model=4, d_sae=10, target_k=3, variant="batchtopk")
    positive_acts = torch.arange(1, 41, dtype=torch.float32).reshape(4, 10)
    sparse = model._batch_topk_nonnegative(positive_acts, target_k=3)
    l0 = (sparse > 0).sum(dim=1)
    assert int(l0.sum().item()) == 12
    assert float(l0.float().mean().item()) == 3.0


def test_matryoshka_prefixes_are_monotonic():
    model = LocalSparseAutoencoder(
        d_model=4,
        d_sae=16,
        target_k=4,
        variant="matryoshka_batchtopk",
        matryoshka_prefixes=[4, 8, 16],
    )
    assert model.matryoshka_prefixes == (4, 8, 16)


def test_invalid_matryoshka_prefixes_are_rejected():
    with pytest.raises(ValueError):
        LocalSparseAutoencoder(
            d_model=4,
            d_sae=16,
            target_k=4,
            variant="matryoshka_batchtopk",
            matryoshka_prefixes=[8, 4, 16],
        )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ddp_stats_worker(rank: int, world_size: int, port: int, out_dir: str) -> None:
    import os

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    init_distributed(torch.device("cpu"))
    stats = RunningFeatureStats(d_sae=4, d_model=4)
    features = (
        torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
        if rank == 0
        else torch.tensor([[0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 4.0]])
    )
    target = torch.tensor(
        [[1.0, 0.0, 0.5, -0.5], [0.0, 2.0, -0.5, 0.5]],
        dtype=torch.float32,
    )
    recon = target.clone()
    stats.update(
        batch_size=2,
        loss=torch.tensor(1.0 + rank, dtype=torch.float32),
        recon_mse=torch.tensor(0.5 + rank, dtype=torch.float32),
        aux_loss=torch.tensor(0.25 * (rank + 1), dtype=torch.float32),
        target=target,
        recon=recon,
        features=features,
    )
    summary, feature_frequency = stats.reduced_summary(device=torch.device("cpu"))
    payload = {
        "loss": summary.loss,
        "recon_mse": summary.recon_mse,
        "aux_loss": summary.aux_loss,
        "mean_l0": summary.mean_l0,
        "max_l0": summary.max_l0,
        "dead_fraction": summary.dead_fraction,
        "feature_frequency": feature_frequency.tolist(),
    }
    Path(out_dir, f"rank{rank}.json").write_text(json.dumps(payload), encoding="utf-8")
    cleanup_distributed()


def test_running_feature_stats_reduce_across_ddp_ranks(tmp_path: Path):
    out_dir = tmp_path / "ddp"
    out_dir.mkdir()
    port = _free_port()
    mp.spawn(
        _ddp_stats_worker,
        args=(2, port, str(out_dir)),
        nprocs=2,
        join=True,
    )
    rank0 = json.loads((out_dir / "rank0.json").read_text(encoding="utf-8"))
    rank1 = json.loads((out_dir / "rank1.json").read_text(encoding="utf-8"))
    assert rank0 == rank1
    assert rank0["loss"] == pytest.approx(1.5)
    assert rank0["recon_mse"] == pytest.approx(1.0)
    assert rank0["aux_loss"] == pytest.approx(0.375)
    assert rank0["mean_l0"] == pytest.approx(1.25)
    assert rank0["max_l0"] == 2
    assert rank0["dead_fraction"] == pytest.approx(0.0)
    assert rank0["feature_frequency"] == pytest.approx([0.25, 0.25, 0.5, 0.25])


def test_running_feature_stats_reports_reconstruction_and_l0_metrics():
    stats = RunningFeatureStats(d_sae=4, d_model=3)
    features = torch.tensor(
        [
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [4.0, 5.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=torch.float32,
    )
    recon = target.clone()
    recon[0, 0] += 1.0
    stats.update(
        batch_size=3,
        loss=torch.tensor(0.5, dtype=torch.float32),
        recon_mse=torch.tensor(0.25, dtype=torch.float32),
        aux_loss=torch.tensor(0.1, dtype=torch.float32),
        target=target,
        recon=recon,
        features=features,
    )
    summary = stats.summary()
    assert summary.loss == pytest.approx(0.5)
    assert summary.recon_mse == pytest.approx(0.25)
    assert summary.aux_loss == pytest.approx(0.1)
    assert summary.mean_l0 == pytest.approx(5.0 / 3.0)
    assert summary.p50_l0 == pytest.approx(2.0)
    assert summary.p90_l0 == pytest.approx(2.0)
    assert summary.p99_l0 == pytest.approx(2.0)
    assert summary.max_l0 == 2
    assert summary.alive_fraction == pytest.approx(0.75)
    assert summary.dead_fraction == pytest.approx(0.25)
    assert summary.dead_feature_count == 1
    assert summary.mse == pytest.approx(1.0 / 9.0)
    zero_baseline = float((target.pow(2).sum() / target.numel()).item())
    assert summary.zero_baseline_mse == pytest.approx(zero_baseline)
    assert summary.normalized_mse == pytest.approx(summary.mse / zero_baseline)
    assert summary.variance_explained < 1.0


def test_load_local_sae_checkpoint_warns_and_falls_back_for_legacy_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = {
        "sae": {
            "variant": "batchtopk",
            "d_model": 4,
            "d_sae": 8,
            "target_k": 2,
        }
    }
    model = LocalSparseAutoencoder(d_model=4, d_sae=8, target_k=2, variant="batchtopk")
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_local_sae_checkpoint(
        checkpoint_path,
        model=model,
        config=config,
        epoch=1,
        step=4,
        best_val_loss=0.1,
        history=[{"epoch": 1}],
    )

    original_load = torch.load
    call_kwargs: list[dict] = []

    def fake_load(*args, **kwargs):
        call_kwargs.append(dict(kwargs))
        if kwargs.get("weights_only"):
            raise RuntimeError("weights_only unsupported in legacy checkpoint")
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", fake_load)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded_model, payload = load_local_sae_checkpoint(checkpoint_path, device="cpu")

    assert isinstance(loaded_model, LocalSparseAutoencoder)
    assert payload["epoch"] == 1
    assert any(kwargs.get("weights_only") for kwargs in call_kwargs)
    assert any(kwargs.get("weights_only") is False for kwargs in call_kwargs)
    assert any("legacy torch.load" in str(item.message) for item in caught)
