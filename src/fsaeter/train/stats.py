"""Token-cache activation statistics for normalized SAE training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fsaeter.config_compat import normalize_train_config
from fsaeter.data.cache import resolve_token_cache_info
from fsaeter.utils.config import resolve_path, save_yaml_config
from fsaeter.utils.repro import resolve_run_seed, seed_everything


def compute_token_stats(config: dict, *, base_root: Path) -> dict:
    config = normalize_train_config(config)
    run_cfg = dict(config.get("run") or {})
    tokens_cfg = dict(config.get("tokens") or {})
    seed_everything(resolve_run_seed(config))

    tokens_dir = resolve_path(tokens_cfg.get("cache_dir", ""), base=base_root)
    token_info = resolve_token_cache_info(tokens_dir)
    tokens = np.load(token_info.tokens_path, mmap_mode="r")
    stats_dir = resolve_path(
        tokens_cfg.get("stats_dir", run_cfg.get("out_dir", "outputs/token_stats")),
        base=base_root,
    )
    stats_dir.mkdir(parents=True, exist_ok=True)

    image_batch_size = max(1, int(tokens_cfg.get("stats_image_batch_size", 64)))
    total_rows = int(token_info.num_images) * int(token_info.tokens_per_image)
    sum_x = np.zeros((int(token_info.d_model),), dtype=np.float64)
    sum_x2 = np.zeros((int(token_info.d_model),), dtype=np.float64)

    for start in range(0, int(token_info.num_images), image_batch_size):
        end = min(start + image_batch_size, int(token_info.num_images))
        flat = np.asarray(tokens[start:end], dtype=np.float32).reshape(-1, int(token_info.d_model))
        sum_x += flat.sum(axis=0, dtype=np.float64)
        sum_x2 += np.square(flat.astype(np.float64, copy=False)).sum(axis=0, dtype=np.float64)

    mean = (sum_x / float(total_rows)).astype(np.float32, copy=False)
    second_moment = sum_x2 / float(total_rows)
    var = np.maximum(0.0, second_moment - np.square(mean.astype(np.float64))).astype(
        np.float32,
        copy=False,
    )
    scale = float(np.sqrt(max(1e-12, float(var.mean()))))

    np.save(stats_dir / "activation_mean.npy", mean)
    np.save(stats_dir / "activation_var.npy", var)
    (stats_dir / "activation_scale.json").write_text(
        json.dumps(
            {
                "scale": scale,
                "num_rows": int(total_rows),
                "d_model": int(token_info.d_model),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (stats_dir / "token_stats.json").write_text(
        json.dumps(
            {
                "tokens_dir": str(tokens_dir),
                "stats_dir": str(stats_dir),
                "num_images": int(token_info.num_images),
                "tokens_per_image": int(token_info.tokens_per_image),
                "d_model": int(token_info.d_model),
                "activation_scale": scale,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    save_yaml_config(config, stats_dir / "config_resolved.yaml")
    return {
        "stats_dir": str(stats_dir),
        "num_rows": int(total_rows),
        "d_model": int(token_info.d_model),
        "activation_scale": scale,
    }


def load_activation_stats(stats_dir: str | Path) -> dict[str, np.ndarray | float]:
    root = Path(stats_dir).expanduser().resolve()
    mean = np.load(root / "activation_mean.npy").astype(np.float32, copy=False)
    scale_payload = json.loads((root / "activation_scale.json").read_text(encoding="utf-8"))
    return {
        "mean": mean,
        "scale": float(scale_payload["scale"]),
    }
