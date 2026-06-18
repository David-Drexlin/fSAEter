"""Reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def resolve_run_seed(config: dict) -> int:
    run_cfg = dict(config.get("run") or {})
    return int(run_cfg.get("seed", 0))


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloader_generator(seed: int, *, rank_offset: int = 0) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(rank_offset))
    return generator


def build_worker_init_fn(seed: int, *, rank_offset: int = 0):
    base_seed = int(seed) + int(rank_offset) * 1000

    def _init_worker(worker_id: int) -> None:
        worker_seed = (base_seed + int(worker_id)) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _init_worker
