"""Minimal DDP helpers."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_distributed() -> bool:
    return world_size() > 1


def init_distributed(device: torch.device) -> None:
    if not is_distributed() or dist.is_initialized():
        return
    backend = "nccl" if device.type == "cuda" else "gloo"
    dist.init_process_group(backend=backend)


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return rank() == 0

