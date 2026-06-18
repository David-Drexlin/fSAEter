"""ImageFolder-based token extraction helpers."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset
from torchvision import datasets, transforms

from fsaeter.utils.config import resolve_path


def select_subset_indices(
    samples: Sequence[tuple[str, int]],
    max_images: int | None,
    *,
    strategy: str = "stratified",
    seed: int = 0,
) -> list[int]:
    if max_images is None or max_images <= 0 or max_images >= len(samples):
        return list(range(len(samples)))

    strategy = str(strategy).lower()
    if strategy == "first":
        return list(range(max_images))

    rng = random.Random(int(seed))
    if strategy == "random":
        indices = list(range(len(samples)))
        rng.shuffle(indices)
        return sorted(indices[:max_images])

    if strategy != "stratified":
        raise ValueError(f"Unknown subset strategy {strategy!r}")

    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, (_, label) in enumerate(samples):
        by_class[int(label)].append(idx)
    for indices in by_class.values():
        rng.shuffle(indices)

    selected: list[int] = []
    class_ids = sorted(by_class)
    while len(selected) < max_images:
        progressed = False
        for class_id in class_ids:
            bucket = by_class[class_id]
            if not bucket:
                continue
            selected.append(bucket.pop())
            progressed = True
            if len(selected) >= max_images:
                break
        if not progressed:
            break
    return selected


class IndexedSubset(Dataset):
    def __init__(self, base: datasets.ImageFolder, indices: Sequence[int]):
        self.base = base
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        dataset_idx = int(self.indices[idx])
        image, label = self.base[dataset_idx]
        path, _ = self.base.samples[dataset_idx]
        return image, int(label), dataset_idx, path


@dataclass(frozen=True)
class ImageRecord:
    row_index: int
    dataset_index: int
    path: str | None
    relative_path: str
    class_index: int
    class_name: str


def build_imagefolder_dataset(
    config: dict[str, Any],
    *,
    base_root: Path,
) -> tuple[datasets.ImageFolder, list[int]]:
    data_cfg = dict(config.get("data") or {})
    root = resolve_path(data_cfg.get("root", ""), base=base_root)
    split = str(data_cfg.get("split", "train"))
    split_root = root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Dataset split root not found: {split_root}")

    image_size = int(data_cfg.get("image_size", 256))
    transform = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.PILToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(str(split_root), transform=transform)

    subset_cfg = dict(data_cfg.get("subset") or {})
    max_images = subset_cfg.get("max_images")
    if max_images is not None:
        max_images = int(max_images)
    indices = select_subset_indices(
        dataset.samples,
        max_images,
        strategy=str(subset_cfg.get("strategy", "stratified")),
        seed=int(subset_cfg.get("seed", config.get("run", {}).get("seed", 0))),
    )
    return dataset, indices


def make_image_records(
    dataset: datasets.ImageFolder,
    selected_indices: Sequence[int],
    *,
    data_root: Path,
    write_absolute_paths: bool = False,
) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    idx_to_class = {idx: name for name, idx in dataset.class_to_idx.items()}
    for row_idx, dataset_idx in enumerate(selected_indices):
        path, label = dataset.samples[int(dataset_idx)]
        path_obj = Path(path)
        try:
            rel_path = str(path_obj.relative_to(data_root))
        except ValueError:
            rel_path = str(path_obj)
        records.append(
            ImageRecord(
                row_index=row_idx,
                dataset_index=int(dataset_idx),
                path=str(path_obj) if write_absolute_paths else None,
                relative_path=rel_path,
                class_index=int(label),
                class_name=idx_to_class[int(label)],
            )
        )
    return records


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def summarize_selection(
    dataset: datasets.ImageFolder,
    selected_indices: Sequence[int],
) -> dict[str, Any]:
    counts = Counter(int(dataset.samples[idx][1]) for idx in selected_indices)
    return {
        "num_images": len(selected_indices),
        "num_classes": len(counts),
        "min_per_class": min(counts.values()) if counts else 0,
        "max_per_class": max(counts.values()) if counts else 0,
        "class_counts": {str(k): int(v) for k, v in sorted(counts.items())},
    }
