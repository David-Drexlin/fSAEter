"""Basic concept-space diagnostics and inspection."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from fsaeter.data.cache import resolve_token_cache_info
from fsaeter.h.helpers import encode_sae
from fsaeter.models.local_sae import load_local_sae_checkpoint


@dataclass(frozen=True)
class ImageRecordLite:
    row_index: int
    dataset_index: int | None = None
    class_index: int | None = None
    class_name: str | None = None
    path: str | None = None
    relative_path: str | None = None


def iter_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc


def load_image_records(path: str | Path) -> list[ImageRecordLite]:
    records: list[ImageRecordLite] = []
    for row in iter_jsonl(path):
        records.append(
            ImageRecordLite(
                row_index=int(row.get("row_index", len(records))),
                dataset_index=(
                    int(row["dataset_index"])
                    if row.get("dataset_index") is not None
                    else None
                ),
                class_index=(
                    int(row["class_index"])
                    if row.get("class_index") is not None
                    else None
                ),
                class_name=str(row["class_name"]) if row.get("class_name") is not None else None,
                path=str(row["path"]) if row.get("path") is not None else None,
                relative_path=(
                    str(row["relative_path"])
                    if row.get("relative_path") is not None
                    else None
                ),
            )
        )
    if not records:
        raise ValueError(f"No image records found in {path}")
    return records


def labels_from_records(records: Sequence[ImageRecordLite]) -> np.ndarray:
    labels = []
    for record in records:
        if record.class_index is None:
            raise ValueError("Image records do not include class_index; pass labels.npy instead.")
        labels.append(int(record.class_index))
    return np.asarray(labels, dtype=np.int64)


def valid_topk_mask(
    top_indices: np.ndarray,
    top_values: np.ndarray,
    vocab_size: int,
    value_threshold: float = 0.0,
) -> np.ndarray:
    return (
        (top_indices >= 0)
        & (top_indices < int(vocab_size))
        & np.isfinite(top_values)
        & (top_values > float(value_threshold))
    )


def class_concept_count_matrix(
    top_indices: np.ndarray,
    top_values: np.ndarray,
    labels: np.ndarray,
    *,
    vocab_size: int,
    value_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    class_ids = np.asarray(sorted(int(v) for v in np.unique(labels)), dtype=np.int64)
    class_to_row = {int(cls): idx for idx, cls in enumerate(class_ids.tolist())}
    class_counts = np.bincount(labels, minlength=int(class_ids.max()) + 1)[
        class_ids
    ].astype(np.int64)
    counts = np.zeros((class_ids.shape[0], int(vocab_size)), dtype=np.int64)
    mask = valid_topk_mask(top_indices, top_values, vocab_size, value_threshold)

    for row_idx, class_id in enumerate(labels.tolist()):
        row_mask = mask[row_idx]
        if not np.any(row_mask):
            continue
        concept_ids = np.unique(top_indices[row_idx, row_mask].astype(np.int64, copy=False))
        counts[class_to_row[int(class_id)], concept_ids] += 1
    return class_ids, class_counts, counts


def select_sparse_topk_rows(
    rows: np.ndarray,
    *,
    k: int,
    active_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    if rows.ndim != 2:
        raise ValueError(f"Expected rows shaped [N,K], got {rows.shape}")
    k = min(int(k), int(rows.shape[1]))
    if k <= 0:
        raise ValueError("k must be positive")
    order = np.argpartition(-rows, kth=k - 1, axis=1)[:, :k]
    values = np.take_along_axis(rows, order, axis=1)
    sort_order = np.argsort(-values, axis=1)
    indices = np.take_along_axis(order, sort_order, axis=1).astype(np.int32, copy=False)
    values = np.take_along_axis(values, sort_order, axis=1)
    if active_threshold > 0:
        keep = values > float(active_threshold)
        values = np.where(keep, values, 0.0)
        indices = np.where(keep, indices, -1)
    return values.astype(rows.dtype, copy=False), indices


def tuple_uniqueness_rates(
    top_indices: np.ndarray,
    sizes: Sequence[int] = (1, 4, 8),
) -> dict[str, float]:
    if top_indices.ndim != 2:
        raise ValueError(f"Expected H_top_indices with shape [N,K], got {top_indices.shape}")
    num_rows, top_k = top_indices.shape
    rates: dict[str, float] = {}
    for size in sizes:
        width = min(int(size), int(top_k))
        tuples = []
        for row in top_indices[:, :width]:
            valid = tuple(int(v) for v in row.tolist() if int(v) >= 0)
            tuples.append(valid)
        unique = len(set(tuples))
        rates[f"top_{int(size)}"] = float(unique / max(1, num_rows))
    return rates


def feature_class_metrics(
    top_indices: np.ndarray,
    top_values: np.ndarray,
    labels: np.ndarray,
    *,
    vocab_size: int,
    value_threshold: float = 0.0,
) -> dict[str, np.ndarray]:
    class_ids, class_counts, counts = class_concept_count_matrix(
        top_indices,
        top_values,
        labels,
        vocab_size=vocab_size,
        value_threshold=value_threshold,
    )
    support = counts.sum(axis=0).astype(np.int64, copy=False)
    active_classes = (counts > 0).sum(axis=0).astype(np.int64, copy=False)
    top_class = counts.argmax(axis=0).astype(np.int64, copy=False)
    top_class_count = counts.max(axis=0).astype(np.int64, copy=False)

    entropy = np.zeros((vocab_size,), dtype=np.float64)
    normalized_entropy = np.zeros((vocab_size,), dtype=np.float64)
    max_class_share = np.zeros((vocab_size,), dtype=np.float64)
    for concept_id in range(vocab_size):
        concept_support = int(support[concept_id])
        if concept_support <= 0:
            continue
        probs = counts[:, concept_id].astype(np.float64) / float(concept_support)
        nz = probs > 0
        entropy[concept_id] = float(-(probs[nz] * np.log(probs[nz])).sum())
        if int(active_classes[concept_id]) > 1:
            normalized_entropy[concept_id] = float(
                entropy[concept_id] / math.log(int(active_classes[concept_id]))
            )
        max_class_share[concept_id] = float(top_class_count[concept_id] / concept_support)

    return {
        "class_ids": class_ids.astype(np.int64, copy=False),
        "class_counts": class_counts.astype(np.int64, copy=False),
        "counts": counts.astype(np.int64, copy=False),
        "support": support,
        "active_classes": active_classes,
        "top_class": top_class,
        "top_class_count": top_class_count,
        "entropy": entropy.astype(np.float32, copy=False),
        "normalized_entropy": normalized_entropy.astype(np.float32, copy=False),
        "max_class_share": max_class_share.astype(np.float32, copy=False),
    }


def select_broad_concepts(
    top_indices: np.ndarray,
    top_values: np.ndarray,
    labels: np.ndarray,
    *,
    vocab_size: int,
    value_threshold: float = 0.0,
    min_support: int = 64,
    min_class_coverage: int = 8,
    min_per_class: int = 4,
    top_n: int = 50,
) -> list[dict]:
    metrics = feature_class_metrics(
        top_indices,
        top_values,
        labels,
        vocab_size=vocab_size,
        value_threshold=value_threshold,
    )
    counts = metrics["counts"]
    candidates: list[dict] = []
    for concept_id in range(vocab_size):
        support = int(metrics["support"][concept_id])
        if support < int(min_support):
            continue
        concept_class_coverage = int((counts[:, concept_id] >= int(min_per_class)).sum())
        if concept_class_coverage < int(min_class_coverage):
            continue
        normalized_entropy = float(metrics["normalized_entropy"][concept_id])
        max_class_share = float(metrics["max_class_share"][concept_id])
        score = float(normalized_entropy * math.log1p(support) * (1.0 - max_class_share))
        candidates.append(
            {
                "concept_id": int(concept_id),
                "support": support,
                "class_coverage": concept_class_coverage,
                "active_classes": int(metrics["active_classes"][concept_id]),
                "entropy": float(metrics["entropy"][concept_id]),
                "normalized_entropy": normalized_entropy,
                "max_class_share": max_class_share,
                "top_class": int(metrics["top_class"][concept_id]),
                "top_class_count": int(metrics["top_class_count"][concept_id]),
                "score": score,
                "miner": "broad",
            }
        )
    candidates.sort(
        key=lambda row: (row["score"], row["normalized_entropy"], row["support"]),
        reverse=True,
    )
    return candidates[: max(1, int(top_n))]


def select_localized_concepts(
    top_indices: np.ndarray,
    top_values: np.ndarray,
    *,
    vocab_size: int,
    image_frequency_max: np.ndarray,
    max_activation: np.ndarray,
    value_threshold: float = 0.0,
    min_support: int = 16,
    top_n: int = 50,
) -> list[dict]:
    mask = valid_topk_mask(top_indices, top_values, vocab_size, value_threshold)
    support = np.zeros((int(vocab_size),), dtype=np.int64)
    score_sum = np.zeros((int(vocab_size),), dtype=np.float64)

    for row_idx in range(int(top_indices.shape[0])):
        row_mask = mask[row_idx]
        if not np.any(row_mask):
            continue
        row_ids = top_indices[row_idx, row_mask].astype(np.int64, copy=False)
        row_vals = top_values[row_idx, row_mask].astype(np.float64, copy=False)
        row_best: dict[int, float] = {}
        for concept_id, score in zip(row_ids.tolist(), row_vals.tolist(), strict=True):
            row_best[concept_id] = max(float(score), row_best.get(int(concept_id), 0.0))
        for concept_id, score in row_best.items():
            support[int(concept_id)] += 1
            score_sum[int(concept_id)] += float(score)

    candidates: list[dict] = []
    effective_min_support = max(1, min(int(min_support), 8))
    for concept_id in range(int(vocab_size)):
        concept_support = int(support[concept_id])
        if concept_support < effective_min_support:
            continue
        image_freq = float(image_frequency_max[concept_id])
        mean_peak = float(score_sum[concept_id] / max(1, concept_support))
        localness = max(0.0, 1.0 - min(1.0, image_freq))
        score = float(mean_peak * math.log1p(concept_support) * (0.5 + localness))
        candidates.append(
            {
                "concept_id": int(concept_id),
                "support": concept_support,
                "image_frequency_max": image_freq,
                "max_activation": float(max_activation[concept_id]),
                "mean_peak_score": mean_peak,
                "score": score,
                "miner": "localized",
            }
        )
    candidates.sort(key=lambda row: (row["score"], row["support"]), reverse=True)
    return candidates[: max(1, int(top_n))]


def select_rare_tuples(
    top_indices: np.ndarray,
    top_values: np.ndarray,
    *,
    vocab_size: int,
    value_threshold: float = 0.0,
    top_n: int = 25,
) -> list[dict]:
    mask = valid_topk_mask(top_indices, top_values, vocab_size, value_threshold)
    feature_counts = Counter()
    pair_counts = Counter()
    num_rows = int(top_indices.shape[0])

    for row_idx in range(num_rows):
        row_ids = sorted(
            {
                int(v)
                for v in top_indices[row_idx, mask[row_idx]].astype(np.int64, copy=False).tolist()
            }
        )
        for concept_id in row_ids:
            feature_counts[concept_id] += 1
        for left_idx in range(len(row_ids)):
            for right_idx in range(left_idx + 1, len(row_ids)):
                pair_counts[(row_ids[left_idx], row_ids[right_idx])] += 1

    rows = []
    for (left, right), observed in pair_counts.items():
        expected = (
            float(feature_counts[left]) * float(feature_counts[right]) / float(max(1, num_rows))
        )
        if observed < 2 or expected <= 0:
            continue
        rows.append(
            {
                "pair": [int(left), int(right)],
                "observed": int(observed),
                "expected": float(expected),
                "lift": float(observed / expected),
                "miner": "rare_tuple",
            }
        )
    rows.sort(key=lambda row: (row["lift"], row["observed"]), reverse=True)
    return rows[: max(1, int(top_n))]


def summarize_feature_frequency(image_frequency: np.ndarray) -> dict[str, float]:
    values = np.asarray(image_frequency, dtype=np.float64).reshape(-1)
    return {
        "min": float(values.min()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def write_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def top_image_rows_for_concepts(
    scores: np.ndarray,
    concept_ids: Sequence[int],
    *,
    top_n: int = 16,
) -> dict[int, list[tuple[int, float]]]:
    result: dict[int, list[tuple[int, float]]] = {}
    for concept_id in concept_ids:
        cid = int(concept_id)
        if cid < 0 or cid >= scores.shape[1]:
            continue
        column = np.asarray(scores[:, cid], dtype=np.float32)
        top_n_eff = min(max(1, int(top_n)), int(column.shape[0]))
        order = np.argpartition(-column, kth=top_n_eff - 1)[:top_n_eff]
        order = order[np.argsort(-column[order])]
        result[cid] = [(int(row), float(column[row])) for row in order.tolist()]
    return result


def top_image_rows_from_sparse_topk(
    top_indices: np.ndarray,
    top_values: np.ndarray,
    concept_ids: Sequence[int],
    *,
    vocab_size: int,
    value_threshold: float,
    top_n: int = 16,
) -> dict[int, list[tuple[int, float]]]:
    mask = valid_topk_mask(top_indices, top_values, vocab_size, value_threshold)
    result: dict[int, list[tuple[int, float]]] = {}
    for concept_id in concept_ids:
        cid = int(concept_id)
        rows: list[tuple[int, float]] = []
        row_hits = np.where(np.any(mask & (top_indices == cid), axis=1))[0]
        for row_idx in row_hits.tolist():
            score = float(top_values[row_idx][top_indices[row_idx] == cid].max())
            rows.append((int(row_idx), score))
        rows.sort(key=lambda item: item[1], reverse=True)
        result[cid] = rows[: max(1, int(top_n))]
    return result


def save_preview_image(image_path: str, *, output_path: Path, target_size: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    if hasattr(Image, "Resampling"):
        resample = Image.Resampling.BICUBIC  # type: ignore[attr-defined]
    else:
        resample = Image.BICUBIC
    preview = ImageOps.fit(image, (target_size, target_size), method=resample, centering=(0.5, 0.5))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(output_path)
    return preview


def resolve_preview_source_path(record: ImageRecordLite, *, data_root: Path | None) -> Path | None:
    if record.path:
        candidate = Path(record.path).expanduser()
        if candidate.is_file():
            return candidate
    if data_root is not None and record.relative_path:
        candidate = (data_root / record.relative_path).expanduser()
        if candidate.is_file():
            return candidate
    return None


def crop_patch(
    preview: Image.Image,
    *,
    patch_index: int,
    patch_grid: tuple[int, int],
    output_path: Path,
) -> tuple[int, int]:
    grid_h, grid_w = (int(patch_grid[0]), int(patch_grid[1]))
    patch_h = preview.height // grid_h
    patch_w = preview.width // grid_w
    row = int(patch_index) // grid_w
    col = int(patch_index) % grid_w
    top = row * patch_h
    left = col * patch_w
    patch = preview.crop((left, top, left + patch_w, top + patch_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    patch.save(output_path)
    return row, col


def load_h_rows(concept_dir: Path, *, num_rows: int) -> np.ndarray:
    path = concept_dir / "h_image_rows.npy"
    if path.exists():
        h_rows = np.load(path).astype(np.int64, copy=False)
    else:
        h_rows = np.arange(num_rows, dtype=np.int64)
    if h_rows.shape[0] != num_rows:
        raise ValueError(
            f"h_image_rows has {h_rows.shape[0]} rows but concept arrays have {num_rows}"
        )
    return h_rows


def load_candidate_sparse_pair(
    concept_dir: Path,
    *,
    score_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    mode = str(score_mode).lower()
    if mode == "max" and (concept_dir / "H_max_top_indices.npy").exists():
        return (
            np.load(concept_dir / "H_max_top_indices.npy", mmap_mode="r"),
            np.load(concept_dir / "H_max_top_values.npy", mmap_mode="r"),
        )
    if mode == "mean" and (concept_dir / "H_mean_top_indices.npy").exists():
        return (
            np.load(concept_dir / "H_mean_top_indices.npy", mmap_mode="r"),
            np.load(concept_dir / "H_mean_top_values.npy", mmap_mode="r"),
        )
    return (
        np.load(concept_dir / "H_top_indices.npy", mmap_mode="r"),
        np.load(concept_dir / "H_top_values.npy", mmap_mode="r"),
    )


def maybe_load_dense_scores(concept_dir: Path, *, score_mode: str) -> np.ndarray | None:
    mode = str(score_mode).lower()
    preferred = concept_dir / ("H_max.npy" if mode == "max" else "H_mean.npy")
    fallback = concept_dir / ("H_mean.npy" if mode == "max" else "H_max.npy")
    if preferred.exists():
        return np.load(preferred, mmap_mode="r")
    if fallback.exists():
        return np.load(fallback, mmap_mode="r")
    return None


def load_train_summary(checkpoint_path: str | Path) -> dict | None:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if checkpoint.parent.name == "checkpoints":
        summary_path = checkpoint.parent.parent / "train_summary.json"
        if summary_path.exists():
            return json.loads(summary_path.read_text(encoding="utf-8"))
    return None


def load_build_summary(concept_dir: Path) -> dict | None:
    summary_path = concept_dir / "build_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return None


def scan_feature_top_tokens(
    *,
    concept_ids: Sequence[int],
    source_rows: np.ndarray,
    tokens: np.ndarray,
    token_info,
    checkpoint_path: str | Path,
    device: torch.device,
    precision: str,
    top_n: int,
) -> dict[str, np.ndarray]:
    concept_ids_arr = np.asarray(sorted({int(v) for v in concept_ids}), dtype=np.int64)
    concept_ids_list = [int(v) for v in concept_ids_arr.tolist()]
    if concept_ids_arr.size == 0:
        return {
            "concept_ids": concept_ids_arr,
            "image_rows": np.empty((0, int(top_n)), dtype=np.int64),
            "patch_indices": np.empty((0, int(top_n)), dtype=np.int32),
            "scores": np.empty((0, int(top_n)), dtype=np.float32),
        }

    model, _ = load_local_sae_checkpoint(checkpoint_path, device=device)
    top_scores = np.full((concept_ids_arr.shape[0], int(top_n)), -np.inf, dtype=np.float32)
    top_rows = np.full((concept_ids_arr.shape[0], int(top_n)), -1, dtype=np.int64)
    top_patches = np.full((concept_ids_arr.shape[0], int(top_n)), -1, dtype=np.int32)
    image_batch_size = max(1, int(max(1, 2048 // max(1, token_info.tokens_per_image))))

    for start in range(0, int(source_rows.shape[0]), image_batch_size):
        end = min(start + image_batch_size, int(source_rows.shape[0]))
        current_rows = source_rows[start:end]
        token_batch = torch.from_numpy(np.asarray(tokens[current_rows], dtype=np.float32)).reshape(
            -1,
            int(token_info.d_model),
        )
        with torch.no_grad():
            acts = (
                encode_sae(model, token_batch.to(device=device, non_blocking=True))
                .float()
                .cpu()
                .reshape(
                    current_rows.shape[0],
                    int(token_info.tokens_per_image),
                    int(model.d_sae),
                )
            )
        selected = acts[:, :, concept_ids_list].numpy()
        for local_idx, concept_id in enumerate(concept_ids_list):
            concept_scores = selected[:, :, local_idx].reshape(-1)
            image_offsets = np.repeat(np.arange(current_rows.shape[0]), int(token_info.tokens_per_image))
            patch_offsets = np.tile(np.arange(int(token_info.tokens_per_image)), current_rows.shape[0])
            candidate_scores = np.concatenate([top_scores[local_idx], concept_scores.astype(np.float32, copy=False)])
            candidate_rows = np.concatenate(
                [top_rows[local_idx], current_rows[image_offsets].astype(np.int64, copy=False)]
            )
            candidate_patches = np.concatenate(
                [top_patches[local_idx], patch_offsets.astype(np.int32, copy=False)]
            )
            keep = min(int(top_n), candidate_scores.shape[0])
            order = np.argpartition(-candidate_scores, kth=keep - 1)[:keep]
            order = order[np.argsort(-candidate_scores[order])]
            top_scores[local_idx] = candidate_scores[order]
            top_rows[local_idx] = candidate_rows[order]
            top_patches[local_idx] = candidate_patches[order]

    valid = np.isfinite(top_scores)
    top_scores = np.where(valid, top_scores, 0.0).astype(np.float32, copy=False)
    top_rows = np.where(valid, top_rows, -1).astype(np.int64, copy=False)
    top_patches = np.where(valid, top_patches, -1).astype(np.int32, copy=False)
    return {
        "concept_ids": concept_ids_arr,
        "image_rows": top_rows,
        "patch_indices": top_patches,
        "scores": top_scores,
    }


def run_basic_qc(
    *,
    concept_dir: str | Path,
    tokens_dir: str | Path,
    checkpoint_path: str | Path,
    device: torch.device,
    precision: str = "fp32",
    preview_score_mode: str = "max",
    candidate_score_mode: str = "max",
    preview_concepts: int = 12,
    preview_images_per_concept: int = 12,
    min_support: int = 64,
    min_class_coverage: int = 8,
    min_per_class: int = 4,
    top_candidate_count: int = 50,
    miners: Sequence[str] = ("localized", "broad"),
    data_root: str | Path | None = None,
) -> dict:
    concept_dir = Path(concept_dir).expanduser().resolve()
    tokens_dir = Path(tokens_dir).expanduser().resolve()
    resolved_data_root = None if data_root is None else Path(data_root).expanduser().resolve()
    token_info = resolve_token_cache_info(tokens_dir)

    candidate_top_indices, candidate_top_values = load_candidate_sparse_pair(
        concept_dir,
        score_mode=candidate_score_mode,
    )
    num_h_rows = int(candidate_top_indices.shape[0])
    h_rows = load_h_rows(concept_dir, num_rows=num_h_rows)
    stats_npz = np.load(concept_dir / "concept_stats.npz")
    mean_activation = stats_npz["mean_activation"]
    max_activation = stats_npz["max_activation"]
    image_frequency_mean = stats_npz["image_frequency_mean"]
    image_frequency_max = stats_npz["image_frequency_max"]
    token_frequency = stats_npz["token_frequency"]
    active_threshold = float(stats_npz["active_threshold"].reshape(()))
    vocab_size = int(mean_activation.shape[0])

    labels_path = tokens_dir / "labels.npy"
    records = load_image_records(tokens_dir / "image_ids.jsonl") if (tokens_dir / "image_ids.jsonl").exists() else None
    if labels_path.exists():
        labels = np.load(labels_path).astype(np.int64, copy=False)
    elif records is not None:
        labels = labels_from_records(records)
    else:
        raise FileNotFoundError("Neither labels.npy nor image_ids.jsonl is available for inspection.")

    labels = labels[h_rows]
    local_records = None
    if records is not None:
        local_records = [records[int(row)] for row in h_rows.tolist()]

    if isinstance(miners, str):
        normalized_miners = [miners.lower()]
    else:
        normalized_miners = [str(value).lower() for value in miners]
    if not normalized_miners:
        normalized_miners = ["localized", "broad"]

    candidate_payload: dict[str, list[dict]] = {}
    flattened_candidates: list[dict] = []
    seen_concepts: set[int] = set()
    for miner in normalized_miners:
        if miner == "localized":
            rows = select_localized_concepts(
                candidate_top_indices,
                candidate_top_values,
                vocab_size=vocab_size,
                image_frequency_max=image_frequency_max,
                max_activation=max_activation,
                value_threshold=active_threshold,
                min_support=min_support,
                top_n=top_candidate_count,
            )
        elif miner == "broad":
            rows = select_broad_concepts(
                candidate_top_indices,
                candidate_top_values,
                labels,
                vocab_size=vocab_size,
                value_threshold=active_threshold,
                min_support=min_support,
                min_class_coverage=min_class_coverage,
                min_per_class=min_per_class,
                top_n=top_candidate_count,
            )
        elif miner == "rare_tuple":
            rows = select_rare_tuples(
                candidate_top_indices,
                candidate_top_values,
                vocab_size=vocab_size,
                value_threshold=active_threshold,
                top_n=top_candidate_count,
            )
        else:
            raise ValueError(f"Unsupported miner {miner!r}")
        candidate_payload[miner] = rows
        for row in rows:
            if "concept_id" not in row:
                continue
            concept_id = int(row["concept_id"])
            if concept_id in seen_concepts:
                continue
            seen_concepts.add(concept_id)
            flattened_candidates.append(row)

    write_json(
        concept_dir / "candidate_concepts.json",
        {
            "concept_dir": str(concept_dir),
            "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
            "candidate_score_mode": str(candidate_score_mode).lower(),
            "miners": normalized_miners,
            "num_candidates": int(len(flattened_candidates)),
            "candidates": flattened_candidates,
            "by_miner": candidate_payload,
        },
    )

    train_summary = load_train_summary(checkpoint_path)
    build_summary = load_build_summary(concept_dir)
    qc_summary = {
        "tuple_uniqueness": tuple_uniqueness_rates(candidate_top_indices, sizes=(1, 4, 8)),
        "feature_image_frequency_mean": summarize_feature_frequency(image_frequency_mean),
        "feature_image_frequency_max": summarize_feature_frequency(image_frequency_max),
        "num_candidates": int(len(flattened_candidates)),
        "candidate_counts": {key: int(len(value)) for key, value in candidate_payload.items()},
        "active_threshold": float(active_threshold),
        "candidate_score_mode": str(candidate_score_mode).lower(),
        "miners": normalized_miners,
        "token_frequency_mean": float(np.asarray(token_frequency, dtype=np.float64).mean()),
        "train_summary": train_summary,
        "build_summary": build_summary,
        "previews": {
            "images_written": 0,
            "images_skipped_missing_source": 0,
            "patches_written": 0,
            "patches_skipped_missing_source": 0,
        },
    }
    write_json(concept_dir / "qc_summary.json", qc_summary)

    top_preview_concepts = [int(row["concept_id"]) for row in flattened_candidates[: int(preview_concepts)]]
    dense_scores = maybe_load_dense_scores(concept_dir, score_mode=preview_score_mode)
    if dense_scores is not None:
        preview_rows = top_image_rows_for_concepts(
            dense_scores,
            top_preview_concepts,
            top_n=int(preview_images_per_concept),
        )
    else:
        preview_top_indices, preview_top_values = load_candidate_sparse_pair(
            concept_dir,
            score_mode=preview_score_mode,
        )
        preview_rows = top_image_rows_from_sparse_topk(
            preview_top_indices,
            preview_top_values,
            top_preview_concepts,
            vocab_size=vocab_size,
            value_threshold=active_threshold,
            top_n=int(preview_images_per_concept),
        )

    image_rows_payload = []
    patch_rows_payload = []
    per_source_image_concepts: dict[int, list[int]] = defaultdict(list)
    for concept_id, ranked_rows in preview_rows.items():
        for rank, (local_h_row, score) in enumerate(ranked_rows):
            source_image_row = int(h_rows[int(local_h_row)])
            row = {
                "concept_id": int(concept_id),
                "rank": int(rank),
                "h_row": int(local_h_row),
                "image_row": source_image_row,
                "score": float(score),
            }
            if local_records is not None:
                record = local_records[int(local_h_row)]
                row.update(
                    dataset_index=None if record.dataset_index is None else int(record.dataset_index),
                    class_index=None if record.class_index is None else int(record.class_index),
                    class_name=record.class_name,
                    path=record.path,
                    relative_path=record.relative_path,
                )
            image_rows_payload.append(row)
            per_source_image_concepts[source_image_row].append(int(concept_id))

    with (concept_dir / "top_images.jsonl").open("w", encoding="utf-8") as handle:
        for row in image_rows_payload:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    if local_records is not None and per_source_image_concepts:
        model, _ = load_local_sae_checkpoint(checkpoint_path, device=device)
        tokens = np.load(token_info.tokens_path, mmap_mode="r")
        unique_rows = np.asarray(sorted(per_source_image_concepts), dtype=np.int64)
        token_batch = torch.from_numpy(
            np.asarray(tokens[unique_rows], dtype=np.float32)
        ).reshape(-1, int(token_info.d_model))
        with torch.no_grad():
            acts = (
                encode_sae(model, token_batch.to(device=device, non_blocking=True))
                .float()
                .cpu()
                .reshape(
                    unique_rows.shape[0],
                    int(token_info.tokens_per_image),
                    int(model.d_sae),
                )
            )

        preview_png_size = int(token_info.encoder_input_size)
        concept_to_rank = {
            (int(row["concept_id"]), int(row["image_row"])): int(row["rank"])
            for row in image_rows_payload
        }
        record_by_source_row = {
            int(h_rows[idx]): local_records[idx] for idx in range(len(local_records))
        }
        for local_idx, image_row in enumerate(unique_rows.tolist()):
            record = record_by_source_row[int(image_row)]
            preview = None
            source_path = resolve_preview_source_path(record, data_root=resolved_data_root)
            if source_path is not None:
                preview_path = concept_dir / "top_images" / f"image_{int(image_row):06d}.png"
                preview = save_preview_image(
                    str(source_path),
                    output_path=preview_path,
                    target_size=preview_png_size,
                )
                qc_summary["previews"]["images_written"] += 1
            else:
                qc_summary["previews"]["images_skipped_missing_source"] += 1
            for concept_id in per_source_image_concepts[int(image_row)]:
                column = acts[local_idx, :, int(concept_id)]
                patch_index = int(column.argmax().item())
                patch_score = float(column[patch_index].item())
                patch_row = patch_col = -1
                patch_png = None
                image_png = None
                if preview is not None:
                    rank = concept_to_rank[(int(concept_id), int(image_row))]
                    patch_png = (
                        concept_dir
                        / "top_patches"
                        / f"concept_{int(concept_id):05d}"
                        / f"rank_{rank:03d}.png"
                    )
                    patch_row, patch_col = crop_patch(
                        preview,
                        patch_index=patch_index,
                        patch_grid=token_info.patch_grid,
                        output_path=patch_png,
                    )
                    image_png = str(concept_dir / "top_images" / f"image_{int(image_row):06d}.png")
                    qc_summary["previews"]["patches_written"] += 1
                else:
                    qc_summary["previews"]["patches_skipped_missing_source"] += 1
                patch_rows_payload.append(
                    {
                        "concept_id": int(concept_id),
                        "image_row": int(image_row),
                        "rank": int(concept_to_rank[(int(concept_id), int(image_row))]),
                        "patch_index": patch_index,
                        "patch_row": patch_row,
                        "patch_col": patch_col,
                        "score": patch_score,
                        "patch_png": None if patch_png is None else str(patch_png),
                        "image_png": image_png,
                        "path": record.path,
                        "relative_path": record.relative_path,
                        "class_index": None if record.class_index is None else int(record.class_index),
                        "class_name": record.class_name,
                    }
                )

    with (concept_dir / "top_patches.jsonl").open("w", encoding="utf-8") as handle:
        for row in patch_rows_payload:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    feature_scan = scan_feature_top_tokens(
        concept_ids=[row["concept_id"] for row in flattened_candidates if "concept_id" in row],
        source_rows=h_rows,
        tokens=np.load(token_info.tokens_path, mmap_mode="r"),
        token_info=token_info,
        checkpoint_path=checkpoint_path,
        device=device,
        precision=precision,
        top_n=max(1, int(preview_images_per_concept)),
    )
    np.savez_compressed(concept_dir / "feature_top_tokens.npz", **feature_scan)

    feature_patch_rows = []
    if local_records is not None and feature_scan["concept_ids"].size > 0:
        preview_png_size = int(token_info.encoder_input_size)
        record_by_source_row = {
            int(h_rows[idx]): local_records[idx] for idx in range(len(local_records))
        }
        preview_cache: dict[int, tuple[Image.Image | None, str | None]] = {}
        for concept_idx, concept_id in enumerate(feature_scan["concept_ids"].tolist()):
            for rank in range(int(feature_scan["scores"].shape[1])):
                image_row = int(feature_scan["image_rows"][concept_idx, rank])
                patch_index = int(feature_scan["patch_indices"][concept_idx, rank])
                if image_row < 0 or patch_index < 0:
                    continue
                record = record_by_source_row.get(image_row)
                preview = None
                image_png = None
                if record is not None:
                    if image_row not in preview_cache:
                        source_path = resolve_preview_source_path(record, data_root=resolved_data_root)
                        if source_path is not None:
                            image_path = concept_dir / "feature_top_images" / f"image_{image_row:06d}.png"
                            preview_cache[image_row] = (
                                save_preview_image(
                                    str(source_path),
                                    output_path=image_path,
                                    target_size=preview_png_size,
                                ),
                                str(image_path),
                            )
                        else:
                            preview_cache[image_row] = (None, None)
                    preview, image_png = preview_cache[image_row]
                patch_png = None
                patch_row = patch_col = -1
                if preview is not None:
                    patch_png_path = (
                        concept_dir
                        / "feature_top_patches"
                        / f"concept_{int(concept_id):05d}"
                        / f"rank_{rank:03d}.png"
                    )
                    patch_row, patch_col = crop_patch(
                        preview,
                        patch_index=patch_index,
                        patch_grid=token_info.patch_grid,
                        output_path=patch_png_path,
                    )
                    patch_png = str(patch_png_path)
                feature_patch_rows.append(
                    {
                        "concept_id": int(concept_id),
                        "rank": int(rank),
                        "image_row": int(image_row),
                        "patch_index": int(patch_index),
                        "patch_row": int(patch_row),
                        "patch_col": int(patch_col),
                        "score": float(feature_scan["scores"][concept_idx, rank]),
                        "patch_png": patch_png,
                        "image_png": image_png,
                        "path": None if record is None else record.path,
                        "relative_path": None if record is None else record.relative_path,
                        "class_index": (
                            None
                            if record is None or record.class_index is None
                            else int(record.class_index)
                        ),
                        "class_name": None if record is None else record.class_name,
                    }
                )

    with (concept_dir / "feature_top_patches.jsonl").open("w", encoding="utf-8") as handle:
        for row in feature_patch_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    metadata_path = concept_dir / "concept_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = {}
    metadata["qc"] = {
        "summary_path": str(concept_dir / "qc_summary.json"),
        "candidate_path": str(concept_dir / "candidate_concepts.json"),
        "top_images_path": str(concept_dir / "top_images.jsonl"),
        "top_patches_path": str(concept_dir / "top_patches.jsonl"),
        "feature_top_tokens_path": str(concept_dir / "feature_top_tokens.npz"),
        "feature_top_patches_path": str(concept_dir / "feature_top_patches.jsonl"),
        **qc_summary,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    write_json(concept_dir / "qc_summary.json", qc_summary)
    return qc_summary
