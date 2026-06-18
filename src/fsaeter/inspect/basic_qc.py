"""Basic concept-space diagnostics and inspection."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import json
import math
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
            }
        )
    candidates.sort(
        key=lambda row: (
            row["score"],
            row["normalized_entropy"],
            row["support"],
        ),
        reverse=True,
    )
    return candidates[: max(1, int(top_n))]


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


def resolve_preview_source_path(
    record: ImageRecordLite,
    *,
    data_root: Path | None,
) -> Path | None:
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


def run_basic_qc(
    *,
    concept_dir: str | Path,
    tokens_dir: str | Path,
    checkpoint_path: str | Path,
    device: torch.device,
    precision: str = "fp32",
    preview_score_mode: str = "max",
    preview_concepts: int = 12,
    preview_images_per_concept: int = 12,
    min_support: int = 64,
    min_class_coverage: int = 8,
    min_per_class: int = 4,
    top_candidate_count: int = 50,
    data_root: str | Path | None = None,
) -> dict:
    concept_dir = Path(concept_dir).expanduser().resolve()
    tokens_dir = Path(tokens_dir).expanduser().resolve()
    resolved_data_root = None if data_root is None else Path(data_root).expanduser().resolve()
    token_info = resolve_token_cache_info(tokens_dir)

    top_indices_arr = np.load(concept_dir / "H_top_indices.npy", mmap_mode="r")
    top_values_arr = np.load(concept_dir / "H_top_values.npy", mmap_mode="r")
    vocab_size = int(np.load(concept_dir / "H_mean.npy", mmap_mode="r").shape[1])
    stats_npz = np.load(concept_dir / "concept_stats.npz")
    image_frequency = stats_npz["image_frequency"]
    active_threshold = float(stats_npz["active_threshold"].reshape(()))

    labels_path = tokens_dir / "labels.npy"
    if labels_path.exists():
        labels = np.load(labels_path).astype(np.int64, copy=False)
        if (tokens_dir / "image_ids.jsonl").exists():
            records = load_image_records(tokens_dir / "image_ids.jsonl")
        else:
            records = None
    else:
        records = load_image_records(tokens_dir / "image_ids.jsonl")
        labels = labels_from_records(records)

    candidates = select_broad_concepts(
        top_indices_arr,
        top_values_arr,
        labels,
        vocab_size=vocab_size,
        value_threshold=active_threshold,
        min_support=min_support,
        min_class_coverage=min_class_coverage,
        min_per_class=min_per_class,
        top_n=top_candidate_count,
    )
    write_json(
        concept_dir / "candidate_concepts.json",
        {
            "concept_dir": str(concept_dir),
            "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
            "num_candidates": int(len(candidates)),
            "candidates": candidates,
        },
    )

    qc_summary = {
        "tuple_uniqueness": tuple_uniqueness_rates(top_indices_arr, sizes=(1, 4, 8)),
        "feature_image_frequency": summarize_feature_frequency(image_frequency),
        "num_candidates": int(len(candidates)),
        "active_threshold": float(active_threshold),
        "previews": {
            "images_written": 0,
            "images_skipped_missing_source": 0,
            "patches_written": 0,
            "patches_skipped_missing_source": 0,
        },
    }
    write_json(concept_dir / "qc_summary.json", qc_summary)

    top_preview_concepts = [int(row["concept_id"]) for row in candidates[: int(preview_concepts)]]
    score_matrix_name = (
        "H_max.npy"
        if preview_score_mode.lower() == "max" and (concept_dir / "H_max.npy").exists()
        else "H_mean.npy"
    )
    score_matrix = np.load(concept_dir / score_matrix_name, mmap_mode="r")
    preview_rows = top_image_rows_for_concepts(
        score_matrix,
        top_preview_concepts,
        top_n=int(preview_images_per_concept),
    )

    image_rows_payload = []
    patch_rows_payload = []
    per_image_concepts: dict[int, list[int]] = defaultdict(list)
    for concept_id, ranked_rows in preview_rows.items():
        for rank, (image_row, score) in enumerate(ranked_rows):
            row = {
                "concept_id": int(concept_id),
                "rank": int(rank),
                "image_row": int(image_row),
                "score": float(score),
            }
            if records is not None:
                record = records[image_row]
                row.update(
                    dataset_index=None if record.dataset_index is None else int(record.dataset_index),
                    class_index=None if record.class_index is None else int(record.class_index),
                    class_name=record.class_name,
                    path=record.path,
                    relative_path=record.relative_path,
                )
            image_rows_payload.append(row)
            per_image_concepts[int(image_row)].append(int(concept_id))

    with (concept_dir / "top_images.jsonl").open("w", encoding="utf-8") as handle:
        for row in image_rows_payload:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    if records is not None and per_image_concepts:
        model, _ = load_local_sae_checkpoint(checkpoint_path, device=device)
        tokens = np.load(token_info.tokens_path, mmap_mode="r")
        unique_rows = np.asarray(sorted(per_image_concepts), dtype=np.int64)
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
        for local_idx, image_row in enumerate(unique_rows.tolist()):
            record = records[int(image_row)]
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
            for concept_id in per_image_concepts[int(image_row)]:
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
                    image_png = str(
                        concept_dir / "top_images" / f"image_{int(image_row):06d}.png"
                    )
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
                        "class_index": (
                            None
                            if record.class_index is None
                            else int(record.class_index)
                        ),
                        "class_name": record.class_name,
                    }
                )

    with (concept_dir / "top_patches.jsonl").open("w", encoding="utf-8") as handle:
        for row in patch_rows_payload:
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
        **qc_summary,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    write_json(concept_dir / "qc_summary.json", qc_summary)
    return qc_summary
