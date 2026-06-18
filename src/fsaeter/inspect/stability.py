"""Read-only stability comparisons across trained SAE runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from fsaeter.models.local_sae import load_local_sae_checkpoint
from fsaeter.utils.config import resolve_path


def _resolve_checkpoint(run_value: str | Path) -> tuple[Path, Path | None]:
    path = Path(run_value).expanduser().resolve()
    if path.is_file():
        train_dir = path.parent.parent if path.parent.name == "checkpoints" else None
        return path, train_dir
    if path.is_dir():
        candidate = path / "checkpoints" / "best.pt"
        if candidate.exists():
            return candidate, path
        raise FileNotFoundError(
            f"Could not infer checkpoint from run directory {path}; expected {candidate}"
        )
    raise FileNotFoundError(f"Run path not found: {path}")


def _resolve_concept_dir(explicit: str | Path | None, *, train_dir: Path | None) -> Path:
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Concept directory not found: {path}")
        return path
    if train_dir is not None:
        candidates = [train_dir / "h", train_dir.parent / "h"]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate.resolve()
    raise FileNotFoundError(
        "Could not infer concept directory; pass --concept-dir-a/--concept-dir-b explicitly."
    )


def _load_feature_top_tokens(concept_dir: Path) -> dict[str, np.ndarray]:
    payload = np.load(concept_dir / "feature_top_tokens.npz")
    return {key: payload[key] for key in payload.files}


def _load_top_image_sets(concept_dir: Path) -> dict[int, set[int]]:
    path = concept_dir / "top_images.jsonl"
    if not path.exists():
        return {}
    grouped: dict[int, set[int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            concept_id = int(row["concept_id"])
            image_row = int(row["image_row"])
            grouped.setdefault(concept_id, set()).add(image_row)
    return grouped


def _load_candidate_ids(concept_dir: Path) -> set[int]:
    path = concept_dir / "candidate_concepts.json"
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(row["concept_id"])
        for row in payload.get("candidates", [])
        if row.get("concept_id") is not None
    }


def _load_image_frequency(concept_dir: Path) -> np.ndarray:
    stats = np.load(concept_dir / "concept_stats.npz")
    return np.asarray(stats["image_frequency_max"], dtype=np.float32)


def _jaccard(left: set, right: set) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 1.0
    return float(len(left & right) / len(union))


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return float("nan")
    if left.size != right.size:
        raise ValueError("Pearson inputs must have matching shape.")
    left_centered = left.astype(np.float64) - float(np.mean(left))
    right_centered = right.astype(np.float64) - float(np.mean(right))
    denom = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left_centered, right_centered) / denom)


def _match_decoders(
    decoder_a: torch.Tensor,
    decoder_b: torch.Tensor,
    *,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    a = torch.nn.functional.normalize(decoder_a.float(), dim=1)
    b = torch.nn.functional.normalize(decoder_b.float(), dim=1)
    matched = np.empty((int(a.shape[0]),), dtype=np.int64)
    cosine = np.empty((int(a.shape[0]),), dtype=np.float32)
    b_t = b.transpose(0, 1).contiguous()
    for start in range(0, int(a.shape[0]), int(chunk_size)):
        end = min(start + int(chunk_size), int(a.shape[0]))
        sims = a[start:end].matmul(b_t)
        values, indices = sims.max(dim=1)
        matched[start:end] = indices.cpu().numpy().astype(np.int64, copy=False)
        cosine[start:end] = values.cpu().numpy().astype(np.float32, copy=False)
    return matched, cosine


def _token_sets(feature_scan: dict[str, np.ndarray]) -> dict[int, set[tuple[int, int]]]:
    concept_ids = np.asarray(feature_scan["concept_ids"], dtype=np.int64)
    image_rows = np.asarray(feature_scan["image_rows"], dtype=np.int64)
    patch_indices = np.asarray(feature_scan["patch_indices"], dtype=np.int64)
    grouped: dict[int, set[tuple[int, int]]] = {}
    for idx, concept_id in enumerate(concept_ids.tolist()):
        rows = image_rows[idx]
        patches = patch_indices[idx]
        grouped[int(concept_id)] = {
            (int(row), int(patch))
            for row, patch in zip(rows.tolist(), patches.tolist(), strict=True)
            if int(row) >= 0 and int(patch) >= 0
        }
    return grouped


def _aggregate_overlap_for_matches(
    matched_b: np.ndarray,
    *,
    sets_a: dict[int, set],
    sets_b: dict[int, set],
) -> tuple[float, float]:
    overlaps: list[float] = []
    for feature_a, feature_b in enumerate(matched_b.tolist()):
        left = sets_a.get(int(feature_a), set())
        right = sets_b.get(int(feature_b), set())
        if not left and not right:
            continue
        overlaps.append(_jaccard(left, right))
    if not overlaps:
        return float("nan"), float("nan")
    arr = np.asarray(overlaps, dtype=np.float64)
    return float(arr.mean()), float(np.median(arr))


def preview_compare_command(config: dict, *, base_root: Path) -> dict:
    compare_cfg = dict(config.get("compare") or {})
    checkpoint_a, train_dir_a = _resolve_checkpoint(resolve_path(compare_cfg["run_a"], base=base_root))
    checkpoint_b, train_dir_b = _resolve_checkpoint(resolve_path(compare_cfg["run_b"], base=base_root))
    concept_dir_a = _resolve_concept_dir(compare_cfg.get("concept_dir_a"), train_dir=train_dir_a)
    concept_dir_b = _resolve_concept_dir(compare_cfg.get("concept_dir_b"), train_dir=train_dir_b)
    out_dir = resolve_path(
        compare_cfg.get("out_dir", str(concept_dir_a / "stability_vs_b")),
        base=base_root,
    )
    return {
        "checkpoint_a": str(checkpoint_a),
        "checkpoint_b": str(checkpoint_b),
        "concept_dir_a": str(concept_dir_a),
        "concept_dir_b": str(concept_dir_b),
        "out_dir": str(out_dir),
    }


def compare_runs(config: dict, *, base_root: Path) -> dict:
    compare_cfg = dict(config.get("compare") or {})
    checkpoint_a, train_dir_a = _resolve_checkpoint(resolve_path(compare_cfg["run_a"], base=base_root))
    checkpoint_b, train_dir_b = _resolve_checkpoint(resolve_path(compare_cfg["run_b"], base=base_root))
    concept_dir_a = _resolve_concept_dir(compare_cfg.get("concept_dir_a"), train_dir=train_dir_a)
    concept_dir_b = _resolve_concept_dir(compare_cfg.get("concept_dir_b"), train_dir=train_dir_b)
    out_dir = resolve_path(
        compare_cfg.get("out_dir", str(concept_dir_a / "stability_vs_b")),
        base=base_root,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    model_a, _ = load_local_sae_checkpoint(checkpoint_a, device="cpu")
    model_b, _ = load_local_sae_checkpoint(checkpoint_b, device="cpu")
    matched_b, cosine = _match_decoders(model_a.W_dec.detach().cpu(), model_b.W_dec.detach().cpu())

    feature_scan_a = _load_feature_top_tokens(concept_dir_a)
    feature_scan_b = _load_feature_top_tokens(concept_dir_b)
    token_sets_a = _token_sets(feature_scan_a)
    token_sets_b = _token_sets(feature_scan_b)
    top_image_sets_a = _load_top_image_sets(concept_dir_a)
    top_image_sets_b = _load_top_image_sets(concept_dir_b)

    token_mean, token_median = _aggregate_overlap_for_matches(
        matched_b,
        sets_a=token_sets_a,
        sets_b=token_sets_b,
    )
    image_mean, image_median = _aggregate_overlap_for_matches(
        matched_b,
        sets_a=top_image_sets_a,
        sets_b=top_image_sets_b,
    )

    image_frequency_a = _load_image_frequency(concept_dir_a)
    image_frequency_b = _load_image_frequency(concept_dir_b)
    aligned_frequency_b = image_frequency_b[matched_b]
    feature_frequency_correlation = _pearson(image_frequency_a, aligned_frequency_b)

    candidate_ids_a = _load_candidate_ids(concept_dir_a)
    candidate_ids_b = _load_candidate_ids(concept_dir_b)
    mapped_candidate_ids_a = {int(matched_b[int(feature_id)]) for feature_id in candidate_ids_a}
    candidate_overlap = _jaccard(mapped_candidate_ids_a, candidate_ids_b)

    summary = {
        "checkpoint_a": str(checkpoint_a),
        "checkpoint_b": str(checkpoint_b),
        "concept_dir_a": str(concept_dir_a),
        "concept_dir_b": str(concept_dir_b),
        "num_features_a": int(model_a.d_sae),
        "num_features_b": int(model_b.d_sae),
        "decoder_cosine_mean": float(np.asarray(cosine, dtype=np.float64).mean()),
        "decoder_cosine_median": float(np.median(cosine)),
        "top_token_jaccard_mean": token_mean,
        "top_token_jaccard_median": token_median,
        "top_image_jaccard_mean": image_mean,
        "top_image_jaccard_median": image_median,
        "feature_frequency_correlation": feature_frequency_correlation,
        "candidate_concept_overlap": candidate_overlap,
        "candidate_concept_counts": {
            "run_a": int(len(candidate_ids_a)),
            "run_b": int(len(candidate_ids_b)),
            "mapped_intersection": int(len(mapped_candidate_ids_a & candidate_ids_b)),
        },
    }
    (out_dir / "stability_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "out_dir": str(out_dir),
        "summary_path": str(out_dir / "stability_summary.json"),
        **summary,
    }
