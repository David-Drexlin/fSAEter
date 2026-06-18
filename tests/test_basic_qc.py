from __future__ import annotations

import numpy as np

from fsaeter.inspect.basic_qc import (
    select_broad_concepts,
    select_sparse_topk_rows,
    tuple_uniqueness_rates,
)


def test_select_sparse_topk_rows_preserves_invalid_slots():
    rows = np.asarray([[0.8, 0.1, 0.0, 0.3], [0.05, 0.04, 0.03, 0.02]], dtype=np.float32)
    values, indices = select_sparse_topk_rows(rows, k=3, active_threshold=0.09)
    assert indices[0].tolist() == [0, 3, 1]
    assert indices[1].tolist() == [-1, -1, -1]
    assert values[1].tolist() == [0.0, 0.0, 0.0]


def test_tuple_uniqueness_rates_report_expected_fingerprint_signal():
    top_indices = np.asarray(
        [[1, 2, 3, -1], [1, 2, 4, -1], [1, 2, 3, -1], [5, 6, 7, -1]],
        dtype=np.int32,
    )
    rates = tuple_uniqueness_rates(top_indices, sizes=(1, 2, 3))
    assert rates["top_1"] == 0.5
    assert rates["top_2"] == 0.5
    assert rates["top_3"] == 0.75


def test_select_broad_concepts_rejects_class_collapsed_feature():
    top_indices = np.asarray(
        [[0, 2, -1], [0, 2, -1], [0, 3, -1], [1, 2, -1], [1, 3, -1], [1, 3, -1]],
        dtype=np.int64,
    )
    top_values = np.where(top_indices >= 0, 1.0, 0.0).astype(np.float32)
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    candidates = select_broad_concepts(
        top_indices,
        top_values,
        labels,
        vocab_size=4,
        min_support=2,
        min_class_coverage=2,
        min_per_class=1,
        top_n=10,
    )
    candidate_ids = [row["concept_id"] for row in candidates]
    assert 0 not in candidate_ids
    assert 1 not in candidate_ids
    assert 2 in candidate_ids
