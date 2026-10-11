from __future__ import annotations

import pytest

from eval.pipeline import ConfigScores, GridRow, select_candidates


def _row(size: int, overlap: int, k: int, recall: float, mrr: float, chunks: int) -> GridRow:
    return GridRow(size, overlap, k, chunks, 1.0, recall, recall, mrr)


def test_candidates_come_from_retrieval_numbers_and_the_threshold_policy() -> None:
    rows = [
        _row(300, 0, 5, 0.80, 0.60, 90), _row(500, 100, 5, 0.90, 0.70, 60), _row(800, 100, 5, 0.90, 0.70, 40),
        _row(500, 100, 3, 0.70, 0.70, 60), _row(800, 100, 3, 0.70, 0.70, 40),
        _row(500, 100, 8, 0.95, 0.70, 60), _row(800, 100, 8, 0.90, 0.70, 40),
    ]
    scores = [
        ConfigScores(500, 100, 60, 1.0, [0.60, 0.50, 0.45], [0.30, 0.20], 0),
        ConfigScores(800, 100, 40, 1.0, [0.60, 0.50, 0.45], [0.30, 0.20], 0),
        ConfigScores(300, 0, 90, 1.0, [0.60], [0.30], 0),
    ]
    picked = select_candidates(rows, scores, rank_k=5, top_ks=[3, 5, 8], max_false_refusal=0.05)

    # equal retrieval quality -> the configuration with fewer chunks (cheaper to embed) ranks first
    assert [(c.chunk_size, c.overlap, c.top_k) for c in picked[:2]] == [(800, 100, 5), (500, 100, 5)]
    # the best configuration is also tried at the other top_k values
    assert sorted(c.top_k for c in picked[2:]) == [3, 8] and all(c.chunk_size == 800 for c in picked[2:])
    # threshold: 0.35 already blocks both unanswerable questions (best scores 0.30 / 0.20) and no
    # answerable one (lowest 0.45); 0.40 and 0.45 catch the same, and ties go to the LOWER value.
    assert picked[0].threshold == pytest.approx(0.35)
