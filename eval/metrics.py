"""Pure metric functions (no I/O, no network) so they can be unit-tested with toy data."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from app.services.citation_validation import normalize
from eval.dataset import Evidence, Question


def chunk_contains(chunk_filename: str, chunk_text: str, evidence: Evidence) -> bool:
    """A chunk is relevant to an evidence fragment if it is from that document and contains it."""
    return chunk_filename == evidence.document and normalize(evidence.text) in normalize(chunk_text)


# ------------------------------------------------------------------ retrieval metrics


def evidence_ranks(question: Question, ranked: Sequence[Dict[str, str]]) -> List[Optional[int]]:
    """For each evidence fragment: 1-based rank of the first retrieved chunk containing it, or None.

    `ranked` is the retrieval result best-first: [{"filename": ..., "text": ...}, ...].
    """
    ranks: List[Optional[int]] = []
    for evidence in question.evidence:
        rank = next(
            (i for i, c in enumerate(ranked, start=1) if chunk_contains(c["filename"], c["text"], evidence)),
            None,
        )
        ranks.append(rank)
    return ranks


@dataclass(frozen=True)
class RetrievalMetrics:
    k: int
    n: int
    hit_at_k: float
    recall_at_k: float
    mrr: float


def retrieval_metrics(per_question_ranks: Sequence[List[Optional[int]]], k: int) -> RetrievalMetrics:
    """Hit@k  : share of questions with at least one evidence fragment in the top k.
    Recall@k : mean over questions of (fragments found in top k) / (fragments needed).
    MRR      : mean of 1 / (rank of the first relevant chunk); 0 when nothing relevant is in the top
               max-k that was retrieved. Rank is NOT cut at k, so MRR is comparable across k.
    Only answerable / multi_chunk questions (those with evidence) belong here.
    """
    n = len(per_question_ranks)
    if n == 0:
        return RetrievalMetrics(k, 0, 0.0, 0.0, 0.0)
    hits = recalls = reciprocal = 0.0
    for ranks in per_question_ranks:
        found = [r for r in ranks if r is not None and r <= k]
        hits += 1.0 if found else 0.0
        recalls += len(found) / len(ranks)
        present = [r for r in ranks if r is not None]
        reciprocal += 1.0 / min(present) if present else 0.0
    return RetrievalMetrics(k, n, hits / n, recalls / n, reciprocal / n)


# ------------------------------------------------------------------ score threshold


@dataclass(frozen=True)
class ThresholdPoint:
    threshold: float
    false_refusal_rate: float  # answerable questions the gate would block (unrecoverable errors)
    caught_unanswerable_rate: float  # unanswerable questions the gate blocks before the LLM
    gate_accuracy: float


def threshold_sweep(
    answerable_best: Sequence[float], unanswerable_best: Sequence[float], thresholds: Sequence[float]
) -> List[ThresholdPoint]:
    """The gate refuses when best_score < threshold. Inputs are the best-chunk scores per question."""
    points: List[ThresholdPoint] = []
    total = len(answerable_best) + len(unanswerable_best)
    for t in thresholds:
        blocked_answerable = sum(1 for s in answerable_best if s < t)
        caught = sum(1 for s in unanswerable_best if s < t)
        points.append(
            ThresholdPoint(
                threshold=round(t, 4),
                false_refusal_rate=blocked_answerable / len(answerable_best) if answerable_best else 0.0,
                caught_unanswerable_rate=caught / len(unanswerable_best) if unanswerable_best else 0.0,
                gate_accuracy=((len(answerable_best) - blocked_answerable) + caught) / total if total else 0.0,
            )
        )
    return points


def choose_threshold(points: Sequence[ThresholdPoint], max_false_refusal: float = 0.05) -> Optional[ThresholdPoint]:
    """Policy: maximize how many unanswerable questions the gate catches, SUBJECT TO blocking at most
    `max_false_refusal` of the answerable ones. Rationale: a gate error on an answerable question is
    unrecoverable (the user gets no answer), whereas an unanswerable question that slips through is
    still refused by the LLM / citation check afterwards. Ties go to the LOWER threshold (safer).
    """
    admissible = [p for p in points if p.false_refusal_rate <= max_false_refusal]
    if not admissible:
        return None
    return max(admissible, key=lambda p: (p.caught_unanswerable_rate, -p.threshold))


# ------------------------------------------------------------------ answer-level metrics


def citation_support(question: Question, citations: Sequence[Dict[str, object]], chunk_text_by_id: Dict[str, str]) -> float:
    """Share of the question's evidence fragments covered by at least one CITED chunk that is from the
    right document and actually contains the fragment. 1.0 = every needed fact is properly cited."""
    if not question.evidence:
        return 0.0
    covered = 0
    for evidence in question.evidence:
        for citation in citations:
            text = chunk_text_by_id.get(str(citation["chunk_id"]), "")
            if chunk_contains(str(citation["filename"]), text, evidence):
                covered += 1
                break
    return covered / len(question.evidence)


def page_match(question: Question, citations: Sequence[Dict[str, object]]) -> Optional[bool]:
    """For PDF evidence: does some citation point to the evidence's page of the right file?
    None when the question has no page-based evidence (DOCX / unanswerable)."""
    paged = [e for e in question.evidence if e.page is not None]
    if not paged:
        return None
    return all(
        any(c["filename"] == e.document and c.get("page") == e.page for c in citations) for e in paged
    )


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(math.ceil(p / 100.0 * len(ordered))) - 1))
    return ordered[index]


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
