from __future__ import annotations

import re
from typing import Any, Dict, List

import pytest

from app.schemas.llm import LLMAnswer, LLMCitation
from app.services.embeddings import Embedder
from eval import metrics
from eval.build_dataset import ROWS, verify
from eval.dataset import ANSWERABLE, MULTI_CHUNK, UNANSWERABLE, Evidence, Question, load_dataset
from eval.judge import JudgeVerdict
from eval.pipeline import EvalConfig, load_corpus, run_full, run_retrieval_grid, summarize
from eval.pricing import embedding_cost, llm_cost
from eval.report import write_full_report, write_grid_report
from tests.fakes import FakeEmbeddingsClient, FakeLLMClient

# ------------------------------------------------------------------ dataset integrity


def test_dataset_file_matches_the_verified_source_rows() -> None:
    assert verify(ROWS) == []  # every evidence fragment occurs verbatim at the stated page/section
    questions = load_dataset()
    assert [q.id for q in questions] == [r["id"] for r in ROWS], "dataset.jsonl is stale: run python -m eval.build_dataset"


def test_dataset_meets_the_spec() -> None:
    questions = load_dataset()
    assert 20 <= len(questions) <= 30
    assert {q.type for q in questions} == {ANSWERABLE, UNANSWERABLE, MULTI_CHUNK}
    unanswerable = sum(1 for q in questions if q.type == UNANSWERABLE)
    assert unanswerable / len(questions) >= 0.20
    assert all(q.expected_answer for q in questions)
    assert {q.language for q in questions} == {"en", "ru"}
    assert any("injection" in q.tags for q in questions)


# ------------------------------------------------------------------ metrics


def _q(*texts: str, document: str = "d.pdf") -> Question:
    return Question("x", "?", "en", ANSWERABLE, "a", document, 1, [Evidence(document, t, 1) for t in texts])


def _chunk(text: str, filename: str = "d.pdf") -> Dict[str, str]:
    return {"filename": filename, "text": text}


def test_evidence_ranks_use_text_and_document() -> None:
    question = _q("the answer is 42")
    ranked = [_chunk("unrelated"), _chunk("The  answer\nis 42 indeed", "other.pdf"), _chunk("so the answer is 42.")]
    assert metrics.evidence_ranks(question, ranked) == [3]  # rank 2 is the right text in the WRONG document


def test_retrieval_metrics_hit_recall_mrr() -> None:
    per_question = [[1], [3], [None], [2, None]]
    at1 = metrics.retrieval_metrics(per_question, 1)
    at3 = metrics.retrieval_metrics(per_question, 3)
    assert (at1.hit_at_k, at3.hit_at_k) == (0.25, 0.75)
    assert at3.recall_at_k == pytest.approx((1 + 1 + 0 + 0.5) / 4)
    assert at1.mrr == at3.mrr == pytest.approx((1 + 1 / 3 + 0 + 1 / 2) / 4)  # rank is not cut at k
    assert metrics.retrieval_metrics([], 3).n == 0


def test_threshold_sweep_and_policy() -> None:
    answerable = [0.62, 0.55, 0.50, 0.41, 0.30]
    unanswerable = [0.35, 0.28, 0.22, 0.15]
    points = metrics.threshold_sweep(answerable, unanswerable, [0.1, 0.3, 0.4, 0.5])
    by_t = {p.threshold: p for p in points}
    assert by_t[0.1].false_refusal_rate == 0 and by_t[0.1].caught_unanswerable_rate == 0
    assert by_t[0.3].false_refusal_rate == 0 and by_t[0.3].caught_unanswerable_rate == 0.75
    assert by_t[0.5].false_refusal_rate == pytest.approx(0.4)
    best = metrics.choose_threshold(points, max_false_refusal=0.05)
    assert best is not None and best.threshold == 0.3  # highest catch rate within the 5% limit
    assert metrics.choose_threshold(points, max_false_refusal=0.0).threshold == 0.3  # type: ignore[union-attr]
    assert metrics.choose_threshold([p for p in points if p.threshold == 0.5], 0.05) is None


def test_threshold_ties_prefer_the_lower_value() -> None:
    points = metrics.threshold_sweep([0.9, 0.9], [0.1, 0.1], [0.3, 0.4, 0.5])
    assert metrics.choose_threshold(points).threshold == 0.3  # type: ignore[union-attr]


def test_citation_support_and_page_match() -> None:
    question = Question("x", "?", "en", ANSWERABLE, "a", "d.pdf", 1, [Evidence("d.pdf", "fact one", 1), Evidence("d.pdf", "fact two", 2)])
    citations = [{"chunk_id": "c1", "filename": "d.pdf", "page": 1}, {"chunk_id": "c2", "filename": "d.pdf", "page": 3}]
    texts = {"c1": "this has fact one", "c2": "irrelevant"}
    assert metrics.citation_support(question, citations, texts) == 0.5
    assert metrics.page_match(question, citations) is False  # fact two lives on page 2 but only page 3 is cited
    assert metrics.page_match(_q("fact one"), citations) is True
    assert metrics.page_match(Question("u", "?", "en", UNANSWERABLE, "-", None, None), citations) is None


def test_percentile_and_mean() -> None:
    assert metrics.percentile([10, 20, 30, 40], 50) == 20 and metrics.percentile([10, 20, 30, 40], 95) == 40
    assert metrics.mean([]) == 0.0


def test_pricing_arithmetic() -> None:
    assert llm_cost("gpt-6-luna", 1_000_000, 1_000_000) == pytest.approx(0.60)
    assert embedding_cost("text-embedding-3-small", 500_000) == pytest.approx(0.01)
    with pytest.raises(KeyError):
        llm_cost("unknown-model", 1, 1)


# ------------------------------------------------------------------ pipeline wiring (fakes: no network)


def _oracle_llm(questions: List[Question], wrong_on: str = "") -> FakeLLMClient:
    """A scripted 'model' that answers correctly from the prompt for answerable questions and declines
    otherwise. It reads the same prompt the real model would get, so it exercises context building,
    alias mapping and citation validation end to end. `wrong_on` makes one question fabricate."""
    by_text = {q.question: q for q in questions}

    def respond(kwargs: Dict[str, Any]) -> LLMAnswer:
        prompt = kwargs["input"][1]["content"]
        asked = re.search(r"<question>\n(.*)\n</question>", prompt, re.S).group(1)  # type: ignore[union-attr]
        question = by_text[asked]
        if not question.is_answerable:
            return LLMAnswer(answerable=False, citations=[], answer="Not in the documents.")
        citations = []
        for evidence in question.evidence:
            for m in re.finditer(r'<chunk id="(C\d+)"[^>]*>\n(.*?)\n</chunk>', prompt, re.S):
                if evidence.text in m.group(2).replace("\n", " "):
                    quote = evidence.text if question.id != wrong_on else "a sentence nobody wrote"
                    citations.append(LLMCitation(chunk_id=m.group(1), quote=quote))
                    break
        return LLMAnswer(answerable=bool(citations), citations=citations, answer=question.expected_answer)

    class Oracle(FakeLLMClient):
        def _parse(self, **kwargs: Any) -> Any:
            self._scripted = [respond]
            return super()._parse(**kwargs)

    return Oracle([])


class _Judge(FakeLLMClient):
    def _parse(self, **kwargs: Any) -> Any:
        self._scripted = [JudgeVerdict(verdict="correct", reason="matches")]
        return super()._parse(**kwargs)


def test_retrieval_grid_runs_on_the_real_corpus_with_fake_embeddings(tmp_path: Any) -> None:
    questions = load_dataset()
    embedder = Embedder(FakeEmbeddingsClient(), "fake-embedding-model", 64)
    rows, scores = run_retrieval_grid(questions, load_corpus(), embedder, [300, 800], [0, 100], [3, 8])
    assert len(rows) == 2 * 2 * 2 and len(scores) == 4
    assert all(0 <= r.hit_at_k <= 1 and 0 <= r.mrr <= 1 and 0 < r.evidence_reachable <= 1 for r in rows)
    by = {(r.chunk_size, r.overlap, r.top_k): r for r in rows}
    assert by[(300, 100, 8)].hit_at_k >= by[(300, 100, 3)].hit_at_k  # more candidates never hurt Hit@k
    assert by[(300, 0, 3)].n_chunks > by[(800, 0, 3)].n_chunks  # smaller chunks => more of them
    s = scores[0]
    assert len(s.answerable_best) == sum(q.is_answerable for q in questions)
    path = write_grid_report(tmp_path, rows, scores, "fake-embedding-model", 0.05)
    text = path.read_text(encoding="utf-8")
    assert "Internal evaluation on a small synthetic set" in text and "Score threshold sweep" in text
    assert (tmp_path / "retrieval_grid.csv").exists()


def test_full_run_summary_and_reports(tmp_path: Any) -> None:
    questions = load_dataset()
    embedder = Embedder(FakeEmbeddingsClient(), "text-embedding-3-small", 64)
    llm = _oracle_llm(questions, wrong_on="q01")
    run = run_full(
        questions, load_corpus(), embedder, llm, "gpt-6-luna", "low", 500,
        EvalConfig(500, 100, 5, 0.15), judge_client=_Judge([]), judge_model="gpt-6.1-sol",
    )
    summary = summarize(run)
    assert summary["n_questions"] == 30 and summary["n_errors"] == 0
    assert summary["n_unanswerable"] == 9 and summary["n_answerable"] == 21
    by_id = {r["id"]: r for r in run.results}
    # q01: the model fabricated a quote -> citation check rejects it twice -> refusal, never a fake citation
    assert by_id["q01"]["answerable"] is False and by_id["q01"]["reason"] == "citation_validation_failed"
    assert by_id["q01"]["usage"]["llm_calls"] == 2
    # a question with strong lexical overlap is answered, cited at the right page, and judged
    # (the bag-of-words fake embedder is weak, so we pick one it retrieves reliably)
    assert by_id["q05"]["answerable"] is True and by_id["q05"]["citation_support"] == 1.0
    assert by_id["q05"]["page_match"] is True and by_id["q05"]["judge"]["verdict"] == "correct"
    assert 0 < summary["refusal_accuracy"] <= 1
    assert summary["cost_per_request_usd"] > 0 and summary["latency_ms_avg"] >= 0
    assert run.judge_tokens[0] > 0

    path = write_full_report(tmp_path, [run], "unit")
    text = path.read_text(encoding="utf-8")
    assert "| 500 | 100 | 5 | 0.15 |" in text and "citation_validation_failed" in text
    assert (tmp_path / "experiments_unit.csv").exists() and (tmp_path / "spot_check_unit.md").exists()


def test_upstream_failure_is_recorded_not_fatal() -> None:
    import httpx
    import openai

    questions = load_dataset()[:2]
    embedder = Embedder(FakeEmbeddingsClient(), "text-embedding-3-small", 64)
    boom = openai.APIConnectionError(request=httpx.Request("POST", "http://x"))
    run = run_full(questions, load_corpus(), embedder, FakeLLMClient([boom, boom]), "gpt-6-luna", "low", 500, EvalConfig(500, 100, 5, 0.0))
    errors = [r for r in run.results if r["error"]]
    assert len(errors) >= 1 and summarize(run)["n_errors"] == len(errors)
