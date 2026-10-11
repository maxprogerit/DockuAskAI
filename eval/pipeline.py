"""Runs the REAL pipeline (parse -> chunk -> embed -> Chroma -> retrieve -> answer) over the eval set.

Everything that talks to OpenAI is injected (`embedder`, `llm_client`, `judge_client`) so the wiring
can be unit-tested with fakes; `eval.run` injects the real clients. Nothing here fakes results.
"""
from __future__ import annotations

import shutil
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


from app.core.errors import UpstreamError
from app.services.answering import AnswerService
from app.services.chunking import Chunk
from app.services.embeddings import Embedder
from app.services.file_validation import FileType, detect_file_type
from app.services.ingestion import build_chunks, embed_and_store
from app.services.retrieval import Retriever
from app.services.vector_store import VectorStore, open_chroma
from eval import metrics
from eval.dataset import SAMPLES_DIR, UNANSWERABLE, Question
from eval.judge import judge_answer
from eval.pricing import embedding_cost, llm_cost


@dataclass(frozen=True)
class EvalConfig:
    chunk_size: int
    overlap: int
    top_k: int
    threshold: float

    @property
    def label(self) -> str:
        return "cs%d_ov%d_k%d_t%.2f" % (self.chunk_size, self.overlap, self.top_k, self.threshold)


@dataclass(frozen=True)
class CorpusFile:
    filename: str
    data: bytes
    file_type: FileType


def load_corpus(directory: Path = SAMPLES_DIR) -> List[CorpusFile]:
    files: List[CorpusFile] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in (".pdf", ".docx"):
            data = path.read_bytes()
            files.append(CorpusFile(path.name, data, detect_file_type(data, path.name)))
    return files


@dataclass
class BuiltIndex:
    store: VectorStore
    chunks: List[Chunk]
    embedding_tokens: int
    workdir: Path

    def close(self) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)


def build_index(cfg: EvalConfig, corpus: Sequence[CorpusFile], embedder: Embedder, max_pages: int = 300) -> BuiltIndex:
    workdir = Path(tempfile.mkdtemp(prefix="docuask_eval_"))
    store = VectorStore(open_chroma(str(workdir)), "eval_%s" % cfg.label.replace(".", "_"), embedder.model)
    chunks: List[Chunk] = []
    tokens = 0
    for file in corpus:
        _, file_chunks = build_chunks(
            "doc:%s" % file.filename, file.filename, file.data, file.file_type, max_pages, cfg.chunk_size, cfg.overlap
        )
        tokens += embed_and_store(file_chunks, embedder, store)
        chunks.extend(file_chunks)
    return BuiltIndex(store, chunks, tokens, workdir)


# ====================================================================== retrieval grid


@dataclass(frozen=True)
class GridRow:
    chunk_size: int
    overlap: int
    top_k: int
    n_chunks: int
    evidence_reachable: float
    hit_at_k: float
    recall_at_k: float
    mrr: float


@dataclass
class ConfigScores:
    chunk_size: int
    overlap: int
    n_chunks: int
    evidence_reachable: float
    answerable_best: List[float]
    unanswerable_best: List[float]
    embedding_tokens: int


def run_retrieval_grid(
    questions: Sequence[Question],
    corpus: Sequence[CorpusFile],
    embedder: Embedder,
    chunk_sizes: Sequence[int],
    overlaps: Sequence[int],
    top_ks: Sequence[int],
) -> Tuple[List[GridRow], List[ConfigScores]]:
    """Retrieval-only (no LLM): embeds the corpus once per (chunk_size, overlap) and the questions once."""
    max_k = max(top_ks)
    qvectors = embedder.embed([q.question for q in questions]).vectors
    rows: List[GridRow] = []
    scores: List[ConfigScores] = []
    with_evidence = [(q, v) for q, v in zip(questions, qvectors) if q.is_answerable]

    for size in chunk_sizes:
        for overlap in overlaps:
            if overlap >= size:
                continue
            index = build_index(EvalConfig(size, overlap, max_k, 0.0), corpus, embedder)
            try:
                needed = [e for q, _ in with_evidence for e in q.evidence]
                reachable = sum(
                    1 for e in needed if any(metrics.chunk_contains(c.filename, c.text, e) for c in index.chunks)
                ) / max(len(needed), 1)

                ranks_per_question: List[List[Optional[int]]] = []
                answerable_best: List[float] = []
                unanswerable_best: List[float] = []
                for question, vector in zip(questions, qvectors):
                    hits = index.store.query(vector, max_k)
                    best = hits[0].score if hits else -1.0
                    if question.is_answerable:
                        answerable_best.append(best)
                        ranks_per_question.append(
                            metrics.evidence_ranks(question, [{"filename": h.filename, "text": h.text} for h in hits])
                        )
                    else:
                        unanswerable_best.append(best)
                for k in top_ks:
                    m = metrics.retrieval_metrics(ranks_per_question, k)
                    rows.append(GridRow(size, overlap, k, len(index.chunks), reachable, m.hit_at_k, m.recall_at_k, m.mrr))
                scores.append(
                    ConfigScores(size, overlap, len(index.chunks), reachable, answerable_best, unanswerable_best, index.embedding_tokens)
                )
            finally:
                index.close()
    return rows, scores


# ====================================================================== full pipeline run


@dataclass
class FullRun:
    config: EvalConfig
    llm_model: str
    embedding_model: str
    judge_model: Optional[str]
    n_chunks: int
    corpus_embedding_tokens: int
    results: List[Dict[str, Any]]
    judge_tokens: Tuple[int, int]


def run_full(
    questions: Sequence[Question],
    corpus: Sequence[CorpusFile],
    embedder: Embedder,
    llm_client: Any,
    llm_model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    cfg: EvalConfig,
    judge_client: Optional[Any] = None,
    judge_model: Optional[str] = None,
) -> FullRun:
    index = build_index(cfg, corpus, embedder)
    try:
        retriever = Retriever(embedder, index.store, cfg.top_k, cfg.threshold)
        service = AnswerService(llm_client, retriever, llm_model, reasoning_effort, max_output_tokens)
        results: List[Dict[str, Any]] = []
        judge_prompt = judge_completion = 0

        for question in questions:
            record: Dict[str, Any] = {
                "id": question.id, "type": question.type, "language": question.language,
                "tags": question.tags, "question": question.question,
                "expected_answer": question.expected_answer, "error": None,
            }
            try:
                response = service.ask(question.question)
            except UpstreamError as exc:
                record["error"] = exc.message
                results.append(record)
                continue

            chunk_text = {r.chunk_id: r.text for r in response.retrieved}
            citations = [c.model_dump() for c in response.citations]
            record.update(
                answerable=response.answerable,
                reason=response.reason,
                answer=response.answer,
                citations=citations,
                retrieved=[
                    {"chunk_id": r.chunk_id, "score": r.score, "filename": r.filename, "page": r.page}
                    for r in response.retrieved
                ],
                best_score=response.retrieved[0].score if response.retrieved else None,
                usage=response.usage.model_dump(),
                citation_support=None,
                page_match=None,
                judge=None,
            )
            if question.is_answerable and response.answerable:
                record["citation_support"] = metrics.citation_support(question, citations, chunk_text)
                record["page_match"] = metrics.page_match(question, citations)
                if judge_client is not None and judge_model:
                    try:
                        judged = judge_answer(judge_client, judge_model, question.question, question.expected_answer, response.answer)
                        judge_prompt += judged.input_tokens
                        judge_completion += judged.output_tokens
                        record["judge"] = judged.verdict.model_dump()
                    except RuntimeError as exc:
                        record["judge"] = {"verdict": "judge_error", "reason": str(exc)}
            results.append(record)
        return FullRun(cfg, llm_model, embedder.model, judge_model, len(index.chunks), index.embedding_tokens, results, (judge_prompt, judge_completion))
    finally:
        index.close()


def summarize(run: FullRun) -> Dict[str, Any]:
    """All headline numbers of one full run. Pure function of `run.results`."""
    ok = [r for r in run.results if r["error"] is None]
    errors = len(run.results) - len(ok)
    answerable_q = [r for r in ok if r["type"] != UNANSWERABLE]
    unanswerable_q = [r for r in ok if r["type"] == UNANSWERABLE]
    answered = [r for r in answerable_q if r["answerable"]]

    verdicts = defaultdict(int)
    for r in answerable_q:
        if not r["answerable"]:
            verdicts["not_answered"] += 1
        else:
            verdicts[(r["judge"] or {}).get("verdict", "not_judged")] += 1

    def rate(num: int, den: int) -> Optional[float]:
        return num / den if den else None

    refusal_correct = sum(1 for r in unanswerable_q if not r["answerable"])
    answer_correct = len(answered)
    reasons = defaultdict(int)
    for r in ok:
        if not r["answerable"]:
            reasons[r["reason"]] += 1

    latencies = [r["usage"]["latency_ms"] for r in ok]
    prompt = [r["usage"]["prompt_tokens"] for r in ok]
    completion = [r["usage"]["completion_tokens"] for r in ok]
    emb = [r["usage"]["embedding_tokens"] for r in ok]
    costs = [
        llm_cost(run.llm_model, r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"])
        + embedding_cost(run.embedding_model, r["usage"]["embedding_tokens"])
        for r in ok
    ]
    supports = [r["citation_support"] for r in answered if r["citation_support"] is not None]
    pages = [r["page_match"] for r in answered if r["page_match"] is not None]
    judged = [r for r in answered if r["judge"] and r["judge"]["verdict"] in ("correct", "partially_correct", "incorrect")]

    return {
        "n_questions": len(run.results),
        "n_errors": errors,
        "n_answerable": len(answerable_q),
        "n_unanswerable": len(unanswerable_q),
        "answered_rate_on_answerable": rate(answer_correct, len(answerable_q)),
        "refusal_rate_on_unanswerable": rate(refusal_correct, len(unanswerable_q)),
        "refusal_accuracy": rate(answer_correct + refusal_correct, len(answerable_q) + len(unanswerable_q)),
        "correct": verdicts.get("correct", 0),
        "partially_correct": verdicts.get("partially_correct", 0),
        "incorrect": verdicts.get("incorrect", 0),
        "not_answered": verdicts.get("not_answered", 0),
        "not_judged": verdicts.get("not_judged", 0) + verdicts.get("judge_error", 0),
        "correctness_strict": rate(verdicts.get("correct", 0), len(answerable_q)) if run.judge_model else None,
        "correctness_when_answered": rate(sum(1 for r in judged if r["judge"]["verdict"] == "correct"), len(judged)) if judged else None,
        "citation_support": metrics.mean(supports) if supports else None,
        "page_match": rate(sum(1 for p in pages if p), len(pages)) if pages else None,
        "refusal_reasons": dict(reasons),
        "latency_ms_avg": metrics.mean(latencies),
        "latency_ms_p50": metrics.percentile(latencies, 50),
        "latency_ms_p95": metrics.percentile(latencies, 95),
        "prompt_tokens_avg": metrics.mean(prompt),
        "completion_tokens_avg": metrics.mean(completion),
        "embedding_tokens_avg": metrics.mean(emb),
        "cost_per_request_usd": metrics.mean(costs),
        "corpus_embedding_cost_usd": embedding_cost(run.embedding_model, run.corpus_embedding_tokens),
    }


def select_candidates(
    rows: Sequence[GridRow],
    scores: Sequence[ConfigScores],
    rank_k: int,
    top_ks: Sequence[int],
    max_false_refusal: float,
    n_best: int = 2,
) -> List[EvalConfig]:
    """Pick which configurations deserve a full (LLM) run, using ONLY retrieval numbers.

    Ranking: Recall@rank_k + MRR (equal weights, simple on purpose), ties -> fewer chunks (cheaper).
    Candidates: the `n_best` best (chunk_size, overlap) at rank_k, plus the single best one at each
    other top_k so the top_k dimension is also compared end to end. Each candidate gets the
    threshold chosen by the policy in eval.metrics.choose_threshold for ITS chunk configuration
    (0.0 = gate off, when no threshold satisfies the false-refusal limit).
    """
    from eval.report import threshold_grid

    at_k = [r for r in rows if r.top_k == rank_k]
    ordered = sorted(at_k, key=lambda r: (-(r.recall_at_k + r.mrr), r.n_chunks))
    scores_by_cfg = {(s.chunk_size, s.overlap): s for s in scores}

    def threshold_for(size: int, overlap: int) -> float:
        s = scores_by_cfg[(size, overlap)]
        points = metrics.threshold_sweep(s.answerable_best, s.unanswerable_best, threshold_grid())
        chosen = metrics.choose_threshold(points, max_false_refusal)
        return chosen.threshold if chosen else 0.0

    picked: List[EvalConfig] = []
    for r in ordered[:n_best]:
        picked.append(EvalConfig(r.chunk_size, r.overlap, rank_k, threshold_for(r.chunk_size, r.overlap)))
    if ordered:
        best = ordered[0]
        for k in top_ks:
            if k != rank_k:
                picked.append(EvalConfig(best.chunk_size, best.overlap, k, threshold_for(best.chunk_size, best.overlap)))
    return picked


def pick_winner(runs: Sequence[FullRun]) -> FullRun:
    """Best full run by (strict correctness, refusal accuracy); a cheaper request breaks ties."""

    def key(run: FullRun) -> Tuple[float, float, float]:
        s = summarize(run)
        return (s["correctness_strict"] or 0.0, s["refusal_accuracy"] or 0.0, -s["cost_per_request_usd"])

    return max(runs, key=key)


def as_json(run: FullRun) -> Dict[str, Any]:
    return {"config": asdict(run.config), "llm_model": run.llm_model, "embedding_model": run.embedding_model,
            "judge_model": run.judge_model, "n_chunks": run.n_chunks, "summary": summarize(run), "results": run.results}
