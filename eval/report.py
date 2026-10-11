from __future__ import annotations

import csv
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from eval import metrics
from eval.pipeline import ConfigScores, FullRun, GridRow, summarize

FRAMING = (
    "> **Internal evaluation on a small synthetic set - not a benchmark.** The questions come from 4 invented sample "
    "documents, the ground truth was drafted by the assistant and must be reviewed by a human, thresholds were tuned on "
    "the same questions they are reported on (in-sample, optimistic), and with ~30 questions one answer moves a rate by "
    "~3 points. Numbers below are from real runs against the real OpenAI API; nothing is simulated.\n"
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else "%.0f%%" % (100 * value)


def _f(value: Optional[float], digits: int = 3) -> str:
    return "n/a" if value is None else ("%." + str(digits) + "f") % value


def threshold_grid() -> List[float]:
    return [round(0.05 * i, 2) for i in range(2, 15)]  # 0.10 .. 0.70


# ------------------------------------------------------------------ retrieval grid


def write_grid_report(
    out_dir: Path, rows: Sequence[GridRow], scores: Sequence[ConfigScores], embedding_model: str, max_false_refusal: float
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "retrieval_grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["chunk_size", "overlap", "top_k", "n_chunks", "evidence_reachable", "hit_at_k", "recall_at_k", "mrr"])
        for r in rows:
            writer.writerow([r.chunk_size, r.overlap, r.top_k, r.n_chunks, "%.4f" % r.evidence_reachable, "%.4f" % r.hit_at_k, "%.4f" % r.recall_at_k, "%.4f" % r.mrr])

    lines = ["# Retrieval experiments (no LLM)", "", FRAMING, "Generated %s - embedding model `%s`." % (_now(), embedding_model), ""]
    lines += [
        "Relevance is judged by text: a retrieved chunk is relevant if it comes from the expected document and contains the "
        "ground-truth evidence fragment. `evidence reachable` is the share of evidence fragments that appear whole inside at "
        "least one chunk of that configuration (a chunker that splits a fact caps recall below 100% before retrieval even starts).",
        "",
        "| chunk_size | overlap | top_k | chunks | evidence reachable | Hit@k | Recall@k | MRR |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append("| %d | %d | %d | %d | %s | %s | %s | %s |" % (r.chunk_size, r.overlap, r.top_k, r.n_chunks, _pct(r.evidence_reachable), _pct(r.hit_at_k), _pct(r.recall_at_k), _f(r.mrr)))

    lines += ["", "## Score threshold sweep", "",
              "The gate refuses when the best chunk's cosine similarity is below the threshold, before any LLM call. "
              "*False refusals* = answerable questions blocked (unrecoverable); *caught* = unanswerable questions blocked. "
              "Policy: maximize *caught* subject to false refusals <= %s; ties go to the lower threshold." % _pct(max_false_refusal), ""]
    for s in scores:
        points = metrics.threshold_sweep(s.answerable_best, s.unanswerable_best, threshold_grid())
        chosen = metrics.choose_threshold(points, max_false_refusal)
        lines += [
            "### chunk_size=%d, overlap=%d" % (s.chunk_size, s.overlap), "",
            "Best-chunk score, answerable questions: min %s, mean %s. Unanswerable questions: mean %s, max %s." % (
                _f(min(s.answerable_best) if s.answerable_best else None), _f(metrics.mean(s.answerable_best)),
                _f(metrics.mean(s.unanswerable_best)), _f(max(s.unanswerable_best) if s.unanswerable_best else None)),
            "", "| threshold | false refusals | caught unanswerable | gate accuracy |", "|---:|---:|---:|---:|",
        ]
        for p in points:
            mark = " **<- chosen**" if chosen is not None and p.threshold == chosen.threshold else ""
            lines.append("| %.2f | %s | %s | %s |%s" % (p.threshold, _pct(p.false_refusal_rate), _pct(p.caught_unanswerable_rate), _pct(p.gate_accuracy), mark))
        lines.append("" if chosen else "\nNo threshold satisfies the false-refusal limit for this configuration.\n")
        lines.append("")
    path = out_dir / "retrieval_grid.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ------------------------------------------------------------------ full runs


def write_full_report(out_dir: Path, runs: Sequence[FullRun], tag: str, spot_seed: int = 7, spot_n: int = 8) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = [summarize(r) for r in runs]

    with (out_dir / ("experiments_%s.csv" % tag)).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["chunk_size", "overlap", "top_k", "threshold", "answered_rate", "refusal_rate_unanswerable", "refusal_accuracy",
                         "correct_strict", "citation_support", "page_match", "latency_ms_avg", "prompt_tokens_avg", "completion_tokens_avg", "cost_per_request_usd"])
        for r, s in zip(runs, summaries):
            c = r.config
            writer.writerow([c.chunk_size, c.overlap, c.top_k, c.threshold, s["answered_rate_on_answerable"], s["refusal_rate_on_unanswerable"],
                             s["refusal_accuracy"], s["correctness_strict"], s["citation_support"], s["page_match"], "%.0f" % s["latency_ms_avg"],
                             "%.0f" % s["prompt_tokens_avg"], "%.0f" % s["completion_tokens_avg"], "%.6f" % s["cost_per_request_usd"]])

    first = runs[0]
    lines = ["# Full-pipeline evaluation: %s" % tag, "", FRAMING,
             "Generated %s. LLM `%s`, embeddings `%s`, judge `%s`." % (_now(), first.llm_model, first.embedding_model, first.judge_model or "none (correctness not graded)"), "",
             "| chunk_size | overlap | top_k | threshold | answered (answerable) | refused (unanswerable) | refusal accuracy | correct (strict) | citation support | page match (PDF) | avg latency | avg tokens in/out | cost / request |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r, s in zip(runs, summaries):
        c = r.config
        lines.append("| %d | %d | %d | %.2f | %s | %s | %s | %s | %s | %s | %.1f s | %.0f / %.0f | $%.5f |" % (
            c.chunk_size, c.overlap, c.top_k, c.threshold, _pct(s["answered_rate_on_answerable"]), _pct(s["refusal_rate_on_unanswerable"]),
            _pct(s["refusal_accuracy"]), _pct(s["correctness_strict"]), _pct(s["citation_support"]), _pct(s["page_match"]),
            s["latency_ms_avg"] / 1000.0, s["prompt_tokens_avg"], s["completion_tokens_avg"], s["cost_per_request_usd"]))

    lines += ["", "Definitions: *answered* = share of answerable+multi_chunk questions for which the system returned a verified answer; "
              "*refused* = share of unanswerable questions correctly refused (by the gate, by the model, or by the citation check); "
              "*correct (strict)* = judge verdict `correct` over ALL answerable questions (a refusal counts as wrong); "
              "*citation support* = share of the needed evidence fragments contained in a cited chunk of the right document; "
              "*cost* = LLM tokens + query embedding at the prices in `eval/pricing.py`, no cache discount, judge excluded.", ""]

    for r, s in zip(runs, summaries):
        c = r.config
        lines += ["## Details: chunk_size=%d overlap=%d top_k=%d threshold=%.2f" % (c.chunk_size, c.overlap, c.top_k, c.threshold), "",
                  "- Questions: %d (%d answerable/multi_chunk, %d unanswerable), errors: %d, chunks indexed: %d" % (s["n_questions"], s["n_answerable"], s["n_unanswerable"], s["n_errors"], r.n_chunks),
                  "- Judge verdicts on answerable questions: correct %d, partially correct %d, incorrect %d, not answered %d, not judged %d" % (s["correct"], s["partially_correct"], s["incorrect"], s["not_answered"], s["not_judged"]),
                  "- Correct when an answer was given: %s" % _pct(s["correctness_when_answered"]),
                  "- Refusal reasons: %s" % (", ".join("%s=%d" % kv for kv in sorted(s["refusal_reasons"].items())) or "none"),
                  "- Latency: avg %.0f ms, p50 %.0f ms, p95 %.0f ms" % (s["latency_ms_avg"], s["latency_ms_p50"], s["latency_ms_p95"]),
                  "- One-time corpus embedding cost: $%.6f; judge tokens in/out: %d / %d" % (s["corpus_embedding_cost_usd"], r.judge_tokens[0], r.judge_tokens[1]),
                  "", "| id | type | expected | outcome | judge | citation support |", "|---|---|---|---|---|---:|"]
        for res in r.results:
            if res["error"]:
                outcome, judge, support = "ERROR: %s" % res["error"], "-", "-"
            else:
                outcome = "answered" if res["answerable"] else "refused (%s)" % res["reason"]
                judge = (res["judge"] or {}).get("verdict", "-")
                support = "-" if res["citation_support"] is None else "%.0f%%" % (100 * res["citation_support"])
            expected = "refuse" if res["type"] == "unanswerable" else "answer"
            lines.append("| %s | %s | %s | %s | %s | %s |" % (res["id"], res["type"], expected, outcome, judge, support))
        lines.append("")
    path = out_dir / ("experiments_%s.md" % tag)
    path.write_text("\n".join(lines), encoding="utf-8")
    write_spot_check(out_dir, runs[0], tag, spot_seed, spot_n)
    return path


def write_spot_check(out_dir: Path, run: FullRun, tag: str, seed: int, n: int) -> Path:
    """A seeded random sample of answered questions for the HUMAN spot check. The judge's verdict is
    shown next to it so disagreements are easy to see."""
    candidates = [r for r in run.results if r["error"] is None and r["type"] != "unanswerable" and r["answerable"]]
    sample = random.Random(seed).sample(candidates, min(n, len(candidates)))
    lines = ["# Manual spot check - %s (first configuration)" % tag, "",
             "Read each answer against the source and fill in your verdict. Compare with the judge to see where the LLM judge is too strict or too lenient.", ""]
    for r in sorted(sample, key=lambda x: x["id"]):
        lines += ["## %s - %s" % (r["id"], r["question"]), "",
                  "- Expected: %s" % r["expected_answer"], "- System answer: %s" % r["answer"],
                  "- Citations: %s" % ("; ".join('%s p.%s "%s"' % (c["filename"], c["page"] if c["page"] else (c["section"] or "-"), c["quote"]) for c in r["citations"]) or "none"),
                  "- LLM judge: %s - %s" % ((r["judge"] or {}).get("verdict", "-"), (r["judge"] or {}).get("reason", "")),
                  "- **Human verdict:** [ ] correct   [ ] partially   [ ] incorrect      Citation supports the answer: [ ] yes  [ ] no", ""]
    path = out_dir / ("spot_check_%s.md" % tag)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_final_defaults(out_dir: Path, winner: FullRun, runs: Sequence[FullRun], tag: str) -> Path:
    """The configuration that scored best on this run, as .env lines. Copy them into .env AND into the
    defaults in app/core/config.py, then record the table that justifies them in the README."""
    c = winner.config
    s = summarize(winner)
    lines = [
        "# Recommended defaults from evaluation run `%s` (%s)" % (tag, _now()),
        "# Winner by strict correctness, then refusal accuracy, then cost, among %d configurations." % len(runs),
        "# strict correctness %s, refusal accuracy %s, citation support %s. In-sample on ~30 questions: see the report."
        % (_pct(s["correctness_strict"]), _pct(s["refusal_accuracy"]), _pct(s["citation_support"])),
        "CHUNK_SIZE=%d" % c.chunk_size,
        "CHUNK_OVERLAP=%d" % c.overlap,
        "TOP_K=%d" % c.top_k,
        "SCORE_THRESHOLD=%.2f" % c.threshold,
        "",
    ]
    path = out_dir / ("final_defaults_%s.env" % tag)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def dump_runs(out_dir: Path, runs: Sequence[FullRun], tag: str) -> Path:
    import json

    from eval.pipeline import as_json

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ("%s.json" % tag)
    path.write_text(json.dumps([as_json(r) for r in runs], ensure_ascii=False, indent=1), encoding="utf-8")
    return path
