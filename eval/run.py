"""Evaluation CLI. Runs against the REAL OpenAI API (needs OPENAI_API_KEY) and writes reports.

    # 1) retrieval-only grid: chunk_size x overlap x top_k, plus threshold sweep (cheap: embeddings only)
    python -m eval.run retrieval-grid --chunk-sizes 300 500 800 --overlaps 0 100 --top-ks 3 5 8

    # 2) full pipeline (LLM answers + judge) for one or more configs "chunk:overlap:top_k:threshold"
    python -m eval.run full --config 500:100:5:0.30 --config 800:100:5:0.30 --tag final

    # 3) everything in one go: grid -> candidates by the documented rule -> full runs -> final defaults
    python -m eval.run auto --tag final

Reports go to eval/reports/ (commit them), raw per-question results to eval/runs/ (git-ignored).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

from app.core.config import get_settings
from app.core.openai_client import build_openai_client
from app.services.embeddings import Embedder
from eval.dataset import DEFAULT_DATASET, load_dataset
from eval.pipeline import EvalConfig, FullRun, load_corpus, pick_winner, run_full, run_retrieval_grid, select_candidates
from eval.report import dump_runs, write_final_defaults, write_full_report, write_grid_report

REPORTS = Path(__file__).parent / "reports"
RUNS = Path(__file__).parent / "runs"
DEFAULT_JUDGE_MODEL = "gpt-6.1-sol"  # stronger than the generator on purpose; see eval/judge.py for limits


def _config(text: str) -> EvalConfig:
    try:
        size, overlap, top_k, threshold = text.split(":")
        return EvalConfig(int(size), int(overlap), int(top_k), float(threshold))
    except ValueError:
        raise argparse.ArgumentTypeError("expected chunk_size:overlap:top_k:threshold, e.g. 500:100:5:0.30")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    sub = parser.add_subparsers(dest="command", required=True)

    grid = sub.add_parser("retrieval-grid")
    grid.add_argument("--chunk-sizes", type=int, nargs="+", default=[300, 500, 800])
    grid.add_argument("--overlaps", type=int, nargs="+", default=[0, 100])
    grid.add_argument("--top-ks", type=int, nargs="+", default=[3, 5, 8])
    grid.add_argument("--max-false-refusal", type=float, default=0.05, help="threshold policy limit (default 5%%)")

    full = sub.add_parser("full")
    full.add_argument("--config", type=_config, action="append", help="chunk:overlap:top_k:threshold (repeatable)")
    full.add_argument("--tag", default="run")
    full.add_argument("--no-judge", action="store_true")
    full.add_argument("--judge-model", default=os.environ.get("EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL))

    if not any(a in ("retrieval-grid", "full", "auto", "-h", "--help") for a in argv):
        argv = list(argv) + ["full"]  # plain `python -m eval.run` = full pipeline, default config
    auto = sub.add_parser("auto", help="grid + full runs on the best candidates + recommended defaults")
    auto.add_argument("--tag", default="final")
    auto.add_argument("--chunk-sizes", type=int, nargs="+", default=[300, 500, 800])
    auto.add_argument("--overlaps", type=int, nargs="+", default=[0, 100])
    auto.add_argument("--top-ks", type=int, nargs="+", default=[3, 5, 8])
    auto.add_argument("--rank-k", type=int, default=5)
    auto.add_argument("--max-false-refusal", type=float, default=0.05)
    auto.add_argument("--judge-model", default=os.environ.get("EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL))

    args = parser.parse_args(argv)
    settings = get_settings()
    if settings.openai_api_key is None:
        print("OPENAI_API_KEY is not set: the evaluation runs against the real API and cannot be faked.", file=sys.stderr)
        return 2

    client = build_openai_client(settings)
    embedder = Embedder(client, settings.embedding_model, settings.embedding_batch_size)
    questions = load_dataset(args.dataset)
    corpus = load_corpus()
    print("dataset: %d questions; corpus: %s" % (len(questions), ", ".join(f.filename for f in corpus)))

    if args.command == "retrieval-grid":
        rows, scores = run_retrieval_grid(questions, corpus, embedder, args.chunk_sizes, args.overlaps, args.top_ks)
        path = write_grid_report(REPORTS, rows, scores, settings.embedding_model, args.max_false_refusal)
        print("wrote", path)
        return 0

    if args.command == "auto":
        rows, scores = run_retrieval_grid(questions, corpus, embedder, args.chunk_sizes, args.overlaps, args.top_ks)
        print("wrote", write_grid_report(REPORTS, rows, scores, settings.embedding_model, args.max_false_refusal))
        candidates = select_candidates(rows, scores, args.rank_k, args.top_ks, args.max_false_refusal)
        auto_runs: List[FullRun] = []
        for cfg in candidates:
            print("running %s ..." % cfg.label)
            auto_runs.append(
                run_full(
                    questions, corpus, embedder, client, settings.openai_model, settings.openai_reasoning_effort,
                    settings.answer_max_output_tokens, cfg, judge_client=client, judge_model=args.judge_model,
                )
            )
        print("wrote", write_full_report(REPORTS, auto_runs, args.tag))
        print("raw results:", dump_runs(RUNS, auto_runs, args.tag))
        print("wrote", write_final_defaults(REPORTS, pick_winner(auto_runs), auto_runs, args.tag))
        return 0

    configs = args.config or [EvalConfig(settings.chunk_size, settings.chunk_overlap, settings.top_k, settings.score_threshold)]
    runs: List[FullRun] = []
    for cfg in configs:
        print("running %s ..." % cfg.label)
        runs.append(
            run_full(
                questions, corpus, embedder, client, settings.openai_model, settings.openai_reasoning_effort,
                settings.answer_max_output_tokens, cfg,
                judge_client=None if args.no_judge else client,
                judge_model=None if args.no_judge else args.judge_model,
            )
        )
    print("wrote", write_full_report(REPORTS, runs, args.tag))
    print("raw results:", dump_runs(RUNS, runs, args.tag))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
