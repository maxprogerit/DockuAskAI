"""Stage-3 helper: ingest ONE file and run retrieval queries, printing real cosine scores.

    python -m scripts.ingest_query tests/samples/employee_handbook.pdf \
        -q "How many vacation days do I get?" -q "What is the dress code?"

Uses the real OpenAI embeddings API (OPENAI_API_KEY) and a throw-away Chroma directory
(data/chroma_dev) - it touches neither PostgreSQL nor the production collection. Use it to see where
answerable and unanswerable questions land on the score scale before choosing SCORE_THRESHOLD.
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path


from app.core.config import get_settings
from app.core.errors import DocuAskError
from app.core.openai_client import build_openai_client
from app.services.embeddings import Embedder
from app.services.file_validation import check_size, detect_file_type, sanitize_filename
from app.services.ingestion import build_chunks, embed_and_store
from app.services.retrieval import Retriever
from app.services.vector_store import VectorStore, open_chroma


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", type=Path)
    parser.add_argument("-q", "--question", action="append", required=True, help="repeatable")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--chroma-dir", type=Path, default=Path("data/chroma_dev"))
    args = parser.parse_args()

    settings = get_settings()
    chunk_size = args.chunk_size or settings.chunk_size
    overlap = settings.chunk_overlap if args.overlap is None else args.overlap

    try:
        client = build_openai_client(settings)
        data = args.file.read_bytes()
        check_size(len(data), settings.max_upload_bytes)
        filename = sanitize_filename(args.file.name)
        file_type = detect_file_type(data, filename)

        document_id = str(uuid.uuid4())
        parsed, chunks = build_chunks(
            document_id, filename, data, file_type, settings.max_pages, chunk_size, overlap
        )
        print("parsed: type=%s pages=%s chunks=%d (chunk_size=%d overlap=%d)"
              % (file_type.value, parsed.page_count, len(chunks), chunk_size, overlap))

        embedder = Embedder(client, settings.embedding_model, settings.embedding_batch_size)
        # One collection per run so repeated runs with other chunk sizes never mix.
        chroma = open_chroma(str(args.chroma_dir))
        store = VectorStore(chroma, "dev_%s" % document_id[:8], settings.embedding_model)
        tokens = embed_and_store(chunks, embedder, store)
        print("embedded with %s: %d tokens; collection distance=%s -> score = 1 - distance"
              % (settings.embedding_model, tokens, store.distance_space))

        retriever = Retriever(embedder, store, args.top_k or settings.top_k, settings.score_threshold)
        for question in args.question:
            outcome = retriever.retrieve(question)
            verdict = "ANSWER (LLM would be called)" if outcome.passed_threshold else "REFUSE (below threshold, no LLM call)"
            print("\nQ: %s\n   best=%s threshold=%.2f -> %s" % (
                question,
                "n/a" if outcome.best_score is None else "%.3f" % outcome.best_score,
                retriever.score_threshold,
                verdict,
            ))
            for rank, chunk in enumerate(outcome.chunks, start=1):
                where = "p.%s" % chunk.page if chunk.page else (chunk.section or "-")
                print("   #%d score=%.3f [%s] %s" % (rank, chunk.score, where, chunk.text[:90].replace("\n", " ")))
    except DocuAskError as exc:
        print("error [%s]: %s" % (exc.code, exc.message), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
