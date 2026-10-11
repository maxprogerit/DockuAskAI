from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from app.services.embeddings import Embedder
from app.services.vector_store import RetrievedChunk, VectorStore


@dataclass(frozen=True)
class RetrievalOutcome:
    chunks: List[RetrievedChunk]  # best first
    best_score: Optional[float]  # None when nothing was retrieved
    passed_threshold: bool  # False => the caller must refuse WITHOUT calling the LLM
    embedding_tokens: int


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, top_k: int, score_threshold: float) -> None:
        self._embedder = embedder
        self._store = store
        self._top_k = top_k
        self._threshold = score_threshold

    @property
    def score_threshold(self) -> float:
        return self._threshold

    def retrieve(
        self, question: str, top_k: Optional[int] = None, document_ids: Optional[Sequence[str]] = None
    ) -> RetrievalOutcome:
        embedded = self._embedder.embed_query(question)
        chunks = self._store.query(embedded.vectors[0], top_k or self._top_k, document_ids)
        best = chunks[0].score if chunks else None
        # The gate looks at the BEST chunk only: if even that is below the threshold, nothing in
        # the library is about this question and calling the LLM would only invite a hallucination.
        return RetrievalOutcome(
            chunks=chunks,
            best_score=best,
            passed_threshold=best is not None and best >= self._threshold,
            embedding_tokens=embedded.tokens,
        )
