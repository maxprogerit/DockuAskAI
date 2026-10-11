from __future__ import annotations

from functools import lru_cache

import chromadb
from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.database import get_db
from app.core.openai_client import LazyOpenAI
from app.repositories.documents import DocumentRepository
from app.repositories.queries import QueryRepository
from app.services.answering import AnswerService
from app.services.documents import DocumentService
from app.services.embeddings import Embedder
from app.services.retrieval import Retriever
from app.services.vector_store import VectorStore, open_chroma

# Wiring lives here (like a Spring @Configuration class). Every provider is a FastAPI dependency,
# so tests replace exactly the piece they need with `app.dependency_overrides[...]`.


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.ClientAPI:
    settings = get_settings()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return open_chroma(str(settings.chroma_dir))


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    settings = get_settings()
    return VectorStore(get_chroma_client(), settings.chroma_collection, settings.embedding_model)


def get_embedder(settings: Settings = Depends(get_settings)) -> Embedder:
    return Embedder(LazyOpenAI(), settings.embedding_model, settings.embedding_batch_size)


def get_document_service(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    embedder: Embedder = Depends(get_embedder),
    store: VectorStore = Depends(get_vector_store),
) -> DocumentService:
    return DocumentService(DocumentRepository(db), settings, embedder, store)


def get_document_repository(db: Session = Depends(get_db)) -> DocumentRepository:
    return DocumentRepository(db)


def get_query_repository(db: Session = Depends(get_db)) -> QueryRepository:
    return QueryRepository(db)


def get_answer_service(
    settings: Settings = Depends(get_settings),
    embedder: Embedder = Depends(get_embedder),
    store: VectorStore = Depends(get_vector_store),
) -> AnswerService:
    retriever = Retriever(embedder, store, settings.top_k, settings.score_threshold)
    return AnswerService(
        LazyOpenAI(),
        retriever,
        settings.openai_model,
        settings.openai_reasoning_effort,
        settings.answer_max_output_tokens,
    )
