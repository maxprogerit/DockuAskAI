from __future__ import annotations

import os

# Must be set before any `app` import: Settings() is built on first use and has no defaults
# for required values. Tests never touch a real database or OpenAI.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # Chroma must not phone home from tests
os.environ.setdefault("API_KEY", "test-api-key-0123456789")
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://test:test@localhost:5432/test")

from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, Iterator  # noqa: E402

import chromadb  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import app.models  # noqa: E402,F401  (registers tables)
from app.api.deps import get_answer_service, get_embedder, get_vector_store  # noqa: E402
from app.core.config import Settings, get_settings  # noqa: E402
from app.core.database import Base, get_db  # noqa: E402
from app.main import app  # noqa: E402
from app.services.answering import AnswerService  # noqa: E402
from app.services.embeddings import Embedder  # noqa: E402
from app.services.retrieval import Retriever  # noqa: E402
from app.services.vector_store import VectorStore  # noqa: E402
from tests.fakes import FakeEmbeddingsClient, FakeLLMClient  # noqa: E402

TEST_API_KEY = "test-api-key-0123456789"
EMBEDDING_MODEL = "fake-embedding-model"


@dataclass
class Env:
    client: TestClient
    settings: Settings
    embeddings: FakeEmbeddingsClient
    llm: FakeLLMClient
    store: VectorStore
    sessions: "sessionmaker[Session]"
    headers: Dict[str, str]


@pytest.fixture
def env(tmp_path: Path) -> Iterator[Env]:
    """The whole app wired to in-memory SQLite, a temp Chroma dir and scripted OpenAI fakes.

    SQLite instead of PostgreSQL keeps the suite hermetic; the real schema/migration is verified
    separately against PostgreSQL (alembic upgrade/check in the README's verification steps).
    """
    settings = Settings(
        api_key=TEST_API_KEY,  # type: ignore[arg-type]
        database_url="sqlite://",
        upload_dir=tmp_path / "uploads",
        chroma_dir=tmp_path / "chroma",
        max_upload_mb=1,
        max_pages=50,
        chunk_size=500,
        chunk_overlap=100,
        top_k=4,
        score_threshold=0.2,
        openai_api_key=None,
    )
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)  # tests only; production schema comes from Alembic
    sessions = sessionmaker(bind=engine, expire_on_commit=False)

    embeddings = FakeEmbeddingsClient()
    llm = FakeLLMClient([])
    embedder = Embedder(embeddings, EMBEDDING_MODEL, 32)
    store = VectorStore(chromadb.PersistentClient(path=str(settings.chroma_dir)), "test_col", EMBEDDING_MODEL)

    def override_db() -> Iterator[Session]:
        session = sessions()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_vector_store] = lambda: store
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_answer_service] = lambda: AnswerService(
        llm, Retriever(embedder, store, settings.top_k, settings.score_threshold), "gpt-test", "low", 500
    )
    # raise_server_exceptions=False: we want to assert on the 500 response the client really gets.
    client = TestClient(app, raise_server_exceptions=False)
    yield Env(client, settings, embeddings, llm, store, sessions, {"X-API-Key": TEST_API_KEY})
    app.dependency_overrides.clear()
    engine.dispose()
