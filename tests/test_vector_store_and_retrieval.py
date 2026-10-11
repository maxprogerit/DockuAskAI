from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

import chromadb
import pytest

from app.services.chunking import Chunk
from app.services.embeddings import Embedder
from app.services.retrieval import Retriever
from app.services.vector_store import VectorStore, VectorStoreConfigError
from tests.fakes import FakeEmbeddingsClient, fake_vector

MODEL = "fake-embedding-model"


def _chunk(doc: str, idx: int, text: str, page: int = 1) -> Chunk:
    return Chunk(document_id=doc, filename="%s.pdf" % doc, chunk_index=idx, text=text, char_start=0, char_end=len(text), page=page)


@pytest.fixture
def chroma(tmp_path: Path) -> Iterator[chromadb.ClientAPI]:
    yield chromadb.PersistentClient(path=str(tmp_path / "chroma"))


@pytest.fixture
def populated(chroma: chromadb.ClientAPI):  # type: ignore[no-untyped-def]
    store = VectorStore(chroma, "test_col", MODEL)
    chunks: List[Chunk] = [
        _chunk("docA", 0, "Annual leave is 25 days per calendar year for every full-time employee."),
        _chunk("docA", 1, "Passwords must be at least 14 characters and use multi-factor authentication.", page=2),
        _chunk("docB", 0, "The uptime guarantee is 99.9 percent measured per calendar month."),
    ]
    embedder = Embedder(FakeEmbeddingsClient(), MODEL, batch_size=8)
    store.add_chunks(chunks, embedder.embed([c.text for c in chunks]).vectors)
    return store, embedder


def test_collection_is_cosine_and_score_is_one_minus_distance(chroma: chromadb.ClientAPI) -> None:
    store = VectorStore(chroma, "cos_col", MODEL)
    assert store.distance_space == "cosine"
    a, b = fake_vector("alpha beta gamma"), fake_vector("alpha delta epsilon")
    store.add_chunks([_chunk("d", 0, "alpha beta gamma")], [a])
    hit = store.query(b, top_k=1)[0]
    expected_cosine = sum(x * y for x, y in zip(a, b))  # vectors are unit length
    assert hit.score == pytest.approx(expected_cosine, abs=1e-5)
    identical = store.query(a, top_k=1)[0]
    assert identical.score == pytest.approx(1.0, abs=1e-5)


def test_existing_l2_collection_is_rejected(chroma: chromadb.ClientAPI) -> None:
    chroma.create_collection("old_col", configuration={"hnsw": {"space": "l2"}}, embedding_function=None)
    with pytest.raises(VectorStoreConfigError, match="cosine"):
        VectorStore(chroma, "old_col", MODEL)


def test_embedding_model_mismatch_is_rejected(chroma: chromadb.ClientAPI) -> None:
    VectorStore(chroma, "m_col", "model-a")
    with pytest.raises(VectorStoreConfigError, match="model-a"):
        VectorStore(chroma, "m_col", "model-b")


def test_metadata_roundtrip_and_null_page(chroma: chromadb.ClientAPI) -> None:
    store = VectorStore(chroma, "meta_col", MODEL)
    docx_chunk = Chunk("d", "a.docx", 0, "Section text here.", 3, 21, page=None, section="1. Intro", para_index=4)
    store.add_chunks([docx_chunk], [fake_vector(docx_chunk.text)])
    got = store.query(fake_vector("section text"), top_k=1)[0]
    assert (got.page, got.section, got.para_index, got.char_start, got.char_end) == (None, "1. Intro", 4, 3, 21)
    assert got.chunk_id == "d:0" and got.text == "Section text here."


def test_document_filter_and_delete(populated) -> None:  # type: ignore[no-untyped-def]
    store, embedder = populated
    query = embedder.embed_query("uptime guarantee").vectors[0]
    assert store.query(query, top_k=3)[0].document_id == "docB"
    only_a = store.query(query, top_k=3, document_ids=["docA"])
    assert {c.document_id for c in only_a} == {"docA"}
    store.delete_document("docA")
    assert store.count() == 1
    assert {c.document_id for c in store.query(query, top_k=5)} == {"docB"}


def test_upsert_is_idempotent(populated) -> None:  # type: ignore[no-untyped-def]
    store, embedder = populated
    chunk = _chunk("docB", 0, "The uptime guarantee is 99.9 percent measured per calendar month.")
    store.add_chunks([chunk], embedder.embed([chunk.text]).vectors)
    assert store.count() == 3


def test_retrieval_passes_threshold_for_relevant_question(populated) -> None:  # type: ignore[no-untyped-def]
    store, embedder = populated
    outcome = Retriever(embedder, store, top_k=3, score_threshold=0.3).retrieve("How many days of annual leave?")
    assert outcome.passed_threshold
    assert outcome.chunks[0].chunk_id == "docA:0"
    assert outcome.best_score == outcome.chunks[0].score
    assert outcome.embedding_tokens > 0


def test_retrieval_below_threshold_for_unrelated_question(populated) -> None:  # type: ignore[no-untyped-def]
    store, embedder = populated
    outcome = Retriever(embedder, store, top_k=3, score_threshold=0.3).retrieve("zebra quantum saxophone")
    assert not outcome.passed_threshold
    assert outcome.best_score is not None and outcome.best_score < 0.3


def test_retrieval_on_empty_store(chroma: chromadb.ClientAPI) -> None:
    store = VectorStore(chroma, "empty_col", MODEL)
    outcome = Retriever(Embedder(FakeEmbeddingsClient(), MODEL, 8), store, 3, 0.3).retrieve("anything")
    assert outcome.chunks == [] and outcome.best_score is None and not outcome.passed_threshold


def test_top_k_override_and_document_filter(populated) -> None:  # type: ignore[no-untyped-def]
    store, embedder = populated
    retriever = Retriever(embedder, store, top_k=3, score_threshold=0.0)
    assert len(retriever.retrieve("leave", top_k=1).chunks) == 1
    assert {c.document_id for c in retriever.retrieve("leave", document_ids=["docB"]).chunks} == {"docB"}
