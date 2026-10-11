from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from app.services.chunking import Chunk

_WRITE_BATCH = 1000


def open_chroma(path: str) -> Any:
    """Embedded persistent Chroma client. Anonymized telemetry is switched off: this service holds
    other people's documents, and nothing should phone home from the process that touches them."""
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    return chromadb.PersistentClient(path=path, settings=ChromaSettings(anonymized_telemetry=False))


class VectorStoreConfigError(RuntimeError):
    """The Chroma collection exists but was created with settings we cannot use."""


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    text: str
    score: float  # cosine SIMILARITY in [-1, 1]; higher is better
    char_start: int
    char_end: int
    page: Optional[int] = None
    section: Optional[str] = None
    para_index: Optional[int] = None


class VectorStore:
    """Thin wrapper over one Chroma collection (cosine space, vectors supplied by us).

    Score derivation: Chroma's "cosine" space returns DISTANCE d = 1 - cos(a, b), in [0, 2].
    We expose score = 1 - d = cos(a, b), so 1.0 is identical direction and ~0 is unrelated.
    (Verified against the real client in tests/test_vector_store.py.)
    """

    def __init__(self, chroma_client: Any, collection_name: str, embedding_model: str) -> None:
        # embedding_function=None: we always pass vectors explicitly. Without it Chroma would
        # attach its default local ONNX embedder, which could silently embed with another model.
        self._collection = chroma_client.get_or_create_collection(
            name=collection_name,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
            metadata={"embedding_model": embedding_model},
        )
        self._verify(embedding_model)

    def _verify(self, embedding_model: str) -> None:
        # get_or_create_collection on an EXISTING collection silently keeps its old settings,
        # so a collection created earlier with the default L2 space would give wrong scores.
        space = self._collection.configuration["hnsw"]["space"]
        if space != "cosine":
            raise VectorStoreConfigError(
                "Chroma collection '%s' uses distance '%s', expected 'cosine'. "
                "Delete the collection (or CHROMA_DIR) and re-ingest." % (self._collection.name, space)
            )
        stored_model = (self._collection.metadata or {}).get("embedding_model")
        if stored_model != embedding_model:
            raise VectorStoreConfigError(
                "Chroma collection '%s' was built with embedding model '%s' but EMBEDDING_MODEL is '%s'. "
                "Vectors from different models are not comparable; re-ingest with a fresh collection."
                % (self._collection.name, stored_model, embedding_model)
            )

    @property
    def distance_space(self) -> str:
        return str(self._collection.configuration["hnsw"]["space"])

    def count(self) -> int:
        return int(self._collection.count())

    def add_chunks(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        for start in range(0, len(chunks), _WRITE_BATCH):
            part = chunks[start : start + _WRITE_BATCH]
            self._collection.upsert(  # upsert: re-ingesting the same document is idempotent
                ids=[c.chunk_id for c in part],
                embeddings=[list(v) for v in vectors[start : start + _WRITE_BATCH]],
                documents=[c.text for c in part],
                metadatas=[_to_metadata(c) for c in part],
            )

    def query(
        self, vector: Sequence[float], top_k: int, document_ids: Optional[Sequence[str]] = None
    ) -> List[RetrievedChunk]:
        where: Optional[Dict[str, Any]] = {"document_id": {"$in": list(document_ids)}} if document_ids else None
        total = self.count()
        if total == 0:
            return []
        result = self._collection.query(
            query_embeddings=[list(vector)],
            n_results=min(top_k, total),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        retrieved: List[RetrievedChunk] = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=meta["document_id"],
                    filename=meta["filename"],
                    chunk_index=int(meta["chunk_index"]),
                    text=text,
                    score=1.0 - float(distance),
                    char_start=int(meta["char_start"]),
                    char_end=int(meta["char_end"]),
                    page=meta.get("page"),
                    section=meta.get("section"),
                    para_index=meta.get("para_index"),
                )
            )
        return retrieved  # Chroma returns them best-first

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})


def _to_metadata(chunk: Chunk) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "document_id": chunk.document_id,
        "filename": chunk.filename,
        "chunk_index": chunk.chunk_index,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
    }
    # Chroma metadata cannot hold None, so nullable fields are simply omitted.
    if chunk.page is not None:
        meta["page"] = chunk.page
    if chunk.section is not None:
        meta["section"] = chunk.section
    if chunk.para_index is not None:
        meta["para_index"] = chunk.para_index
    return meta
