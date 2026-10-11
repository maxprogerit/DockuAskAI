from __future__ import annotations

from typing import List, Tuple

from app.services.chunking import Chunk, chunk_segments
from app.services.embeddings import Embedder
from app.services.file_validation import FileType
from app.services.parsing import ParsedDocument, parse_document
from app.services.vector_store import VectorStore


def build_chunks(
    document_id: str,
    filename: str,
    data: bytes,
    file_type: FileType,
    max_pages: int,
    chunk_size: int,
    overlap: int,
) -> Tuple[ParsedDocument, List[Chunk]]:
    """Pure CPU work (no network, no DB): parse -> chunk. Raises the 422-type errors."""
    parsed = parse_document(data, file_type, max_pages)
    chunks = chunk_segments(document_id, filename, parsed.segments, chunk_size, overlap)
    return parsed, chunks


def embed_and_store(chunks: List[Chunk], embedder: Embedder, store: VectorStore) -> int:
    """Embed all chunks (batched) and upsert them into Chroma. Returns embedding tokens used."""
    result = embedder.embed([c.embedding_text for c in chunks])
    store.add_chunks(chunks, result.vectors)
    return result.tokens
