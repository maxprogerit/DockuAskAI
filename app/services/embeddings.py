from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List

import openai

from app.core.errors import UpstreamError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: List[List[float]]
    tokens: int


class Embedder:
    """Batched OpenAI embeddings. `client` is anything with `.embeddings.create(...)` (mockable)."""

    def __init__(self, client: Any, model: str, batch_size: int) -> None:
        self._client = client
        self._model = model
        self._batch_size = batch_size

    @property
    def model(self) -> str:
        return self._model

    def embed(self, texts: List[str]) -> EmbeddingResult:
        vectors: List[List[float]] = []
        tokens = 0
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            try:
                response = self._client.embeddings.create(
                    model=self._model, input=batch, encoding_format="float"
                )
            except openai.OpenAIError as exc:
                # Log the exception TYPE and status only: messages may echo request content.
                logger.error(
                    "embeddings request failed",
                    extra={"error_type": type(exc).__name__, "status": getattr(exc, "status_code", None)},
                )
                raise UpstreamError("The embeddings service is unavailable. Please try again later.") from exc

            data = sorted(response.data, key=lambda item: item.index)
            if len(data) != len(batch):
                raise UpstreamError("The embeddings service returned an unexpected number of vectors.")
            vectors.extend(list(item.embedding) for item in data)
            tokens += response.usage.total_tokens
        return EmbeddingResult(vectors=vectors, tokens=tokens)

    def embed_query(self, text: str) -> EmbeddingResult:
        return self.embed([text])
