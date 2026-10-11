from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.query import Query
from app.schemas.ask import AskResponse


class QueryRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def log(self, question: str, response: AskResponse) -> Query:
        """Persist metrics only: no answer text, no document text."""
        row = Query(
            question=question,
            answerable=response.answerable,
            retrieved_count=len(response.retrieved),
            model_name=response.usage.model,  # NULL when the LLM was not called
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            latency_ms=response.usage.latency_ms,
        )
        self._session.add(row)
        self._session.commit()
        return row
