from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

# Why an answer is not an answer. None when answerable=True.
#   no_documents               - the library (or the selected subset) is empty
#   below_score_threshold      - best chunk too dissimilar; the LLM was NOT called
#   no_answer_in_documents     - the LLM looked at the excerpts and said they do not contain it
#   citation_validation_failed - the LLM answered but its citations did not verify after one retry
#   llm_refused                - the model produced no parsable output (safety refusal)
REASONS = (
    "no_documents",
    "below_score_threshold",
    "no_answer_in_documents",
    "citation_validation_failed",
    "llm_refused",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_ids: Optional[List[UUID]] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class Citation(BaseModel):
    document_id: str
    filename: str
    page: Optional[int] = None  # PDF only; DOCX has no pages
    section: Optional[str] = None  # DOCX heading path
    para_index: Optional[int] = None  # DOCX paragraph number where the chunk starts
    chunk_id: str
    quote: str  # verified to occur in the chunk text


class RetrievedItem(BaseModel):
    chunk_id: str
    score: float  # cosine similarity
    document_id: str
    filename: str
    page: Optional[int] = None
    section: Optional[str] = None
    text: str


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    embedding_tokens: int = 0
    llm_calls: int = 0  # 0 = refused before the LLM; 2 = citation retry happened
    model: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    answerable: bool
    reason: Optional[str] = None
    citations: List[Citation] = Field(default_factory=list)
    retrieved: List[RetrievedItem] = Field(default_factory=list)
    usage: Usage
