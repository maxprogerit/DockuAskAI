from __future__ import annotations

from typing import List

from pydantic import BaseModel

# Schema the LLM must fill (OpenAI structured outputs). Every field is required and nothing is
# Optional: strict schemas are simplest that way. Only what the model genuinely has to decide is
# here - document_id / filename / page come from OUR chunk metadata after validation, never from
# the model, so it cannot invent them.
#
# Field order matters: the model emits citations before the answer, which nudges it to ground the
# answer in quotes instead of writing first and rationalizing later.


class LLMCitation(BaseModel):
    chunk_id: str  # alias from the prompt, e.g. "C2"
    quote: str  # verbatim excerpt from that chunk


class LLMAnswer(BaseModel):
    answerable: bool
    citations: List[LLMCitation]
    answer: str
