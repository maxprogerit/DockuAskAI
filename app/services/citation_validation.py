"""Deterministic citation checks. Plain Python on purpose: the LLM must never grade its own homework.

A citation is accepted only if
  1. its chunk id is one of the chunks we actually retrieved for this request, and
  2. its quote really occurs in THAT chunk's text (after whitespace/unicode normalization), and
  3. the quote is long enough to be evidence (a 2-letter "quote" matches everything).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from app.schemas.ask import Citation
from app.schemas.llm import LLMAnswer
from app.services.vector_store import RetrievedChunk

MIN_QUOTE_CHARS = 8

_QUOTE_TRANSLATION = str.maketrans(
    {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "}
)
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """NFKC (also folds the full-width '<' used when escaping prompt delimiters), typographic quotes
    and dashes to ASCII, and runs of whitespace to one space. Case is NOT folded."""
    text = unicodedata.normalize("NFKC", text).translate(_QUOTE_TRANSLATION)
    return _WHITESPACE.sub(" ", text).strip()


@dataclass
class ValidationResult:
    citations: List[Citation] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and bool(self.citations)


def validate_citations(answer: LLMAnswer, chunks_by_alias: Dict[str, RetrievedChunk]) -> ValidationResult:
    result = ValidationResult()
    if not answer.answer.strip():
        result.errors.append("The answer text is empty.")
    if not answer.citations:
        result.errors.append("An answerable response must include at least one citation.")

    seen: Set[Tuple[str, str]] = set()
    for number, cited in enumerate(answer.citations, start=1):
        alias = cited.chunk_id.strip()
        chunk = chunks_by_alias.get(alias)
        if chunk is None:
            result.errors.append(
                "Citation %d: chunk id '%s' is not one of the provided chunks (%s)."
                % (number, cited.chunk_id, ", ".join(chunks_by_alias))
            )
            continue
        quote = normalize(cited.quote)
        if len(quote) < MIN_QUOTE_CHARS:
            result.errors.append("Citation %d: the quote is too short to be evidence." % number)
            continue
        if quote not in normalize(chunk.text):
            result.errors.append(
                "Citation %d: the quote does not occur verbatim in chunk %s. "
                "Copy it exactly, without ellipses or paraphrasing." % (number, alias)
            )
            continue
        key = (chunk.chunk_id, quote)
        if key in seen:
            continue
        seen.add(key)
        result.citations.append(
            Citation(
                document_id=chunk.document_id,
                filename=chunk.filename,
                page=chunk.page,
                section=chunk.section,
                para_index=chunk.para_index,
                chunk_id=chunk.chunk_id,
                quote=quote,
            )
        )
    return result
