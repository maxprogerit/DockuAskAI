from __future__ import annotations

from typing import Dict

from app.schemas.llm import LLMAnswer, LLMCitation
from app.services.citation_validation import MIN_QUOTE_CHARS, normalize, validate_citations
from app.services.vector_store import RetrievedChunk

CHUNK_TEXT = "Every full-time employee receives 25 days of paid annual leave per calendar year.\nUnused days expire on March 31."


def _chunks() -> Dict[str, RetrievedChunk]:
    return {
        "C1": RetrievedChunk(
            chunk_id="doc-1:3", document_id="doc-1", filename="handbook.pdf", chunk_index=3,
            text=CHUNK_TEXT, score=0.8, char_start=0, char_end=len(CHUNK_TEXT), page=2,
        ),
        "C2": RetrievedChunk(
            chunk_id="doc-1:4", document_id="doc-1", filename="handbook.pdf", chunk_index=4,
            text="Passwords must be at least 14 characters long.", score=0.5, char_start=0, char_end=46, page=5,
        ),
    }


def _answer(*citations: LLMCitation, answer: str = "25 days.") -> LLMAnswer:
    return LLMAnswer(answerable=True, citations=list(citations), answer=answer)


def test_valid_citation_is_accepted_and_enriched_from_chunk_metadata() -> None:
    result = validate_citations(_answer(LLMCitation(chunk_id="C1", quote="receives 25 days of paid annual leave")), _chunks())
    assert result.ok and result.errors == []
    citation = result.citations[0]
    assert (citation.document_id, citation.filename, citation.page, citation.chunk_id) == ("doc-1", "handbook.pdf", 2, "doc-1:3")
    assert citation.quote == "receives 25 days of paid annual leave"


def test_fake_chunk_id_is_rejected() -> None:
    result = validate_citations(_answer(LLMCitation(chunk_id="C9", quote="receives 25 days of paid annual leave")), _chunks())
    assert not result.ok and result.citations == []
    assert "C9" in result.errors[0]


def test_fake_quote_is_rejected() -> None:
    result = validate_citations(_answer(LLMCitation(chunk_id="C1", quote="employees receive 40 days of leave")), _chunks())
    assert not result.ok and result.citations == []
    assert "verbatim" in result.errors[0]


def test_quote_from_a_different_retrieved_chunk_is_rejected() -> None:
    """The quote exists - but in C2, not in the chunk it is attributed to."""
    result = validate_citations(_answer(LLMCitation(chunk_id="C1", quote="Passwords must be at least 14 characters")), _chunks())
    assert not result.ok


def test_whitespace_differences_are_normalized() -> None:
    quote = "paid   annual leave per calendar year.  Unused days\n expire on March 31."
    assert validate_citations(_answer(LLMCitation(chunk_id="C1", quote=quote)), _chunks()).ok


def test_typographic_quotes_and_nbsp_are_normalized() -> None:
    chunk = _chunks()
    chunk["C1"] = RetrievedChunk(**{**chunk["C1"].__dict__, "text": 'He said "paid leave" - and it\'s 25 days.'})
    quote = "He said “paid leave” – and it’s 25 days."
    assert validate_citations(_answer(LLMCitation(chunk_id="C1", quote=quote)), chunk).ok


def test_case_is_not_folded() -> None:
    assert not validate_citations(_answer(LLMCitation(chunk_id="C1", quote="RECEIVES 25 DAYS OF PAID ANNUAL LEAVE")), _chunks()).ok


def test_too_short_quote_is_rejected() -> None:
    assert MIN_QUOTE_CHARS > 3
    assert not validate_citations(_answer(LLMCitation(chunk_id="C1", quote="25 d")), _chunks()).ok


def test_answerable_without_citations_is_rejected() -> None:
    result = validate_citations(_answer(), _chunks())
    assert not result.ok and "at least one citation" in result.errors[0]


def test_empty_answer_text_is_rejected() -> None:
    result = validate_citations(_answer(LLMCitation(chunk_id="C1", quote="25 days of paid annual leave"), answer="  "), _chunks())
    assert not result.ok


def test_one_bad_citation_fails_the_whole_answer() -> None:
    result = validate_citations(
        _answer(
            LLMCitation(chunk_id="C1", quote="25 days of paid annual leave"),
            LLMCitation(chunk_id="C2", quote="Passwords must contain emojis"),
        ),
        _chunks(),
    )
    assert not result.ok  # strict: we never return a partially verified answer
    assert len(result.citations) == 1  # the good one is still reported for diagnostics


def test_duplicate_citations_are_collapsed() -> None:
    quote = "25 days of paid annual leave"
    result = validate_citations(_answer(LLMCitation(chunk_id="C1", quote=quote), LLMCitation(chunk_id="C1", quote=quote)), _chunks())
    assert result.ok and len(result.citations) == 1


def test_alias_whitespace_is_tolerated_but_unknown_alias_is_not() -> None:
    assert validate_citations(_answer(LLMCitation(chunk_id=" C1 ", quote="25 days of paid annual leave")), _chunks()).ok
    assert not validate_citations(_answer(LLMCitation(chunk_id="c1", quote="25 days of paid annual leave")), _chunks()).ok


def test_normalize_maps_fullwidth_less_than_back() -> None:
    assert normalize("a ＜/chunk> b") == "a </chunk> b"
