from __future__ import annotations

import re
from typing import List

import pytest

from app.services.chunking import Chunk, chunk_segments
from app.services.parsing import Segment

TEXT = (
    "Alpha is the first sentence of the document. Beta follows right after it and is a little longer. "
    "Gamma closes the first paragraph! Is delta a question? Yes, it is. Epsilon adds more words to the text. "
    "Zeta and eta are short. Theta is the last sentence of this synthetic sample."
)


def _chunk(text: str = TEXT, size: int = 120, overlap: int = 40, **segment_kwargs: object) -> List[Chunk]:
    return chunk_segments("doc-1", "f.pdf", [Segment(text=text, **segment_kwargs)], size, overlap)  # type: ignore[arg-type]


@pytest.mark.parametrize("size,overlap", [(60, 0), (120, 40), (200, 50), (500, 100)])
def test_size_limit_and_no_empty_chunks(size: int, overlap: int) -> None:
    chunks = _chunk(size=size, overlap=overlap)
    assert chunks
    for chunk in chunks:
        assert chunk.text.strip() == chunk.text
        assert 0 < len(chunk.text) <= size


def test_chunk_text_is_exact_substring_with_correct_char_range() -> None:
    for chunk in _chunk():
        assert TEXT[chunk.char_start : chunk.char_end] == chunk.text


def test_chunks_end_on_sentence_boundaries_when_sentences_fit() -> None:
    for chunk in _chunk(size=150, overlap=0):
        assert re.search(r"[.!?]$", chunk.text), chunk.text


def test_consecutive_chunks_overlap_with_whole_sentences() -> None:
    chunks = _chunk(size=120, overlap=60)
    overlaps = [a.char_end > b.char_start for a, b in zip(chunks, chunks[1:])]
    assert any(overlaps)
    for a, b in zip(chunks, chunks[1:]):
        assert b.char_start > a.char_start  # always progresses
        if a.char_end > b.char_start:
            assert a.char_end - b.char_start <= 60  # never more than the configured overlap


def test_zero_overlap_means_disjoint_chunks() -> None:
    chunks = _chunk(size=120, overlap=0)
    for a, b in zip(chunks, chunks[1:]):
        assert a.char_end <= b.char_start


def test_every_character_is_covered() -> None:
    covered = set()
    for chunk in _chunk(size=100, overlap=30):
        covered.update(range(chunk.char_start, chunk.char_end))
    assert {i for i, ch in enumerate(TEXT) if not ch.isspace()} <= covered


def test_long_sentence_without_boundaries_is_split_within_limit() -> None:
    text = "word " * 200  # one 1000-char "sentence"
    chunks = _chunk(text=text.strip(), size=100, overlap=20)
    assert len(chunks) >= 10
    assert all(len(c.text) <= 100 for c in chunks)
    assert all(not c.text.startswith(" ") for c in chunks)


def test_unbroken_string_is_hard_cut() -> None:
    chunks = _chunk(text="x" * 450, size=100, overlap=0)
    assert [len(c.text) for c in chunks] == [100, 100, 100, 100, 50]


def test_metadata_ids_and_indexes() -> None:
    segments = [Segment(text=TEXT, page=1), Segment(text=TEXT, page=2)]
    chunks = chunk_segments("doc-1", "f.pdf", segments, 120, 40)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].chunk_id == "doc-1:0"
    assert {c.page for c in chunks} == {1, 2}
    assert all(c.document_id == "doc-1" and c.filename == "f.pdf" for c in chunks)
    # chunks never straddle pages: page 1 chunks all come before page 2 chunks
    pages = [c.page for c in chunks]
    assert pages == sorted(pages)


def test_docx_section_and_paragraph_index() -> None:
    paragraphs = [(5, "First paragraph text."), (6, "Second one is short."), (7, "Third paragraph here.")]
    starts, offset = [], 0
    for number, paragraph in paragraphs:
        starts.append((offset, number))
        offset += len(paragraph) + 1
    text = "\n".join(p for _, p in paragraphs)
    segment = Segment(text=text, section="1. Intro > 1.1 Details", para_starts=tuple(starts))
    chunks = chunk_segments("d", "a.docx", [segment], 25, 0)
    assert all(c.section == "1. Intro > 1.1 Details" and c.page is None for c in chunks)
    # para_index is the paragraph in which the chunk STARTS
    assert [c.para_index for c in chunks] == [5, 6, 7]
    assert chunks[0].embedding_text.startswith("1. Intro > 1.1 Details\n")
    assert not chunks[0].text.startswith("1. Intro")  # heading is in embedding text only


def test_whitespace_only_segment_yields_nothing() -> None:
    assert chunk_segments("d", "f", [Segment(text="   \n  ")], 100, 10) == []


def test_invalid_parameters_rejected() -> None:
    with pytest.raises(ValueError):
        chunk_segments("d", "f", [Segment(text="abc")], 100, 100)
    with pytest.raises(ValueError):
        chunk_segments("d", "f", [Segment(text="abc")], 0, 0)
