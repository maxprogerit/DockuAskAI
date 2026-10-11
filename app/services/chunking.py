from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from app.services.parsing import Segment

# A sentence ends at ., !, ? or … (optionally followed by closing quotes/brackets) + whitespace,
# or at a newline. Abbreviations ("e.g. ", "Dr. ") over-split, which only makes units smaller.
_SENTENCE_END = re.compile(r"[.!?…]+[\"'”’)\]]*\s+|\n+")


@dataclass(frozen=True)
class Chunk:
    document_id: str
    filename: str
    chunk_index: int
    text: str
    char_start: int  # offsets into the segment text (normalized page text / DOCX section text)
    char_end: int
    page: Optional[int] = None  # PDF only
    section: Optional[str] = None  # DOCX heading path
    para_index: Optional[int] = None  # DOCX: 1-based paragraph where the chunk starts

    @property
    def chunk_id(self) -> str:
        return "%s:%d" % (self.document_id, self.chunk_index)

    @property
    def embedding_text(self) -> str:
        """What gets embedded. Heading context is prepended so 'Termination' chunks that never
        repeat the word still match; it is NOT part of `text`, so quotes are validated against
        exactly what the user's document says."""
        return "%s\n%s" % (self.section, self.text) if self.section else self.text


def chunk_segments(
    document_id: str,
    filename: str,
    segments: Sequence[Segment],
    chunk_size: int,
    overlap: int,
) -> List[Chunk]:
    """Fixed-size chunks with overlap that prefer sentence boundaries.

    Guarantees: every chunk is a stripped, non-empty, exact substring of its segment and is at most
    `chunk_size` characters. Chunks end on sentence boundaries unless a single sentence exceeds
    `chunk_size` (then it is split at whitespace). Overlap is made of WHOLE trailing sentences that
    fit in `overlap` characters, so the real overlap can be smaller than configured (or zero if the
    last sentence alone is longer than `overlap`) - this is the price of never cutting mid-sentence.
    """
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("require chunk_size > 0 and 0 <= overlap < chunk_size")

    chunks: List[Chunk] = []
    for segment in segments:
        text = segment.text
        units = _split_units(text, chunk_size)
        para_offsets = [offset for offset, _ in segment.para_starts]

        i = 0
        while i < len(units):
            start = units[i][0]
            j = i
            while j < len(units) and units[j][1] - start <= chunk_size:
                j += 1  # j > i always: every unit is <= chunk_size by construction
            end = units[j - 1][1]

            raw = text[start:end]
            stripped = raw.strip()
            if stripped:
                char_start = start + (len(raw) - len(raw.lstrip()))
                char_end = char_start + len(stripped)
                chunks.append(
                    Chunk(
                        document_id=document_id,
                        filename=filename,
                        chunk_index=len(chunks),
                        text=stripped,
                        char_start=char_start,
                        char_end=char_end,
                        page=segment.page,
                        section=segment.section,
                        para_index=_paragraph_at(segment, para_offsets, char_start),
                    )
                )

            if j >= len(units):
                break
            # Next chunk starts at the earliest unit whose tail still fits into `overlap`,
            # but strictly after unit i so we always make progress.
            next_i = j
            for k in range(i + 1, j):
                if end - units[k][0] <= overlap:
                    next_i = k
                    break
            i = next_i
    return chunks


def _paragraph_at(segment: Segment, offsets: List[int], position: int) -> Optional[int]:
    if not segment.para_starts:
        return None
    index = bisect.bisect_right(offsets, position) - 1
    return segment.para_starts[max(index, 0)][1]


def _split_units(text: str, max_len: int) -> List[Tuple[int, int]]:
    """Contiguous (start, end) spans covering `text`, one per sentence, each <= max_len."""
    spans: List[Tuple[int, int]] = []
    position = 0
    for match in _SENTENCE_END.finditer(text):
        spans.append((position, match.end()))
        position = match.end()
    if position < len(text):
        spans.append((position, len(text)))

    units: List[Tuple[int, int]] = []
    for start, end in spans:
        while end - start > max_len:
            cut = _cut_point(text, start, max_len)
            units.append((start, cut))
            start = cut
        if end > start:
            units.append((start, end))
    return units


def _cut_point(text: str, start: int, max_len: int) -> int:
    """Split an over-long sentence at the last whitespace in its second half, else hard-cut."""
    limit = start + max_len
    index = text.rfind(" ", start + max_len // 2, limit)
    return index + 1 if index != -1 else limit
