from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from app.core.errors import NoExtractableTextError, TooManyPagesError, UnreadableDocumentError
from app.services.file_validation import FileType

# Below this many characters in the whole document we treat it as "no text layer" (a scan).
MIN_TEXT_CHARS = 20

_HEADING_STYLE = re.compile(r"^Heading\s+(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class Segment:
    """A run of text the chunker may split freely, but never merge with its neighbours.

    PDF: one segment per page, so a chunk never straddles pages and `page` is always exact.
    DOCX: one segment per heading section; `page` is None because DOCX has no stable pages.
    """

    text: str
    page: Optional[int] = None
    section: Optional[str] = None
    # DOCX only: (char offset in `text`, 1-based paragraph number) for each paragraph start.
    para_starts: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ParsedDocument:
    file_type: FileType
    page_count: Optional[int]  # None for DOCX
    segments: List[Segment]

    @property
    def total_chars(self) -> int:
        return sum(len(s.text) for s in self.segments)


def parse_document(data: bytes, file_type: FileType, max_pages: int) -> ParsedDocument:
    parsed = _parse_pdf(data, max_pages) if file_type == FileType.PDF else _parse_docx(data)
    if parsed.total_chars < MIN_TEXT_CHARS:
        raise NoExtractableTextError(
            "No extractable text found. Scanned documents are not supported (OCR is out of scope)."
        )
    return parsed


def _parse_pdf(data: bytes, max_pages: int) -> ParsedDocument:
    import fitz  # PyMuPDF; imported lazily so unit tests of other modules stay light

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several unrelated types for corrupt input
        raise UnreadableDocumentError("The PDF could not be opened (corrupt file?).") from exc

    try:
        if doc.needs_pass:
            raise UnreadableDocumentError("The PDF is password-protected.")
        if doc.page_count > max_pages:
            raise TooManyPagesError(
                "The PDF has %d pages; the limit is %d." % (doc.page_count, max_pages)
            )
        segments: List[Segment] = []
        for index in range(doc.page_count):
            try:
                raw = doc[index].get_text("text")
            except Exception as exc:
                raise UnreadableDocumentError("Page %d could not be read." % (index + 1)) from exc
            # PDF line breaks are layout, not structure: collapse all whitespace so sentences
            # that wrap across lines stay whole. char_range refers to THIS normalized page text.
            text = " ".join(raw.split())
            if text:
                segments.append(Segment(text=text, page=index + 1))
        return ParsedDocument(FileType.PDF, doc.page_count, segments)
    finally:
        doc.close()


def _parse_docx(data: bytes) -> ParsedDocument:
    from docx import Document as open_docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        document = open_docx(io.BytesIO(data))
    except Exception as exc:
        raise UnreadableDocumentError("The DOCX could not be opened (corrupt file?).") from exc

    segments: List[Segment] = []
    heading_stack: List[str] = []
    paragraphs: List[Tuple[int, str]] = []  # (paragraph number, text) of the current section
    counter = 0
    has_title = False

    def flush() -> None:
        if not paragraphs:
            return
        text_parts: List[str] = []
        starts: List[Tuple[int, int]] = []
        offset = 0
        for number, text in paragraphs:
            starts.append((offset, number))
            text_parts.append(text)
            offset += len(text) + 1  # +1 for the "\n" joiner
        section = " > ".join(heading_stack) if heading_stack else None
        segments.append(Segment(text="\n".join(text_parts), section=section, para_starts=tuple(starts)))
        paragraphs.clear()

    try:
        for child in document.element.body.iterchildren():
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "p":
                paragraph = Paragraph(child, document)
                text = " ".join(paragraph.text.split())
                if not text:
                    continue
                counter += 1
                style = paragraph.style.name if paragraph.style is not None else ""
                level = _heading_level(style)
                if level is not None:
                    flush()
                    if style.lower() == "title":
                        heading_stack[:] = [text]  # the document title is the root of every path
                        has_title = True
                    else:
                        # Heading 1 nests under the title when there is one.
                        del heading_stack[level - 1 + (1 if has_title else 0) :]
                        heading_stack.append(text)
                else:
                    paragraphs.append((counter, text))
            elif tag == "tbl":
                # Tables are flattened to one line per row: "cell | cell | cell".
                # python-docx returns the SAME cell object for every grid position a merged cell
                # covers (horizontal gridSpan and vertical vMerge). Keep one per underlying cell,
                # across rows too, or a vertically merged column is indexed once per row.
                seen_cells = set()
                for row in Table(child, document).rows:
                    cells = []
                    for cell in row.cells:
                        if cell._tc in seen_cells:
                            continue
                        seen_cells.add(cell._tc)
                        cell_text = " ".join(cell.text.split())
                        if cell_text:
                            cells.append(cell_text)
                    if cells:
                        counter += 1
                        paragraphs.append((counter, " | ".join(cells)))
        flush()
    except Exception as exc:
        raise UnreadableDocumentError("The DOCX structure could not be read.") from exc

    return ParsedDocument(FileType.DOCX, None, segments)


def _heading_level(style_name: str) -> Optional[int]:
    if style_name.lower() == "title":
        return 0
    match = _HEADING_STYLE.match(style_name)
    if match:
        return max(1, int(match.group(1)))
    return None
