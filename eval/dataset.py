from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_DATASET = Path(__file__).parent / "dataset.jsonl"
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "tests" / "samples"

ANSWERABLE = "answerable"
UNANSWERABLE = "unanswerable"
MULTI_CHUNK = "multi_chunk"
TYPES = (ANSWERABLE, UNANSWERABLE, MULTI_CHUNK)


@dataclass(frozen=True)
class Evidence:
    """A verbatim fragment of the source that proves (part of) the answer.

    Retrieval relevance is judged by TEXT, not by chunk id: a chunk is relevant if it comes from
    `document` and contains `text`. That keeps the ground truth valid when chunk_size changes.
    """

    document: str
    text: str
    page: Optional[int] = None  # PDF page; None for DOCX
    section: Optional[str] = None  # DOCX heading (used only to verify the dataset)


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    language: str
    type: str
    expected_answer: str
    expected_document: Optional[str]
    expected_page: Optional[int]
    evidence: List[Evidence] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    @property
    def is_answerable(self) -> bool:
        return self.type != UNANSWERABLE


def load_dataset(path: Path = DEFAULT_DATASET) -> List[Question]:
    questions: List[Question] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw: Dict[str, Any] = json.loads(line)
            questions.append(
                Question(
                    id=raw["id"],
                    question=raw["question"],
                    language=raw["language"],
                    type=raw["type"],
                    expected_answer=raw["expected_answer"],
                    expected_document=raw.get("expected_document"),
                    expected_page=raw.get("expected_page"),
                    evidence=[Evidence(**e) for e in raw.get("evidence", [])],
                    tags=list(raw.get("tags", [])),
                )
            )
    return questions
