from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    file_type: str  # "pdf" | "docx"
    page_count: Optional[int]  # null for DOCX (no stable pages)
    chunk_count: int
    status: str
    created_at: datetime


class UploadResponse(DocumentOut):
    # True when the same bytes were uploaded before: the existing record is returned and the
    # embeddings API is not called again.
    already_processed: bool = False
