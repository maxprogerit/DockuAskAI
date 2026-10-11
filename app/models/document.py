from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Enum, Integer, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class DocumentStatus(str, enum.Enum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String(255))
    # Unique: the same bytes uploaded twice must map to one document (no second embeddings call).
    file_hash: Mapped[str] = mapped_column(String(64), unique=True)
    # Relative to UPLOAD_DIR, never user-controlled (we store as <id>.<ext>), so no path traversal.
    file_path: Mapped[str] = mapped_column(Text)
    # NULL for DOCX: the format has no stable page concept.
    page_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(
            DocumentStatus,
            native_enum=False,
            length=16,
            create_constraint=True,
            # store "ready", not the member name "READY"
            values_callable=lambda e: [m.value for m in e],
        ),
        default=DocumentStatus.PROCESSING,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    @property
    def file_type(self) -> str:
        return self.file_path.rsplit(".", 1)[-1].lower()  # stored as <uuid>.<ext>, ext validated at upload
