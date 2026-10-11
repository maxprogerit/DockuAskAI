from __future__ import annotations

import uuid
from typing import Iterable, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.document import Document


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, document_id: uuid.UUID) -> Optional[Document]:
        return self._session.get(Document, document_id)

    def get_many(self, ids: Iterable[uuid.UUID]) -> List[Document]:
        return list(self._session.scalars(select(Document).where(Document.id.in_(list(ids)))))

    def get_by_hash(self, file_hash: str) -> Optional[Document]:
        return self._session.scalars(select(Document).where(Document.file_hash == file_hash)).first()

    def list_all(self) -> List[Document]:
        return list(self._session.scalars(select(Document).order_by(Document.created_at.desc())))

    def add(self, document: Document) -> None:
        self._session.add(document)
        self._session.commit()  # may raise IntegrityError on a duplicate file_hash

    def commit(self) -> None:
        self._session.commit()

    def rollback(self) -> None:
        self._session.rollback()

    def delete(self, document: Document) -> None:
        self._session.delete(document)
        self._session.commit()
