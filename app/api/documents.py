from __future__ import annotations

import uuid
from typing import List

from fastapi import APIRouter, Depends, File, Response, UploadFile
from fastapi.responses import FileResponse

from app.api.deps import get_document_service
from app.core.config import Settings, get_settings
from app.core.security import require_api_key
from app.schemas.documents import DocumentOut, UploadResponse
from app.services.documents import MEDIA_TYPES, DocumentService

router = APIRouter(prefix="/documents", tags=["documents"], dependencies=[Depends(require_api_key)])


@router.post("", response_model=UploadResponse, status_code=201)
def upload_document(
    response: Response,
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service),
    settings: Settings = Depends(get_settings),
) -> UploadResponse:
    """Parse, chunk, embed and index a PDF or DOCX.

    201 = newly ingested. 200 with `already_processed=true` = identical bytes were uploaded before,
    the existing record is returned and nothing is re-embedded.
    """
    # Read at most limit+1 bytes: enough to detect "too large" without buffering an unbounded body.
    data = file.file.read(settings.max_upload_bytes + 1)
    document, created = service.upload(file.filename or "document", data)
    if not created:
        response.status_code = 200
    return UploadResponse.model_validate(document, from_attributes=True).model_copy(
        update={"already_processed": not created}
    )


@router.get("", response_model=List[DocumentOut])
def list_documents(service: DocumentService = Depends(get_document_service)) -> List[DocumentOut]:
    return [DocumentOut.model_validate(d) for d in service.list_documents()]


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: uuid.UUID, service: DocumentService = Depends(get_document_service)) -> DocumentOut:
    return DocumentOut.model_validate(service.get(document_id))


@router.get("/{document_id}/file")
def get_document_file(document_id: uuid.UUID, service: DocumentService = Depends(get_document_service)) -> FileResponse:
    """The original upload, for in-browser preview. The path comes from our DB row (never from the
    request) and is re-checked to be inside UPLOAD_DIR."""
    document = service.get(document_id)
    return FileResponse(
        service.file_for(document),
        media_type=MEDIA_TYPES[document.file_type],
        filename=document.filename,
        content_disposition_type="inline",
        headers={"X-Content-Type-Options": "nosniff"},
    )


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: uuid.UUID, service: DocumentService = Depends(get_document_service)) -> Response:
    service.delete(document_id)
    return Response(status_code=204)
