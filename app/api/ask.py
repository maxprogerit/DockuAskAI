from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_answer_service, get_document_repository, get_query_repository
from app.core.errors import DocumentNotFoundError
from app.core.security import require_api_key
from app.repositories.documents import DocumentRepository
from app.repositories.queries import QueryRepository
from app.schemas.ask import AskRequest, AskResponse
from app.services.answering import AnswerService

router = APIRouter(tags=["ask"], dependencies=[Depends(require_api_key)])


@router.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    answers: AnswerService = Depends(get_answer_service),
    documents: DocumentRepository = Depends(get_document_repository),
    queries: QueryRepository = Depends(get_query_repository),
) -> AskResponse:
    """Answer strictly from the indexed documents, with verified citations - or refuse."""
    document_ids = None
    if request.document_ids:
        wanted = set(request.document_ids)
        found = {d.id for d in documents.get_many(wanted)}
        if found != wanted:
            raise DocumentNotFoundError("Unknown document id(s): %s" % ", ".join(sorted(str(i) for i in wanted - found)))
        document_ids = [str(i) for i in wanted]

    response = answers.ask(request.question, request.top_k, document_ids)
    queries.log(request.question, response)
    return response
