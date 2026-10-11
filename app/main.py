from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import ask, documents, health
from app.api.deps import get_vector_store
from app.core.config import get_settings
from app.core.errors import DocuAskError
from app.core.logging import request_context_middleware, setup_logging
from app.services.vector_store import VectorStoreConfigError

logger = logging.getLogger(__name__)

# multipart framing adds a little to the file itself
_MULTIPART_OVERHEAD_BYTES = 64 * 1024


def _error(request: Request, status_code: int, code: str, message: str, **extra: Any) -> JSONResponse:
    """The single error shape of this API: {"error": {"code", "message"}, "request_id": ...}."""
    body: Dict[str, Any] = {
        "error": {"code": code, "message": message, **extra},
        "request_id": getattr(request.state, "request_id", "-"),
    }
    return JSONResponse(status_code=status_code, content=body)


async def reject_oversized_upload(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Cheap early 413 from Content-Length, before the multipart body is parsed and spooled.

    Not a security boundary (the header can be absent/chunked; the service re-checks the real size),
    and a reverse proxy should enforce its own body limit too (nginx client_max_body_size).
    """
    if request.method == "POST" and request.url.path == "/documents":
        declared = request.headers.get("content-length", "")
        limit = get_settings().max_upload_bytes + _MULTIPART_OVERHEAD_BYTES
        if declared.isdigit() and int(declared) > limit:
            return _error(request, 413, "file_too_large", "File is larger than the limit of %d MB." % get_settings().max_upload_mb)
    return await call_next(request)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    get_vector_store()  # fail fast if the Chroma collection is not cosine / built with another model
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level)

    app = FastAPI(
        title="DocuAsk",
        version="0.1.0",
        description="Upload PDF/DOCX documents and ask questions. Answers are grounded only in the "
        "documents, with verified citations - or an honest refusal.",
        lifespan=lifespan,
    )
    # Starlette runs the LAST registered middleware first: request context must be outermost.
    app.middleware("http")(reject_oversized_upload)
    app.middleware("http")(request_context_middleware)

    @app.exception_handler(DocuAskError)
    async def handle_domain_error(request: Request, exc: DocuAskError) -> JSONResponse:
        return _error(request, exc.status_code, exc.code, exc.message)

    @app.exception_handler(VectorStoreConfigError)
    async def handle_vector_config(request: Request, exc: VectorStoreConfigError) -> JSONResponse:
        logger.error("vector store misconfigured: %s", exc)
        return _error(request, 503, "vector_store_misconfigured", str(exc))

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        codes = {401: "unauthorized", 404: "not_found", 405: "method_not_allowed"}
        return _error(request, exc.status_code, codes.get(exc.status_code, "http_error"), str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Field locations and messages only: never echo the submitted values back.
        problems: List[Dict[str, str]] = [
            {"field": ".".join(str(p) for p in err["loc"]), "message": err["msg"]} for err in exc.errors()
        ]
        return _error(request, 422, "validation_error", "Invalid request.", problems=problems)

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error")  # stack trace goes to the log, never to the client
        return _error(request, 500, "internal_error", "Internal server error.")

    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(ask.router)
    return app


app = create_app()
