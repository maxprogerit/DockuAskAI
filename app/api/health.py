from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.deps import get_vector_store
from app.core.database import get_db
from app.schemas.health import HealthResponse
from app.services.vector_store import VectorStore

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db), store: VectorStore = Depends(get_vector_store)) -> JSONResponse:
    """Liveness + dependency reachability. Intentionally unauthenticated (docker healthchecks).

    Returns 503 when PostgreSQL or the vector store is down so orchestrators stop routing here.
    It deliberately does NOT call OpenAI: a health probe must not cost money or depend on a
    third party's uptime.
    """
    database = "ok"
    vector_store = "ok"
    try:
        db.execute(text("SELECT 1"))
    except SQLAlchemyError:
        logger.exception("health check: database unavailable")
        database = "unavailable"
    try:
        store.count()
    except Exception:  # noqa: BLE001 - any failure here means "not healthy", whatever its type
        logger.exception("health check: vector store unavailable")
        vector_store = "unavailable"

    healthy = database == "ok" and vector_store == "ok"
    body = HealthResponse(
        status="ok" if healthy else "degraded", database=database, vector_store=vector_store
    )
    return JSONResponse(status_code=200 if healthy else 503, content=body.model_dump())
