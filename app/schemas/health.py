from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    database: str  # "ok" | "unavailable"
    vector_store: str  # "ok" | "unavailable"
