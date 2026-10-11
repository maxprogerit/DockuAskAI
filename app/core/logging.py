from __future__ import annotations

import json
import logging
import re
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any, Awaitable, Callable, Dict

from fastapi import Request, Response

# Request id is stored in a ContextVar so any log line emitted while handling a request
# (services, repositories) carries it without passing it through every function signature.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

# Client-supplied ids are accepted only in this shape; anything else is replaced.
# Otherwise a client could inject newlines/arbitrary text into our logs.
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Attributes every LogRecord has; everything else was passed via `extra=` and goes into the JSON.
_STANDARD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id_var.get(),
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    # uvicorn's own access log would duplicate ours (and lacks request_id).
    logging.getLogger("uvicorn.access").disabled = True


_access_logger = logging.getLogger("app.access")


async def request_context_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    incoming = request.headers.get("x-request-id", "")
    request_id = incoming if _SAFE_REQUEST_ID.match(incoming) else uuid.uuid4().hex
    token = request_id_var.set(request_id)
    # Also on request.state: exception handlers run outside this middleware's context scope.
    request.state.request_id = request_id
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        _access_logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,  # path only: query strings may carry user text
                "status": status_code,
                "latency_ms": int((time.perf_counter() - started) * 1000),
            },
        )
        request_id_var.reset(token)
