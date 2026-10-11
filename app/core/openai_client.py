from __future__ import annotations

from functools import lru_cache
from typing import Any

from openai import OpenAI

from app.core.config import Settings, get_settings
from app.core.errors import ServiceNotConfiguredError


def build_openai_client(settings: Settings) -> OpenAI:
    """One client for embeddings + LLM.

    Timeouts and retries: the SDK retries 408/409/429/5xx and connection errors with exponential
    backoff up to `max_retries`; `timeout` applies per attempt. We rely on that instead of a
    hand-written retry loop (less code, fewer bugs) and map whatever finally escapes to a 502.
    """
    if settings.openai_api_key is None:
        raise ServiceNotConfiguredError("OPENAI_API_KEY is not configured on the server.")
    return OpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        timeout=settings.openai_timeout_s,
        max_retries=settings.openai_max_retries,
    )


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    return build_openai_client(get_settings())


class LazyOpenAI:
    """Resolves the real client on first use instead of at dependency-wiring time.

    Without this, merely building a service (e.g. for GET /documents or DELETE) would fail with
    503 when OPENAI_API_KEY is missing, although those endpoints never talk to OpenAI.
    """

    @property
    def embeddings(self) -> Any:
        return get_openai_client().embeddings

    @property
    def responses(self) -> Any:
        return get_openai_client().responses
