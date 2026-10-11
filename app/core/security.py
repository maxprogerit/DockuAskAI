from __future__ import annotations

import hmac
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

from app.core.config import Settings, get_settings

# auto_error=False: we raise our own 401 so it uses the same JSON error envelope as everything else.
# The scheme still shows up in /docs as an "Authorize" button.
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(
    provided: Optional[str] = Depends(_api_key_header), settings: Settings = Depends(get_settings)
) -> None:
    expected = settings.api_key.get_secret_value().encode("utf-8")
    # compare_digest: constant-time comparison, so response timing does not leak the key prefix.
    if provided is None or not hmac.compare_digest(provided.encode("utf-8"), expected):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")
