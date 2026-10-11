from __future__ import annotations

from sqlalchemy.exc import OperationalError

from app.api.deps import get_vector_store
from app.core.database import get_db
from app.main import app
from tests.conftest import Env


class _BrokenSession:
    def execute(self, *_args: object, **_kwargs: object) -> None:
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))


def test_health_ok(env: Env) -> None:
    response = env.client.get("/health")  # no API key needed
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok", "vector_store": "ok"}


def test_health_degraded_when_db_down(env: Env) -> None:
    app.dependency_overrides[get_db] = lambda: _BrokenSession()
    response = env.client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "database": "unavailable", "vector_store": "ok"}


def test_health_degraded_when_vector_store_down(env: Env) -> None:
    class BrokenStore:
        def count(self) -> int:
            raise RuntimeError("chroma is gone")

    app.dependency_overrides[get_vector_store] = lambda: BrokenStore()
    response = env.client.get("/health")
    assert response.status_code == 503
    assert response.json()["vector_store"] == "unavailable"


def test_request_id_is_generated_and_echoed(env: Env) -> None:
    assert len(env.client.get("/health").headers["x-request-id"]) == 32
    assert env.client.get("/health", headers={"X-Request-ID": "abc-123"}).headers["x-request-id"] == "abc-123"


def test_unsafe_request_id_is_replaced(env: Env) -> None:
    response = env.client.get("/health", headers={"X-Request-ID": "bad id with spaces"})
    assert response.headers["x-request-id"] != "bad id with spaces"
