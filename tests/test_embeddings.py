from __future__ import annotations

import json
from typing import Callable, List

import httpx
import openai
import pytest

from app.core.errors import UpstreamError
from app.services.embeddings import Embedder
from tests.fakes import FakeEmbeddingsClient


def test_batches_preserve_order_even_if_response_is_shuffled() -> None:
    client = FakeEmbeddingsClient(shuffle_response=True)
    texts = ["t%d" % i for i in range(5)]
    result = Embedder(client, "m", batch_size=2).embed(texts)
    assert [len(batch) for batch in client.calls] == [2, 2, 1]
    from tests.fakes import fake_vector

    assert result.vectors == [fake_vector(t) for t in texts]
    assert result.tokens == 5


def test_empty_input_makes_no_call() -> None:
    client = FakeEmbeddingsClient()
    assert Embedder(client, "m", 4).embed([]).vectors == []
    assert client.calls == []


def test_openai_errors_are_mapped_to_upstream_error() -> None:
    class Boom:
        class embeddings:
            @staticmethod
            def create(**_: object) -> None:
                raise openai.APIConnectionError(request=httpx.Request("POST", "http://x"))

    with pytest.raises(UpstreamError):
        Embedder(Boom, "m", 4).embed(["a"])


# ---- through the REAL openai SDK, with the network replaced by httpx.MockTransport ----


def _sdk_client(handler: Callable[[httpx.Request], httpx.Response], max_retries: int) -> openai.OpenAI:
    return openai.OpenAI(
        api_key="sk-test",
        base_url="http://mock.local/v1",
        max_retries=max_retries,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _embeddings_payload(n: int) -> dict:
    return {
        "object": "list",
        "model": "text-embedding-3-small",
        "data": [{"object": "embedding", "index": i, "embedding": [0.1, 0.2, 0.3]} for i in range(n)],
        "usage": {"prompt_tokens": 7, "total_tokens": 7},
    }


def test_sdk_retries_transient_5xx_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda _s: None)  # skip real backoff delays
    attempts: List[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(503, json={"error": {"message": "overloaded"}})
        body = json.loads(request.content)
        assert body["model"] == "text-embedding-3-small" and body["input"] == ["hello"]
        return httpx.Response(200, json=_embeddings_payload(1))

    result = Embedder(_sdk_client(handler, max_retries=3), "text-embedding-3-small", 8).embed(["hello"])
    assert len(attempts) == 3
    assert result.vectors == [[0.1, 0.2, 0.3]] and result.tokens == 7


def test_sdk_gives_up_after_limited_retries_and_maps_to_502(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda _s: None)
    attempts: List[int] = []

    def handler(_: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(500, json={"error": {"message": "boom"}})

    with pytest.raises(UpstreamError) as info:
        Embedder(_sdk_client(handler, max_retries=2), "m", 8).embed(["x"])
    assert len(attempts) == 3  # 1 try + 2 retries
    assert "boom" not in info.value.message  # upstream text is not leaked to the client


def test_auth_error_is_not_retried_and_mapped(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: List[int] = []

    def handler(_: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(401, json={"error": {"message": "Incorrect API key provided: sk-secret"}})

    with pytest.raises(UpstreamError) as info:
        Embedder(_sdk_client(handler, max_retries=3), "m", 8).embed(["x"])
    assert len(attempts) == 1
    assert "sk-secret" not in info.value.message
