"""Test doubles. These live under tests/ ONLY: the product never fakes embeddings or LLM output."""
from __future__ import annotations

import hashlib
import math
import re
from types import SimpleNamespace
from typing import Any, Dict, List

DIM = 8192  # large on purpose: hash collisions between unrelated words must not fake a similarity


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "of", "to", "in", "on", "at", "by", "for", "and", "or", "it",
    "its", "what", "how", "many", "much", "does", "do", "did", "which", "who", "when", "with", "as", "that", "this",
    "per", "from", "up", "can", "i", "we", "you", "has", "have", "if", "not", "no",
}


def fake_vector(text: str) -> List[float]:
    """Deterministic bag-of-words hashing vector, L2-normalized (like OpenAI's). Texts sharing
    words get a high cosine; texts with no common words get ~0. Good enough to exercise retrieval."""
    vector = [0.0] * DIM
    for word in re.findall(r"\w+", text.lower()):
        if word in _STOPWORDS:
            continue  # without this, "the/is/of" overlap would dominate the fake similarity
        digest = hashlib.md5(word.encode("utf-8")).digest()
        vector[int.from_bytes(digest[:4], "little") % DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class FakeEmbeddingsClient:
    """Mimics `openai.OpenAI().embeddings.create` and records calls."""

    def __init__(self, shuffle_response: bool = False) -> None:
        self.calls: List[List[str]] = []
        self._shuffle = shuffle_response
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, model: str, input: List[str], encoding_format: str = "float") -> Any:
        self.calls.append(list(input))
        data = [SimpleNamespace(index=i, embedding=fake_vector(text)) for i, text in enumerate(input)]
        if self._shuffle:
            data.reverse()  # real API returns by index; Embedder must not depend on arrival order
        tokens = sum(len(t.split()) for t in input)
        return SimpleNamespace(data=data, usage=SimpleNamespace(total_tokens=tokens, prompt_tokens=tokens))


class FakeLLMClient:
    """Mimics `client.responses.parse`. Feed it scripted LLMAnswer objects (or exceptions)."""

    def __init__(self, scripted: List[Any]) -> None:
        self._scripted = list(scripted)
        self.calls: List[Dict[str, Any]] = []
        self.responses = SimpleNamespace(parse=self._parse)

    def script(self, *items: Any) -> None:
        self._scripted.extend(items)

    def _parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self._scripted:
            raise AssertionError("LLM was called more times than scripted")
        item = self._scripted.pop(0)
        if isinstance(item, Exception):
            raise item
        status = "completed"
        if isinstance(item, tuple):  # ("incomplete", None) to simulate a truncated response
            status, item = item
        if callable(item):  # lets a test build its answer from the prompt that was actually sent
            item = item(kwargs)
        return SimpleNamespace(
            output_parsed=item,
            status=status,
            incomplete_details=None,
            usage=SimpleNamespace(input_tokens=1000, output_tokens=100),
            output=[],
        )
