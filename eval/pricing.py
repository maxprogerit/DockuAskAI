"""USD per 1M tokens, copied from the official OpenAI pricing page on 2026-10-10
(https://developers.openai.com/api/docs/pricing, standard tier, prompts <= 272K tokens).

Prices change: re-check before quoting cost numbers. Cached-input discounts are NOT applied, so
the reported cost is an upper bound for the LLM part.
"""
from __future__ import annotations

from typing import Dict, Tuple

# model -> (input, output) per 1M tokens
LLM_PRICES: Dict[str, Tuple[float, float]] = {
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6.1-sol": (2.00, 10.00),
    "gpt-6-astra": (10.00, 50.00),
}
EMBEDDING_PRICES: Dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}


def llm_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    if model not in LLM_PRICES:
        raise KeyError("no price known for model %r; add it to eval/pricing.py" % model)
    price_in, price_out = LLM_PRICES[model]
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


def embedding_cost(model: str, tokens: int) -> float:
    if model not in EMBEDDING_PRICES:
        raise KeyError("no price known for embedding model %r; add it to eval/pricing.py" % model)
    return tokens * EMBEDDING_PRICES[model] / 1_000_000
