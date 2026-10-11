from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}


class Settings(BaseSettings):
    """All configuration comes from environment variables (or a local .env file).

    SecretStr keeps keys out of repr()/logs; call .get_secret_value() only at the point of use.
    DATABASE_URL and API_KEY have no defaults on purpose: the app must fail fast when misconfigured.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API
    api_key: SecretStr = Field(min_length=16)
    log_level: str = "INFO"

    # Database
    database_url: str

    # Storage
    upload_dir: Path = Path("./data/uploads")
    chroma_dir: Path = Path("./data/chroma")
    chroma_collection: str = "docuask_chunks"

    # Upload limits
    max_upload_mb: int = Field(default=20, gt=0)
    max_pages: int = Field(default=300, gt=0)

    # OpenAI. The key is optional so /health, migrations and unit tests work without it.
    openai_api_key: Optional[SecretStr] = None
    openai_model: str = "gpt-6-luna"
    # Reasoning tokens are billed as output and add latency; extraction-style RAG needs little.
    openai_reasoning_effort: str = "low"
    embedding_model: str = "text-embedding-3-small"
    openai_timeout_s: float = Field(default=30.0, gt=0)
    # Retries with exponential backoff are done by the OpenAI SDK (429 / 5xx / connection errors).
    openai_max_retries: int = Field(default=3, ge=0, le=6)
    embedding_batch_size: int = Field(default=64, gt=0, le=2048)
    # Reasoning tokens count against this limit, so keep it generous.
    answer_max_output_tokens: int = Field(default=2000, gt=0)

    # Chunking (characters, not tokens: keeps the chunker dependency-free and explainable)
    chunk_size: int = Field(default=500, gt=50)
    chunk_overlap: int = Field(default=100, ge=0)

    # Retrieval
    top_k: int = Field(default=5, gt=0, le=50)
    # Cosine similarity of the best chunk below which we refuse WITHOUT calling the LLM.
    # NOTE: this default is a placeholder until it is tuned on the evaluation set (stage 6).
    score_threshold: float = Field(default=0.30, ge=-1.0, le=1.0)

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def _blank_key_is_none(cls, value: object) -> object:
        # `OPENAI_API_KEY=` in .env arrives as "" - that means "not configured", not a key.
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("openai_reasoning_effort")
    @classmethod
    def _check_effort(cls, value: str) -> str:
        if value not in REASONING_EFFORTS:
            raise ValueError("OPENAI_REASONING_EFFORT must be one of %s" % sorted(REASONING_EFFORTS))
        return value

    @model_validator(mode="after")
    def _check_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # values come from the environment
