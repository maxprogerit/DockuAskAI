from __future__ import annotations

from functools import lru_cache
from typing import Iterator

from sqlalchemy import MetaData, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings

# Explicit constraint naming: without it Postgres generates names and Alembic
# cannot reliably drop/alter constraints by name later.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    # pool_pre_ping: transparently replaces connections killed by a DB restart.
    # connect_timeout: without it a dead DB makes every request (incl. /health) hang indefinitely.
    return create_engine(
        get_settings().database_url, pool_pre_ping=True, connect_args={"connect_timeout": 3}
    )


@lru_cache(maxsize=1)
def get_sessionmaker() -> "sessionmaker[Session]":
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one session per request (like @Transactional at the controller level)."""
    session = get_sessionmaker()()
    try:
        yield session
    finally:
        session.close()
