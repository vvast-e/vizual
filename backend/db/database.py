"""SQLAlchemy engine and session factory."""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


def get_database_url() -> str:
    # По умолчанию для локальной разработки (uvicorn в терминале) -> localhost
    return os.getenv(
        "DATABASE_URL",
        "postgresql://vizual:vizual@localhost:5433/vizual",
    )


class Base(DeclarativeBase):
    """Declarative base for ORM models."""


engine = create_engine(get_database_url(), pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
