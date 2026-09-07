"""
RailBlock AI — PostgreSQL Database Connection

This module creates the SQLAlchemy connection to PostgreSQL.

Database credentials are loaded from the private .env file.
Nothing sensitive is hardcoded here.
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.orm import declarative_base, sessionmaker


# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------

load_dotenv()


DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# ---------------------------------------------------------------------------
# Validate configuration
# ---------------------------------------------------------------------------

required_variables = {
    "DB_HOST": DB_HOST,
    "DB_NAME": DB_NAME,
    "DB_USER": DB_USER,
    "DB_PASSWORD": DB_PASSWORD,
}

missing_variables = [
    name
    for name, value in required_variables.items()
    if not value
]

if missing_variables:
    raise RuntimeError(
        "Missing database environment variables: "
        + ", ".join(missing_variables)
    )


# ---------------------------------------------------------------------------
# Build safe PostgreSQL connection URL
# ---------------------------------------------------------------------------

DATABASE_URL = URL.create(
    drivername="postgresql+psycopg",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=int(DB_PORT),
    database=DB_NAME,
)


# ---------------------------------------------------------------------------
# SQLAlchemy engine
# ---------------------------------------------------------------------------

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)


# ---------------------------------------------------------------------------
# Database session
# ---------------------------------------------------------------------------

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


# ---------------------------------------------------------------------------
# Base class for database models
# ---------------------------------------------------------------------------

Base = declarative_base()


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

def get_db():
    """
    Provide one database session.

    FastAPI will later use this inside API endpoints.
    """

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()


# ---------------------------------------------------------------------------
# Connection test
# ---------------------------------------------------------------------------

def test_database_connection() -> bool:
    """
    Verify that Python can connect to PostgreSQL.

    Returns True if PostgreSQL responds successfully.
    """

    with engine.connect() as connection:

        result = connection.execute(
            text("SELECT 1")
        )

        value = result.scalar()

        return value == 1