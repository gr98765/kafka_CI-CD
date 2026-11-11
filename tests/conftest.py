import os
import uuid
from pathlib import Path

# Must be set before app.main is imported so the lifespan hook skips
# starting the Kafka consumer thread.
os.environ.setdefault("DISABLE_KAFKA_CONSUMER", "1")

import psycopg2
import pytest
from fastapi.testclient import TestClient

from app import main
from app.main import app


@pytest.fixture(autouse=True)
def reset_shared_connections():
    """The app keeps one producer and one API DB connection per process;
    make sure no test sees another test's (mock) objects."""
    main._producer = None
    main._api_conn = None
    yield
    main._producer = None
    main._api_conn = None


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# --- a real database, for the SQL tests ----------------------------------------
# Each test gets its own throwaway schema (never the real tables). Point the
# tests at a server with TEST_DB_HOST (CI does); without it they are skipped,
# unless REQUIRE_DB_TESTS=1, which turns the skip into a failure.

INIT_SQL = Path(__file__).resolve().parent.parent / "db-init-script.sql"


def _connect(**kwargs):
    return psycopg2.connect(
        host=os.environ["TEST_DB_HOST"],
        port=os.getenv("TEST_DB_PORT", "5432"),
        user=os.getenv("DB_USER", "fintech"),
        password=os.getenv("DB_PASSWORD", "fintech"),
        dbname=os.getenv("DB_NAME", "transactions_db"),
        connect_timeout=5,
        **kwargs,
    )


@pytest.fixture
def db():
    if not os.getenv("TEST_DB_HOST"):
        if os.getenv("REQUIRE_DB_TESTS") == "1":
            pytest.fail("REQUIRE_DB_TESTS=1 but TEST_DB_HOST is not set")
        pytest.skip("set TEST_DB_HOST to run database tests")

    schema = f"test_{uuid.uuid4().hex[:12]}"
    admin = _connect()
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(f"CREATE SCHEMA {schema}")
    conn = _connect(options=f"-c search_path={schema}")
    try:
        with conn.cursor() as cur:
            cur.execute(INIT_SQL.read_text())
        conn.commit()
        yield conn
    finally:
        conn.close()
        with admin.cursor() as cur:
            cur.execute(f"DROP SCHEMA {schema} CASCADE")
        admin.close()
