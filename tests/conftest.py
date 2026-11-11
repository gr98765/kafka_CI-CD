import os

# Must be set before app.main is imported so the lifespan hook skips
# starting the Kafka consumer thread.
os.environ.setdefault("DISABLE_KAFKA_CONSUMER", "1")

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
