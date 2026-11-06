import os

# Must be set before app.main is imported so the lifespan hook skips
# starting the Kafka consumer thread.
os.environ.setdefault("DISABLE_KAFKA_CONSUMER", "1")

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
