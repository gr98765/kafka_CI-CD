from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from app.main import FLAG_THRESHOLD, TOPIC_NAME, classify

VALID_TXN = {
    "account_id": "acct_123",
    "merchant": "Amazon",
    "amount": 49.99,
    "location": "Tucson, AZ",
}


# --- risk rule -------------------------------------------------------------

@pytest.mark.parametrize(
    "amount, expected",
    [
        (0.01, "cleared"),
        (999.99, "cleared"),
        (FLAG_THRESHOLD, "cleared"),  # threshold itself is not flagged
        (1000.01, "flagged"),
        (25000, "flagged"),
    ],
)
def test_classify(amount, expected):
    assert classify(amount) == expected


# --- POST /transactions ----------------------------------------------------

def test_create_transaction_publishes_to_kafka(client):
    producer = MagicMock()
    with patch("app.main.get_producer", return_value=producer):
        resp = client.post("/transactions", json=VALID_TXN)

    assert resp.status_code == 202
    assert resp.json() == {"status": "submitted", "transaction": VALID_TXN}
    producer.send.assert_called_once_with(TOPIC_NAME, value=VALID_TXN)
    producer.flush.assert_called_once()


def test_create_transaction_defaults_location(client):
    producer = MagicMock()
    payload = {k: v for k, v in VALID_TXN.items() if k != "location"}
    with patch("app.main.get_producer", return_value=producer):
        resp = client.post("/transactions", json=payload)

    assert resp.status_code == 202
    assert resp.json()["transaction"]["location"] == "unknown"


def test_create_transaction_kafka_down_returns_503(client):
    with patch("app.main.get_producer", side_effect=Exception("no brokers")):
        resp = client.post("/transactions", json=VALID_TXN)

    assert resp.status_code == 503
    assert "Kafka unavailable" in resp.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [
        {**VALID_TXN, "amount": 0},
        {**VALID_TXN, "amount": -5},
        {k: v for k, v in VALID_TXN.items() if k != "account_id"},
        {k: v for k, v in VALID_TXN.items() if k != "merchant"},
    ],
)
def test_create_transaction_rejects_invalid_payload(client, payload):
    with patch("app.main.get_producer") as get_producer:
        resp = client.post("/transactions", json=payload)

    assert resp.status_code == 422
    get_producer.assert_not_called()


# --- GET /transactions -----------------------------------------------------

def _mock_db(rows):
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = rows
    return conn, cursor


def test_list_transactions_returns_rows(client):
    created = datetime(2026, 1, 1, 12, 0, 0)
    rows = [(1, "acct_123", "Amazon", 1500.0, "Tucson, AZ", "flagged", created)]
    conn, cursor = _mock_db(rows)

    with patch("app.main.get_db_connection", return_value=conn):
        resp = client.get("/transactions")

    assert resp.status_code == 200
    assert resp.json() == [
        {
            "id": 1,
            "account_id": "acct_123",
            "merchant": "Amazon",
            "amount": 1500.0,
            "location": "Tucson, AZ",
            "status": "flagged",
            "created_at": created.isoformat(),
        }
    ]
    sql, = cursor.execute.call_args.args
    assert "WHERE" not in sql
    conn.close.assert_called_once()


def test_list_transactions_filters_by_status(client):
    conn, cursor = _mock_db([])

    with patch("app.main.get_db_connection", return_value=conn):
        resp = client.get("/transactions", params={"status": "flagged"})

    assert resp.status_code == 200
    assert resp.json() == []
    sql, params = cursor.execute.call_args.args
    assert "WHERE status = %s" in sql
    assert params == ("flagged",)


def test_list_transactions_closes_connection_on_error(client):
    conn, cursor = _mock_db([])
    cursor.execute.side_effect = RuntimeError("db exploded")

    with (
        patch("app.main.get_db_connection", return_value=conn),
        pytest.raises(RuntimeError),
    ):
        client.get("/transactions")

    conn.close.assert_called_once()


# --- misc ------------------------------------------------------------------

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_dashboard_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
