from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import psycopg2
import pytest

from app import main
from app.main import COLUMNS, FLAG_THRESHOLD, TOPIC_NAME, classify

VALID_TXN = {
    "account_id": "acct_123",
    "merchant": "Amazon",
    "amount": 49.99,
    "location": "Tucson, AZ",
}

GENERATOR_TXN = {
    "payment_id": "pay_00000001",
    "account_id": "ACC00012",
    "merchant_id": "M0042",
    "amount": 12.5,
    "currency": "USD",
    "timestamp": "2026-10-01T12:00:00Z",
    "country": "US",
    "channel": "online",
}


# --- amount rule -----------------------------------------------------------

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
        resp = client.post("/transactions", json=GENERATOR_TXN)

    assert resp.status_code == 202
    sent = resp.json()["transaction"]
    assert sent["payment_id"] == "pay_00000001"
    assert sent["amount"] == 12.5
    assert datetime.fromisoformat(sent["timestamp"]) == datetime(
        2026, 10, 1, 12, tzinfo=UTC
    )
    producer.send.assert_called_once_with(TOPIC_NAME, key=b"ACC00012", value=sent)
    producer.send.return_value.get.assert_called_once()  # waits for Kafka's ack


def test_create_transaction_fills_in_id_time_and_location(client):
    producer = MagicMock()
    payload = {k: v for k, v in VALID_TXN.items() if k != "location"}
    with patch("app.main.get_producer", return_value=producer):
        first = client.post("/transactions", json=payload).json()["transaction"]
        second = client.post("/transactions", json=payload).json()["transaction"]

    assert first["payment_id"].startswith("pay_")
    assert first["payment_id"] != second["payment_id"]
    assert first["location"] == "unknown"
    assert datetime.fromisoformat(first["timestamp"]).tzinfo is not None


def test_timestamp_without_timezone_is_utc(client):
    producer = MagicMock()
    payload = {**GENERATOR_TXN, "timestamp": "2026-10-01T12:00:00"}
    with patch("app.main.get_producer", return_value=producer):
        sent = client.post("/transactions", json=payload).json()["transaction"]

    assert datetime.fromisoformat(sent["timestamp"]) == datetime(
        2026, 10, 1, 12, tzinfo=UTC
    )


def test_create_transaction_kafka_down_returns_503(client):
    with patch("app.main.get_producer", side_effect=Exception("no brokers")):
        resp = client.post("/transactions", json=VALID_TXN)

    assert resp.status_code == 503
    assert "Kafka unavailable" in resp.json()["detail"]


def test_create_transaction_kafka_not_acking_returns_503(client):
    producer = MagicMock()
    producer.send.return_value.get.side_effect = TimeoutError("no ack")
    with patch("app.main.get_producer", return_value=producer):
        resp = client.post("/transactions", json=VALID_TXN)

    assert resp.status_code == 503


@pytest.mark.parametrize(
    "payload",
    [
        {**VALID_TXN, "amount": 0},
        {**VALID_TXN, "amount": -5},
        {**VALID_TXN, "amount": "$12.50"},
        {**VALID_TXN, "account_id": ""},
        {**VALID_TXN, "timestamp": "not a time"},
        {k: v for k, v in VALID_TXN.items() if k != "account_id"},
    ],
)
def test_create_transaction_rejects_invalid_payload(client, payload):
    with patch("app.main.get_producer") as get_producer:
        resp = client.post("/transactions", json=payload)

    assert resp.status_code == 422
    get_producer.assert_not_called()


def test_producer_is_created_once_and_reused():
    with patch("app.main.KafkaProducer") as kafka_producer:
        assert main.get_producer() is main.get_producer()
    kafka_producer.assert_called_once()


def test_producer_creation_is_retried_after_failure():
    with patch("app.main.KafkaProducer", side_effect=[Exception("down"), MagicMock()]):
        with pytest.raises(Exception, match="down"):
            main.get_producer()
        assert main.get_producer() is not None


# --- GET /transactions -----------------------------------------------------

def _mock_db(rows):
    conn = MagicMock()
    conn.closed = 0
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = rows
    return conn, cursor


def test_list_transactions_returns_rows(client):
    event_time = datetime(2026, 10, 1, 12, tzinfo=UTC)
    created = datetime(2026, 10, 1, 12, 0, 2, tzinfo=UTC)
    row = ("pay_1", "ACC00012", "M0042", None, 1500.0, "USD", "US", "online",
           "unknown", event_time, "flagged", "amount", created)  # fmt: skip
    conn, cursor = _mock_db([row])

    with patch("app.main.get_db_connection", return_value=conn):
        resp = client.get("/transactions")

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert set(body[0]) == set(COLUMNS)
    assert body[0]["payment_id"] == "pay_1"
    assert body[0]["status"] == "flagged"
    assert body[0]["reason"] == "amount"
    sql, _ = cursor.execute.call_args.args
    assert "WHERE" not in sql


def test_list_transactions_filters_by_status(client):
    conn, cursor = _mock_db([])

    with patch("app.main.get_db_connection", return_value=conn):
        resp = client.get("/transactions", params={"status": "flagged"})

    assert resp.status_code == 200
    assert resp.json() == []
    sql, params = cursor.execute.call_args.args
    assert "WHERE status = %s" in sql
    assert params == ("flagged",)


def test_list_transactions_reuses_one_connection(client):
    conn, _ = _mock_db([])

    with patch("app.main.get_db_connection", return_value=conn) as connect:
        client.get("/transactions")
        client.get("/transactions")
        client.get("/transactions", params={"status": "cleared"})

    connect.assert_called_once()
    conn.close.assert_not_called()
    assert conn.autocommit is True  # reads never hold a transaction open


def test_list_transactions_reconnects_after_db_drops(client):
    broken, broken_cursor = _mock_db([])
    broken_cursor.execute.side_effect = psycopg2.OperationalError("server closed")
    healthy, _ = _mock_db([])

    with patch("app.main.get_db_connection", side_effect=[broken, healthy]):
        first = client.get("/transactions")
        second = client.get("/transactions")

    assert first.status_code == 503
    assert "Database unavailable" in first.json()["detail"]
    broken.close.assert_called_once()
    assert second.status_code == 200


# --- misc ------------------------------------------------------------------

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_dashboard_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
