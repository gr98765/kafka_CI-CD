"""Consumer reliability: the raw layer, re-deliveries and dead-lettering.

The consumer stores every message as received in raw_payments; record problems
(amounts, timestamps, IDs) are left to the clean step (tests/test_cleaning.py).
The database tests use the `db` fixture from conftest.py: a throwaway schema,
skipped without TEST_DB_HOST.
"""

import json
from unittest.mock import MagicMock, patch

import psycopg2
import pytest

from app.main import DEAD_LETTER_TOPIC, process_message

GOOD = {
    "payment_id": "pay_1",
    "account_id": "ACC00001",
    "merchant_id": "M0001",
    "amount": 20.0,
    "currency": "USD",
    "timestamp": "2026-10-01T12:00:00Z",
    "country": "US",
    "channel": "online",
}


def message(**changes) -> bytes:
    return json.dumps({**GOOD, **changes}).encode()


@pytest.fixture
def producer():
    """Stands in for Kafka; dead-letter sends land here."""
    fake = MagicMock()
    with patch("app.main.get_producer", return_value=fake):
        yield fake


def dead_letters(producer):
    return [c.kwargs["value"] for c in producer.send.call_args_list if c.args[0] == DEAD_LETTER_TOPIC]


def raw_rows(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT payload, kafka_partition, kafka_offset FROM raw_payments ORDER BY raw_id")
        return cur.fetchall()


# --- dead letter (no database needed) ---------------------------------------


@pytest.mark.parametrize(
    "raw",
    [b"this is not json", b"\xff\xfe\x00", b"[1, 2, 3]", b'"a string"', b"null", b'{"amount": NaN}'],
    ids=["not-json", "not-utf8", "array", "string", "null", "nan"],
)
def test_non_object_goes_to_dead_letter(producer, raw):
    conn = MagicMock()

    outcome = process_message(conn, raw, 0, 42)

    assert outcome == "dead_letter"
    [letter] = dead_letters(producer)
    assert letter["raw"] == raw.decode("utf-8", errors="replace")
    assert letter["source"] == "transactions/0@42"
    assert letter["error"]
    producer.send.return_value.get.assert_called_once()  # waited for Kafka's ack
    conn.commit.assert_not_called()


def test_db_rejecting_a_record_rolls_back_and_dead_letters(producer):
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value.execute.side_effect = psycopg2.DataError("bad jsonb")

    assert process_message(conn, message(), 0, 1) == "dead_letter"
    conn.rollback.assert_called_once()
    assert "DataError" in dead_letters(producer)[0]["error"]


def test_lost_db_connection_is_not_dead_lettered(producer):
    """A DB outage is not the record's fault: re-raise so the consumer
    reconnects and retries, instead of parking good payments."""
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value.execute.side_effect = psycopg2.OperationalError(
        "server closed the connection"
    )

    with pytest.raises(psycopg2.OperationalError):
        process_message(conn, message(), 0, 1)
    assert dead_letters(producer) == []


def test_dead_letter_send_failure_is_raised_not_swallowed(producer):
    """If Kafka can't take the dead letter either, the message must not be
    acknowledged, so the error propagates and the consumer retries it."""
    producer.send.return_value.get.side_effect = TimeoutError("no ack")

    with pytest.raises(TimeoutError):
        process_message(MagicMock(), b"not json", 0, 1)


# --- against a real database --------------------------------------------------


def test_record_is_stored_exactly_as_received(db, producer):
    assert process_message(db, message(), 3, 17) == "stored"

    assert raw_rows(db) == [(GOOD, 3, 17)]


@pytest.mark.parametrize(
    "changes",
    [
        {"amount": "$12.50"},
        {"amount": -3},
        {"timestamp": 1790856000},
        {"timestamp": "yesterday-ish"},
        {"account_id": " acc00001 "},
        {"currency": "XXX"},
        {"payment_id": None},
        {"account_id": "A" * 500},  # too long for the old VARCHAR(50): fine in raw
    ],
    ids=["amount-string", "negative", "epoch", "bad-time", "messy-id", "currency", "no-payment-id", "long-id"],
)
def test_messy_records_are_stored_raw_not_dead_lettered(db, producer, changes):
    assert process_message(db, message(**changes), 0, 1) == "stored"

    [(payload, _, _)] = raw_rows(db)
    assert payload == {**GOOD, **changes}
    assert dead_letters(producer) == []


def test_kafka_redelivery_is_stored_once(db, producer):
    """Same partition and offset = the same message delivered again."""
    first = process_message(db, message(), 0, 5)
    again = process_message(db, message(), 0, 5)

    assert (first, again) == ("stored", "duplicate")
    assert len(raw_rows(db)) == 1


def test_resent_payment_is_a_new_raw_record(db, producer):
    """The same payment_id at a different offset is a separate arrival: the raw
    layer keeps both, and the clean step keeps the first."""
    process_message(db, message(), 0, 5)
    process_message(db, message(), 0, 6)

    assert [r[2] for r in raw_rows(db)] == [5, 6]


def test_bad_record_dead_lettered_and_consumer_keeps_going(db, producer):
    """Postgres refuses the record (\\u0000 is not allowed in JSONB); the
    transaction is rolled back and the next good record is stored normally."""
    bad = process_message(db, b'{"payment_id": "pay_bad", "merchant": "\\u0000"}', 0, 1)
    good = process_message(db, message(), 0, 2)

    assert (bad, good) == ("dead_letter", "stored")
    assert [r[0]["payment_id"] for r in raw_rows(db)] == ["pay_1"]
    [letter] = dead_letters(producer)
    assert "pay_bad" in letter["raw"]
