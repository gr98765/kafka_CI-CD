"""Consumer reliability: duplicates, dead-lettering, and the velocity rule.

The velocity rule and ON CONFLICT live in SQL, so those tests run against a
real Postgres. Each test gets its own throwaway schema (never the real table).
Point them at a server with TEST_DB_HOST (CI does); without it they are
skipped, unless REQUIRE_DB_TESTS=1, which turns the skip into a failure.
"""

import json
import os
import random
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import psycopg2
import pytest

from app.main import DEAD_LETTER_TOPIC, process_message

BASE = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
INIT_SQL = Path(__file__).resolve().parent.parent / "db-init-script.sql"


def payment(pid, seconds=0, account="ACC00001", amount=20.0, **extra):
    """One Kafka message body, shaped like the generator's records."""
    record = {
        "payment_id": pid,
        "account_id": account,
        "merchant_id": "M0001",
        "amount": amount,
        "currency": "USD",
        "timestamp": (BASE + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "country": "US",
        "channel": "online",
        **extra,
    }
    return json.dumps(record).encode()


@pytest.fixture
def producer():
    """Stands in for Kafka; dead-letter sends land here."""
    fake = MagicMock()
    with patch("app.main.get_producer", return_value=fake):
        yield fake


def dead_letters(producer):
    return [
        c.kwargs["value"]
        for c in producer.send.call_args_list
        if c.args[0] == DEAD_LETTER_TOPIC
    ]


# --- dead letter (no database needed) ---------------------------------------

@pytest.mark.parametrize(
    "raw",
    [
        b"this is not json",
        b"\xff\xfe\x00",  # not even UTF-8
        b'{"account_id": "ACC00001", "amount": 5, "timestamp": "2026-10-01T12:00:00Z"}',
        payment("pay_1", amount="$12.50"),
        payment("pay_1", amount=-3),
        payment("pay_1", timestamp="yesterday-ish"),
    ],
    ids=["not-json", "not-utf8", "no-payment-id", "amount-string", "negative", "bad-time"],
)
def test_bad_record_goes_to_dead_letter(producer, raw):
    conn = MagicMock()

    outcome = process_message(conn, raw, source="transactions/0@42")

    assert outcome == "dead_letter"
    [letter] = dead_letters(producer)
    assert letter["raw"] == raw.decode("utf-8", errors="replace")
    assert letter["source"] == "transactions/0@42"
    assert letter["error"]
    producer.send.return_value.get.assert_called_once()  # waited for Kafka's ack
    conn.commit.assert_not_called()


def test_db_rejecting_a_record_rolls_back_and_dead_letters(producer):
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value.execute.side_effect = (
        psycopg2.DataError("numeric field overflow")
    )

    assert process_message(conn, payment("pay_1")) == "dead_letter"
    conn.rollback.assert_called_once()
    assert "DataError" in dead_letters(producer)[0]["error"]


def test_lost_db_connection_is_not_dead_lettered(producer):
    """A DB outage is not the record's fault: re-raise so the consumer
    reconnects and retries, instead of parking good payments."""
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value.execute.side_effect = (
        psycopg2.OperationalError("server closed the connection")
    )

    with pytest.raises(psycopg2.OperationalError):
        process_message(conn, payment("pay_1"))
    assert dead_letters(producer) == []


def test_dead_letter_send_failure_is_raised_not_swallowed(producer):
    """If Kafka can't take the dead letter either, the message must not be
    acknowledged, so the error propagates and the consumer retries it."""
    producer.send.return_value.get.side_effect = TimeoutError("no ack")

    with pytest.raises(TimeoutError):
        process_message(MagicMock(), b"not json")


# --- against a real database --------------------------------------------------

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


def rows(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT payment_id, status, reason, amount FROM transactions "
            "ORDER BY event_time, payment_id"
        )
        return cur.fetchall()


def statuses(conn):
    return {pid: (status, reason) for pid, status, reason, _ in rows(conn)}


def flagged(conn):
    return sorted(pid for pid, (status, _) in statuses(conn).items() if status == "flagged")


# duplicates

def test_duplicate_payment_is_ignored(db, producer):
    first = process_message(db, payment("pay_1", amount=20.0))
    again = process_message(db, payment("pay_1", amount=20.0))

    assert (first, again) == ("cleared", "duplicate")
    assert len(rows(db)) == 1


def test_duplicate_never_overwrites_the_stored_payment(db, producer):
    process_message(db, payment("pay_1", amount=20.0))
    assert process_message(db, payment("pay_1", amount=5000.0)) == "duplicate"

    [(pid, status, reason, amount)] = rows(db)
    assert (pid, status, reason, float(amount)) == ("pay_1", "cleared", None, 20.0)


def test_duplicates_do_not_count_towards_velocity(db, producer):
    """6 distinct payments, each delivered twice, are still only 6."""
    for i in range(6):
        for _ in range(2):
            process_message(db, payment(f"pay_{i}", seconds=i * 10))

    assert len(rows(db)) == 6
    assert flagged(db) == []


# amount rule

def test_amount_over_1000_is_flagged_with_reason_amount(db, producer):
    process_message(db, payment("pay_big", amount=1000.01))
    process_message(db, payment("pay_edge", seconds=1, amount=1000.00))

    assert statuses(db) == {
        "pay_big": ("flagged", "amount"),
        "pay_edge": ("cleared", None),
    }


# velocity rule

def test_velocity_7_payments_in_180s_flags_the_7th(db, producer):
    outcomes = [
        process_message(db, payment(f"pay_{i}", seconds=i * 30))  # 0s .. 180s
        for i in range(7)
    ]

    assert outcomes == ["cleared"] * 6 + ["flagged"]
    assert statuses(db)["pay_6"] == ("flagged", "velocity")
    assert flagged(db) == ["pay_6"]


def test_velocity_6_payments_in_180s_is_not_flagged(db, producer):
    for i in range(6):
        process_message(db, payment(f"pay_{i}", seconds=i * 30))

    assert flagged(db) == []


def test_velocity_7th_and_later_are_all_flagged(db, producer):
    for i in range(9):
        process_message(db, payment(f"pay_{i}", seconds=i * 10))

    assert flagged(db) == ["pay_6", "pay_7", "pay_8"]


def test_velocity_window_is_180_seconds_not_181(db, producer):
    seconds = [0, 30, 60, 90, 120, 150, 181]  # 7 payments, but spread over 181s
    for i, s in enumerate(seconds):
        process_message(db, payment(f"pay_{i}", seconds=s))

    assert flagged(db) == []


def test_velocity_counts_per_account(db, producer):
    for i in range(7):
        account = "ACC00001" if i % 2 else "ACC00002"
        process_message(db, payment(f"pay_{i}", seconds=i * 10, account=account))

    assert flagged(db) == []


def test_velocity_uses_event_time_not_arrival_order(db, producer):
    """Late arrivals: the same 7 payments in shuffled arrival order still flag
    exactly the 7th by timestamp, even though it was stored before others."""
    messages = [payment(f"pay_{i}", seconds=i * 30) for i in range(7)]
    random.Random(3).shuffle(messages)
    for raw in messages:
        process_message(db, raw)

    assert flagged(db) == ["pay_6"]
    assert statuses(db)["pay_6"] == ("flagged", "velocity")


def test_amount_flagged_payment_keeps_reason_amount_and_still_counts(db, producer):
    for i in range(7):
        amount = 2500.0 if i == 2 else 20.0
        process_message(db, payment(f"pay_{i}", seconds=i * 10, amount=amount))

    assert statuses(db)["pay_2"] == ("flagged", "amount")
    assert statuses(db)["pay_6"] == ("flagged", "velocity")
    assert flagged(db) == ["pay_2", "pay_6"]


# dead letter with a real database

def test_bad_record_dead_lettered_and_consumer_keeps_going(db, producer):
    """The DB rejects the record (account_id too long for VARCHAR(50)); the
    transaction is rolled back and the next good record is stored normally."""
    bad = process_message(db, payment("pay_bad", account="A" * 60))
    good = process_message(db, payment("pay_good"))

    assert (bad, good) == ("dead_letter", "cleared")
    assert [r[0] for r in rows(db)] == ["pay_good"]
    [letter] = dead_letters(producer)
    assert "pay_bad" in letter["raw"]
