"""The clean step (sql/clean.sql via app/clean.py): fixes, de-duplication,
quarantine reasons and the two fraud rules on payments_clean.

Every test writes raw_payments rows into a throwaway schema (the `db` fixture
in conftest.py, skipped without TEST_DB_HOST) and runs the real clean step.
"""

import json
import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from app.clean import run_clean

BASE = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)


def record(pid="pay_1", seconds=0, account="ACC00001", amount=20.0, **extra):
    """One raw record, shaped like the generator's output."""
    return {
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


def load(conn, *records, received=None):
    """Insert records into raw_payments in arrival order (offsets 0, 1, ...).
    `received` optionally gives each record's arrival time in seconds after BASE."""
    with conn.cursor() as cur:
        for i, r in enumerate(records):
            at = BASE + timedelta(seconds=received[i]) if received else BASE + timedelta(milliseconds=i)
            cur.execute(
                "INSERT INTO raw_payments (payload, received_at, kafka_partition, kafka_offset) "
                "VALUES (%s::jsonb, %s, 0, %s)",
                (json.dumps(r), at, i),
            )
    conn.commit()


def clean(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT payment_id, account_id, amount, currency, event_time, status, reason "
            "FROM payments_clean ORDER BY event_time, payment_id"
        )
        cols = ("payment_id", "account_id", "amount", "currency", "event_time", "status", "reason")
        return {r[0]: dict(zip(cols, r, strict=True)) for r in cur.fetchall()}


def quarantine(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT raw_id, payment_id, reason FROM payments_quarantine ORDER BY raw_id")
        return [(pid, reason) for _, pid, reason in cur.fetchall()]


def flagged(conn):
    return sorted(p for p, r in clean(conn).items() if r["status"] == "flagged")


def cleaned_one(db, **changes):
    load(db, record(**changes))
    run_clean(db)
    assert quarantine(db) == []
    [row] = clean(db).values()
    return row


# --- fixes ------------------------------------------------------------------


@pytest.mark.parametrize(
    "amount, expected",
    [
        (12.5, "12.50"),
        ("12.50", "12.50"),
        ("$12.50", "12.50"),
        (" $ 12.50 ", "12.50"),
        ("1,234.50", "1234.50"),
        ("$1,234.50", "1234.50"),
        ("7", "7.00"),
        ("12.505", "12.51"),  # rounded to cents
    ],
)
def test_amount_is_fixed(db, amount, expected):
    assert cleaned_one(db, amount=amount)["amount"] == Decimal(expected)


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-10-01T12:00:00Z",
        "2026-10-01T12:00:00",  # no zone: UTC
        "2026-10-01 12:00:00",  # the generator's "YYYY-MM-DD HH:MM:SS"
        "2026-10-01T14:00:00+02:00",  # other zone: converted to UTC
        "2026-10-01t12:00:00z",
        "2026-10-01T12:00:00.000Z",
        1790856000,  # epoch seconds
        "1790856000",  # epoch seconds as a string
    ],
)
def test_timestamp_is_normalized_to_utc(db, timestamp):
    row = cleaned_one(db, timestamp=timestamp)
    assert row["event_time"] == BASE
    assert row["event_time"].utcoffset() == timedelta(0)


@pytest.mark.parametrize("account", [" acc00001 ", "acc00001", "ACC00001\t", "Acc00001"])
def test_account_id_is_trimmed_and_uppercased(db, account):
    assert cleaned_one(db, account=account)["account_id"] == "ACC00001"


@pytest.mark.parametrize("currency", ["usd", " USD ", "Usd"])
def test_currency_is_uppercased(db, currency):
    assert cleaned_one(db, currency=currency)["currency"] == "USD"


def test_duplicate_payment_id_keeps_the_first_arrival(db):
    load(
        db,
        record("pay_1", amount=99.0),  # arrives second
        record("pay_1", amount=20.0),  # arrives first
        record("pay_1", amount=55.0),  # arrives third
        received=[5, 1, 9],
    )
    counts = run_clean(db)

    assert clean(db)["pay_1"]["amount"] == Decimal("20.00")
    assert (counts["clean"], counts["quarantined"], counts["duplicates_dropped"]) == (1, 0, 2)


def test_duplicate_with_same_arrival_time_keeps_the_lower_offset(db):
    load(db, record("pay_1", amount=20.0), record("pay_1", amount=99.0), received=[1, 1])
    run_clean(db)
    assert clean(db)["pay_1"]["amount"] == Decimal("20.00")


def test_every_raw_record_is_accounted_for(db):
    load(db, record("pay_1"), record("pay_1"), record("pay_2", amount=-1), record("pay_3"))
    counts = run_clean(db)
    assert counts["raw"] == counts["clean"] + counts["quarantined"] + counts["duplicates_dropped"] == 4
    assert (counts["clean"], counts["quarantined"], counts["duplicates_dropped"]) == (2, 1, 1)


def test_clean_step_is_repeatable(db):
    load(db, record("pay_1"), record("pay_2", amount=-1))
    first = run_clean(db)
    second = run_clean(db)
    assert first == second
    assert len(clean(db)) == 1 and len(quarantine(db)) == 1


# --- quarantine ---------------------------------------------------------------


@pytest.mark.parametrize(
    "changes, reason",
    [
        ({"payment_id": None}, "missing_payment_id"),
        ({"payment_id": ""}, "missing_payment_id"),
        ({"payment_id": "   "}, "missing_payment_id"),
        ({"account_id": None}, "missing_account_id"),
        ({"account_id": "  "}, "missing_account_id"),
        ({"amount": -12.5}, "non_positive_amount"),
        ({"amount": "-$12.50"}, "non_positive_amount"),
        ({"amount": 0}, "non_positive_amount"),
        ({"amount": "0.001"}, "non_positive_amount"),  # rounds to 0.00
        ({"amount": "abc"}, "bad_amount"),
        ({"amount": None}, "bad_amount"),
        ({"amount": [12]}, "bad_amount"),
        ({"amount": 1e12}, "bad_amount"),  # too large for NUMERIC(12,2)
        ({"currency": "XXX"}, "currency_not_usd"),
        ({"currency": "EUR"}, "currency_not_usd"),
        ({"currency": None}, "currency_not_usd"),
        ({"timestamp": "yesterday-ish"}, "bad_timestamp"),
        ({"timestamp": "2026-13-01T12:00:00Z"}, "bad_timestamp"),
        ({"timestamp": None}, "bad_timestamp"),
        ({"timestamp": -5}, "bad_timestamp"),
        ({"timestamp": 1790856000000}, "bad_timestamp"),  # milliseconds: year > 2099
        ({"timestamp": "2026-10-01"}, "bad_timestamp"),  # no time of day
    ],
)
def test_unfixable_record_is_quarantined_with_reason(db, changes, reason):
    load(db, record(**changes))
    run_clean(db)

    assert clean(db) == {}
    [(_pid, got)] = quarantine(db)
    assert got == reason


def test_missing_key_is_quarantined_too(db):
    load(db, {k: v for k, v in record().items() if k != "account_id"})
    run_clean(db)
    assert quarantine(db) == [("pay_1", "missing_account_id")]


def test_first_problem_wins_when_there_are_several(db):
    load(db, record(account=None, amount=-5, currency="XXX"))
    run_clean(db)
    assert quarantine(db) == [("pay_1", "missing_account_id")]


def test_records_without_payment_id_are_each_quarantined(db):
    """They cannot be de-duplicated, so none of them is dropped."""
    load(db, record(pid=None), record(pid=None))
    counts = run_clean(db)
    assert quarantine(db) == [(None, "missing_payment_id")] * 2
    assert counts["duplicates_dropped"] == 0


def test_quarantine_keeps_the_raw_payload(db):
    bad = record(amount="abc")
    load(db, bad)
    run_clean(db)
    with db.cursor() as cur:
        cur.execute("SELECT payload FROM payments_quarantine")
        assert cur.fetchone()[0] == bad


# --- fraud rules on payments_clean -------------------------------------------


def test_amount_over_1000_is_flagged_with_reason_amount(db):
    load(db, record("pay_big", amount=1000.01), record("pay_edge", seconds=1, amount="$1000.00"))
    run_clean(db)
    rows = clean(db)
    assert (rows["pay_big"]["status"], rows["pay_big"]["reason"]) == ("flagged", "amount")
    assert (rows["pay_edge"]["status"], rows["pay_edge"]["reason"]) == ("cleared", None)


def test_velocity_7_payments_in_180s_flags_the_7th(db):
    load(db, *(record(f"pay_{i}", seconds=i * 30) for i in range(7)))  # 0s .. 180s
    run_clean(db)
    assert flagged(db) == ["pay_6"]
    assert clean(db)["pay_6"]["reason"] == "velocity"


def test_velocity_6_payments_in_180s_is_not_flagged(db):
    load(db, *(record(f"pay_{i}", seconds=i * 30) for i in range(6)))
    run_clean(db)
    assert flagged(db) == []


def test_velocity_7th_and_later_are_all_flagged(db):
    load(db, *(record(f"pay_{i}", seconds=i * 10) for i in range(9)))
    run_clean(db)
    assert flagged(db) == ["pay_6", "pay_7", "pay_8"]


def test_velocity_window_is_180_seconds_not_181(db):
    load(db, *(record(f"pay_{i}", seconds=s) for i, s in enumerate([0, 30, 60, 90, 120, 150, 181])))
    run_clean(db)
    assert flagged(db) == []


def test_velocity_counts_per_account_after_normalizing_ids(db):
    """' acc00001 ' and 'ACC00001' are one account once cleaned."""
    load(db, *(record(f"pay_{i}", seconds=i * 10, account=" acc00001 " if i % 2 else "ACC00001") for i in range(7)))
    run_clean(db)
    assert flagged(db) == ["pay_6"]


def test_velocity_counts_per_account(db):
    load(db, *(record(f"pay_{i}", seconds=i * 10, account="ACC00001" if i % 2 else "ACC00002") for i in range(7)))
    run_clean(db)
    assert flagged(db) == []


def test_velocity_uses_event_time_not_arrival_order(db):
    records = [record(f"pay_{i}", seconds=i * 30) for i in range(7)]
    random.Random(3).shuffle(records)
    load(db, *records)
    run_clean(db)
    assert flagged(db) == ["pay_6"]


def test_velocity_uses_fixed_timestamps(db):
    """Mixed timestamp formats are normalized before the rule counts them."""
    formats = [
        lambda t: t.strftime("%Y-%m-%dT%H:%M:%SZ"),
        lambda t: t.strftime("%Y-%m-%d %H:%M:%S"),
        lambda t: int(t.timestamp()),
    ]
    load(db, *(record(f"pay_{i}", timestamp=formats[i % 3](BASE + timedelta(seconds=i * 10))) for i in range(7)))
    run_clean(db)
    assert flagged(db) == ["pay_6"]


def test_duplicates_and_quarantined_records_do_not_count_towards_velocity(db):
    """6 distinct good payments, each delivered twice, plus one quarantined: not 7."""
    records = [record(f"pay_{i}", seconds=i * 10) for i in range(6) for _ in range(2)]
    records.append(record("pay_bad", seconds=65, amount=-1))
    load(db, *records)
    run_clean(db)
    assert len(clean(db)) == 6
    assert flagged(db) == []


def test_amount_flagged_payment_keeps_reason_amount_and_still_counts(db):
    load(db, *(record(f"pay_{i}", seconds=i * 10, amount="$2,500.00" if i == 2 else 20.0) for i in range(7)))
    run_clean(db)
    rows = clean(db)
    assert rows["pay_2"]["reason"] == "amount"
    assert rows["pay_6"]["reason"] == "velocity"
    assert flagged(db) == ["pay_2", "pay_6"]
