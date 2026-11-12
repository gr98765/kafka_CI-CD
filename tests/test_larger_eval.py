"""scripts/larger_eval.py: rules, loading, the split, learning-curve days,
aggregation and the SQL used for timing."""

import json
import random
import uuid
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from generator.generate import generate
from scripts.larger_eval import (
    INDEX_NAME,
    LEARNING_FRACTIONS,
    SPLIT,
    LabelArrays,
    arrays_from_records,
    compact_features,
    csv_chunks,
    daily_summary_sql,
    evaluate_compact,
    fixed_rules,
    home_by_account,
    learning_days,
    load_arrays,
    load_label_arrays,
    load_schema,
    pattern_counts,
    plan_uses_index,
    rule_flags,
    rule_mask,
    spread,
    truth_to_csv_row,
    tuned_rules,
    velocity_sql,
)
from scripts.score import rule_hits
from scripts.train_model import (
    Payment,
    build_features,
    day_of,
    evaluate,
    load_home,
    make_models,
    threshold_for_false_alarms,
    window_of,
)

T0 = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)


def pay(pid, secs, amount=20.0, account="A"):
    return Payment(pid, account, T0 + timedelta(seconds=secs), amount, "online", "US")


# --- rules ----------------------------------------------------------------------


def test_fixed_rules_match_score_rule_hits():
    rng = random.Random(7)
    payments = [
        pay(f"p{i:04d}", rng.randrange(0, 3000), rng.choice([2.0, 30.0, 1500.0]), rng.choice("ABC"))
        for i in range(400)
    ]
    hits = rule_hits([(p.payment_id, p.account_id, p.event_time, p.amount) for p in payments])
    assert fixed_rules(payments) == {pid for pid, h in hits.items() if h}


def test_tuned_velocity_flags_the_fifth_payment_in_three_minutes():
    payments = [pay(f"p{i}", i * 30) for i in range(5)]  # 0..120 s
    assert tuned_rules(payments) == {"p4"}
    assert fixed_rules(payments) == set()


def test_tiny_rule_flags_from_the_fifth_tiny_payment_including_a_cash_out():
    """Five tiny probes then a big payment within 3 minutes: the 5th probe and
    the cash-out are flagged. Spaced so velocity 5 alone would not fire."""
    probes = [pay(f"t{i}", i * 40, 1.5) for i in range(5)]  # 0..160 s
    cash_out = pay("cash", 170, 250.0)
    flags = rule_flags([*probes, cash_out], velocity_limit=99, tiny_limit=5)
    assert flags == {"t4", "cash"}


def test_tiny_rule_needs_five_tiny_payments_inside_the_window():
    four = [pay(f"t{i}", i * 30, 1.0) for i in range(4)] + [pay("big", 130, 50.0)]
    assert rule_flags(four, velocity_limit=99, tiny_limit=5) == set()

    spread_out = [pay(f"s{i}", i * 50, 1.0) for i in range(5)]  # 0..200 s: s0 drops out at 200
    assert rule_flags(spread_out, velocity_limit=99, tiny_limit=5) == set()

    five_dollars = [pay(f"f{i}", i, 5.0) for i in range(6)]  # "under $5": $5.00 is not tiny
    assert rule_flags(five_dollars, velocity_limit=99, tiny_limit=5) == set()


def test_tiny_rule_counts_per_account():
    mixed = [pay(f"a{i}", i, 1.0, "A") for i in range(3)] + [pay(f"b{i}", i, 1.0, "B") for i in range(3)]
    assert rule_flags(mixed, velocity_limit=99, tiny_limit=5) == set()


def test_amount_rule_is_in_both_rule_sets():
    big = [pay("big", 0, 1000.01), pay("edge", 10, 1000.00)]
    assert fixed_rules(big) == tuned_rules(big) == {"big"}


# --- split, learning curve, counts, aggregation -----------------------------------------


def test_split_covers_sixty_days_with_no_gaps():
    windows = [window_of(d, SPLIT) for d in range(1, 61)]
    assert windows.count("train") == 36
    assert windows.count("validation") == 10
    assert windows.count("test") == 14
    assert window_of(61, SPLIT) is None
    last_val, first_test = datetime(2026, 11, 15, 23, 59, 59, tzinfo=UTC), datetime(2026, 11, 16, tzinfo=UTC)
    assert (day_of(last_val), window_of(day_of(last_val), SPLIT)) == (46, "validation")
    assert (day_of(first_test), window_of(day_of(first_test), SPLIT)) == (47, "test")


def test_learning_days_end_at_the_last_training_day():
    assert [learning_days(f) for f in LEARNING_FRACTIONS] == [(28, 36), (19, 36), (1, 36)]


def test_pattern_counts_payments_and_incidents():
    lab = LabelArrays(
        fraud=np.array([True, True, True, False, True]),
        pattern=np.array([1, 1, 0, -1, 1], dtype=np.int8),  # card_testing x3, velocity_burst, legit
        incident=np.array([4, 4, 7, -1, 9], dtype=np.int32),
    )
    win = np.array([True, True, True, True, False])  # the last one is outside the window
    c = pattern_counts(win, lab)
    assert c["card_testing"] == {"payments": 2, "incidents": 1}
    assert c["velocity_burst"] == {"payments": 1, "incidents": 1}
    assert c["amount_outlier"] == {"payments": 0, "incidents": 0}


def test_spread_ignores_missing_values():
    assert spread([10.0, None, 30.0]) == (20.0, 10.0, 30.0)
    assert spread([None]) is None


# --- loading ----------------------------------------------------------------------------


RECORD = {
    "payment_id": "pay_1",
    "account_id": "ACC00001",
    "merchant_id": "M0001",
    "amount": 12.5,
    "currency": "USD",
    "timestamp": "2026-10-01T03:00:59Z",
    "country": "US",
    "channel": "online",
}


def test_truth_record_becomes_a_clean_row():
    row = truth_to_csv_row(RECORD, 7)
    assert row == [
        "pay_1", "ACC00001", "M0001", 12.5, "USD", "US", "online",
        "unknown", "2026-10-01T03:00:59Z", "cleared", "2026-10-01T03:00:59Z", 7,
    ]  # fmt: skip


def test_csv_chunks_split_and_skip_blank_lines():
    lines = [json.dumps({**RECORD, "payment_id": f"p{i}"}) for i in range(5)]
    chunks = [c.read() for c in csv_chunks([*lines[:2], "\n", *lines[2:]], size=2)]
    assert [c.count("\n") for c in chunks] == [2, 2, 1]
    assert chunks[-1].startswith("p4,")


@pytest.fixture
def eval_schema(db):
    """A schema name for load_schema, dropped afterwards."""
    name = f"eval_test_{uuid.uuid4().hex[:8]}"
    yield name
    db.rollback()
    with db.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {name} CASCADE")
    db.commit()


def write_truth(path, n):
    with open(path, "w") as f:
        for i in range(n):
            ts = (T0 + timedelta(seconds=10 * i)).strftime("%Y-%m-%dT%H:%M:%SZ")
            f.write(json.dumps({**RECORD, "payment_id": f"p{i}", "account_id": f"A{i % 3}", "timestamp": ts}) + "\n")


def test_load_schema_copies_every_record_and_resets_on_reload(db, eval_schema, tmp_path):
    truth = tmp_path / "truth.jsonl"
    write_truth(truth, 25)
    assert load_schema(db, eval_schema, truth) == 25
    assert load_schema(db, eval_schema, truth) == 25  # dropped and reloaded, not appended

    with db.cursor() as cur:
        cur.execute(f"SELECT account_id, amount, event_time, status FROM {eval_schema}.payments_clean WHERE payment_id = 'p4'")
        assert cur.fetchone() == ("A1", 12.50, T0 + timedelta(seconds=40), "cleared")
        cur.execute(
            "SELECT count(*) FROM pg_indexes WHERE schemaname = %s AND indexname <> %s",
            (eval_schema, "payments_clean_pkey"),
        )
        assert cur.fetchone()[0] == 0  # no account/time index until sql-speed adds one


def test_velocity_sql_matches_the_python_rule(db, eval_schema, tmp_path):
    truth = tmp_path / "truth.jsonl"
    with open(truth, "w") as f:
        for i in range(12):  # A: 12 payments 10 s apart -> 7th..12th flagged; B: spaced out
            ts = (T0 + timedelta(seconds=10 * i)).strftime("%Y-%m-%dT%H:%M:%SZ")
            f.write(json.dumps({**RECORD, "payment_id": f"a{i:02d}", "account_id": "A", "timestamp": ts}) + "\n")
            ts = (T0 + timedelta(seconds=100 * i)).strftime("%Y-%m-%dT%H:%M:%SZ")
            f.write(json.dumps({**RECORD, "payment_id": f"b{i:02d}", "account_id": "B", "timestamp": ts}) + "\n")
    load_schema(db, eval_schema, truth)
    params = {"window": 180, "limit": 7, "accounts": ["A"]}
    with db.cursor() as cur:
        cur.execute(velocity_sql(eval_schema), params)
        assert cur.fetchone()[0] == 6
        cur.execute(velocity_sql(eval_schema, accounts_filter=True), params)
        assert cur.fetchone()[0] == 6
        cur.execute(
            daily_summary_sql(eval_schema),
            {"account": "B", "start": "2026-10-01T00:00:00Z", "end": "2026-10-31T00:00:00Z"},
        )
        [(_, n, total, largest)] = cur.fetchall()
        assert (n, float(total), float(largest)) == (12, 150.0, 12.5)


def test_plan_uses_index():
    assert plan_uses_index(f"Index Scan using {INDEX_NAME} on payments_clean w", INDEX_NAME)
    assert plan_uses_index(f"Bitmap Index Scan on {INDEX_NAME}", INDEX_NAME)
    assert not plan_uses_index("Seq Scan on payments_clean w", INDEX_NAME)


# --- compact arrays give the same answers as the reference path -----------------------


def random_payments(seed=11, n=1500):
    """Several accounts, bursts, same-second ties, tiny and big amounts."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        secs = rng.randrange(0, 4000) if rng.random() < 0.9 else rng.randrange(0, 40) * 30
        amount = rng.choice([0.75, 2.0, 4.99, 5.0, 20.0, 37.5, 250.0, 1000.0, 1000.01, 3000.0])
        out.append(
            Payment(f"p{i:05d}", rng.choice(["A", "B", "C", "D"]), T0 + timedelta(seconds=secs), amount,
                    rng.choice(["online", "in_person"]), rng.choice(["US", "US", "DE", "FR"]))
        )  # fmt: skip
    return out


def to_arrays(payments):
    return arrays_from_records(
        {"payment_id": p.payment_id, "account_id": p.account_id, "amount": p.amount,
         "timestamp": p.event_time.strftime("%Y-%m-%dT%H:%M:%SZ"), "channel": p.channel, "country": p.country}
        for p in payments
    )  # fmt: skip


@pytest.mark.parametrize(
    ("velocity_limit", "tiny_limit"), [(7, None), (5, 5), (3, 2), (5, None)], ids=["fixed", "tuned", "tight", "v5"]
)
def test_rule_mask_matches_reference_rule_flags(velocity_limit, tiny_limit):
    payments = random_payments()
    a = to_arrays(payments)
    expected = rule_flags(payments, velocity_limit, tiny_limit=tiny_limit)
    mask = rule_mask(a, velocity_limit, tiny_limit=tiny_limit)
    assert set(a.pid[mask].tolist()) == expected
    assert expected  # the data does trip the rules


def test_compact_features_match_build_features():
    payments = random_payments()
    home = {"A": "US", "B": "DE", "C": "US"}  # D has no home country
    channels = ["in_person", "online"]
    ids, X_ref = build_features(payments, home, channels)
    a = to_arrays(payments)
    X = compact_features(a, home_by_account(a, home), channels)
    assert a.pid.tolist() == ids  # same row order
    np.testing.assert_allclose(X, X_ref, rtol=1e-12, atol=1e-9)


def test_arrays_are_sorted_by_time_then_payment_id():
    a = to_arrays([pay("p2", 5), pay("p1", 5), pay("p0", 9), pay("p3", 0)])
    assert a.pid.tolist() == ["p3", "p1", "p2", "p0"]


def test_labels_must_cover_every_payment(tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text("payment_id,is_fraud,pattern,incident_id\np1,0,normal,\n")
    with pytest.raises(ValueError, match="no label"):
        load_label_arrays(labels, to_arrays([pay("p1", 0), pay("p2", 1)]))


SMALL_SPLIT = {"train": (1, 6), "validation": (7, 8), "test": (9, 10)}


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("small")
    generate(out, accounts=400, days=10, seed=5)
    return out


def reference_evaluation(data, split):
    """The old path: one Payment per payment, sets of ids, train_model.evaluate."""
    labels, truth = {}, []
    with open(data / "labels.csv") as f:
        next(f)
        for line in f:
            pid, is_fraud, pattern, _ = line.rstrip("\n").split(",")
            labels[pid] = (is_fraud == "1", pattern)
    with open(data / "payments_truth.jsonl") as f:
        for r in map(json.loads, f):
            truth.append(
                Payment(r["payment_id"], r["account_id"],
                        datetime.strptime(r["timestamp"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC),
                        float(r["amount"]), r["channel"], r["country"])
            )  # fmt: skip
    home = load_home(data / "accounts.csv")
    amounts = {p.payment_id: p.amount for p in truth}
    windows = {w: {p.payment_id for p in truth if window_of(day_of(p.event_time), split) == w} for w in split}
    rules = {"fixed rules": fixed_rules(truth), "hand-tuned rules": tuned_rules(truth)}
    out = {name: {w: evaluate(flags, windows[w], labels, amounts) for w in split} for name, flags in rules.items()}
    budget = out["fixed rules"]["validation"]["false_alarms"]

    channels = sorted({p.channel for p in truth})
    ids, X = build_features(truth, home, channels)
    y = np.array([labels[p][0] for p in ids])
    win_of = np.array([window_of(day_of(p.event_time), split) or "" for p in sorted(truth, key=lambda p: (p.event_time, p.payment_id))])
    train, val = np.flatnonzero(win_of == "train"), np.flatnonzero(win_of == "validation")
    for name, model in make_models(channels).items():
        model.fit(X[train], y[train])
        scores = model.predict_proba(X)[:, 1]
        threshold = threshold_for_false_alarms(scores[val], y[val], budget)
        flags = {ids[i] for i in np.flatnonzero(scores >= threshold)}
        out[name] = {w: evaluate(flags, windows[w], labels, amounts) for w in split}
    return out


def test_compact_evaluation_matches_the_reference_path(small_dataset):
    """Same recall, precision and false alarms for every setup and window."""
    expected = reference_evaluation(small_dataset, SMALL_SPLIT)
    with open(small_dataset / "payments_truth.jsonl") as f:
        a = arrays_from_records(map(json.loads, f))
    lab = load_label_arrays(small_dataset / "labels.csv", a)
    home = home_by_account(a, load_home(small_dataset / "accounts.csv"))
    got = evaluate_compact(a, lab, home, SMALL_SPLIT)["setups"]

    assert set(got) == set(expected) == {"fixed rules", "hand-tuned rules", "logistic_regression", "gradient_boosting"}
    for name in expected:
        for w in SMALL_SPLIT:
            e, g = expected[name][w], got[name][w]
            assert (g["recall"], g["precision"], g["false_alarms"]) == (e["recall"], e["precision"], e["false_alarms"]), (name, w)
            assert g["by_pattern"] == e["by_pattern"], (name, w)
    assert expected["fixed rules"]["test"]["fraud"] > 0  # the comparison is not vacuous


def test_load_arrays_reads_what_was_copied(db, eval_schema, tmp_path):
    truth = tmp_path / "truth.jsonl"
    write_truth(truth, 25)
    load_schema(db, eval_schema, truth)
    from_db = load_arrays(db, eval_schema)
    with open(truth) as f:
        from_file = arrays_from_records(map(json.loads, f))
    for field in ("pid", "acct", "acct_names", "t", "amount", "channel", "country"):
        np.testing.assert_array_equal(getattr(from_db, field), getattr(from_file, field))
