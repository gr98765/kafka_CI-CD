"""scripts/train_model.py: features without future leakage, the time split,
threshold choice and scoring. No database needed."""

from datetime import UTC, datetime, timedelta

import joblib
import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from scripts.train_model import (
    FIRST_GAP,
    Payment,
    build_features,
    day_of,
    evaluate,
    feature_names,
    make_models,
    threshold_for_false_alarms,
    window_of,
)

CHANNELS = ["in_person", "online"]
T0 = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
HOME = {"A": "US", "B": "DE"}


def pay(pid, secs, amount=10.0, account="A", channel="online", country="US"):
    return Payment(pid, account, T0 + timedelta(seconds=secs), amount, channel, country)


def features(payments):
    ids, X = build_features(payments, HOME, CHANNELS)
    names = feature_names(CHANNELS)
    return {pid: dict(zip(names, row, strict=True)) for pid, row in zip(ids, X, strict=True)}


def test_window_counts_only_earlier_payments_of_the_same_account():
    f = features(
        [
            pay("p1", 0),
            pay("p2", 60),
            pay("p3", 179),
            pay("p4", 240),  # p1 is now 240 s back: outside 3 minutes, inside the hour
            pay("p5", 3700),  # p3 and p4 are within the hour; p1 and p2 are not
            pay("other", 61, account="B", country="DE"),
        ]
    )
    assert [f[p]["n_prev_3m"] for p in ["p1", "p2", "p3", "p4", "p5"]] == [0, 1, 2, 2, 0]
    assert [f[p]["n_prev_1h"] for p in ["p1", "p2", "p3", "p4", "p5"]] == [0, 1, 2, 3, 2]
    assert f["other"]["n_prev_3m"] == 0


def test_same_second_payments_are_ordered_by_payment_id():
    f = features([pay("p_b", 0), pay("p_a", 0)])
    assert (f["p_a"]["n_prev_3m"], f["p_b"]["n_prev_3m"]) == (0, 1)


def test_amount_ratio_and_gap_use_only_earlier_payments():
    f = features([pay("p1", 0, 10.0), pay("p2", 30, 30.0), pay("p3", 100, 40.0)])
    assert f["p1"]["amount_to_avg"] == 1.0  # nothing earlier
    assert f["p1"]["secs_since_prev"] == FIRST_GAP
    assert f["p2"]["amount_to_avg"] == pytest.approx(30 / 10)
    assert f["p3"]["amount_to_avg"] == pytest.approx(40 / 20)
    assert (f["p2"]["secs_since_prev"], f["p3"]["secs_since_prev"]) == (30, 70)


def test_later_payments_never_change_earlier_features():
    early = [pay("p1", 0, 10.0), pay("p2", 50, 25.0)]
    future = [pay("p3", 60, 9999.0), pay("p4", 61, 1.0, country="FR")]
    before, after = features(early), features(early + future)
    for pid in before:
        assert before[pid] == after[pid]


def test_hour_channel_and_foreign():
    f = features(
        [
            pay("home", 0, channel="in_person"),
            pay("abroad", 10, country="FR"),
            pay("b_home", 20, account="B", country="DE"),
        ]
    )
    assert f["home"]["hour"] == 12
    assert (f["home"]["channel_in_person"], f["home"]["channel_online"]) == (1.0, 0.0)
    assert (f["home"]["foreign"], f["abroad"]["foreign"], f["b_home"]["foreign"]) == (0.0, 1.0, 0.0)


@pytest.mark.parametrize(
    ("ts", "day", "window"),
    [
        ("2026-10-01T00:00:00Z", 1, "train"),
        ("2026-10-21T23:59:59Z", 21, "train"),
        ("2026-10-22T00:00:00Z", 22, "validation"),
        ("2026-10-25T23:59:59Z", 25, "validation"),
        ("2026-10-26T00:00:00Z", 26, "test"),
        ("2026-10-30T23:59:59Z", 30, "test"),
        ("2026-10-31T00:00:00Z", 31, None),
    ],
)
def test_time_split(ts, day, window):
    d = day_of(datetime.fromisoformat(ts))
    assert (d, window_of(d)) == (day, window)


def test_threshold_allows_exactly_the_budget_of_false_alarms():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.95])
    fraud = np.array([False, False, False, False, False, True])
    for budget in range(6):
        t = threshold_for_false_alarms(scores, fraud, budget)
        assert int(((scores >= t) & ~fraud).sum()) == budget


def test_threshold_never_exceeds_the_budget_on_ties():
    scores = np.array([0.9, 0.7, 0.7, 0.1])
    fraud = np.zeros(4, dtype=bool)
    t = threshold_for_false_alarms(scores, fraud, 2)
    assert int((scores >= t).sum()) == 1  # taking both 0.7s would make 3


def test_evaluate_counts_unscored_fraud_as_missed():
    labels = {
        "f1": (True, "card_testing"),
        "f2": (True, "velocity_burst"),
        "f3": (True, "card_testing"),  # in the window but never stored
        "l1": (False, "normal"),
        "l2": (False, "normal"),
        "late": (True, "card_testing"),  # outside the window
    }
    amounts = {"f1": 10.0, "f2": 20.0, "f3": 70.0, "l1": 5.0, "l2": 5.0, "late": 1.0}
    s = evaluate({"f1", "l1", "late"}, {"f1", "f2", "f3", "l1", "l2"}, labels, amounts)
    assert (s["fraud"], s["caught"], s["flagged"], s["false_alarms"]) == (3, 1, 2, 1)
    assert s["recall"] == pytest.approx(100 / 3)
    assert s["precision"] == 50.0
    assert (s["dollars_blocked"], s["fraud_dollars"]) == (10.0, 100.0)
    assert s["by_pattern"]["card_testing"] == (1, 2)
    assert s["by_pattern"]["velocity_burst"] == (0, 1)
    assert s["by_pattern"]["impossible_travel"] == (0, 0)


def test_models_learn_a_burst_and_survive_a_joblib_round_trip(tmp_path):
    """Bursts (many payments seconds apart) are fraud; spaced payments are not."""
    payments, labels = [], {}
    for a in range(40):
        acct, burst = f"ACC{a:03d}", a % 4 == 0
        for k in range(8):
            pid = f"{acct}_{k}"
            secs = a * 10_000 + (k * 15 if burst else k * 1_200)
            payments.append(pay(pid, secs, 20.0 + k, account=acct))
            labels[pid] = burst and k >= 2
    ids, X = build_features(payments, {}, CHANNELS)
    y = np.array([labels[p] for p in ids])

    for name, model in make_models(CHANNELS).items():
        model.fit(X, y)
        scores = model.predict_proba(X)[:, 1]
        assert average_precision_score(y, scores) > 0.9, name

        path = tmp_path / f"{name}.joblib"
        joblib.dump({"model": model}, path)
        assert np.allclose(joblib.load(path)["model"].predict_proba(X)[:, 1], scores)
