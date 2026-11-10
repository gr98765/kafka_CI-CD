"""The pure parts of scripts/: scoring maths, 422 parsing, kill-test tagging."""

import pytest

from scripts.kill_test import tag_record
from scripts.replay import Result, rejection_reason, summarize
from scripts.score import compute

LABELS = {
    "p1": (True, "velocity_burst"),  # fraud, flagged        -> caught
    "p2": (True, "velocity_burst"),  # fraud, cleared        -> missed
    "p3": (True, "amount_outlier"),  # fraud, never stored   -> missed
    "p4": (False, "normal"),  # legit, flagged           -> false alarm
    "p5": (False, "normal"),  # legit, cleared
    "p6": (False, "normal"),  # legit, cleared
}
AMOUNTS = {"p1": 100.0, "p2": 50.0, "p3": 2000.0, "p4": 1500.0, "p5": 10.0, "p6": 10.0}
STORED = {
    "p1": ("flagged", "velocity", 100.0),
    "p2": ("cleared", None, 50.0),
    "p4": ("flagged", "amount", 1500.0),
    "p5": ("cleared", None, 10.0),
    "p6": ("cleared", None, 10.0),
    "kt123_p9": ("flagged", "velocity", 5.0),  # not labelled: ignored
}


def test_compute_scores():
    s = compute(LABELS, AMOUNTS, STORED)

    assert (s["fraud"], s["legit"], s["stored"], s["fraud_stored"]) == (3, 3, 5, 2)
    assert (s["flagged"], s["caught"], s["false_alarms"]) == (2, 1, 1)
    assert s["recall"] == pytest.approx(100 / 3)  # never-stored fraud counts as missed
    assert s["precision"] == pytest.approx(50.0)
    assert s["false_alarm_rate"] == pytest.approx(100 / 3)  # 1 of 3 stored legit
    assert s["fraud_dollars"] == 2150.0
    assert s["dollars_blocked"] == 100.0
    assert s["legit_dollars_blocked"] == 1500.0
    assert s["by_pattern"]["velocity_burst"] == {"total": 2, "caught": 1, "not_stored": 0}
    assert s["by_pattern"]["amount_outlier"] == {"total": 1, "caught": 0, "not_stored": 1}
    assert s["by_reason"] == {
        "amount": {"flagged": 1, "fraud": 0},
        "velocity": {"flagged": 1, "fraud": 1},
    }


def test_compute_with_nothing_flagged_has_no_precision():
    stored = {p: ("cleared", None, 1.0) for p in LABELS}
    s = compute(LABELS, AMOUNTS, stored)
    assert s["precision"] is None
    assert s["recall"] == 0


def test_rejection_reason():
    body = {"detail": [{"loc": ["body", "amount"], "msg": "Input should be greater than 0"}]}
    assert rejection_reason(body) == "amount: Input should be greater than 0"
    assert rejection_reason({"detail": "nope"}) == "unrecognised 422 body"


def test_summarize_counts_unique_accepted_ids():
    results = [
        Result("p1", 202),
        Result("p1", 202),  # resent duplicate
        Result("p2", 422, "amount: bad"),
        Result("p3", 0, "ConnectError"),
    ]
    s = summarize(results)
    assert (s["posted"], s["accepted"], s["rejected"], s["failed"]) == (4, 2, 1, 1)
    assert s["unique_accepted"] == 1


def test_tag_record_tags_ids_only():
    record = {"payment_id": "pay_1", "account_id": " acc00012 ", "amount": "$5"}
    assert tag_record(record, "kt1") == {
        "payment_id": "kt1_pay_1",
        "account_id": "kt1_ acc00012 ",
        "amount": "$5",
    }
    assert record["payment_id"] == "pay_1"  # original untouched
