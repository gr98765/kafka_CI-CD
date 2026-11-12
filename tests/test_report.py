"""scripts/report.py: reading Locust's CSVs and comparing sent with stored.
Uses the small sample files in tests/data (same columns as Locust 2.47 writes)."""

import shutil
from collections import Counter
from pathlib import Path

import pytest

from scripts.report import (
    HANDOFF_NOTE,
    compare,
    parse_failures,
    parse_stats,
    run_tag,
    stats_files,
    stats_lines,
)

DATA = Path(__file__).parent / "data"
STATS = DATA / "locust_stats_sample.csv"
FAILURES = DATA / "locust_failures_sample.csv"


def test_parse_stats_reads_the_aggregated_row():
    s = parse_stats(STATS)
    assert s == {
        "requests": 1200,
        "failures": 3,
        "rps": 300.25,
        "p50": 140.0,
        "p95": 250.0,
        "p99": 350.0,
        "avg": 152.5,
        "max": 931.4,
    }


def test_parse_stats_can_read_a_named_row_and_rejects_a_missing_one():
    assert parse_stats(STATS, name="POST /transactions")["requests"] == 1200
    with pytest.raises(ValueError, match="no 'GET /x' row"):
        parse_stats(STATS, name="GET /x")


def test_parse_failures_sums_by_error():
    assert parse_failures(FAILURES) == Counter({"HTTP 503": 2, "ConnectionResetError(10054, 'reset')": 1})
    assert parse_failures(DATA / "does_not_exist.csv") == Counter()


def test_stats_lines_show_the_numbers_and_failure_reasons():
    text = "\n".join(stats_lines(parse_stats(STATS), parse_failures(FAILURES)))
    for expected in ["requests/s      300.2", "p50     140 ms", "p95     250 ms", "p99     350 ms",
                     "failures        3 (HTTP 503: 2, ConnectionResetError(10054, 'reset'): 1)"]:  # fmt: skip
        assert expected in text


def test_stats_files_prefer_the_final_csv(tmp_path):
    prefix = tmp_path / "locust"
    shutil.copy(STATS, f"{prefix}_stats.csv")
    assert stats_files(prefix) == (Path(f"{prefix}_stats.csv"), Path(f"{prefix}_failures.csv"), False)
    shutil.copy(STATS, f"{prefix}_stats_final.csv")
    assert stats_files(prefix) == (Path(f"{prefix}_stats_final.csv"), Path(f"{prefix}_failures_final.csv"), True)


def test_compare_reports_lost_duplicated_and_unacknowledged():
    sent = ["lt1_001", "lt1_002", "lt1_003", "lt1_004"]
    copies = Counter({"lt1_001": 1, "lt1_002": 3, "lt1_004": 1, "lt1_009": 1})  # 003 missing, 009 never acked
    assert compare(sent, copies) == {
        "sent": 4,
        "stored": 3,
        "rows": 6,
        "lost": 1,
        "duplicated": 2,
        "stored_not_acknowledged": 1,
    }


def test_run_tag_needs_exactly_one_tag():
    assert run_tag(["ltabc_0000001", "ltabc_0000002"]) == "ltabc"
    with pytest.raises(ValueError, match="one run tag"):
        run_tag(["lta_1", "ltb_1"])


def test_handoff_note_wording():
    assert HANDOFF_NOTE == (
        "The API replies after handing the payment to Kafka, so p95 measures the hand-off, not saving to Postgres."
    )
