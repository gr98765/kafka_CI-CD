import csv
import json
from collections import defaultdict
from datetime import datetime

import pytest

from generator.generate import MESS_RATES, TS_FORMAT, UNRECOVERABLE, generate, poisson


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    manifest = generate(out, accounts=400, days=30, seed=7)
    return out, manifest


def test_same_seed_gives_same_files(tmp_path, data):
    out, _ = data
    generate(tmp_path, accounts=400, days=30, seed=7)
    for name in ("payments_raw.jsonl", "labels.csv", "mess_log.csv"):
        assert (tmp_path / name).read_bytes() == (out / name).read_bytes()


def test_different_seed_gives_different_data(tmp_path, data):
    out, _ = data
    generate(tmp_path, accounts=400, days=30, seed=8)
    assert (tmp_path / "payments_raw.jsonl").read_bytes() != (
        out / "payments_raw.jsonl"
    ).read_bytes()


def test_truth_ids_unique_and_time_sorted(data):
    out, manifest = data
    truth = read_jsonl(out / "payments_truth.jsonl")
    ids = [r["payment_id"] for r in truth]
    assert len(ids) == len(set(ids)) == manifest["payments"]
    times = [datetime.strptime(r["timestamp"], TS_FORMAT) for r in truth]
    assert times == sorted(times)


def test_raw_has_exactly_the_logged_duplicates(data):
    _, manifest = data
    dupes = manifest["mess_by_issue"]["duplicate"]
    assert manifest["raw_records"] == manifest["payments"] + dupes


def test_every_mess_kind_appears_and_is_logged(data):
    _, manifest = data
    assert set(manifest["mess_by_issue"]) == set(MESS_RATES)


def test_fraud_never_gets_unfixable_mess(data):
    out, _ = data
    fraud = {r["payment_id"] for r in read_csv(out / "labels.csv") if r["is_fraud"] == "1"}
    bad = [
        m
        for m in read_csv(out / "mess_log.csv")
        if m["issue"] in UNRECOVERABLE and m["payment_id"] in fraud
    ]
    assert bad == []


def test_all_four_fraud_patterns_present(data):
    _, manifest = data
    assert set(manifest["fraud_by_pattern"]) == {
        "velocity_burst",
        "card_testing",
        "impossible_travel",
        "amount_outlier",
    }


def test_fraud_share_is_small(data):
    _, manifest = data
    share = manifest["fraud_payments"] / manifest["payments"]
    assert 0.002 < share < 0.03


def test_bursts_are_dense_and_amount_outliers_are_big(data):
    out, _ = data
    truth = {r["payment_id"]: r for r in read_jsonl(out / "payments_truth.jsonl")}
    incidents = defaultdict(list)
    for label in read_csv(out / "labels.csv"):
        if label["is_fraud"] == "1":
            incidents[(label["pattern"], label["incident_id"])].append(
                truth[label["payment_id"]]
            )
    for (pattern, _), payments in incidents.items():
        if pattern in ("velocity_burst", "card_testing"):
            assert len(payments) >= 6
        if pattern == "amount_outlier":
            assert len(payments) == 1


def test_impossible_travel_is_abroad(data):
    out, _ = data
    home = {a["account_id"]: a["home_country"] for a in read_csv(out / "accounts.csv")}
    truth = {r["payment_id"]: r for r in read_jsonl(out / "payments_truth.jsonl")}
    travel = [
        truth[x["payment_id"]]
        for x in read_csv(out / "labels.csv")
        if x["pattern"] == "impossible_travel"
    ]
    assert travel
    assert all(p["country"] != home[p["account_id"]] for p in travel)


def test_some_legit_account_has_7_in_180_seconds(data):
    """Heavy legit bursts exist, so a 7-in-180s velocity rule has honest false alarms."""
    out, manifest = data
    assert manifest["legit_heavy_bursts"] > 0
    legit = {r["payment_id"] for r in read_csv(out / "labels.csv") if r["is_fraud"] == "0"}
    times = defaultdict(list)
    for r in read_jsonl(out / "payments_truth.jsonl"):
        if r["payment_id"] in legit:
            times[r["account_id"]].append(datetime.strptime(r["timestamp"], TS_FORMAT).timestamp())

    def max_in_window(ts, window=180):
        ts.sort()
        left, best = 0, 0
        for i, t in enumerate(ts):
            while ts[left] < t - window:
                left += 1
            best = max(best, i - left + 1)
        return best

    assert max(max_in_window(ts) for ts in times.values()) >= 7


def test_poisson_average():
    import random

    rng = random.Random(1)
    mean = sum(poisson(rng, 2.0) for _ in range(5000)) / 5000
    assert 1.9 < mean < 2.1
