"""Train fraud models on payments_clean and compare them with the rules.

    python -m scripts.train_model [--out results/model_comparison.md] [--model-out models/fraud_model.joblib]

Run after the clean step (python -m app.clean). Reads payments_clean, the
generator's labels.csv, payments_truth.jsonl (true amounts and timestamps,
also for payments that were quarantined) and accounts.csv (home countries).

Features use only the account's earlier payments, never later ones:
  amount, hour (UTC), channel (one-hot), foreign (country != home country),
  payments by the account in the previous 3 minutes and in the previous hour,
  amount / mean of all the account's earlier amounts, seconds since the
  account's previous payment. "Earlier" means before in (event_time,
  payment_id) order, the same order the velocity rule uses.

Split by event day: train days 1-21, validation 22-25, test 26-30.

Thresholds: the rules' false alarms on the test window are the budget. Each
model's threshold is chosen on the validation window only, to give the same
number of false alarms per legit payment there (budget scaled by legit
validation / legit test payments), then applied unchanged to the test window.
So the model's test false alarms land close to the rules', not exactly on them.

Everything printed is for the test window. Payments that never reached
payments_clean (quarantined) cannot be flagged, and count as missed fraud.
"""

from __future__ import annotations

import argparse
import csv
import json
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from generator.generate import TS_FORMAT
from scripts.score import (
    DATA,
    RESULTS,
    ROOT,
    _fmt_pct,
    _pct,
    db_connect,
    load_amounts,
    load_labels,
)

START = date(2026, 10, 1)  # day 1 (manifest.json "start")
WINDOWS = {"train": (1, 21), "validation": (22, 25), "test": (26, 30)}
PATTERNS = ["velocity_burst", "card_testing", "impossible_travel", "amount_outlier"]
FIRST_GAP = 30 * 24 * 3600.0  # "seconds since previous" for an account's first payment
SEED = 0

NUMERIC = ["amount", "n_prev_3m", "n_prev_1h", "amount_to_avg", "secs_since_prev"]


@dataclass(frozen=True)
class Payment:
    payment_id: str
    account_id: str
    event_time: datetime
    amount: float
    channel: str
    country: str


def feature_names(channels: list[str]) -> list[str]:
    return ["amount", "hour", *(f"channel_{c}" for c in channels), "foreign", *NUMERIC[1:]]


def build_features(
    payments: list[Payment], home: dict[str, str], channels: list[str]
) -> tuple[list[str], np.ndarray]:
    """One row per payment, in (event_time, payment_id) order. Returns (ids, X)."""
    ordered = sorted(payments, key=lambda p: (p.event_time, p.payment_id))
    by_account: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(ordered):
        by_account[p.account_id].append(i)

    n_prev_3m = np.zeros(len(ordered))
    n_prev_1h = np.zeros(len(ordered))
    to_avg = np.ones(len(ordered))
    since_prev = np.full(len(ordered), FIRST_GAP)
    for idx in by_account.values():
        ts = [ordered[i].event_time.timestamp() for i in idx]
        total = 0.0
        for k, i in enumerate(idx):
            n_prev_3m[i] = k - bisect_left(ts, ts[k] - 180, 0, k)
            n_prev_1h[i] = k - bisect_left(ts, ts[k] - 3600, 0, k)
            if k:
                avg = total / k
                to_avg[i] = ordered[i].amount / avg if avg > 0 else 1.0
                since_prev[i] = ts[k] - ts[k - 1]
            total += ordered[i].amount

    rows = []
    for i, p in enumerate(ordered):
        rows.append(
            [
                p.amount,
                p.event_time.astimezone(UTC).hour,
                *(1.0 if p.channel == c else 0.0 for c in channels),
                1.0 if p.country != home.get(p.account_id) else 0.0,
                n_prev_3m[i],
                n_prev_1h[i],
                to_avg[i],
                since_prev[i],
            ]
        )
    return [p.payment_id for p in ordered], np.array(rows, dtype=float)


def day_of(ts: datetime) -> int:
    return (ts.astimezone(UTC).date() - START).days + 1


def window_of(day: int) -> str | None:
    for name, (first, last) in WINDOWS.items():
        if first <= day <= last:
            return name
    return None


def make_models(channels: list[str]) -> dict[str, Pipeline]:
    names = feature_names(channels)
    col = {n: i for i, n in enumerate(names)}
    passthrough = [col[f"channel_{c}"] for c in channels] + [col["foreign"]]
    logistic = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        # Counts, amounts and gaps are heavily skewed: log, then scale.
                        ("num", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()),
                         [col[n] for n in NUMERIC]),
                        ("hour", OneHotEncoder(categories=[list(range(24))]), [col["hour"]]),
                        ("flags", "passthrough", passthrough),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )  # fmt: skip
    boosting = Pipeline([("clf", HistGradientBoostingClassifier(random_state=SEED))])
    return {"logistic_regression": logistic, "gradient_boosting": boosting}


def threshold_for_false_alarms(scores: np.ndarray, is_fraud: np.ndarray, budget: int) -> float:
    """Lowest threshold with at most `budget` legit scores at or above it."""
    legit = np.sort(scores[~is_fraud])[::-1]
    if budget <= 0 or not len(legit):
        return float(np.nextafter(legit[0], np.inf)) if len(legit) else 0.5
    if budget >= len(legit):
        return float(legit[-1])
    # Ties at the cut would push past the budget: move just above them.
    cut = legit[budget - 1]
    return float(cut) if legit[budget] < cut else float(np.nextafter(cut, np.inf))


def evaluate(flagged: set[str], window: set[str], labels: dict, amounts: dict) -> dict:
    """Metrics over every labelled payment in the window (stored or not)."""
    fraud = {p for p in window if labels[p][0]}
    hit = flagged & window
    caught = hit & fraud
    fraud_dollars = sum(amounts[p] for p in fraud)
    blocked = sum(amounts[p] for p in caught)
    return {
        "fraud": len(fraud),
        "flagged": len(hit),
        "caught": len(caught),
        "false_alarms": len(hit - fraud),
        "recall": _pct(len(caught), len(fraud)),
        "precision": _pct(len(caught), len(hit)),
        "fraud_dollars": fraud_dollars,
        "dollars_blocked": blocked,
        "by_pattern": {
            name: (
                sum(1 for p in caught if labels[p][1] == name),
                sum(1 for p in fraud if labels[p][1] == name),
            )
            for name in PATTERNS
        },
    }


def results_table(results: dict[str, dict]) -> str:
    lines = [
        "| Setup | Recall | Precision | False alarms | Dollars blocked |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, s in results.items():
        lines.append(
            f"| {name} | {_fmt_pct(s['recall'])} ({s['caught']:,}/{s['fraud']:,}) "
            f"| {_fmt_pct(s['precision'])} ({s['caught']:,}/{s['flagged']:,}) | {s['false_alarms']:,} "
            f"| ${s['dollars_blocked']:,.2f} ({_fmt_pct(_pct(s['dollars_blocked'], s['fraud_dollars']))}) |"
        )
    return "\n".join(lines)


def pattern_table(results: dict[str, dict]) -> str:
    lines = ["| Setup | " + " | ".join(PATTERNS) + " |", "|---|" + "---:|" * len(PATTERNS)]
    for name, s in results.items():
        cells = []
        for pattern in PATTERNS:
            caught, total = s["by_pattern"][pattern]
            cells.append(f"{_fmt_pct(_pct(caught, total))} ({caught}/{total})" if total else "none in window")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_payments(conn, labelled: set[str]) -> tuple[list[Payment], set[str]]:
    """Clean payments from this dataset, and the ids the rules flagged."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT payment_id, account_id, event_time, amount, channel, country, status FROM payments_clean"
        )
        payments, flagged = [], set()
        for pid, acct, ts, amount, channel, country, status in cur:
            if pid not in labelled:
                continue
            payments.append(Payment(pid, acct, ts, float(amount), channel or "", country or ""))
            if status == "flagged":
                flagged.add(pid)
    return payments, flagged


def load_true_times(path: Path) -> dict[str, datetime]:
    with open(path) as f:
        return {
            r["payment_id"]: datetime.strptime(r["timestamp"], TS_FORMAT).replace(tzinfo=UTC)
            for r in map(json.loads, f)
            if r
        }


def load_home(path: Path) -> dict[str, str]:
    with open(path, newline="") as f:
        return {r["account_id"]: r["home_country"] for r in csv.DictReader(f)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--labels", type=Path, default=DATA / "labels.csv")
    p.add_argument("--truth", type=Path, default=DATA / "payments_truth.jsonl")
    p.add_argument("--accounts", type=Path, default=DATA / "accounts.csv")
    p.add_argument("--out", type=Path, default=RESULTS / "model_comparison.md")
    p.add_argument("--model-out", type=Path, default=ROOT / "models" / "fraud_model.joblib")
    args = p.parse_args()

    labels, amounts = load_labels(args.labels), load_amounts(args.truth)
    true_time, home = load_true_times(args.truth), load_home(args.accounts)
    conn = db_connect()
    try:
        payments, rule_flagged = load_payments(conn, set(labels))
    finally:
        conn.close()

    # Window membership by true event day, so quarantined payments are counted too.
    windows: dict[str, set[str]] = defaultdict(set)
    for pid, ts in true_time.items():
        if (w := window_of(day_of(ts))) is not None:
            windows[w].add(pid)

    channels = sorted({pay.channel for pay in payments})
    ids, X = build_features(payments, home, channels)
    y = np.array([labels[pid][0] for pid in ids])
    split = np.array([window_of(day_of(true_time[pid])) for pid in ids])
    idx = {w: np.flatnonzero(split == w) for w in WINDOWS}
    test_ids = [ids[i] for i in idx["test"]]

    rules = evaluate(rule_flagged, windows["test"], labels, amounts)
    legit_val, legit_test = int((~y[idx["validation"]]).sum()), int((~y[idx["test"]]).sum())
    budget = round(rules["false_alarms"] * legit_val / legit_test)

    print("SPLIT (clean payments / fraud among them)")
    for w, (first, last) in WINDOWS.items():
        print(f"  {w:<10} days {first:>2}-{last:<2}  {len(idx[w]):>7,} / {int(y[idx[w]].sum()):>4,}")
    print(f"budget: rules make {rules['false_alarms']:,} false alarms on {legit_test:,} legit test payments; "
          f"thresholds allow {budget:,} on {legit_val:,} legit validation payments")  # fmt: skip
    print()

    fitted, val_ap, thresholds, model_flags = {}, {}, {}, {}
    for name, model in make_models(channels).items():
        model.fit(X[idx["train"]], y[idx["train"]])
        val_scores = model.predict_proba(X[idx["validation"]])[:, 1]
        val_ap[name] = average_precision_score(y[idx["validation"]], val_scores)
        thresholds[name] = threshold_for_false_alarms(val_scores, y[idx["validation"]], budget)
        test_scores = model.predict_proba(X[idx["test"]])[:, 1]
        model_flags[name] = {pid for pid, s in zip(test_ids, test_scores, strict=True) if s >= thresholds[name]}
        fitted[name] = model
        val_fa = int(((val_scores >= thresholds[name]) & ~y[idx["validation"]]).sum())
        print(f"{name}: validation average precision {val_ap[name]:.3f}, threshold {thresholds[name]:.4f} "
              f"({val_fa:,} validation false alarms)")  # fmt: skip

    results = {"rules (amount + velocity)": rules}
    for name, flags in model_flags.items():
        results[name] = evaluate(flags, windows["test"], labels, amounts)
    for name, flags in model_flags.items():
        results[f"rules + {name}"] = evaluate(rule_flagged | flags, windows["test"], labels, amounts)

    best = max(val_ap, key=val_ap.get)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "name": best,
            "model": fitted[best],
            "threshold": thresholds[best],
            "features": feature_names(channels),
            "channels": channels,
            "validation_average_precision": val_ap[best],
            "trained_on_days": WINDOWS["train"],
        },
        args.model_out,
    )

    n_test, fraud_test = len(windows["test"]), rules["fraud"]
    stored_test = len(idx["test"])
    print()
    print(f"TEST WINDOW (days 26-30): {n_test:,} labelled payments, {stored_test:,} in payments_clean, "
          f"{fraud_test:,} fraud (${rules['fraud_dollars']:,.2f})")  # fmt: skip
    print(results_table(results))
    print()
    print("recall by fraud pattern (test window)")
    print(pattern_table(results))
    print()
    print(f"best model by validation average precision: {best} -> {args.model_out.relative_to(ROOT)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(
            [
                "# Model comparison (test window, days 26-30)",
                "",
                (f"Train days 1-21 ({len(idx['train']):,} clean payments, {int(y[idx['train']].sum()):,} fraud), "
                f"validation days 22-25 ({len(idx['validation']):,}, {int(y[idx['validation']].sum()):,}), "
                f"test days 26-30 ({stored_test:,} clean of {n_test:,} labelled, {fraud_test:,} fraud)."),
                "",
                (f"Thresholds were chosen on validation to allow {budget:,} false alarms on {legit_val:,} legit "
                f"payments: the rules' {rules['false_alarms']:,} test false alarms per legit test payment "
                f"({legit_test:,}). Test false alarms are what each threshold produced on unseen days."),
                "",
                *(f"- {n}: validation average precision {val_ap[n]:.3f}, threshold {thresholds[n]:.4f}" for n in val_ap),
                "",
                results_table(results),
                "",
                "## Recall by fraud pattern",
                "",
                pattern_table(results),
                "",
                ("Quarantined payments are never scored and count as missed fraud. "
                f"Saved model: {best} (best validation average precision)."),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
