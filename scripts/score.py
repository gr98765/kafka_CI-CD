"""Score the flags stored in Postgres against the generator's labels.

    python -m scripts.score

Run after scripts.replay once the consumer has caught up. Only payment_ids that
appear in labels.csv are scored, so dashboard or kill-test rows are ignored.
Labelled payments that never reached Postgres (rejected by the API) count as
not flagged: a fraud payment that was never stored was not caught.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import psycopg2

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "synthetic"


def db_connect():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5433"),  # docker-compose maps Postgres to 5433
        user=os.getenv("DB_USER", "fintech"),
        password=os.getenv("DB_PASSWORD", "fintech"),
        dbname=os.getenv("DB_NAME", "transactions_db"),
        connect_timeout=5,
    )
    conn.autocommit = True
    return conn


def load_labels(path: Path) -> dict[str, tuple[bool, str]]:
    """payment_id -> (is_fraud, pattern)."""
    with open(path, newline="") as f:
        return {r["payment_id"]: (r["is_fraud"] == "1", r["pattern"]) for r in csv.DictReader(f)}


def load_amounts(path: Path) -> dict[str, float]:
    """payment_id -> true amount in USD, from the clean (pre-mess) file."""
    with open(path) as f:
        return {r["payment_id"]: float(r["amount"]) for r in map(json.loads, f) if r}


def load_stored(conn) -> dict[str, tuple[str, str | None, float]]:
    """payment_id -> (status, reason, amount) for every stored payment."""
    with conn.cursor() as cur:
        cur.execute("SELECT payment_id, status, reason, amount FROM transactions")
        return {pid: (status, reason, float(amount)) for pid, status, reason, amount in cur}


def _pct(part: float, whole: float) -> float | None:
    return 100 * part / whole if whole else None


def compute(labels: dict, amounts: dict, stored: dict) -> dict:
    fraud = {p for p, (is_fraud, _) in labels.items() if is_fraud}
    legit = set(labels) - fraud
    scored = {p: v for p, v in stored.items() if p in labels}
    flagged = {p for p, (status, _, _) in scored.items() if status == "flagged"}
    caught, false_alarms = flagged & fraud, flagged & legit

    pattern = {p: labels[p][1] for p in fraud}
    by_pattern = {
        name: {
            "total": sum(1 for p in fraud if pattern[p] == name),
            "caught": sum(1 for p in caught if pattern[p] == name),
            "not_stored": sum(1 for p in fraud - set(scored) if pattern[p] == name),
        }
        for name in sorted(set(pattern.values()))
    }
    by_reason = {
        reason: {
            "flagged": sum(1 for p in flagged if scored[p][1] == reason),
            "fraud": sum(1 for p in caught if scored[p][1] == reason),
        }
        for reason in sorted({scored[p][1] for p in flagged}, key=str)
    }
    stored_legit = legit & set(scored)
    return {
        "labelled": len(labels),
        "fraud": len(fraud),
        "legit": len(legit),
        "stored": len(scored),
        "fraud_stored": len(fraud & set(scored)),
        "flagged": len(flagged),
        "caught": len(caught),
        "false_alarms": len(false_alarms),
        "recall": _pct(len(caught), len(fraud)),
        "precision": _pct(len(caught), len(flagged)),
        "false_alarm_rate": _pct(len(false_alarms), len(stored_legit)),
        "fraud_dollars": sum(amounts.get(p, 0.0) for p in fraud),
        "dollars_blocked": sum(scored[p][2] for p in caught),
        "legit_dollars_blocked": sum(scored[p][2] for p in false_alarms),
        "by_pattern": by_pattern,
        "by_reason": by_reason,
    }


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def print_report(s: dict) -> None:
    print(f"labelled payments        {s['labelled']:>8,}  ({s['fraud']:,} fraud, {s['legit']:,} legit)")
    print(f"stored in Postgres       {s['stored']:>8,}  ({s['fraud_stored']:,} of the fraud); "
          f"{s['labelled'] - s['stored']:,} never stored")  # fmt: skip
    print()
    print(f"fraud caught (recall)    {_fmt_pct(s['recall']):>8}  {s['caught']:,} of {s['fraud']:,} fraud payments")
    print(f"correct flags (precision){_fmt_pct(s['precision']):>8}  {s['caught']:,} of {s['flagged']:,} flags were fraud")
    print(f"false alarms             {s['false_alarms']:>8,}  legit payments flagged "
          f"({_fmt_pct(s['false_alarm_rate'])} of stored legit)")  # fmt: skip
    print(f"dollars blocked          ${s['dollars_blocked']:>11,.2f} of ${s['fraud_dollars']:,.2f} fraud "
          f"({_fmt_pct(_pct(s['dollars_blocked'], s['fraud_dollars']))})")  # fmt: skip
    print(f"legit dollars blocked    ${s['legit_dollars_blocked']:>11,.2f}  (by the false alarms)")
    print()
    print("by fraud pattern          caught / total   (never stored)")
    for name, c in s["by_pattern"].items():
        print(f"  {name:<22} {c['caught']:>5} / {c['total']:<5}   ({c['not_stored']})")
    print()
    print("by flag reason            flagged   fraud   false alarms")
    for reason, c in s["by_reason"].items():
        print(f"  {reason!s:<22} {c['flagged']:>8,} {c['fraud']:>7,} {c['flagged'] - c['fraud']:>14,}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--labels", type=Path, default=DATA / "labels.csv")
    p.add_argument("--truth", type=Path, default=DATA / "payments_truth.jsonl")
    args = p.parse_args()

    conn = db_connect()
    try:
        stored = load_stored(conn)
    finally:
        conn.close()
    print_report(compute(load_labels(args.labels), load_amounts(args.truth), stored))


if __name__ == "__main__":
    main()
