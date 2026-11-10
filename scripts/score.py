"""Score the flags stored in Postgres against the generator's labels.

    python -m scripts.score

Run after scripts.replay once the consumer has caught up. Only payment_ids that
appear in labels.csv are scored, so dashboard or kill-test rows are ignored.
Labelled payments that never reached Postgres (rejected by the API) count as
not flagged: a fraud payment that was never stored was not caught.

It also compares three rule setups on the same stored payments: amount rule
only, velocity rule only, and both. The table is printed and written to
results/rules_comparison.md. Postgres keeps one reason per row (amount wins
when both rules fire), so each rule is re-run here on the stored rows with the
consumer's own thresholds; a check line confirms "both" matches Postgres.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import psycopg2

from app.main import FLAG_THRESHOLD, VELOCITY_LIMIT, VELOCITY_WINDOW_SECONDS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "synthetic"
RESULTS = ROOT / "results"

# Rule setups to compare: name -> the rules allowed to flag a payment.
SETUPS = {
    "amount only": {"amount"},
    "velocity only": {"velocity"},
    "both": {"amount", "velocity"},
}


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


def load_rule_inputs(conn) -> list[tuple[str, str, datetime, float]]:
    """(payment_id, account_id, event_time, amount) for every stored payment.

    All rows, labelled or not: any stored payment counts towards an account's
    velocity window, exactly as it does in the consumer.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT payment_id, account_id, event_time, amount FROM transactions")
        return [(pid, acct, t, float(amount)) for pid, acct, t, amount in cur]


def rule_hits(rows: list[tuple[str, str, datetime, float]]) -> dict[str, set[str]]:
    """payment_id -> the rules it trips, a subset of {'amount', 'velocity'}.

    Same definitions as app.main: amount > FLAG_THRESHOLD; velocity when the
    payment is the VELOCITY_LIMIT-th or later from its account within
    [event_time - window, event_time], ties on event_time ordered by payment_id.
    """
    by_account: dict[str, list[tuple[datetime, str, float]]] = defaultdict(list)
    for pid, acct, t, amount in rows:
        by_account[acct].append((t, pid, amount))
    window = timedelta(seconds=VELOCITY_WINDOW_SECONDS)
    hits: dict[str, set[str]] = {}
    for payments in by_account.values():
        payments.sort()
        left = 0
        for i, (t, pid, amount) in enumerate(payments):
            while payments[left][0] < t - window:
                left += 1
            hits[pid] = set()
            if amount > FLAG_THRESHOLD:
                hits[pid].add("amount")
            if i - left + 1 >= VELOCITY_LIMIT:
                hits[pid].add("velocity")
    return hits


def apply_setup(hits: dict[str, set[str]], stored: dict, rules: set[str]) -> dict:
    """The stored rows with status/reason recomputed as if only `rules` were on."""
    out = {}
    for pid, (_, _, amount) in stored.items():
        fired = sorted(hits.get(pid, set()) & rules)
        out[pid] = ("flagged", fired[0], amount) if fired else ("cleared", None, amount)
    return out


def compare_setups(labels: dict, amounts: dict, stored: dict, hits: dict) -> dict[str, dict]:
    """Setup name -> compute() summary, every setup scored on the same stored payments."""
    return {name: compute(labels, amounts, apply_setup(hits, stored, rules)) for name, rules in SETUPS.items()}


def status_mismatches(labels: dict, stored: dict, hits: dict) -> int:
    """Labelled stored payments whose Postgres status differs from re-running both rules."""
    both = apply_setup(hits, stored, SETUPS["both"])
    return sum(1 for p in stored if p in labels and stored[p][0] != both[p][0])


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


def comparison_table(results: dict[str, dict]) -> str:
    """The setups as one Markdown table (also readable as plain text)."""
    lines = [
        "| Setup | Recall | Precision | False alarms | Dollars blocked |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, s in results.items():
        blocked = _pct(s["dollars_blocked"], s["fraud_dollars"])
        lines.append(
            f"| {name} "
            f"| {_fmt_pct(s['recall'])} ({s['caught']:,}/{s['fraud']:,}) "
            f"| {_fmt_pct(s['precision'])} ({s['caught']:,}/{s['flagged']:,}) "
            f"| {s['false_alarms']:,} "
            f"| ${s['dollars_blocked']:,.2f} ({_fmt_pct(blocked)}) |"
        )
    return "\n".join(lines)


def comparison_markdown(results: dict[str, dict], mismatches: int) -> str:
    s = results["both"]
    return "\n".join(
        [
            "# Rule comparison",
            "",
            f"Scored on {s['labelled']:,} labelled payments ({s['fraud']:,} fraud, "
            f"${s['fraud_dollars']:,.2f} in fraud); {s['stored']:,} of them were stored in Postgres.",
            "",
            f"- **amount**: amount > ${FLAG_THRESHOLD:,.2f}",
            f"- **velocity**: {VELOCITY_LIMIT}th-or-later payment by one account within "
            f"{VELOCITY_WINDOW_SECONDS} s (by event time)",
            "",
            comparison_table(results),
            "",
            "Recall = fraud flagged / all labelled fraud (fraud that was never stored counts as missed). "
            "Precision = fraud flagged / all flagged. False alarms = legit payments flagged. "
            "Dollars blocked = amount of the fraud payments flagged, and its share of all fraud dollars.",
            "",
            f"Check: re-running both rules disagrees with the status stored in Postgres on {mismatches:,} payments.",
            "",
        ]
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--labels", type=Path, default=DATA / "labels.csv")
    p.add_argument("--truth", type=Path, default=DATA / "payments_truth.jsonl")
    p.add_argument("--out", type=Path, default=RESULTS / "rules_comparison.md")
    args = p.parse_args()

    conn = db_connect()
    try:
        stored = load_stored(conn)
        hits = rule_hits(load_rule_inputs(conn))
    finally:
        conn.close()
    labels, amounts = load_labels(args.labels), load_amounts(args.truth)
    print_report(compute(labels, amounts, stored))

    results = compare_setups(labels, amounts, stored, hits)
    mismatches = status_mismatches(labels, stored, hits)
    print()
    print(comparison_table(results))
    print(f"\ncheck: re-running both rules disagrees with Postgres status on {mismatches:,} payments")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(comparison_markdown(results, mismatches), encoding="utf-8")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
