"""Score the clean step against the generator's own record of what it broke.

    python -m scripts.clean_score

Run after scripts.replay (once the consumer has caught up) and app.clean, on
tables that hold only that replay. Compares payments_clean and
payments_quarantine with data/synthetic/mess_log.csv, labels.csv and the clean
payments_truth.jsonl, and prints:
  - per mess type: injected, fixed (kept AND equal to the truth), quarantined,
    kept but wrong, missing;
  - fixable records wrongly quarantined and unfixable records wrongly kept;
  - payments in clean + quarantine versus unique payments generated, with every
    raw record accounted for (kept, quarantined, or a dropped later copy);
  - the rules comparison (amount only, velocity only, both) on payments_clean;
  - a before/after line against the pre-cleaning run.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from generator.generate import MESS_RATES, TS_FORMAT, UNRECOVERABLE
from scripts.score import (
    DATA,
    _fmt_pct,
    _pct,
    compare_setups,
    comparison_table,
    db_connect,
    load_amounts,
    load_labels,
    rule_hits,
    status_mismatches,
)

# The run before the clean step existed: the API validated records and
# rejected these, and the consumer applied the rules to what was left.
BEFORE = {
    "rejected": 727,
    "rejected_amount_unreadable": 571,
    "rejected_amount_negative": 156,
    "recall": 39.7,
    "precision": 45.8,
}

# What the clean step should do with each unfixable mess type.
EXPECTED_REASON = {"unknown_currency": "currency_not_usd", "negative_amount": "non_positive_amount"}


@dataclass
class Tables:
    raw_rows: int
    raw_copies: Counter  # payment_id -> raw rows with that id
    clean: dict  # payment_id -> (account_id, amount, currency, event_time, status, reason)
    quarantine: list  # (payment_id, reason), one per quarantined raw record


def load_tables(conn) -> Tables:
    with conn.cursor() as cur:
        cur.execute("SELECT payload ->> 'payment_id', count(*) FROM raw_payments GROUP BY 1")
        copies = Counter({pid: n for pid, n in cur.fetchall()})
        cur.execute(
            "SELECT payment_id, account_id, amount, currency, event_time, status, reason FROM payments_clean"
        )
        clean = {r[0]: r[1:] for r in cur.fetchall()}
        cur.execute("SELECT payment_id, reason FROM payments_quarantine")
        quarantine = [tuple(r) for r in cur.fetchall()]
    return Tables(sum(copies.values()), copies, clean, quarantine)


def load_truth(path: Path) -> dict[str, dict]:
    with open(path) as f:
        return {r["payment_id"]: r for r in map(json.loads, f) if r}


def load_mess(path: Path) -> dict[str, str]:
    with open(path, newline="") as f:
        return {r["payment_id"]: r["issue"] for r in csv.DictReader(f)}


def count_lines(path: Path) -> int:
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def matches_truth(row: tuple, truth: dict) -> bool:
    """A kept payment is fixed only if every cleaned field equals the clean original."""
    account_id, amount, currency, event_time, _, _ = row
    return (
        account_id == truth["account_id"]
        and amount == Decimal(str(truth["amount"])).quantize(Decimal("0.01"))
        and currency == truth["currency"]
        and event_time == datetime.strptime(truth["timestamp"], TS_FORMAT).replace(tzinfo=UTC)
    )


def expected_quarantine(pid: str, truth: dict, mess: dict) -> bool:
    return mess.get(pid) in UNRECOVERABLE or truth[pid]["currency"] != "USD"


def evaluate(truth: dict, mess: dict, t: Tables) -> dict:
    q_reason = {pid: reason for pid, reason in t.quarantine if pid in truth}
    outcome = {}
    for pid, original in truth.items():
        if pid in t.clean:
            outcome[pid] = "fixed" if matches_truth(t.clean[pid], original) else "kept_wrong"
        elif pid in q_reason:
            outcome[pid] = "quarantined"
        else:
            outcome[pid] = "missing"

    rows = []
    for kind in [*MESS_RATES, None]:
        pids = [p for p in truth if mess.get(p) == kind]
        c = Counter(outcome[p] for p in pids)
        expected = (
            sum(1 for p in pids if q_reason.get(p) == EXPECTED_REASON[kind]) if kind in UNRECOVERABLE else None
        )
        rows.append({"kind": kind or "(no mess)", "injected": len(pids), **c, "expected_reason": expected})

    wrongly_quarantined = [p for p in truth if outcome[p] == "quarantined" and not expected_quarantine(p, truth, mess)]
    wrongly_kept = [p for p in truth if p in t.clean and expected_quarantine(p, truth, mess)]

    in_clean = sum(1 for p in t.clean if p in truth)
    in_quarantine = sum(1 for p in q_reason)
    missing = [p for p in truth if outcome[p] == "missing"]
    later_copies = sum(n - 1 for p, n in t.raw_copies.items() if p in truth and n > 1)
    return {
        "outcome": outcome,
        "rows": rows,
        "wrongly_quarantined": wrongly_quarantined,
        "wrongly_kept": wrongly_kept,
        "in_clean": in_clean,
        "in_quarantine": in_quarantine,
        "extra_clean": len(t.clean) - in_clean,
        "extra_quarantine": len(t.quarantine) - in_quarantine,
        "missing_never_received": sum(1 for p in missing if t.raw_copies.get(p, 0) == 0),
        "missing_received": sum(1 for p in missing if t.raw_copies.get(p, 0) > 0),
        "later_copies": later_copies,
        "received_labelled": sum(n for p, n in t.raw_copies.items() if p in truth),
        "quarantine_by_reason": Counter(reason for _, reason in t.quarantine),
    }


def mess_table(rows: list[dict]) -> str:
    lines = [
        "| Mess type | Injected | Fixed | Quarantined | Kept but wrong | Missing |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        unfixable = r["kind"] in UNRECOVERABLE
        quarantined = f"{r.get('quarantined', 0):,}"
        if unfixable:
            quarantined += f" ({r['expected_reason']:,} as {EXPECTED_REASON[r['kind']]})"
        lines.append(
            f"| {r['kind']} | {r['injected']:,} | {'n/a (unfixable)' if unfixable else format(r.get('fixed', 0), ',')} "
            f"| {quarantined} | {r.get('kept_wrong', 0):,} | {r.get('missing', 0):,} |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--labels", type=Path, default=DATA / "labels.csv")
    p.add_argument("--truth", type=Path, default=DATA / "payments_truth.jsonl")
    p.add_argument("--mess", type=Path, default=DATA / "mess_log.csv")
    p.add_argument("--raw-file", type=Path, default=DATA / "payments_raw.jsonl")
    args = p.parse_args()

    truth, mess = load_truth(args.truth), load_mess(args.mess)
    labels, amounts = load_labels(args.labels), load_amounts(args.truth)
    sent = count_lines(args.raw_file)
    conn = db_connect()
    try:
        t = load_tables(conn)
    finally:
        conn.close()
    e = evaluate(truth, mess, t)

    print("CLEANING: what happened to each kind of injected mess")
    print("(fixed = kept in payments_clean with account_id, amount, currency and event_time equal to the truth)")
    print(mess_table(e["rows"]))
    print()
    print("quarantine reasons:  " + ", ".join(f"{r} {n:,}" for r, n in sorted(e["quarantine_by_reason"].items())))
    print(f"fixable records wrongly quarantined:  {len(e['wrongly_quarantined']):,}"
          + (f"  e.g. {e['wrongly_quarantined'][:5]}" if e["wrongly_quarantined"] else ""))  # fmt: skip
    print(f"unfixable records wrongly kept:       {len(e['wrongly_kept']):,}"
          + (f"  e.g. {e['wrongly_kept'][:5]}" if e["wrongly_kept"] else ""))  # fmt: skip

    total = e["in_clean"] + e["in_quarantine"]
    gap = len(truth) - total
    print()
    print("ACCOUNTING")
    print(f"unique payments generated            {len(truth):>8,}")
    print(f"  in payments_clean                  {e['in_clean']:>8,}")
    print(f"  in payments_quarantine             {e['in_quarantine']:>8,}")
    print(f"  clean + quarantine                 {total:>8,}   gap {gap:,}")
    print(f"    never received (not in raw)      {e['missing_never_received']:>8,}")
    print(f"    received but in neither table    {e['missing_received']:>8,}")
    print(f"records in the replay file           {sent:>8,}")
    print(f"  received in raw_payments           {e['received_labelled']:>8,}   "
          f"(not received: {sent - e['received_labelled']:,}, i.e. rejected by the API or failed)")  # fmt: skip
    print(f"raw rows = clean + quarantine + dropped later copies: {t.raw_rows:,} = {len(t.clean):,} + "
          f"{len(t.quarantine):,} + {t.raw_rows - len(t.clean) - len(t.quarantine):,}")  # fmt: skip
    print(f"  later copies of a payment_id in raw: {e['later_copies']:,} "
          f"(generator injected {sum(1 for k in mess.values() if k == 'duplicate'):,} re-sends)")  # fmt: skip
    if e["extra_clean"] or e["extra_quarantine"]:
        print(f"  rows not from this dataset: {e['extra_clean']:,} clean, {e['extra_quarantine']:,} quarantined")

    stored = {pid: (row[4], row[5], float(row[1])) for pid, row in t.clean.items()}
    rows = [(pid, row[0], row[3], float(row[1])) for pid, row in t.clean.items()]
    hits = rule_hits(rows)
    results = compare_setups(labels, amounts, stored, hits)
    mismatches = status_mismatches(labels, stored, hits)
    print()
    print("FRAUD RULES on payments_clean (quarantined or missing fraud counts as not caught)")
    print(comparison_table(results))
    print(f"check: re-running both rules disagrees with payments_clean status on {mismatches:,} payments")

    b, both = BEFORE, results["both"]
    fixable = {"amount_string", "timestamp_format", "id_whitespace", "currency_case"}
    fixed_messy = sum(r.get("fixed", 0) for r in e["rows"] if r["kind"] in fixable)
    print()
    print(f"BEFORE  API rejected {b['rejected']:,} records ({b['rejected_amount_unreadable']:,} unreadable amounts, "
          f"{b['rejected_amount_negative']:,} negative amounts); both rules: recall {b['recall']:.1f}%, "
          f"precision {b['precision']:.1f}%")  # fmt: skip
    print(f"AFTER   API rejected {sent - e['received_labelled']:,} records; {e['in_quarantine']:,} quarantined "
          f"with a reason, {fixed_messy:,} reformatted records fixed and kept; both rules: recall {_fmt_pct(both['recall'])}, "
          f"precision {_fmt_pct(both['precision'])}")  # fmt: skip
    print(f"        ({both['caught']:,} of {both['fraud']:,} fraud caught, {both['false_alarms']:,} false alarms, "
          f"${both['dollars_blocked']:,.2f} blocked = {_fmt_pct(_pct(both['dollars_blocked'], both['fraud_dollars']))} "
          f"of fraud dollars)")  # fmt: skip


if __name__ == "__main__":
    main()
