"""Synthetic payments generator.

Makes fake-but-realistic card payments, with three things on purpose:
  1. normal behaviour (people shop, pay bills, travel),
  2. normal-looking things that LOOK suspicious (a shopping trip, a bill batch,
     a one-off big purchase), so rules produce honest false alarms,
  3. labelled fraud (velocity bursts, card testing, impossible travel,
     big amounts), plus deliberate data mess that we log, so a cleaning step
     can be scored against the truth.

Same seed -> same files. Standard library only.

Run:  python -m generator.generate --accounts 1000 --days 30
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from generator.catalog import CATEGORIES, COUNTRIES, Merchant, build_merchants

TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# --- knobs -----------------------------------------------------------------
SHOPPING_TRIP_PER_DAY = 0.02  # chance per account per day (3-4 payments, minutes apart)
BILLING_BATCH_ON_FIRST = 0.08  # chance per account on the 1st (3-5 bills in seconds)
BIG_PURCHASE_RATE = 0.001  # share of normal payments that are legit and > $1000
FRAUD_PATTERN_WEIGHTS = {
    "velocity_burst": 0.35,
    "card_testing": 0.25,
    "impossible_travel": 0.20,
    "amount_outlier": 0.20,
}

# Chance that one payment gets one kind of mess. Rolled once per payment.
MESS_RATES = {
    "duplicate": 0.015,  # same payment delivered twice
    "amount_string": 0.020,  # "$12.50" instead of 12.5
    "timestamp_format": 0.020,  # epoch number or "2026-10-01 12:00:00"
    "currency_case": 0.010,  # "usd"
    "id_whitespace": 0.010,  # " acc00012 " instead of "ACC00012"
    "late_arrival": 0.020,  # arrives 1-30 minutes after it happened
    "unknown_currency": 0.005,  # "XXX": cannot be fixed -> quarantine
    "negative_amount": 0.003,  # -12.50: cannot be fixed -> quarantine
}
UNRECOVERABLE = {"unknown_currency", "negative_amount"}  # never put on fraud rows


# --- small helpers ---------------------------------------------------------
def poisson(rng: random.Random, lam: float) -> int:
    """How many events happen in a day if they average `lam` per day (Knuth)."""
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


@dataclass
class Account:
    account_id: str
    home_country: str
    spend_factor: float  # 0.5 = frugal, 2.0 = big spender
    daily_rate: float  # average payments per day
    start_hour: int  # active window (UTC hours)
    end_hour: int
    favorites: list[Merchant]


@dataclass
class Event:
    account: Account
    merchant: Merchant
    when: datetime
    amount_usd: float
    channel: str
    pattern: str = "normal"
    is_fraud: bool = False
    incident_id: str = ""


@dataclass
class World:
    start: datetime
    days: int
    by_cc: dict[tuple[str, str], list[Merchant]]  # (country, category) -> merchants
    by_country: dict[str, list[Merchant]]
    everyone: list[Merchant]


def category_pick(rng: random.Random) -> str:
    return rng.choices(
        list(CATEGORIES), weights=[c.weight for c in CATEGORIES.values()]
    )[0]


def make_accounts(rng: random.Random, n: int, world: World) -> list[Account]:
    codes = list(COUNTRIES)
    shares = [COUNTRIES[c][0] for c in codes]
    accounts = []
    for i in range(1, n + 1):
        country = rng.choices(codes, weights=shares)[0]
        favorites = [
            rng.choice(world.by_cc[(country, category_pick(rng))]) for _ in range(6)
        ]
        accounts.append(
            Account(
                account_id=f"ACC{i:05d}",
                home_country=country,
                spend_factor=round(rng.lognormvariate(0, 0.35), 2),
                daily_rate=round(rng.uniform(0.5, 3.0), 2),
                start_hour=rng.randint(7, 11),
                end_hour=rng.randint(19, 23),
                favorites=favorites,
            )
        )
    return accounts


def pick_merchant(rng: random.Random, world: World, acct: Account) -> Merchant:
    if rng.random() < 0.75:
        return rng.choice(acct.favorites)
    return rng.choice(world.by_cc[(acct.home_country, category_pick(rng))])


def normal_amount(rng: random.Random, category: str, spend_factor: float) -> float:
    raw = CATEGORIES[category].median_usd * spend_factor * rng.lognormvariate(0, 0.6)
    return round(min(max(raw, 0.5), 900.0), 2)


def channel_for(rng: random.Random, merchant: Merchant) -> str:
    if merchant.category == "subscriptions":
        return "online"
    return rng.choices(["in_person", "online"], weights=[0.55, 0.45])[0]


# --- normal behaviour ------------------------------------------------------
def normal_events(rng: random.Random, world: World, acct: Account) -> list[Event]:
    events: list[Event] = []
    for d in range(world.days):
        day = world.start + timedelta(days=d)
        for _ in range(poisson(rng, acct.daily_rate)):
            secs = int(rng.uniform(acct.start_hour * 3600, acct.end_hour * 3600))
            when = day + timedelta(seconds=secs)
            if rng.random() < BIG_PURCHASE_RATE:  # legit, but over $1000
                category = rng.choice(["travel", "retail"])
                merchant = rng.choice(world.by_cc[(acct.home_country, category)])
                amount = round(rng.uniform(1000, 3000), 2)
            else:
                merchant = pick_merchant(rng, world, acct)
                amount = normal_amount(rng, merchant.category, acct.spend_factor)
            events.append(
                Event(acct, merchant, when, amount, channel_for(rng, merchant))
            )
        if rng.random() < SHOPPING_TRIP_PER_DAY:
            events += shopping_trip(rng, world, acct, day)
        if day.day == 1 and rng.random() < BILLING_BATCH_ON_FIRST:
            events += billing_batch(rng, world, acct, day)
    return events


def shopping_trip(
    rng: random.Random, world: World, acct: Account, day: datetime
) -> list[Event]:
    """3-4 stops in a mall: legit, close together, a little suspicious-looking."""
    when = day + timedelta(seconds=rng.randint(11 * 3600, 18 * 3600))
    out = []
    for _ in range(rng.randint(3, 4)):
        category = rng.choice(["retail", "dining", "groceries"])
        merchant = rng.choice(world.by_cc[(acct.home_country, category)])
        amount = normal_amount(rng, category, acct.spend_factor)
        out.append(Event(acct, merchant, when, amount, "in_person"))
        when += timedelta(seconds=rng.randint(60, 240))
    return out


def billing_batch(
    rng: random.Random, world: World, acct: Account, day: datetime
) -> list[Event]:
    """Several bills auto-charged within seconds on the 1st: legit, very bursty."""
    when = day + timedelta(seconds=rng.randint(3 * 3600, 4 * 3600))
    out = []
    for _ in range(rng.randint(3, 5)):
        category = rng.choice(["subscriptions", "utilities"])
        merchant = rng.choice(world.by_cc[(acct.home_country, category)])
        amount = normal_amount(rng, category, acct.spend_factor)
        out.append(Event(acct, merchant, when, amount, "online"))
        when += timedelta(seconds=rng.randint(5, 25))
    return out


# --- fraud -----------------------------------------------------------------
def burst_offsets(rng: random.Random, n: int, span: int) -> list[int]:
    """n second-offsets from 0 to span (first and last pinned), sorted."""
    return sorted([0, span] + [rng.randrange(span) for _ in range(n - 2)])


def velocity_burst(
    rng: random.Random, world: World, acct: Account, base: datetime, inc: str
) -> list[Event]:
    """Many payments very quickly. 70% fast (under 3 min), 30% slow (4-9 min)."""
    n = rng.randint(6, 12)
    span = rng.randint(30, 150) if rng.random() < 0.7 else rng.randint(240, 540)
    category = rng.choice(["retail", "entertainment"])
    merchants = world.by_cc[(acct.home_country, category)]
    return [
        Event(
            acct,
            rng.choice(merchants),
            base + timedelta(seconds=off),
            round(rng.uniform(30, 300), 2),
            "online",
            "velocity_burst",
            True,
            inc,
        )
        for off in burst_offsets(rng, n, span)
    ]


def card_testing(
    rng: random.Random, world: World, acct: Account, base: datetime, inc: str
) -> list[Event]:
    """Tiny payments to see if a stolen card works, then one real cash-out."""
    n = rng.randint(8, 15)
    span = rng.randint(60, 170)
    out = [
        Event(
            acct,
            rng.choice(world.everyone),
            base + timedelta(seconds=off),
            round(rng.uniform(0.5, 4.0), 2),
            "online",
            "card_testing",
            True,
            inc,
        )
        for off in burst_offsets(rng, n, span)
    ]
    cash_out = base + timedelta(seconds=span + rng.randint(30, 120))
    out.append(
        Event(
            acct,
            rng.choice(world.by_cc[(acct.home_country, "retail")]),
            cash_out,
            round(rng.uniform(100, 400), 2),
            "online",
            "card_testing",
            True,
            inc,
        )
    )
    return out


def impossible_travel(
    rng: random.Random, world: World, acct: Account, base: datetime, inc: str
) -> list[Event]:
    """A normal payment at home, then payments in another country minutes later."""
    abroad = rng.choice([c for c in COUNTRIES if c != acct.home_country])
    home_merchant = rng.choice(world.by_cc[(acct.home_country, "dining")])
    out = [Event(acct, home_merchant, base, 25.0, "in_person")]  # innocent
    when = base + timedelta(minutes=rng.randint(5, 25))
    for _ in range(2):
        merchant = rng.choice(world.by_country[abroad])
        out.append(
            Event(
                acct,
                merchant,
                when,
                round(rng.uniform(80, 600), 2),
                "in_person",
                "impossible_travel",
                True,
                inc,
            )
        )
        when += timedelta(minutes=rng.randint(2, 8))
    return out


def amount_outlier(
    rng: random.Random, world: World, acct: Account, base: datetime, inc: str
) -> list[Event]:
    """One huge payment far above this account's normal."""
    category = rng.choice(["travel", "retail", "entertainment"])
    merchant = rng.choice(world.by_cc[(acct.home_country, category)])
    return [
        Event(
            acct,
            merchant,
            base,
            round(rng.uniform(1100, 5000), 2),
            "online",
            "amount_outlier",
            True,
            inc,
        )
    ]


PATTERNS = {
    "velocity_burst": velocity_burst,
    "card_testing": card_testing,
    "impossible_travel": impossible_travel,
    "amount_outlier": amount_outlier,
}


def fraud_events(
    rng: random.Random, world: World, acct: Account, counter: list[int]
) -> list[Event]:
    events: list[Event] = []
    for _ in range(2 if rng.random() < 0.3 else 1):
        pattern = rng.choices(
            list(FRAUD_PATTERN_WEIGHTS), weights=list(FRAUD_PATTERN_WEIGHTS.values())
        )[0]
        counter[0] += 1
        base = world.start + timedelta(seconds=rng.randrange(world.days * 86400 - 3600))
        events += PATTERNS[pattern](rng, world, acct, base, f"inc_{counter[0]:04d}")
    return events


# --- turn events into records ---------------------------------------------
def to_records(events: list[Event]) -> list[tuple[datetime, dict, Event]]:
    events.sort(key=lambda e: (e.when, e.account.account_id))
    out = []
    for i, e in enumerate(events, start=1):
        _, currency, usd_per_unit = COUNTRIES[e.merchant.country]
        record = {
            "payment_id": f"pay_{i:08d}",
            "account_id": e.account.account_id,
            "merchant_id": e.merchant.merchant_id,
            "amount": max(round(e.amount_usd / usd_per_unit, 2), 0.01),
            "currency": currency,
            "timestamp": e.when.strftime(TS_FORMAT),
            "country": e.merchant.country,
            "channel": e.channel,
        }
        out.append((e.when, record, e))
    return out


def apply_mess(
    rng: random.Random, rows: list[tuple[datetime, dict, Event]]
) -> tuple[list[dict], list[dict]]:
    """Return (dirty records in arrival order, log of every mess we added)."""
    arrivals: list[tuple[datetime, int, dict]] = []
    log: list[dict] = []
    for when, clean, event in rows:
        rec = dict(clean)
        roll, acc, issue = rng.random(), 0.0, None
        for kind, rate in MESS_RATES.items():
            acc += rate
            if roll < acc:
                issue = kind
                break
        if issue in UNRECOVERABLE and event.is_fraud:
            issue = None  # keep fraud rows usable so recall is about rules, not luck
        delay = rng.randint(0, 3)
        if issue == "duplicate":
            arrivals.append((when + timedelta(seconds=delay + rng.randint(1, 30)), 0, dict(rec)))
        elif issue == "amount_string":
            rec["amount"] = rng.choice([f"${rec['amount']:.2f}", f"{rec['amount']:.2f}"])
        elif issue == "timestamp_format":
            rec["timestamp"] = (
                int(when.timestamp())
                if rng.random() < 0.5
                else when.strftime("%Y-%m-%d %H:%M:%S")
            )
        elif issue == "currency_case":
            rec["currency"] = rec["currency"].lower()
        elif issue == "id_whitespace":
            rec["account_id"] = f" {rec['account_id'].lower()} "
        elif issue == "late_arrival":
            delay = rng.randint(60, 1800)
        elif issue == "unknown_currency":
            rec["currency"] = "XXX"
        elif issue == "negative_amount":
            rec["amount"] = -rec["amount"]
        if issue:
            log.append({"payment_id": clean["payment_id"], "issue": issue})
        arrivals.append((when + timedelta(seconds=delay), 1, rec))
    arrivals.sort(key=lambda a: (a[0], a[1]))
    return [a[2] for a in arrivals], log


# --- main ------------------------------------------------------------------
def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def generate(
    out_dir: str | Path,
    accounts: int = 1000,
    days: int = 30,
    seed: int = 42,
    start: str = "2026-10-01",
    fraud_account_rate: float = 0.04,
) -> dict:
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    merchants = build_merchants(rng)
    by_cc: dict[tuple[str, str], list[Merchant]] = {}
    by_country: dict[str, list[Merchant]] = {}
    for m in merchants:
        by_cc.setdefault((m.country, m.category), []).append(m)
        by_country.setdefault(m.country, []).append(m)
    world = World(
        start=datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC),
        days=days,
        by_cc=by_cc,
        by_country=by_country,
        everyone=merchants,
    )

    accts = make_accounts(rng, accounts, world)
    events: list[Event] = []
    for acct in accts:
        events += normal_events(rng, world, acct)
    counter = [0]
    for acct in rng.sample(accts, round(accounts * fraud_account_rate)):
        events += fraud_events(rng, world, acct, counter)

    rows = to_records(events)
    raw, mess_log = apply_mess(rng, rows)

    write_jsonl(out / "payments_raw.jsonl", raw)  # what the pipeline receives
    write_jsonl(out / "payments_truth.jsonl", [r for _, r, _ in rows])  # the clean answer
    write_csv(
        out / "labels.csv",
        [
            {
                "payment_id": rec["payment_id"],
                "is_fraud": int(e.is_fraud),
                "pattern": e.pattern,
                "incident_id": e.incident_id,
            }
            for _, rec, e in rows
        ],
        ["payment_id", "is_fraud", "pattern", "incident_id"],
    )
    write_csv(out / "mess_log.csv", mess_log, ["payment_id", "issue"])
    write_csv(
        out / "accounts.csv",
        [
            {
                "account_id": a.account_id,
                "home_country": a.home_country,
                "spend_factor": a.spend_factor,
                "daily_rate": a.daily_rate,
            }
            for a in accts
        ],
        ["account_id", "home_country", "spend_factor", "daily_rate"],
    )
    write_csv(
        out / "merchants.csv",
        [m.__dict__ for m in merchants],
        ["merchant_id", "name", "category", "country"],
    )

    manifest = {
        "seed": seed,
        "accounts": accounts,
        "days": days,
        "start": start,
        "payments": len(rows),
        "raw_records": len(raw),
        "fraud_payments": sum(e.is_fraud for _, _, e in rows),
        "fraud_incidents": counter[0],
        "fraud_by_pattern": dict(Counter(e.pattern for _, _, e in rows if e.is_fraud)),
        "mess_by_issue": dict(Counter(m["issue"] for m in mess_log)),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic payments.")
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--accounts", type=int, default=1000)
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    print(json.dumps(generate(args.out, args.accounts, args.days, args.seed), indent=2))


if __name__ == "__main__":
    main()
