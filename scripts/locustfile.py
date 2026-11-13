"""Load test: simulated users POST payments to the API as fast as they can.

    locust -f scripts/locustfile.py --headless --host http://localhost:8000 \
        -u 50 -r 10 -t 2m --csv results/locust

Each request is a record sampled at random from data/synthetic/payments_raw.jsonl
(messy records included, exactly as generated) with a fresh payment_id
"<run tag>_<n>", so the consumer's duplicate protection never skips one. Users
do not wait between requests (closed loop): this measures how much the stack
takes, not a realistic arrival rate.

When the test stops, after every user has stopped, two files are written:
  - results/locust_sent.txt: the payment_ids the API accepted (202), which
    scripts/report.py compares with what was stored;
  - <csv prefix>_stats_final.csv and _failures_final.csv: Locust's own CSV
    format, written once at the end. (Locust rewrites <prefix>_stats.csv once a
    second while running and not again at shutdown, so that file can miss the
    last second of requests.)
Set LOCUST_RECORDS to use another file, LOCUST_RUN_TAG to fix the tag.
"""

from __future__ import annotations

import csv
import itertools
import json
import os
import random
import uuid
from pathlib import Path

from locust import HttpUser, events, task
from locust.stats import PERCENTILES_TO_REPORT, StatsCSV

ROOT = Path(__file__).resolve().parent.parent
RECORDS = Path(os.getenv("LOCUST_RECORDS", ROOT / "data" / "synthetic" / "payments_raw.jsonl"))
SENT_FILE = ROOT / "results" / "locust_sent.txt"
RUN_TAG = os.getenv("LOCUST_RUN_TAG") or f"lt{uuid.uuid4().hex[:8]}"

with open(RECORDS) as f:
    LINES = [line for line in f if line.strip()]  # kept as text; parsed per request
_counter = itertools.count(1)
accepted: list[str] = []


def next_payment() -> dict:
    record = json.loads(random.choice(LINES))
    record["payment_id"] = f"{RUN_TAG}_{next(_counter):07d}"
    return record


class PaymentUser(HttpUser):
    @task
    def post_payment(self) -> None:
        record = next_payment()
        with self.client.post("/transactions", json=record, name="POST /transactions", catch_response=True) as r:
            if r.status_code == 202:
                accepted.append(record["payment_id"])
                r.success()
            else:
                r.failure(f"HTTP {r.status_code}")


@events.test_start.add_listener
def _start(environment, **_kwargs) -> None:
    print(f"run tag {RUN_TAG}: sampling {len(LINES):,} records from {RECORDS}")


@events.test_stop.add_listener
def _stop(environment, **_kwargs) -> None:
    SENT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SENT_FILE.write_text("\n".join(accepted) + ("\n" if accepted else ""))
    print(f"{len(accepted):,} accepted payment_ids written to {SENT_FILE}")

    prefix = getattr(environment.parsed_options, "csv_prefix", None)
    if prefix:
        writer = StatsCSV(environment, PERCENTILES_TO_REPORT)
        for suffix, write in (("_stats_final.csv", writer.requests_csv), ("_failures_final.csv", writer.failures_csv)):
            with open(f"{prefix}{suffix}", "w", newline="") as out:
                write(csv.writer(out))
        print(f"final stats written to {prefix}_stats_final.csv")
