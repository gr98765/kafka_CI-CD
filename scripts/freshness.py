"""Freshness: how long from POSTing one payment until it is in the database.

    python -m scripts.freshness [--n 20] [--api http://localhost:8000]

Sends one payment at a time (a fresh payment_id each), then polls raw_payments
until it appears, and prints the seconds from just before the POST to the
first poll that sees it. Polls only rows newer than the last raw_id seen
before the POST (primary-key lookup), so a big table does not slow the poll.
Run it with the consumer idle (after a load test has drained), or the number
includes the backlog.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid

import httpx

from scripts.replay import DEFAULT_API, DEFAULT_FILE
from scripts.report import HANDOFF_NOTE, docker_memory_note
from scripts.score import db_connect

POLL_SECONDS = 0.005
TIMEOUT_SECONDS = 60


def sample_record() -> dict:
    with open(DEFAULT_FILE) as f:
        return json.loads(f.readline())


def measure_once(client: httpx.Client, conn, record: dict, payment_id: str) -> tuple[float, float]:
    """(seconds until stored, seconds for the API to answer)."""
    with conn.cursor() as cur:
        cur.execute("SELECT coalesce(max(raw_id), 0) FROM raw_payments")
        (last_id,) = cur.fetchone()
        start = time.perf_counter()
        r = client.post("/transactions", json={**record, "payment_id": payment_id})
        api = time.perf_counter() - start
        r.raise_for_status()
        while time.perf_counter() - start < TIMEOUT_SECONDS:
            cur.execute(
                "SELECT 1 FROM raw_payments WHERE raw_id > %s AND payload ->> 'payment_id' = %s",
                (last_id, payment_id),
            )
            if cur.fetchone():
                return time.perf_counter() - start, api
            time.sleep(POLL_SECONDS)
    raise TimeoutError(f"{payment_id} not stored within {TIMEOUT_SECONDS} s")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--api", default=DEFAULT_API)
    p.add_argument("--pause", type=float, default=0.5, help="seconds between payments")
    args = p.parse_args()

    record = sample_record()
    tag = f"fr{uuid.uuid4().hex[:8]}"
    conn = db_connect()
    stored, api = [], []
    try:
        with httpx.Client(base_url=args.api, timeout=10) as client:
            for i in range(1, args.n + 1):
                s, a = measure_once(client, conn, record, f"{tag}_{i:03d}")
                stored.append(s)
                api.append(a)
                print(f"  {i:>2}: stored after {s:.3f} s (API answered in {a:.3f} s)")
                time.sleep(args.pause)
    finally:
        conn.close()
    print()
    print(f"FRESHNESS over {len(stored)} payments, one at a time (poll every {POLL_SECONDS * 1000:.0f} ms)")
    print(f"  POST -> in raw_payments: median {statistics.median(stored):.3f} s, max {max(stored):.3f} s, "
          f"min {min(stored):.3f} s")  # fmt: skip
    print(f"  API answer only:         median {statistics.median(api):.3f} s, max {max(api):.3f} s")
    print()
    print(docker_memory_note())
    print(HANDOFF_NOTE)


if __name__ == "__main__":
    main()
