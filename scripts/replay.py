"""Replay the synthetic payments through the API, exactly as generated (mess included).

    python -m scripts.replay [--file data/synthetic/payments_raw.jsonl] [--api http://localhost:8000]

Every record is POSTed to /transactions. Records the API rejects (422) are
counted by reason and not retried; 503s and connection errors are retried.
Replaying twice is harmless: the consumer ignores payment_ids it already has.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILE = ROOT / "data" / "synthetic" / "payments_raw.jsonl"
DEFAULT_API = "http://localhost:8000"
ATTEMPTS = 6  # per record, for 5xx / connection errors


@dataclass
class Result:
    payment_id: str | None
    status: int  # final HTTP status; 0 = never got a response
    reason: str = ""


def load_records(path: Path, limit: int | None = None) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                if limit and len(records) >= limit:
                    break
    return records


def rejection_reason(body: object) -> str:
    """'field: message' from a FastAPI 422 body."""
    try:
        err = body["detail"][0]
        return f"{err['loc'][-1]}: {err['msg']}"
    except (KeyError, IndexError, TypeError):
        return "unrecognised 422 body"


def post_one(client: httpx.Client, record: dict) -> Result:
    pid = record.get("payment_id")
    status, reason = 0, ""
    for attempt in range(ATTEMPTS):
        try:
            resp = client.post("/transactions", json=record)
        except httpx.TransportError as exc:
            status, reason = 0, type(exc).__name__
        else:
            if resp.status_code == 202:
                return Result(pid, 202)
            if resp.status_code == 422:
                return Result(pid, 422, rejection_reason(resp.json()))
            status, reason = resp.status_code, resp.text[:200]
            if resp.status_code < 500:
                return Result(pid, status, reason)
        time.sleep(min(2**attempt, 10))
    return Result(pid, status, reason)


def post_all(records: list[dict], api: str = DEFAULT_API, workers: int = 16,
             on_result=None) -> list[Result]:  # fmt: skip
    """POST every record (several at a time); results are in input order."""
    results: list[Result | None] = [None] * len(records)
    limits = httpx.Limits(max_connections=workers, max_keepalive_connections=workers)
    with (
        httpx.Client(base_url=api, timeout=30, limits=limits) as client,
        ThreadPoolExecutor(workers) as pool,
    ):

        def run(i: int) -> None:
            results[i] = post_one(client, records[i])
            if on_result:
                on_result(results[i])

        list(pool.map(run, range(len(records))))
    return results


def summarize(results: list[Result]) -> dict:
    by_status = Counter(r.status for r in results)
    return {
        "posted": len(results),
        "accepted": by_status[202],
        "rejected": by_status[422],
        "failed": len(results) - by_status[202] - by_status[422],
        "unique_accepted": len({r.payment_id for r in results if r.status == 202}),
        "reasons": Counter(r.reason for r in results if r.status == 422),
        "failures": Counter(
            f"{r.status} {r.reason}" for r in results if r.status not in (202, 422)
        ),
    }


def print_summary(s: dict, seconds: float) -> None:
    print(f"posted            {s['posted']:>8,}  in {seconds:.1f}s "
          f"({s['posted'] / max(seconds, 1e-9):,.0f}/s)")  # fmt: skip
    print(f"accepted (202)    {s['accepted']:>8,}  "
          f"({s['unique_accepted']:,} unique payment_ids; the rest are resent duplicates)")  # fmt: skip
    print(f"rejected (422)    {s['rejected']:>8,}")
    for reason, n in s["reasons"].most_common():
        print(f"    {n:>6,}  {reason}")
    print(f"failed            {s['failed']:>8,}")
    for reason, n in s["failures"].most_common(5):
        print(f"    {n:>6,}  {reason}")


def check_api(api: str) -> None:
    try:
        httpx.get(f"{api}/health", timeout=5).raise_for_status()
    except httpx.HTTPError as exc:
        sys.exit(f"API not reachable at {api} ({exc}). Is `docker compose up` running?")


def progress_printer(total: int, every: int = 5000):
    lock, done = threading.Lock(), [0]

    def on_result(_: Result) -> None:
        with lock:
            done[0] += 1
            if done[0] % every == 0 or done[0] == total:
                print(f"  {done[0]:,}/{total:,}", flush=True)

    return on_result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--file", type=Path, default=DEFAULT_FILE)
    p.add_argument("--api", default=DEFAULT_API)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, help="only the first N records")
    args = p.parse_args()

    check_api(args.api)
    records = load_records(args.file, args.limit)
    print(f"replaying {len(records):,} records from {args.file}")
    start = time.monotonic()
    results = post_all(records, args.api, args.workers, progress_printer(len(records)))
    print_summary(summarize(results), time.monotonic() - start)


if __name__ == "__main__":
    main()
