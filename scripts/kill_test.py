"""Crash test: kill the consumer halfway through a replay, restart it, and
check that no accepted payment was lost or stored twice.

    python -m scripts.kill_test [--down-seconds 15]

Needs the stack running (docker compose up -d --build). Every payment_id and
account_id gets a unique run tag, so this run is counted on its own and never
touches the velocity counts of payments replayed by scripts.replay.
The consumer is killed with SIGKILL (docker compose kill): no clean shutdown.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
import uuid

from scripts.replay import (
    DEFAULT_API,
    DEFAULT_FILE,
    ROOT,
    check_api,
    load_records,
    post_all,
    summarize,
)
from scripts.score import db_connect

DEAD_LETTER_TOPIC = "payments.dead-letter"


def compose(*args: str, check: bool = True) -> str:
    out = subprocess.run(
        ["docker", "compose", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if check and out.returncode != 0:
        sys.exit(f"docker compose {' '.join(args)} failed:\n{out.stderr}")
    return out.stdout


def tag_record(record: dict, tag: str) -> dict:
    tagged = dict(record)
    for field in ("payment_id", "account_id"):
        if isinstance(tagged.get(field), str):
            tagged[field] = f"{tag}_{tagged[field]}"
    return tagged


def stored_counts(conn, tag: str) -> tuple[int, int]:
    """(rows, distinct payment_ids) stored for this run."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*), count(DISTINCT payment_id) FROM transactions WHERE payment_id LIKE %s",
            (f"{tag}\\_%",),
        )
        rows, distinct = cur.fetchone()
    return rows, distinct


def stored_ids(conn, tag: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT payment_id FROM transactions WHERE payment_id LIKE %s", (f"{tag}\\_%",))
        return {r[0] for r in cur}


def dead_lettered(tag: str) -> int:
    """How many of this run's records are on the dead-letter topic."""
    out = compose(
        "exec", "-T", "kafka", "kafka-console-consumer",
        "--bootstrap-server", "localhost:29092", "--topic", DEAD_LETTER_TOPIC,
        "--from-beginning", "--timeout-ms", "8000",
        check=False,
    )  # fmt: skip
    return sum(1 for line in out.splitlines() if f"{tag}_" in line)


def wait_for_drain(conn, tag: str, expected: int, idle_limit: float) -> int:
    """Wait until every accepted payment is stored, or nothing new arrives for
    `idle_limit` seconds. Returns the final row count."""
    last, last_change = -1, time.monotonic()
    while True:
        rows, _ = stored_counts(conn, tag)
        if rows >= expected:
            return rows
        if rows != last:
            last, last_change = rows, time.monotonic()
            print(f"  stored {rows:,}/{expected:,}", flush=True)
        elif time.monotonic() - last_change > idle_limit:
            return rows
        time.sleep(2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--file", default=DEFAULT_FILE)
    p.add_argument("--api", default=DEFAULT_API)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, help="only the first N records")
    p.add_argument("--down-seconds", type=float, default=15, help="how long the consumer stays dead")
    p.add_argument("--idle-limit", type=float, default=90, help="give up waiting after this long with no progress")
    args = p.parse_args()

    check_api(args.api)
    if not compose("ps", "--status", "running", "-q", "consumer").strip():
        sys.exit("consumer container is not running; start the stack with `docker compose up -d --build`")

    tag = f"kt{uuid.uuid4().hex[:8]}"
    records = [tag_record(r, tag) for r in load_records(args.file, args.limit)]
    half = len(records) // 2
    print(f"run tag {tag}: replaying {len(records):,} records; killing the consumer after {half:,}")

    done, lock, halfway = [0], threading.Lock(), threading.Event()

    def on_result(_) -> None:
        with lock:
            done[0] += 1
            if done[0] >= half:
                halfway.set()

    results: list = []
    replay = threading.Thread(
        target=lambda: results.extend(post_all(records, args.api, args.workers, on_result))
    )
    start = time.monotonic()
    replay.start()

    conn = db_connect()
    while not halfway.wait(1):
        if not replay.is_alive():
            sys.exit("replay stopped before reaching halfway")
    compose("kill", "consumer")
    stored_at_kill, _ = stored_counts(conn, tag)  # after the kill returns: it takes ~1s
    print(f"  killed consumer (SIGKILL) at {done[0]:,} posted, {stored_at_kill:,} stored")
    time.sleep(args.down_seconds)
    stored_while_down, _ = stored_counts(conn, tag)
    compose("start", "consumer")
    print(f"  restarted consumer after {args.down_seconds:.0f}s "
          f"({stored_while_down - stored_at_kill:,} stored while it was down)")  # fmt: skip

    replay.join()
    summary = summarize(results)
    print(f"  replay finished in {time.monotonic() - start:.0f}s; waiting for the consumer to catch up")

    accepted = {r.payment_id for r in results if r.status == 202}
    wait_for_drain(conn, tag, len(accepted), args.idle_limit)
    rows, distinct = stored_counts(conn, tag)
    ids = stored_ids(conn, tag)
    conn.close()
    lost_ids = accepted - ids
    unexpected = ids - accepted
    dlq = dead_lettered(tag) if lost_ids else 0

    print()
    print(f"records posted           {summary['posted']:>8,}")
    print(f"  rejected by API (422)  {summary['rejected']:>8,}  (never sent to Kafka)")
    print(f"  failed (no 202/422)    {summary['failed']:>8,}")
    print(f"  accepted (202)         {summary['accepted']:>8,}")
    print(f"sent (unique accepted)   {len(accepted):>8,}  payment_ids")
    print(f"stored                   {distinct:>8,}  payment_ids ({rows:,} rows)")
    print(f"lost                     {len(lost_ids):>8,}" + (f"  ({dlq:,} of them on {DEAD_LETTER_TOPIC})" if lost_ids else ""))
    print(f"duplicated               {rows - distinct:>8,}")
    if unexpected:
        print(f"stored but never accepted {len(unexpected):>7,}")
    if lost_ids:
        print(f"  e.g. {sorted(lost_ids)[:5]}")
    sys.exit(1 if lost_ids or rows != distinct or unexpected else 0)


if __name__ == "__main__":
    main()
