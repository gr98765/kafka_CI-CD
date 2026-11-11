"""Crash test: kill the consumer during a replay, restart it, and check that no
accepted record was lost from, or stored twice in, the raw layer (raw_payments). Repeated at several kill points,
then once more with duplicate protection off to show what it prevents.

    python -m scripts.kill_test [--limit 6000] [--kill-at 0.25,0.5,0.75] [--no-plain-run]

Needs the stack running (docker compose up -d --build). Each run tags every
payment_id and account_id with its own run tag, so runs are counted on their
own and never touch the velocity counts of payments replayed by scripts.replay.
The consumer is killed with SIGKILL (docker compose kill): no clean shutdown.

Exact re-sends already in the input file (the generator's "duplicate" mess) are
dropped, so every duplicate the consumer ignores after the restart is a Kafka
re-delivery: messages it had stored but not yet committed when it was killed.
The consumer logs each one ("duplicate ignored: <payment_id>").

The plain-INSERT run recreates the consumer with DEDUP_DISABLED=1: plain INSERT
into raw_payments_nodedup (no (partition, offset) unique key), then puts the
normal consumer back.
"""

from __future__ import annotations

import argparse
import os
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
RAW_TABLE = "raw_payments"
NODEDUP_TABLE = "raw_payments_nodedup"
PID = "payload ->> 'payment_id'"


def compose(*args: str, check: bool = True, env: dict | None = None) -> str:
    out = subprocess.run(
        ["docker", "compose", *args],
        cwd=ROOT, capture_output=True, text=True, check=False,
        env={**os.environ, **(env or {})},
    )  # fmt: skip
    if check and out.returncode != 0:
        sys.exit(f"docker compose {' '.join(args)} failed:\n{out.stderr}")
    return out.stdout


def tag_record(record: dict, tag: str) -> dict:
    tagged = dict(record)
    for field in ("payment_id", "account_id"):
        if isinstance(tagged.get(field), str):
            tagged[field] = f"{tag}_{tagged[field]}"
    return tagged


def drop_resends(records: list[dict]) -> list[dict]:
    """Keep the first record per payment_id (removes the input's exact re-sends)."""
    seen, out = set(), []
    for r in records:
        pid = r.get("payment_id")
        if pid in seen:
            continue
        seen.add(pid)
        out.append(r)
    return out


def stored_counts(conn, tag: str, table: str = RAW_TABLE) -> tuple[int, int]:
    """(rows, distinct payment_ids) stored for this run."""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT count(*), count(DISTINCT {PID}) FROM {table} WHERE {PID} LIKE %s",
            (f"{tag}\\_%",),
        )
        rows, distinct = cur.fetchone()
    return rows, distinct


def stored_ids(conn, tag: str, table: str = RAW_TABLE) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT {PID} FROM {table} WHERE {PID} LIKE %s", (f"{tag}\\_%",))
        return {r[0] for r in cur}


def ignored_duplicates(tag: str) -> int:
    """How many of this run's payments the consumer has logged as ignored duplicates."""
    out = compose("logs", "--no-log-prefix", "consumer", check=False)
    return sum(1 for line in out.splitlines() if f"duplicate ignored: {tag}_" in line)


def dead_lettered(tag: str) -> int:
    """How many of this run's records are on the dead-letter topic."""
    out = compose(
        "exec", "-T", "kafka", "kafka-console-consumer",
        "--bootstrap-server", "localhost:29092", "--topic", DEAD_LETTER_TOPIC,
        "--from-beginning", "--timeout-ms", "8000",
        check=False,
    )  # fmt: skip
    return sum(1 for line in out.splitlines() if f"{tag}_" in line)


def wait_for_drain(conn, tag: str, expected: int, idle_limit: float, table: str) -> int:
    """Wait until every accepted payment is stored (distinct ids), or nothing new
    arrives for `idle_limit` seconds. Returns the final row count."""
    last, last_change = -1, time.monotonic()
    while True:
        rows, distinct = stored_counts(conn, tag, table)
        if distinct >= expected:
            time.sleep(5)  # let late re-deliveries land before counting duplicates
            return stored_counts(conn, tag, table)[0]
        if rows != last:
            last, last_change = rows, time.monotonic()
            print(f"    stored {distinct:,}/{expected:,}", flush=True)
        elif time.monotonic() - last_change > idle_limit:
            return rows
        time.sleep(2)


def wait_for_consumer_running() -> None:
    for _ in range(30):
        if compose("ps", "--status", "running", "-q", "consumer").strip():
            return
        time.sleep(1)
    sys.exit("consumer container did not start")


def run_once(args, base_records: list[dict], kill_at: float, dedup: bool) -> dict:
    table = RAW_TABLE if dedup else NODEDUP_TABLE
    tag = f"kt{uuid.uuid4().hex[:8]}"
    records = [tag_record(r, tag) for r in base_records]
    kill_after = int(len(records) * kill_at)
    print(f"run {tag} ({'dedup on' if dedup else 'PLAIN INSERT'}): {len(records):,} records, "
          f"kill after {kill_after:,} posted ({kill_at:.0%})")  # fmt: skip

    done, lock, reached = [0], threading.Lock(), threading.Event()

    def on_result(_) -> None:
        with lock:
            done[0] += 1
            if done[0] >= kill_after:
                reached.set()

    results: list = []
    replay = threading.Thread(target=lambda: results.extend(post_all(records, args.api, args.workers, on_result)))
    replay.start()

    conn = db_connect()
    while not reached.wait(1):
        if not replay.is_alive():
            sys.exit("replay stopped before the kill point")
    compose("kill", "consumer")
    posted_at_kill = done[0]
    stored_at_kill, _ = stored_counts(conn, tag, table)
    ignored_before = ignored_duplicates(tag) if dedup else 0
    print(f"    killed consumer (SIGKILL) at {posted_at_kill:,} posted, {stored_at_kill:,} stored")
    time.sleep(args.down_seconds)
    compose("start", "consumer")
    print(f"    restarted consumer after {args.down_seconds:.0f}s")

    replay.join()
    summary = summarize(results)
    accepted = {r.payment_id for r in results if r.status == 202}
    rows = wait_for_drain(conn, tag, len(accepted), args.idle_limit, table)
    _, distinct = stored_counts(conn, tag, table)
    ids = stored_ids(conn, tag, table)
    conn.close()
    lost = accepted - ids
    return {
        "tag": tag,
        "mode": "dedup (ON CONFLICT)" if dedup else "plain INSERT",
        "kill_at": kill_at,
        "posted_at_kill": posted_at_kill,
        "stored_at_kill": stored_at_kill,
        "ignored_before": ignored_before,
        "redelivered_ignored": (ignored_duplicates(tag) - ignored_before) if dedup else None,
        "accepted_posts": summary["accepted"],
        "sent": len(accepted),
        "stored": distinct,
        "rows": rows,
        "lost": len(lost),
        "dead_lettered": dead_lettered(tag) if lost else 0,
        "duplicated": rows - distinct,
        "unexpected": len(ids - accepted),
    }


def print_table(runs: list[dict]) -> None:
    head = ("run", "mode", "kill at", "stored at kill", "re-delivered+ignored",
            "sent", "stored", "lost", "duplicated")  # fmt: skip
    rows = [
        (
            str(i),
            r["mode"],
            f"{r['kill_at']:.0%} ({r['posted_at_kill']:,} posted)",
            f"{r['stored_at_kill']:,}",
            "n/a (no dedup)" if r["redelivered_ignored"] is None else f"{r['redelivered_ignored']:,}",
            f"{r['sent']:,}",
            f"{r['stored']:,}",
            f"{r['lost']:,}",
            f"{r['duplicated']:,}",
        )
        for i, r in enumerate(runs, start=1)
    ]
    widths = [max(len(x) for x in col) for col in zip(head, *rows, strict=True)]
    for line in (head, tuple("-" * w for w in widths), *rows):
        print("  ".join(x.rjust(w) if i > 1 else x.ljust(w) for i, (x, w) in enumerate(zip(line, widths, strict=True))))


def ensure_nodedup_table() -> None:
    conn = db_connect()
    try:
        with conn.cursor() as cur:
            # LIKE copies columns, NOT NULLs and defaults, but no unique keys or indexes.
            cur.execute(f"CREATE TABLE IF NOT EXISTS {NODEDUP_TABLE} (LIKE {RAW_TABLE} INCLUDING DEFAULTS)")
            # Non-unique, so duplicates still go in; it only keeps the per-run
            # counts (WHERE payment_id LIKE 'tag\_%') from scanning the whole table.
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {NODEDUP_TABLE}_payment_id ON {NODEDUP_TABLE} "
                f"(({PID}) text_pattern_ops)"
            )
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--file", default=DEFAULT_FILE)
    p.add_argument("--api", default=DEFAULT_API)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=6000, help="records per run (first N of the file)")
    p.add_argument("--kill-at", default="0.25,0.5,0.75", help="kill points as fractions of the records posted")
    p.add_argument("--plain-kill-at", type=float, default=0.5, help="kill point for the plain-INSERT run")
    p.add_argument("--no-plain-run", action="store_true", help="skip the plain-INSERT run")
    p.add_argument("--down-seconds", type=float, default=15, help="how long the consumer stays dead")
    p.add_argument("--idle-limit", type=float, default=90, help="give up waiting after this long with no progress")
    args = p.parse_args()

    check_api(args.api)
    if not compose("ps", "--status", "running", "-q", "consumer").strip():
        sys.exit("consumer container is not running; start the stack with `docker compose up -d --build`")

    loaded = load_records(args.file, args.limit)
    base = drop_resends(loaded)
    print(f"{len(loaded):,} records loaded; dropped {len(loaded) - len(base):,} exact re-sends from the input\n")

    runs = [run_once(args, base, float(k), dedup=True) for k in args.kill_at.split(",") if k.strip()]

    if not args.no_plain_run:
        ensure_nodedup_table()
        print("recreating the consumer with DEDUP_DISABLED=1 (plain INSERT into raw_payments_nodedup)")
        compose("up", "-d", "consumer", env={"DEDUP_DISABLED": "1"})
        wait_for_consumer_running()
        try:
            runs.append(run_once(args, base, args.plain_kill_at, dedup=False))
        finally:
            print("restoring the normal consumer (DEDUP_DISABLED=0)")
            compose("up", "-d", "consumer", env={"DEDUP_DISABLED": "0"})

    print()
    print_table(runs)
    for i, r in enumerate(runs, start=1):
        notes = []
        if r["ignored_before"]:
            notes.append(f"{r['ignored_before']:,} duplicates ignored before the kill")
        if r["dead_lettered"]:
            notes.append(f"{r['dead_lettered']:,} of the lost are on {DEAD_LETTER_TOPIC}")
        if r["unexpected"]:
            notes.append(f"{r['unexpected']:,} stored but never accepted")
        if r["accepted_posts"] != r["sent"]:
            notes.append(f"{r['accepted_posts'] - r['sent']:,} payment_ids accepted twice by the API (replay retries)")
        if notes:
            print(f"run {i}: " + "; ".join(notes))

    protected = [r for r in runs if r["redelivered_ignored"] is not None]
    ok = all(r["lost"] == 0 and r["duplicated"] == 0 and not r["unexpected"] for r in protected)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
