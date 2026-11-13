"""Smoke and scaling test for the pipeline on the throwaway kind cluster in CI
(the k8s-smoke job in .github/workflows/ci.yml). Not meant for the laptop.

    python -m scripts.k8s_smoke [--smoke 500] [--payments 10000] [--replicas 1,3]

Needs kubectl pointed at the cluster, and `kubectl port-forward` running for the
API (svc/api -> localhost:8080) and Postgres (svc/postgres -> localhost:5432).
The CI job starts both and restarts them if they drop; database reads here
reconnect if that happens mid-run, and the HTTP sender retries.

1. Smoke: with 1 consumer running, POST --smoke payments and wait until they are
   all in raw_payments. Prints sent, stored, lost, duplicated.
2. Backlog, for each replica count: scale the consumer to 0, empty the tables,
   POST --payments payments as fast as possible (they wait in Kafka), then
   scale to N and time until all are stored. The clock starts at the scale
   command, so pod start-up and joining the consumer group are included.
3. Recovery: 3 consumers, --payments payments sent as fast as possible; one
   consumer pod is deleted once a third are stored. Prints the seconds until the
   backlog is back to its level before the deletion, and lost / duplicated.

Exits non-zero if any accepted payment is lost, or if the smoke run is not
stored exactly once.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
import uuid
from contextlib import suppress
from datetime import UTC, datetime

import httpx
import psycopg2

from scripts.replay import post_all, summarize

API = os.getenv("K8S_API", "http://localhost:8080")
PAYMENT_TABLES = ("raw_payments", "payments_clean", "payments_quarantine")
PID = "payload ->> 'payment_id'"
GROUP = "transaction-consumer-group"
SAMPLE_SECONDS = 0.5


# --- small pure helpers (unit-tested) ---------------------------------------------------


def make_records(n: int, tag: str, seed: int = 0) -> list[dict]:
    """n payments with unique payment_ids '<tag>_000001'..., spread over 1,000
    accounts (the API keys messages by account, so all partitions get work)."""
    rng = random.Random(seed)
    now = datetime.now(UTC).isoformat()
    return [
        {
            "payment_id": f"{tag}_{i:06d}",
            "account_id": f"ACC{rng.randrange(1000):05d}",
            "merchant_id": f"M{rng.randrange(200):04d}",
            "amount": round(rng.uniform(1, 500), 2),
            "currency": "USD",
            "country": "US",
            "channel": rng.choice(["online", "pos", "mobile"]),
            "event_time": now,
        }
        for i in range(1, n + 1)
    ]


def total_lag(describe_output: str) -> int | None:
    """Sum of the LAG column from `kafka-consumer-groups --describe` ('-' counts as unknown)."""
    lines = [x.split() for x in describe_output.splitlines() if x.strip()]
    header = next((x for x in lines if "LAG" in x and "PARTITION" in x), None)
    if header is None:
        return None
    col = header.index("LAG")
    total, seen = 0, False
    for row in lines[lines.index(header) + 1 :]:
        if len(row) > col and row[col].isdigit():
            total += int(row[col])
            seen = True
    return total if seen else None


def catch_up_seconds(samples: list[tuple[float, int, int]], t_event: float, margin: int = 50) -> float | None:
    """samples: (time, accepted so far, stored so far). The backlog is accepted
    minus stored. Returns seconds after t_event until the backlog is back at or
    below its median over the 10 s before t_event (plus `margin`), or None."""
    before = [a - s for t, a, s in samples if t_event - 10 <= t < t_event]
    if not before:
        return None
    level = statistics.median(before) + margin
    for t, a, s in samples:
        if t > t_event and a - s <= level:
            return t - t_event
    return None


def failures(smoke: dict, runs: list[dict], recovery: dict | None) -> list[str]:
    """Why the job should fail (empty list: pass)."""
    out = []
    if smoke["lost"] or smoke["stored"] != smoke["sent"] or smoke["duplicated"]:
        out.append(f"smoke: sent {smoke['sent']}, stored {smoke['stored']}, lost {smoke['lost']}, "
                   f"duplicated {smoke['duplicated']}")  # fmt: skip
    out += [f"backlog with {r['replicas']} replica(s): lost {r['lost']}" for r in runs if r["lost"]]
    if recovery and recovery["lost"]:
        out.append(f"recovery: lost {recovery['lost']}")
    return out


# --- cluster and database ---------------------------------------------------------------


def run(*cmd: str, check: bool = True, timeout: float = 120) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if check and out.returncode != 0:
        sys.exit(f"{' '.join(cmd)} failed:\n{out.stderr.strip()}")
    return out.stdout


def kubectl(*args: str, check: bool = True) -> str:
    return run("kubectl", *args, check=check)


def consumer_pods() -> list[str]:
    out = json.loads(kubectl("get", "pods", "-l", "app=consumer", "-o", "json"))
    return [p["metadata"]["name"] for p in out["items"] if p["metadata"].get("deletionTimestamp") is None]


def ready_consumers() -> int:
    return int(kubectl("get", "deployment", "consumer", "-o", "jsonpath={.status.readyReplicas}") or 0)


def scale(replicas: int, wait: bool = True, timeout: float = 180) -> None:
    kubectl("scale", "deployment/consumer", f"--replicas={replicas}")
    if not wait:
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pods = kubectl("get", "pods", "-l", "app=consumer", "-o", "name").split()
        if ready_consumers() == replicas and len(pods) == replicas:
            return
        time.sleep(1)
    sys.exit(f"consumer did not reach {replicas} ready replicas in {timeout:.0f}s")


def kafka_lag() -> int | None:
    # Small heap: the tool runs inside the broker's pod, which would otherwise
    # give it the broker's KAFKA_HEAP_OPTS and risk the pod's memory limit.
    out = kubectl("exec", "deploy/kafka", "--", "env", "KAFKA_HEAP_OPTS=-Xmx64m",
                  "/opt/kafka/bin/kafka-consumer-groups.sh",
                  "--bootstrap-server", "localhost:9092", "--describe", "--group", GROUP, check=False)  # fmt: skip
    return total_lag(out)


def wait_for_api(api: str, seconds: float = 60) -> None:
    """Poll {api}/health until it answers 200, or exit with a Kubernetes-specific hint."""
    deadline, error = time.monotonic() + seconds, None
    while time.monotonic() < deadline:
        try:
            httpx.get(f"{api}/health", timeout=5).raise_for_status()
            return
        except httpx.HTTPError as exc:
            error = exc
            time.sleep(2)
    sys.exit(f"API not reachable at {api} after {seconds:.0f}s ({error}). Is `kubectl port-forward svc/api "
             "8080:8000` running, and is the api pod Ready (kubectl get pods -l app=api)?")  # fmt: skip


class Db:
    """Autocommit Postgres connection that reconnects (a few times) if the
    port-forward restarts under it. Every query here is a read or an idempotent
    TRUNCATE, so running one again is safe."""

    def __init__(self, attempts: int = 5) -> None:
        self.attempts = attempts
        self.conn = None

    def _connect(self) -> None:
        self.conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            user=os.getenv("DB_USER", "fintech"),
            password=os.getenv("DB_PASSWORD", "fintech"),
            dbname=os.getenv("DB_NAME", "transactions_db"),
            connect_timeout=5,
        )
        self.conn.autocommit = True

    def query(self, sql: str, params: tuple = (), fetch: bool = True) -> list[tuple]:
        for attempt in range(1, self.attempts + 1):
            try:
                if self.conn is None or self.conn.closed:
                    self._connect()
                with self.conn.cursor() as cur:
                    cur.execute(sql, params)
                    return cur.fetchall() if fetch else []
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as exc:
                if attempt == self.attempts:
                    raise
                print(f"  database connection lost ({type(exc).__name__}); reconnecting")
                self.close()
                time.sleep(2)
        return []

    def close(self) -> None:
        if self.conn is not None:
            with suppress(psycopg2.Error):
                self.conn.close()
            self.conn = None


def reset_tables(db: Db) -> None:
    db.query(f"TRUNCATE {', '.join(PAYMENT_TABLES)} RESTART IDENTITY", fetch=False)


def stored(db: Db, tag: str) -> tuple[int, int]:
    """(rows, distinct payment_ids) for this run."""
    [row] = db.query(f"SELECT count(*), count(DISTINCT {PID}) FROM raw_payments WHERE {PID} LIKE %s", (f"{tag}\\_%",))
    return row


def stored_ids(db: Db, tag: str) -> set[str]:
    return {r[0] for r in db.query(f"SELECT {PID} FROM raw_payments WHERE {PID} LIKE %s", (f"{tag}\\_%",))}


def send(records: list[dict], workers: int) -> tuple[set[str], float, dict]:
    t0 = time.monotonic()
    results = post_all(records, API, workers)
    seconds = time.monotonic() - t0
    return {r.payment_id for r in results if r.status == 202}, seconds, summarize(results)


def wait_stored(conn, tag: str, want: int, idle_limit: float) -> float:
    """Poll until `want` distinct payments of this run are stored, or no progress
    for idle_limit seconds. Returns the monotonic time it stopped."""
    last, last_change = -1, time.monotonic()
    while True:
        rows, distinct = stored(conn, tag)
        now = time.monotonic()
        if distinct >= want:
            return now
        if rows != last:
            last, last_change = rows, now
        elif now - last_change > idle_limit:
            print(f"  no progress for {idle_limit:.0f}s; stopping at {distinct:,}/{want:,}")
            return now
        time.sleep(SAMPLE_SECONDS)


def final_counts(conn, tag: str, accepted: set[str]) -> dict:
    time.sleep(3)  # late re-deliveries, before counting duplicates
    rows, distinct = stored(conn, tag)
    ids = stored_ids(conn, tag)
    return {"sent": len(accepted), "stored": len(ids & accepted), "lost": len(accepted - ids),
            "duplicated": rows - distinct, "stored_not_acknowledged": len(ids - accepted)}  # fmt: skip


# --- the three tests ----------------------------------------------------------------------


def smoke_run(n: int, workers: int, idle_limit: float) -> dict:
    tag = f"smoke{uuid.uuid4().hex[:6]}"
    print(f"\nSMOKE: {n:,} payments with 1 consumer, tag {tag}")
    scale(1)
    conn = Db()
    try:
        reset_tables(conn)
        accepted, send_s, s = send(make_records(n, tag), workers)
        print(f"  sent {len(accepted):,} accepted of {n:,} in {send_s:.1f}s, {s['failed']:,} failed")
        t0 = time.monotonic()
        t_end = wait_stored(conn, tag, len(accepted), idle_limit)
        result = {"stored_after": t_end - t0, **final_counts(conn, tag, accepted)}
    finally:
        conn.close()
    print(f"  sent {result['sent']:,}  stored {result['stored']:,}  lost {result['lost']:,}  "
          f"duplicated {result['duplicated']:,}  (all stored {result['stored_after']:.1f}s after the last send)")  # fmt: skip
    return result


def backlog_run(n: int, replicas: int, workers: int, idle_limit: float) -> dict:
    tag = f"bl{replicas}r{uuid.uuid4().hex[:6]}"
    print(f"\nBACKLOG: {n:,} payments, then {replicas} consumer replica(s), tag {tag}")
    scale(0)
    conn = Db()
    try:
        reset_tables(conn)
        accepted, send_s, s = send(make_records(n, tag, seed=replicas), workers)
        print(f"  sent {len(accepted):,} accepted of {n:,} in {send_s:.1f}s "
              f"({len(accepted) / send_s:,.0f}/s), {s['failed']:,} failed; consumers off")  # fmt: skip
        lag = kafka_lag()
        print(f"  Kafka backlog before scaling: {lag if lag is not None else 'n/a'} messages")
        t0 = time.monotonic()
        scale(replicas, wait=False)
        t_end = wait_stored(conn, tag, len(accepted), idle_limit)
        result = {"replicas": replicas, "seconds": t_end - t0, **final_counts(conn, tag, accepted)}
    finally:
        conn.close()
    result["rate"] = result["stored"] / result["seconds"]
    print(f"  {replicas} replica(s): all stored {result['seconds']:.1f}s after scaling "
          f"({result['rate']:,.0f} payments/s); lost {result['lost']:,}, duplicated {result['duplicated']:,}")  # fmt: skip
    return result


def recovery_run(n: int, workers: int, idle_limit: float) -> dict:
    tag = f"rec{uuid.uuid4().hex[:6]}"
    print(f"\nRECOVERY: {n:,} payments with 3 replicas, one consumer pod deleted, tag {tag}")
    scale(3)
    conn = Db()
    reset_tables(conn)
    records = make_records(n, tag, seed=99)
    lock, accepted_so_far = threading.Lock(), [0]

    def on_result(r) -> None:
        if r.status == 202:
            with lock:
                accepted_so_far[0] += 1

    results: list = []
    sender = threading.Thread(target=lambda: results.extend(post_all(records, API, workers, on_result)))
    t0 = time.monotonic()
    sender.start()
    samples: list[tuple[float, int, int]] = []
    t_delete = t_sent = t_replacement = None
    deleted = None
    accepted: set[str] = set()
    last, last_change = -1, time.monotonic()
    try:
        while True:
            _, distinct = stored(conn, tag)
            now = time.monotonic() - t0
            with lock:
                samples.append((now, accepted_so_far[0], distinct))
            if deleted is None and distinct >= n // 3:
                deleted = consumer_pods()[0]
                kubectl("delete", "pod", deleted, "--wait=false")
                t_delete = now
                print(f"  deleted pod {deleted} at {now:.1f}s ({distinct:,} stored)")
            if deleted and t_replacement is None and len(consumer_pods()) == 3 and ready_consumers() == 3:
                t_replacement = now
            if t_sent is None and not sender.is_alive():
                t_sent = now
                accepted = {r.payment_id for r in results if r.status == 202}
            if t_sent is not None and distinct >= len(accepted):
                break
            if distinct != last:
                last, last_change = distinct, time.monotonic()
            elif t_sent is not None and time.monotonic() - last_change > idle_limit:
                print(f"  no progress for {idle_limit:.0f}s; stopping")
                break
            time.sleep(SAMPLE_SECONDS)
        t_done = time.monotonic() - t0
        counts = final_counts(conn, tag, accepted)
    finally:
        conn.close()
    result = {
        "deleted_pod": deleted,
        "deleted_at": t_delete,
        "replacement_ready_after": (t_replacement - t_delete) if t_replacement and t_delete else None,
        "catch_up_after": catch_up_seconds(samples, t_delete) if t_delete is not None else None,
        "sending_done_at": t_sent,
        "all_stored_at": t_done,
        **counts,
    }
    print(f"  replacement pod ready {fmt_s(result['replacement_ready_after'])} after the deletion; "
          f"backlog back to its earlier level {fmt_s(result['catch_up_after'])} after the deletion")  # fmt: skip
    print(f"  sending finished at {fmt_s(t_sent)}, all stored at {fmt_s(t_done)}")
    print(f"  sent {counts['sent']:,}  stored {counts['stored']:,}  lost {counts['lost']:,}  "
          f"duplicated {counts['duplicated']:,}")  # fmt: skip
    return result


def fmt_s(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1f}s"


def summary(smoke: dict, runs: list[dict], recovery: dict | None, n: int) -> str:
    lines = [
        "## k8s smoke (throwaway kind cluster on a GitHub runner)",
        "",
        (f"Smoke: sent {smoke['sent']:,}, stored {smoke['stored']:,}, lost {smoke['lost']:,}, "
        f"duplicated {smoke['duplicated']:,}"),
        "",
        f"Backlog: {n:,} payments waiting in Kafka, then the consumer scaled up (clock starts at the scale command)",
        "",
        "| Consumer replicas | Seconds until all stored | Payments/s | Lost | Duplicated |",
        "|---:|---:|---:|---:|---:|",
        *(f"| {r['replicas']} | {r['seconds']:.1f} | {r['rate']:,.0f} | {r['lost']:,} | {r['duplicated']:,} |"
          for r in runs),
    ]  # fmt: skip
    if recovery:
        r = recovery
        lines += [
            "",
            (f"Recovery (3 replicas, pod deleted under load): replacement ready after "
            f"{fmt_s(r['replacement_ready_after'])}, backlog caught up after {fmt_s(r['catch_up_after'])}; "
            f"sent {r['sent']:,}, stored {r['stored']:,}, lost {r['lost']:,}, duplicated {r['duplicated']:,}"),
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--smoke", type=int, default=500)
    p.add_argument("--payments", type=int, default=10_000)
    p.add_argument("--replicas", default="1,3")
    p.add_argument("--workers", type=int, default=32, help="concurrent HTTP senders")
    p.add_argument("--idle-limit", type=float, default=60)
    p.add_argument("--skip-recovery", action="store_true")
    args = p.parse_args()

    wait_for_api(API)
    smoke = smoke_run(args.smoke, args.workers, args.idle_limit)
    runs = [backlog_run(args.payments, int(n), args.workers, args.idle_limit) for n in args.replicas.split(",")]
    recovery = None if args.skip_recovery else recovery_run(args.payments, args.workers, args.idle_limit)

    text = summary(smoke, runs, recovery, args.payments)
    print("\n" + text)
    if path := os.getenv("GITHUB_STEP_SUMMARY"):
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)
    if problems := failures(smoke, runs, recovery):
        sys.exit("FAILED: " + "; ".join(problems))


if __name__ == "__main__":
    main()
