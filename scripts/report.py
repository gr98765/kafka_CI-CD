"""Load-test report: Locust's numbers, then what actually reached the database.

    python -m scripts.report reset      # before the run: empty the payments tables
    locust -f scripts/locustfile.py --headless --host http://localhost:8000 -u 50 -r 10 -t 2m --csv results/locust
    python -m scripts.report summary    # after the run

`summary` reads results/locust_stats.csv (requests per second, p50/p95/p99,
failures), then waits until the consumer has stored everything the API
accepted (results/locust_sent.txt, written by the locustfile) or stops making
progress, and counts sent, stored, lost and duplicated in raw_payments, the
table the consumer writes. (payments_clean is filled later by the batch clean
step, python -m app.clean.)
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

from scripts.score import RESULTS, ROOT, db_connect

DEFAULT_PREFIX = RESULTS / "locust"
DEFAULT_SENT = RESULTS / "locust_sent.txt"
PAYMENT_TABLES = ("raw_payments", "payments_clean", "payments_quarantine")
HANDOFF_NOTE = (
    "The API replies after handing the payment to Kafka, so p95 measures the hand-off, not saving to Postgres."
)


# --- Locust CSV -----------------------------------------------------------------


def parse_stats(path: Path, name: str = "Aggregated") -> dict:
    """The `name` row of a Locust *_stats.csv (default: the all-requests row)."""
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["Name"] == name:
                return {
                    "requests": int(row["Request Count"]),
                    "failures": int(row["Failure Count"]),
                    "rps": float(row["Requests/s"]),
                    "p50": float(row["50%"]),
                    "p95": float(row["95%"]),
                    "p99": float(row["99%"]),
                    "avg": float(row["Average Response Time"]),
                    "max": float(row["Max Response Time"]),
                }
    raise ValueError(f"no {name!r} row in {path}")


def parse_failures(path: Path) -> Counter:
    """Failure counts by error message from a Locust *_failures.csv (empty if none)."""
    out: Counter = Counter()
    if path.exists():
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                out[row["Error"]] += int(row["Occurrences"])
    return out


def stats_lines(s: dict, failures: Counter) -> list[str]:
    return [
        f"requests        {s['requests']:,}",
        f"requests/s      {s['rps']:.1f}  (whole run, ramp-up included)",
        f"latency p50     {s['p50']:.0f} ms",
        f"latency p95     {s['p95']:.0f} ms",
        f"latency p99     {s['p99']:.0f} ms",
        f"latency max     {s['max']:.0f} ms",
        f"failures        {s['failures']:,}"
        + (" (" + ", ".join(f"{e}: {n:,}" for e, n in failures.most_common()) + ")" if failures else ""),
    ]


# --- sent vs stored ---------------------------------------------------------------


def read_sent(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def run_tag(sent: list[str]) -> str:
    tags = {pid.rsplit("_", 1)[0] for pid in sent}
    if len(tags) != 1:
        raise ValueError(f"expected one run tag in the sent file, found {sorted(tags)[:3]}")
    return tags.pop()


def stored_copies(conn, tag: str) -> Counter:
    """payment_id -> raw rows, for this run's payments."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT payload ->> 'payment_id', count(*) FROM raw_payments "
            "WHERE payload ->> 'payment_id' LIKE %s GROUP BY 1",
            (tag.replace("_", "\\_") + "\\_%",),
        )
        return Counter(dict(cur.fetchall()))


def compare(sent: list[str], copies: Counter) -> dict:
    sent_set = set(sent)
    stored = set(copies)
    return {
        "sent": len(sent_set),
        "stored": len(stored & sent_set),
        "rows": sum(copies.values()),
        "lost": len(sent_set - stored),
        "duplicated": sum(n - 1 for n in copies.values() if n > 1),
        "stored_not_acknowledged": len(stored - sent_set),  # client saw an error, Kafka got it anyway
    }


def wait_for_consumer(conn, tag: str, expected: int, idle_limit: float, log=print) -> tuple[Counter, float]:
    """Poll until every accepted payment is stored or nothing new arrives for
    idle_limit seconds. Returns (copies, seconds waited)."""
    start = time.monotonic()
    last, last_change = -1, start
    while True:
        copies = stored_copies(conn, tag)
        rows = sum(copies.values())
        if len(copies) >= expected:
            time.sleep(3)  # let any late re-delivery land before counting duplicates
            return stored_copies(conn, tag), time.monotonic() - start
        if rows != last:
            log(f"  waiting for the consumer: {len(copies):,}/{expected:,} stored")
            last, last_change = rows, time.monotonic()
        elif time.monotonic() - last_change > idle_limit:
            return copies, time.monotonic() - start
        time.sleep(2)


# --- environment ------------------------------------------------------------------


def docker_memory_note() -> str:
    """Where this ran and how much memory Docker had (read, not assumed)."""
    parts = []
    wslconfig = Path(os.environ.get("USERPROFILE", Path.home())) / ".wslconfig"
    if wslconfig.exists():
        lines = [x.strip() for x in wslconfig.read_text().splitlines()]
        setting = next((x for x in lines if x.lower().startswith("memory")), None)
        if setting:
            parts.append(f"{setting} in {wslconfig}")
    try:
        out = subprocess.run(
            ["docker", "info", "--format", "{{.MemTotal}}|{{.NCPU}}|{{.OperatingSystem}}"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip()  # fmt: skip
        mem, cpus, os_name = out.split("|")
        parts.append(f"docker info: {int(mem) / 2**30:.2f} GiB memory, {cpus} CPUs ({os_name})")
    except (OSError, subprocess.SubprocessError, ValueError):
        parts.append("docker info unavailable")
    return "Ran on one laptop: every service in Docker, with the test client on the same machine; " + "; ".join(parts) + "."


# --- commands -----------------------------------------------------------------------


def reset_tables() -> None:
    conn = db_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE {', '.join(PAYMENT_TABLES)} RESTART IDENTITY")
    finally:
        conn.close()
    print(f"emptied {', '.join(PAYMENT_TABLES)}")


def stats_files(prefix: Path) -> tuple[Path, Path, bool]:
    """(stats csv, failures csv, is_final). Prefers the files the locustfile
    writes once at the end; Locust's own <prefix>_stats.csv is rewritten once a
    second while running and can miss the last second of requests."""
    final = Path(f"{prefix}_stats_final.csv")
    if final.exists():
        return final, Path(f"{prefix}_failures_final.csv"), True
    return Path(f"{prefix}_stats.csv"), Path(f"{prefix}_failures.csv"), False


def summary(prefix: Path, sent_path: Path, idle_limit: float) -> None:
    stats_path, failures_path, final = stats_files(prefix)
    s = parse_stats(stats_path)
    failures = parse_failures(failures_path)
    shown = stats_path.relative_to(ROOT) if stats_path.is_relative_to(ROOT) else stats_path
    print(f"LOCUST ({shown})")
    if not final:
        print("  note: no *_stats_final.csv found; Locust's periodic CSV can miss the last second of requests")
    for line in stats_lines(s, failures):
        print("  " + line)

    sent = read_sent(sent_path)
    tag = run_tag(sent)
    conn = db_connect()
    try:
        copies, waited = wait_for_consumer(conn, tag, len(set(sent)), idle_limit)
    finally:
        conn.close()
    c = compare(sent, copies)
    print()
    print(f"DATABASE (raw_payments, run tag {tag}; waited {waited:.0f} s after starting this report)")
    print(f"  sent (API answered 202)   {c['sent']:,}")
    print(f"  stored                    {c['stored']:,}")
    print(f"  lost                      {c['lost']:,}")
    print(f"  duplicated rows           {c['duplicated']:,}")
    if c["stored_not_acknowledged"]:
        print(f"  stored but not acknowledged to Locust: {c['stored_not_acknowledged']:,}")
    print()
    print(docker_memory_note())
    print(HANDOFF_NOTE)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("reset", help="empty raw_payments, payments_clean and payments_quarantine")
    s = sub.add_parser("summary", help="Locust numbers, then sent vs stored")
    s.add_argument("--prefix", type=Path, default=DEFAULT_PREFIX, help="Locust --csv prefix")
    s.add_argument("--sent", type=Path, default=DEFAULT_SENT)
    s.add_argument("--idle-limit", type=float, default=60, help="give up after this long with no progress")
    args = p.parse_args()
    if args.cmd == "reset":
        reset_tables()
    else:
        summary(args.prefix.resolve(), args.sent, args.idle_limit)


if __name__ == "__main__":
    main()
