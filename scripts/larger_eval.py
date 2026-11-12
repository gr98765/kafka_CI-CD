"""Larger, repeatable model evaluation and SQL timing, one seed at a time.

    python -m generator.generate --out data/large    --accounts 5000 --days 60 --seed 1
    python -m generator.generate --out data/large_s2 --accounts 5000 --days 60 --seed 2
    python -m generator.generate --out data/large_s3 --accounts 5000 --days 60 --seed 3

    python -m scripts.larger_eval load     --seed 1      # COPY into schema eval_s1 (dropped first)
    python -m scripts.larger_eval evaluate --seed 1      # -> results/larger_eval/seed1.json
    python -m scripts.larger_eval sql-speed --seed 1     # -> results/larger_eval/sql_speed.json
    python -m scripts.larger_eval report                 # all seeds -> results/larger_eval.md

Each seed lives in its own schema (eval_s1, eval_s2, eval_s3) with its own
payments_clean, so seeds never mix and the app's public tables are untouched.
`load` drops and recreates the schema, then streams the generator's clean
payments_truth.jsonl through COPY in chunks (no API, no Kafka). That file is
already clean, so nothing is quarantined: every payment is present, and the
rules are computed here in Python (status is loaded as 'cleared'). The table
gets its primary key but no (account_id, event_time) index; `sql-speed` times
the queries before and after adding one.

Run `evaluate` once per seed, each in its own process, so memory is freed
between seeds. The split is the same for every seed (60 days): train days
1-36, validation 37-46, test 47-60. Each model's threshold is chosen on the
validation window so its false alarms there equal the fixed rules' false
alarms on the same validation window; the test window is only scored.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import sys
import time
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from math import ceil
from pathlib import Path

import numpy as np
from psycopg2.errors import QueryCanceled
from sklearn.metrics import average_precision_score

from app.main import FLAG_THRESHOLD, VELOCITY_LIMIT, VELOCITY_WINDOW_SECONDS
from generator.generate import TS_FORMAT
from scripts.score import RESULTS, ROOT, _fmt_pct, _pct, db_connect
from scripts.train_model import (
    FIRST_GAP,
    PATTERNS,
    START,
    Payment,
    feature_names,
    load_home,
    make_models,
    threshold_for_false_alarms,
)

SPLIT = {"train": (1, 36), "validation": (37, 46), "test": (47, 60)}
LEARNING_FRACTIONS = (0.25, 0.5, 1.0)
TUNED_VELOCITY_LIMIT = 5
TINY_AMOUNT = 5.00  # a "tiny" payment is under this many dollars
TINY_LIMIT = 5  # this many tiny payments by one account within the velocity window
DATA_DIRS = {1: ROOT / "data" / "large", 2: ROOT / "data" / "large_s2", 3: ROOT / "data" / "large_s3"}
OUT_DIR = RESULTS / "larger_eval"
COPY_CHUNK = 50_000
INDEX_NAME = "idx_eval_account_time"
SETUPS = ["fixed rules", "hand-tuned rules", "logistic_regression", "gradient_boosting"]
MODELS = ["logistic_regression", "gradient_boosting"]


def schema_for(seed: int) -> str:
    return f"eval_s{int(seed)}"


def peak_memory_mb() -> float | None:
    """This process's peak memory so far (Windows: peak working set; Linux/macOS: max RSS)."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            *((f"pad{i}", ctypes.c_size_t) for i in range(6))]  # fmt: skip

            c = Counters()
            c.cb = ctypes.sizeof(c)
            kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
            if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
                return None
            return c.PeakWorkingSetSize / 2**20
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / 2**20 if sys.platform == "darwin" else rss / 1024  # bytes on macOS, KiB on Linux
    except (OSError, AttributeError, ImportError):
        return None


# --- loading ------------------------------------------------------------------


COPY_COLUMNS = (
    "payment_id, account_id, merchant_id, amount, currency, country, channel, "
    "location, event_time, status, created_at, raw_id"
)


def truth_to_csv_row(record: dict, raw_id: int) -> list:
    """One generator truth record as a payments_clean row (already clean)."""
    ts = record["timestamp"]
    return [
        record["payment_id"],
        record["account_id"],
        record.get("merchant_id"),
        record["amount"],
        record["currency"],
        record.get("country"),
        record.get("channel"),
        "unknown",
        ts,
        "cleared",
        ts,  # no arrival time for bulk-loaded data: use the event time
        raw_id,
    ]


def csv_chunks(lines: Iterable[str], size: int = COPY_CHUNK) -> Iterator[io.StringIO]:
    """Stream records as CSV, `size` rows per buffer, so a file never sits in memory whole."""
    buf, n = io.StringIO(), 0
    writer = csv.writer(buf)
    for raw_id, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        writer.writerow(truth_to_csv_row(json.loads(line), raw_id))
        n += 1
        if n == size:
            buf.seek(0)
            yield buf
            buf, n = io.StringIO(), 0
            writer = csv.writer(buf)
    if n:
        buf.seek(0)
        yield buf


def load_schema(conn, schema: str, truth_path: Path, log=lambda _msg: None) -> int:
    """Drop and recreate `schema`, COPY the truth file into schema.payments_clean."""
    t = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        cur.execute(f"CREATE SCHEMA {schema}")
        # Same columns as the app's payments_clean; no keys or indexes yet.
        cur.execute(f"CREATE TABLE {schema}.payments_clean (LIKE payments_clean INCLUDING DEFAULTS)")
        with open(truth_path) as f:
            for chunk in csv_chunks(f):
                cur.copy_expert(
                    f"COPY {schema}.payments_clean ({COPY_COLUMNS}) FROM STDIN WITH (FORMAT csv)", chunk
                )
        log(f"  COPY: {time.perf_counter() - t:.1f}s")
        t = time.perf_counter()
        cur.execute(f"ALTER TABLE {schema}.payments_clean ADD PRIMARY KEY (payment_id)")
        log(f"  primary key: {time.perf_counter() - t:.1f}s")
        t = time.perf_counter()
        cur.execute(f"ANALYZE {schema}.payments_clean")
        log(f"  analyze: {time.perf_counter() - t:.1f}s")
        cur.execute(f"SELECT count(*) FROM {schema}.payments_clean")
        (n,) = cur.fetchone()
    conn.commit()
    return n


# --- rules (reference: one Python object per payment) ---------------------------


def rule_flags(
    payments: Iterable[Payment],
    velocity_limit: int = VELOCITY_LIMIT,
    tiny_limit: int | None = None,
    amount_threshold: float = FLAG_THRESHOLD,
    window_seconds: int = VELOCITY_WINDOW_SECONDS,
    tiny_amount: float = TINY_AMOUNT,
) -> set[str]:
    """Payment ids flagged by: amount > threshold; or the velocity_limit-th or
    later payment by the account in [t - window, t]; or (tiny_limit set) the
    account has made tiny_limit or more payments under tiny_amount in
    [t - window, t], this one included. That last rule flags whatever payment
    completes the count, tiny or not (so a card-testing cash-out right after
    the tiny probes is flagged too). Ties on event_time ordered by payment_id.

    The reference version, easy to read; rule_mask() is the same rules on
    compact arrays, used for the large datasets."""
    by_account: dict[str, list[Payment]] = defaultdict(list)
    for p in payments:
        by_account[p.account_id].append(p)
    flagged: set[str] = set()
    for rows in by_account.values():
        rows.sort(key=lambda p: (p.event_time, p.payment_id))
        ts = [p.event_time.timestamp() for p in rows]
        left, tiny = 0, deque()
        for i, p in enumerate(rows):
            while ts[left] < ts[i] - window_seconds:
                left += 1
            if p.amount < tiny_amount:
                tiny.append(ts[i])
            while tiny and tiny[0] < ts[i] - window_seconds:
                tiny.popleft()
            if (
                p.amount > amount_threshold
                or i - left + 1 >= velocity_limit
                or (tiny_limit is not None and len(tiny) >= tiny_limit)
            ):
                flagged.add(p.payment_id)
    return flagged


def fixed_rules(payments: list[Payment]) -> set[str]:
    return rule_flags(payments, VELOCITY_LIMIT)


def tuned_rules(payments: list[Payment]) -> set[str]:
    return rule_flags(payments, TUNED_VELOCITY_LIMIT, tiny_limit=TINY_LIMIT)


# --- compact arrays -------------------------------------------------------------
# 550K payments as Python objects (strings, datetimes, dicts keyed by id) did
# not fit in this laptop's free memory. Here every field is one numpy array.

FIELDS = ("pid", "acct", "t", "amount", "channel", "country")


@dataclass
class Arrays:
    """One entry per payment, sorted by (event_time, payment_id): the row order
    train_model.build_features uses, so the models see the same rows in the
    same order as the reference path."""

    pid: np.ndarray  # str
    acct: np.ndarray  # int code into acct_names
    acct_names: np.ndarray  # sorted unique account ids
    t: np.ndarray  # event time, epoch seconds (float64)
    amount: np.ndarray  # dollars (float64)
    channel: np.ndarray  # str
    country: np.ndarray  # str

    def __len__(self) -> int:
        return len(self.pid)


def make_arrays(cols: dict[str, np.ndarray]) -> Arrays:
    order = np.lexsort((cols["pid"], cols["t"]))
    cols = {k: v[order] for k, v in cols.items()}
    names, codes = np.unique(cols["acct"], return_inverse=True)
    return Arrays(
        pid=cols["pid"],
        acct=codes.astype(np.int32),
        acct_names=names,
        t=cols["t"].astype(np.float64),
        amount=cols["amount"].astype(np.float64),
        channel=cols["channel"],
        country=cols["country"],
    )


def arrays_from_batches(batches: Iterable[list[tuple]]) -> Arrays:
    """Batches of (payment_id, account_id, epoch_seconds, amount, channel, country)."""
    parts: dict[str, list[np.ndarray]] = defaultdict(list)
    for batch in batches:
        for name, col in zip(FIELDS, zip(*batch, strict=True), strict=True):
            parts[name].append(np.array(col, dtype=np.float64 if name in ("t", "amount") else str))
    return make_arrays({k: np.concatenate(v) for k, v in parts.items()})


def arrays_from_records(records: Iterable[dict], batch: int = 50_000) -> Arrays:
    """From generator truth records (used by the tests; the DB path is load_arrays)."""

    def batches() -> Iterator[list[tuple]]:
        buf = []
        for r in records:
            ts = datetime.strptime(r["timestamp"], TS_FORMAT).replace(tzinfo=UTC).timestamp()
            buf.append(
                (r["payment_id"], r["account_id"], ts, float(r["amount"]), r.get("channel") or "", r.get("country") or "")
            )
            if len(buf) == batch:
                yield buf
                buf = []
        if buf:
            yield buf

    return arrays_from_batches(batches())


def load_arrays(conn, schema: str, batch: int = 50_000) -> Arrays:
    def batches() -> Iterator[list[tuple]]:
        with conn.cursor(name="eval_arrays") as cur:  # server-side: streamed in batches
            cur.itersize = batch
            cur.execute(
                "SELECT payment_id, account_id, extract(epoch FROM event_time)::float8, amount::float8, "
                f"coalesce(channel, ''), coalesce(country, '') FROM {schema}.payments_clean"
            )
            while rows := cur.fetchmany(batch):
                yield rows

    autocommit = conn.autocommit
    conn.autocommit = False  # a server-side cursor needs a transaction
    try:
        return arrays_from_batches(batches())
    finally:
        conn.rollback()  # read-only
        conn.autocommit = autocommit


@dataclass
class LabelArrays:
    """Labels aligned with an Arrays (same order)."""

    fraud: np.ndarray  # bool
    pattern: np.ndarray  # index into PATTERNS, -1 for legit (or an unknown pattern)
    incident: np.ndarray  # int code, -1 for legit


def load_label_arrays(path: Path, a: Arrays) -> LabelArrays:
    pids, fraud, pattern, incident = [], [], [], []
    code = {p: i for i, p in enumerate(PATTERNS)}
    incidents: dict[str, int] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            pids.append(r["payment_id"])
            is_fraud = r["is_fraud"] == "1"
            fraud.append(is_fraud)
            pattern.append(code.get(r["pattern"], -1) if is_fraud else -1)
            incident.append(incidents.setdefault(r["incident_id"], len(incidents)) if is_fraud else -1)
    pid_arr = np.array(pids)
    del pids
    order = np.argsort(pid_arr)
    pid_sorted = pid_arr[order]
    pos = np.minimum(np.searchsorted(pid_sorted, a.pid), len(pid_sorted) - 1)
    if not (pid_sorted[pos] == a.pid).all():
        raise ValueError("some payments in the table have no label")
    take = order[pos]
    return LabelArrays(
        np.array(fraud)[take], np.array(pattern, dtype=np.int8)[take], np.array(incident, dtype=np.int32)[take]
    )


def home_by_account(a: Arrays, home: dict[str, str]) -> np.ndarray:
    return np.array([home.get(str(name), "") for name in a.acct_names])


@dataclass
class AccountOrder:
    """Payments regrouped by account, each account's payments in (event_time,
    payment_id) order. `key` keeps accounts far apart, so a searchsorted on
    key - window never reaches into another account."""

    order: np.ndarray  # grouped row -> position in Arrays order
    key: np.ndarray
    start: np.ndarray  # grouped row -> grouped index where its account starts


def account_order(a: Arrays, max_window: float = 3600) -> AccountOrder:
    order = np.argsort(a.acct, kind="stable")  # Arrays order is (t, pid): stable sort keeps it per account
    acct = a.acct[order]
    t = a.t[order]
    t0 = t.min() if len(t) else 0.0
    span = (t.max() - t0) + 2 * max_window + 1 if len(t) else 1.0
    key = acct.astype(np.float64) * span + (t - t0)
    pos = np.arange(len(order))
    first = np.r_[True, acct[1:] != acct[:-1]] if len(order) else np.zeros(0, dtype=bool)
    start = np.maximum.accumulate(np.where(first, pos, 0)) if len(order) else pos
    return AccountOrder(order, key, start)


def earlier_in_window(g: AccountOrder, window: float) -> np.ndarray:
    """For each grouped row, how many of the account's earlier payments fall in
    [t - window, t] (same-second payments with a smaller payment_id count)."""
    return np.arange(len(g.key)) - np.searchsorted(g.key, g.key - window, side="left")


def rule_mask(
    a: Arrays,
    velocity_limit: int = VELOCITY_LIMIT,
    tiny_limit: int | None = None,
    amount_threshold: float = FLAG_THRESHOLD,
    window_seconds: int = VELOCITY_WINDOW_SECONDS,
    tiny_amount: float = TINY_AMOUNT,
    g: AccountOrder | None = None,
) -> np.ndarray:
    """rule_flags() on compact arrays: one boolean per payment, in Arrays order."""
    g = g or account_order(a)
    amount = a.amount[g.order]
    lo = np.searchsorted(g.key, g.key - window_seconds, side="left")
    in_window = np.arange(len(g.key)) - lo + 1  # this payment included
    hit = (amount > amount_threshold) | (in_window >= velocity_limit)
    if tiny_limit is not None:
        tiny = np.cumsum(amount < tiny_amount)
        before = np.where(lo > 0, tiny[np.maximum(lo - 1, 0)], 0)
        hit |= (tiny - before) >= tiny_limit
    out = np.zeros(len(a), dtype=bool)
    out[g.order] = hit
    return out


def compact_features(a: Arrays, home: np.ndarray, channels: list[str], g: AccountOrder | None = None) -> np.ndarray:
    """train_model.build_features() on compact arrays: same columns, rows in Arrays order."""
    g = g or account_order(a)
    names = feature_names(channels)
    col = {n: j for j, n in enumerate(names)}
    n = len(a)
    X = np.zeros((n, len(names)))
    X[:, col["amount"]] = a.amount
    X[:, col["hour"]] = (a.t // 3600) % 24
    for c in channels:
        X[:, col[f"channel_{c}"]] = a.channel == c
    X[:, col["foreign"]] = a.country != home[a.acct]

    k = np.arange(n) - g.start  # how many earlier payments the account has
    t = a.t[g.order]
    amount = a.amount[g.order]
    cents = np.rint(amount * 100).astype(np.int64)  # exact running totals
    total = np.cumsum(cents)
    prev_total = (total - cents - (total[g.start] - cents[g.start])) / 100.0
    avg = prev_total / np.maximum(k, 1)
    to_avg = np.ones(n)
    ok = (k > 0) & (avg > 0)
    to_avg[ok] = amount[ok] / avg[ok]
    gap = np.full(n, FIRST_GAP)
    later = k > 0
    gap[later] = t[later] - t[np.flatnonzero(later) - 1]

    grouped = {
        "n_prev_3m": earlier_in_window(g, 180),
        "n_prev_1h": earlier_in_window(g, 3600),
        "amount_to_avg": to_avg,
        "secs_since_prev": gap,
    }
    for name, values in grouped.items():
        X[g.order, col[name]] = values
    return X


# --- evaluation ---------------------------------------------------------------


def days_of(a: Arrays) -> np.ndarray:
    start = datetime(START.year, START.month, START.day, tzinfo=UTC).timestamp()
    return ((a.t - start) // 86400).astype(np.int32) + 1


def metrics(flag: np.ndarray, win: np.ndarray, lab: LabelArrays, amount: np.ndarray) -> dict:
    """train_model.evaluate() on boolean masks (every payment in the window is labelled)."""
    fraud = win & lab.fraud
    hit = flag & win
    caught = hit & lab.fraud
    n_fraud, n_hit, n_caught = int(fraud.sum()), int(hit.sum()), int(caught.sum())
    return {
        "fraud": n_fraud,
        "flagged": n_hit,
        "caught": n_caught,
        "false_alarms": n_hit - n_caught,
        "recall": _pct(n_caught, n_fraud),
        "precision": _pct(n_caught, n_hit),
        "fraud_dollars": float(amount[fraud].sum()),
        "dollars_blocked": float(amount[caught].sum()),
        "by_pattern": {
            name: (int((caught & (lab.pattern == i)).sum()), int((fraud & (lab.pattern == i)).sum()))
            for i, name in enumerate(PATTERNS)
        },
        "legit": int(win.sum()) - n_fraud,
    }


def pattern_counts(win: np.ndarray, lab: LabelArrays) -> dict[str, dict[str, int]]:
    """Per fraud pattern: fraud payments and distinct incidents in the window."""
    out = {}
    for i, name in enumerate(PATTERNS):
        m = win & lab.fraud & (lab.pattern == i)
        out[name] = {"payments": int(m.sum()), "incidents": len(np.unique(lab.incident[m]))}
    return out


def learning_days(fraction: float, train: tuple[int, int] = SPLIT["train"]) -> tuple[int, int]:
    """The most recent `fraction` of the training days (always ending at the
    last training day, so the gap to validation stays the same)."""
    first, last = train
    n = max(1, ceil((last - first + 1) * fraction))
    return last - n + 1, last


def evaluate_compact(
    a: Arrays, lab: LabelArrays, home: np.ndarray, split: dict = SPLIT, log=lambda _msg: None
) -> dict:
    days = days_of(a)
    win = {w: (days >= lo) & (days <= hi) for w, (lo, hi) in split.items()}

    t0 = time.perf_counter()
    g = account_order(a)
    rules = {
        "fixed rules": rule_mask(a, VELOCITY_LIMIT, g=g),
        "hand-tuned rules": rule_mask(a, TUNED_VELOCITY_LIMIT, tiny_limit=TINY_LIMIT, g=g),
    }
    log(f"  rules: {time.perf_counter() - t0:.1f}s")
    channels = sorted(str(c) for c in np.unique(a.channel))
    t0 = time.perf_counter()
    X = compact_features(a, home, channels, g)
    del g
    log(f"  features: {X.shape[0]:,} x {X.shape[1]} in {time.perf_counter() - t0:.1f}s")
    y = lab.fraud

    result = {
        "payments": len(a),
        "split": split,
        "counts": {w: pattern_counts(m, lab) for w, m in win.items()},
        "window_sizes": {w: {"payments": int(m.sum()), "fraud": int((m & y).sum())} for w, m in win.items()},
    }
    budget = metrics(rules["fixed rules"], win["validation"], lab, a.amount)["false_alarms"]
    result["budget"] = budget
    result["setups"] = {
        name: {w: metrics(flag, m, lab, a.amount) for w, m in win.items()} for name, flag in rules.items()
    }

    val = win["validation"]
    result["models"], result["learning_curve"] = {}, {}
    for name in MODELS:
        curve = []
        for fraction in LEARNING_FRACTIONS:
            first, last = learning_days(fraction, split["train"])
            rows = np.flatnonzero((days >= first) & (days <= last))
            entry = {
                "fraction": fraction,
                "days": [first, last],
                "train_payments": len(rows),
                "train_fraud": int(y[rows].sum()),
                "validation_ap": None,
                "validation_recall": None,
                "validation_precision": None,
                "validation_false_alarms": None,
            }
            if len(np.unique(y[rows])) < 2:  # nothing to learn from
                curve.append(entry)
                continue
            t0 = time.perf_counter()
            model = make_models(channels)[name]
            model.fit(X[rows], y[rows])
            scores = model.predict_proba(X)[:, 1]
            threshold = threshold_for_false_alarms(scores[val], y[val], budget)
            flag = scores >= threshold
            vm = metrics(flag, val, lab, a.amount)
            ap = float(average_precision_score(y[val], scores[val])) if y[val].any() else None
            entry.update(
                validation_ap=ap,
                validation_recall=vm["recall"],
                validation_precision=vm["precision"],
                validation_false_alarms=vm["false_alarms"],
            )
            curve.append(entry)
            shown = "n/a" if ap is None else f"{ap:.3f}"
            log(f"  {name} on days {first}-{last}: validation AP {shown} ({time.perf_counter() - t0:.1f}s)")
            if fraction == 1.0:
                result["models"][name] = {"threshold": threshold, "validation_ap": ap}
                result["setups"][name] = {w: metrics(flag, m, lab, a.amount) for w, m in win.items()}
            del model, scores, flag
        result["learning_curve"][name] = curve
    return result


def evaluate_seed(seed: int, data_dir: Path, log=print) -> dict:
    conn = db_connect()
    try:
        a = load_arrays(conn, schema_for(seed))
    finally:
        conn.close()
    log(f"seed {seed}: {len(a):,} payments loaded from {schema_for(seed)}.payments_clean")
    lab = load_label_arrays(data_dir / "labels.csv", a)
    home = home_by_account(a, load_home(data_dir / "accounts.csv"))
    return {"seed": seed, **evaluate_compact(a, lab, home, SPLIT, log)}


# --- SQL speed ----------------------------------------------------------------


def velocity_sql(schema: str, accounts_filter: bool = False) -> str:
    """The clean step's velocity rule (sql/clean.sql) as a read-only count."""
    return f"""
        SELECT count(*)
        FROM {schema}.payments_clean AS t
        WHERE {"t.account_id = ANY(%(accounts)s) AND" if accounts_filter else ""}
          (
              SELECT count(*)
              FROM {schema}.payments_clean AS w
              WHERE w.account_id = t.account_id
                AND w.event_time >= t.event_time - make_interval(secs => %(window)s)
                AND (w.event_time, w.payment_id) <= (t.event_time, t.payment_id)
          ) >= %(limit)s
    """


def daily_summary_sql(schema: str) -> str:
    """One account's payments per day over a date range (an account page)."""
    return f"""
        SELECT date_trunc('day', event_time) AS day, count(*) AS payments,
               sum(amount) AS total, max(amount) AS largest
        FROM {schema}.payments_clean
        WHERE account_id = %(account)s AND event_time >= %(start)s AND event_time < %(end)s
        GROUP BY 1
        ORDER BY 1
    """


def plan_uses_index(plan: str, index: str) -> bool:
    return index in plan and ("Index Scan" in plan or "Index Only Scan" in plan or "Bitmap Index Scan" in plan)


def _timed(cur, sql: str, params: dict, timeout_s: float) -> tuple[float | None, object]:
    """Seconds to run and fetch, or (None, None) if it hit the timeout."""
    cur.execute(f"SET statement_timeout = {int(timeout_s * 1000)}")
    t = time.perf_counter()
    try:
        cur.execute(sql, params)
        rows = cur.fetchall()
    except QueryCanceled:
        return None, None
    finally:
        cur.execute("SET statement_timeout = 0")  # autocommit: a cancelled query leaves no failed transaction
    return time.perf_counter() - t, rows


def _median_time(cur, sql: str, params: dict, runs: int, timeout_s: float) -> tuple[float | None, object]:
    times, rows = [], None
    for _ in range(runs):
        seconds, rows = _timed(cur, sql, params, timeout_s)
        if seconds is None:
            return None, None
        times.append(seconds)
    return statistics.median(times), rows


def sql_speed(seed: int, timeout_s: float, subset_accounts: int, log=print) -> dict:
    schema = schema_for(seed)
    conn = db_connect()  # autocommit
    out: dict = {"seed": seed, "schema": schema, "timeout_s": timeout_s, "index": f"{INDEX_NAME} (account_id, event_time)"}
    try:
        with conn.cursor() as cur:
            cur.execute(f"DROP INDEX IF EXISTS {schema}.{INDEX_NAME}")
            cur.execute(f"ANALYZE {schema}.payments_clean")
            cur.execute(f"SELECT count(*) FROM {schema}.payments_clean")
            out["rows"] = cur.fetchone()[0]
            cur.execute(f"SELECT account_id FROM {schema}.payments_clean GROUP BY 1 ORDER BY 1 LIMIT %s",
                        (subset_accounts,))  # fmt: skip
            accounts = [r[0] for r in cur.fetchall()]
            cur.execute(f"SELECT account_id FROM {schema}.payments_clean GROUP BY 1 ORDER BY count(*) DESC LIMIT 1")
            busiest = cur.fetchone()[0]
            cur.execute(f"SELECT count(*) FROM {schema}.payments_clean WHERE account_id = ANY(%s)", (accounts,))
            out["subset"] = {"accounts": len(accounts), "payments": cur.fetchone()[0]}

            vel = {"window": VELOCITY_WINDOW_SECONDS, "limit": VELOCITY_LIMIT}
            vel_subset = {**vel, "accounts": accounts}
            summary = {"account": busiest, "start": "2026-10-01T00:00:00Z", "end": "2026-10-31T00:00:00Z"}
            out["summary_account"] = busiest
            queries = {
                "velocity (all accounts)": (velocity_sql(schema), vel, 1),
                f"velocity ({len(accounts)} accounts)": (velocity_sql(schema, True), vel_subset, 1),
                "account daily summary (30 days, 1 account)": (daily_summary_sql(schema), summary, 5),
            }

            def run_all(label: str) -> dict:
                res = {}
                for name, (sql, params, runs) in queries.items():
                    seconds, rows = _median_time(cur, sql, params, runs, timeout_s)
                    res[name] = {"seconds": seconds, "runs": runs,
                                 "result": rows[0][0] if rows and name.startswith("velocity") else
                                 (len(rows) if rows is not None else None)}  # fmt: skip
                    shown = f"{seconds:.4f}s" if seconds is not None else f"cancelled after {timeout_s:.0f}s"
                    log(f"  {label:<6} {name}: {shown}")
                return res

            cur.execute(f"EXPLAIN {velocity_sql(schema, True)}", vel_subset)
            out["plan_before"] = {"velocity": "\n".join(r[0] for r in cur.fetchall())}
            cur.execute(f"EXPLAIN {daily_summary_sql(schema)}", summary)
            out["plan_before"]["summary"] = "\n".join(r[0] for r in cur.fetchall())
            out["before"] = run_all("before")

            t = time.perf_counter()
            cur.execute(f"CREATE INDEX {INDEX_NAME} ON {schema}.payments_clean (account_id, event_time)")
            cur.execute(f"ANALYZE {schema}.payments_clean")
            out["index_build_seconds"] = time.perf_counter() - t
            log(f"  index built in {out['index_build_seconds']:.1f}s")
            out["after"] = run_all("after")

            out["explain_analyze"] = {}
            for key, sql, params in [
                ("velocity", velocity_sql(schema, True), vel_subset),
                ("summary", daily_summary_sql(schema), summary),
            ]:
                cur.execute(f"EXPLAIN (ANALYZE, BUFFERS) {sql}", params)
                plan = "\n".join(r[0] for r in cur.fetchall())
                out["explain_analyze"][key] = {"plan": plan, "uses_index": plan_uses_index(plan, INDEX_NAME)}
    finally:
        conn.close()
    return out


# --- report -------------------------------------------------------------------


def spread(values: list[float | None]) -> tuple[float, float, float] | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return statistics.mean(vals), min(vals), max(vals)


def fmt_spread(values: list[float | None], pct: bool = True) -> str:
    s = spread(values)
    if s is None:
        return "n/a"
    mean, lo, hi = s
    if pct:
        return f"{mean:.1f}% ({lo:.1f}-{hi:.1f})"
    return f"{mean:,.1f} ({lo:,.0f}-{hi:,.0f})"


def pattern_recall(s: dict, pattern: str) -> float | None:
    caught, total = s["by_pattern"][pattern]
    return _pct(caught, total)


def counts_table(r: dict) -> str:
    lines = [
        "| Window (days) | Payments | " + " | ".join(PATTERNS) + " |",
        "|---|---:|" + "---:|" * len(PATTERNS),
    ]
    for w, (a, b) in r["split"].items():
        c = r["counts"][w]
        cells = [f"{c[p]['payments']:,} pay / {c[p]['incidents']} inc" for p in PATTERNS]
        lines.append(f"| {w} ({a}-{b}) | {r['window_sizes'][w]['payments']:,} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def overfit_table(r: dict) -> str:
    lines = [
        "| Setup | Window | Recall | Precision | False alarms | per 1,000 legit |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name in SETUPS:
        for w in SPLIT:
            s = r["setups"][name][w]
            lines.append(
                f"| {name} | {w} | {_fmt_pct(s['recall'])} ({s['caught']}/{s['fraud']}) | "
                f"{_fmt_pct(s['precision'])} | {s['false_alarms']:,} | {1000 * s['false_alarms'] / s['legit']:.2f} |"
            )
    return "\n".join(lines)


def learning_table(runs: list[dict]) -> str:
    lines = [
        "| Model | Training days | " + " | ".join(f"seed {r['seed']}" for r in runs) + " | mean |",
        "|---|---|" + "---:|" * (len(runs) + 1),
    ]
    for name in MODELS:
        for k, fraction in enumerate(LEARNING_FRACTIONS):
            first, last = runs[0]["learning_curve"][name][k]["days"]
            aps = [r["learning_curve"][name][k]["validation_ap"] for r in runs]
            recalls = [r["learning_curve"][name][k]["validation_recall"] for r in runs]
            def ap_text(a: float | None) -> str:
                return "n/a" if a is None else f"{a:.3f}"

            cells = [f"AP {ap_text(a)} / recall {_fmt_pct(rc)}" for a, rc in zip(aps, recalls, strict=True)]
            mean_ap, mean_recall = spread(aps), spread(recalls)
            lines.append(
                f"| {name} | {fraction:.0%} (days {first}-{last}) | " + " | ".join(cells)
                + f" | AP {ap_text(mean_ap and mean_ap[0])} / recall {_fmt_pct(mean_recall and mean_recall[0])} |"
            )  # fmt: skip
    return "\n".join(lines)


def final_table(runs: list[dict]) -> str:
    head = ["Setup", "Recall", "Precision", "False alarms", *(f"Recall {p}" for p in PATTERNS)]
    lines = ["| " + " | ".join(head) + " |", "|---|" + "---:|" * (len(head) - 1)]
    for name in SETUPS:
        tests = [r["setups"][name]["test"] for r in runs]
        cells = [
            fmt_spread([s["recall"] for s in tests]),
            fmt_spread([s["precision"] for s in tests]),
            fmt_spread([s["false_alarms"] for s in tests], pct=False),
            *(fmt_spread([pattern_recall(s, p) for s in tests]) for p in PATTERNS),
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def speed_table(sp: dict) -> str:
    lines = ["| Query | Before index (s) | After index (s) |", "|---|---:|---:|"]
    for name, before in sp["before"].items():
        after = sp["after"][name]

        def show(m: dict) -> str:
            if m["seconds"] is None:
                return f"did not finish in {sp['timeout_s']:.0f}"
            return f"{m['seconds']:.4f}" + (f" (median of {m['runs']})" if m["runs"] > 1 else "")

        lines.append(f"| {name} | {show(before)} | {show(after)} |")
    return "\n".join(lines)


def report(seeds: list[int]) -> str:
    runs = [json.loads((OUT_DIR / f"seed{s}.json").read_text()) for s in seeds]
    speed_path = OUT_DIR / "sql_speed.json"
    sp = json.loads(speed_path.read_text()) if speed_path.exists() else None
    a = runs[0]["split"]
    parts = [
        "# Larger evaluation",
        "",
        f"Seeds {', '.join(str(r['seed']) for r in runs)}: 5,000 accounts, 60 days each "
        f"({', '.join(f'{r['payments']:,}' for r in runs)} payments). Split by event day: "
        + ", ".join(f"{w} {x}-{y}" for w, (x, y) in a.items())
        + ". Data bulk-loaded from the generator's clean file, so nothing is quarantined.",
        "",
        (f"- **fixed rules**: amount > ${FLAG_THRESHOLD:,.0f}, or the {VELOCITY_LIMIT}th payment in "
        f"{VELOCITY_WINDOW_SECONDS} s (what the pipeline runs)."),
        (f"- **hand-tuned rules**: amount > ${FLAG_THRESHOLD:,.0f}, or the {TUNED_VELOCITY_LIMIT}th payment in "
        f"{VELOCITY_WINDOW_SECONDS} s, or {TINY_LIMIT}+ payments under ${TINY_AMOUNT:.0f} in "
        f"{VELOCITY_WINDOW_SECONDS} s."),
        "- **models**: threshold set on validation so their validation false alarms equal the fixed rules' "
        "validation false alarms (" + ", ".join(f"seed {r['seed']}: {r['budget']}" for r in runs) + ").",
        ("- **training data**: every payment in the training days, fraud and legit (no sampling). "
        "Validation and test are scored on every payment in their windows."),
        "- **memory**: payments held as compact numpy arrays; peak per seed "
        + ", ".join(f"seed {r['seed']}: {r['peak_memory_mb']:,.0f} MB" for r in runs if r.get("peak_memory_mb"))
        + " (process peak working set).",
        "",
        "## Test window (days 47-60): mean (min-max) across seeds",
        "",
        final_table(runs),
        "",
        "## Fraud per pattern in each window (payments / incidents)",
        "",
    ]
    for r in runs:
        parts += [f"Seed {r['seed']}", "", counts_table(r), ""]
    parts += ["## Train vs validation vs test at the same threshold", ""]
    for r in runs:
        parts += [f"Seed {r['seed']}", "", overfit_table(r), ""]
    parts += [
        "## Learning curve (validation average precision / recall at the false-alarm budget)",
        "",
        "Trained on the most recent 25%, 50% and 100% of the training days.",
        "",
        learning_table(runs),
        "",
    ]
    if sp:
        parts += [
            f"## SQL speed (seed {sp['seed']}, {sp['rows']:,} rows)",
            "",
            (f"Index {sp['index']}, built in {sp['index_build_seconds']:.1f} s. Subset: the first "
            f"{sp['subset']['accounts']} accounts ({sp['subset']['payments']:,} payments) as the outer rows; "
            f"the summary is for the busiest account ({sp['summary_account']})."),
            "",
            speed_table(sp),
            "",
            "EXPLAIN ANALYZE after the index: "
            + ", ".join(f"{k} uses index: {v['uses_index']}" for k, v in sp["explain_analyze"].items()),
            "",
        ]
    return "\n".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("load", "evaluate", "sql-speed"):
        s = sub.add_parser(name)
        s.add_argument("--seed", type=int, required=True, choices=sorted(DATA_DIRS))
    sub.choices["sql-speed"].add_argument("--timeout", type=float, default=300, help="seconds per query")
    sub.choices["sql-speed"].add_argument("--subset-accounts", type=int, default=10)
    rep = sub.add_parser("report")
    rep.add_argument("--seeds", default="1,2,3")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.cmd == "load":
        conn = db_connect()
        conn.autocommit = False
        try:
            t = time.perf_counter()
            n = load_schema(conn, schema_for(args.seed), DATA_DIRS[args.seed] / "payments_truth.jsonl", print)
            print(f"loaded {n:,} rows into {schema_for(args.seed)}.payments_clean in {time.perf_counter() - t:.1f}s")
        finally:
            conn.close()
    elif args.cmd == "evaluate":
        result = evaluate_seed(args.seed, DATA_DIRS[args.seed])
        result["peak_memory_mb"] = peak_memory_mb()
        if result["peak_memory_mb"] is not None:
            print(f"peak memory (working set): {result['peak_memory_mb']:,.0f} MB")
        (OUT_DIR / f"seed{args.seed}.json").write_text(json.dumps(result, indent=1))
        print(f"wrote {(OUT_DIR / f'seed{args.seed}.json').relative_to(ROOT)}")
    elif args.cmd == "sql-speed":
        result = sql_speed(args.seed, args.timeout, args.subset_accounts)
        (OUT_DIR / "sql_speed.json").write_text(json.dumps(result, indent=1))
        print(speed_table(result))
        for k, v in result["explain_analyze"].items():
            print(f"\nEXPLAIN ANALYZE {k} (uses {INDEX_NAME}: {v['uses_index']}):\n{v['plan']}")
    else:
        text = report([int(s) for s in args.seeds.split(",")])
        out = RESULTS / "larger_eval.md"
        out.write_text(text, encoding="utf-8")
        print(text)
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
