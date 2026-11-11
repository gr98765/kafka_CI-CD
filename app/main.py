import json
import logging
import os
import threading
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from typing import Annotated, Any

import psycopg2
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from kafka import KafkaConsumer, KafkaProducer

log = logging.getLogger("pipeline")

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "fintech")
DB_PASSWORD = os.getenv("DB_PASSWORD", "fintech")
DB_NAME = os.getenv("DB_NAME", "transactions_db")
TOPIC_NAME = "transactions"
DEAD_LETTER_TOPIC = "payments.dead-letter"
CONSUMER_GROUP = "transaction-consumer-group"

# Fraud rules. Everything is in US dollars; no currency conversion.
FLAG_THRESHOLD = 1000.00  # rule 1: amount > this -> flagged ('amount')
VELOCITY_LIMIT = 7  # rule 2: the 7th-or-later payment by one account...
VELOCITY_WINDOW_SECONDS = 180  # ...within this many seconds -> flagged ('velocity')

RETRY_SECONDS = 3

# These mean "the database is unreachable", not "this record is bad": the
# consumer retries them instead of dead-lettering perfectly good payments.
DB_CONNECTION_ERRORS = (psycopg2.OperationalError, psycopg2.InterfaceError)

COLUMNS = [
    "payment_id", "account_id", "merchant_id", "merchant", "amount", "currency",
    "country", "channel", "location", "event_time", "status", "reason", "created_at",
]  # fmt: skip


def _utcnow() -> datetime:
    return datetime.now(UTC)


# --- shared connections ------------------------------------------------------

_producer: KafkaProducer | None = None
_producer_lock = threading.Lock()
_api_conn = None
_api_conn_lock = threading.Lock()


def get_db_connection():
    return psycopg2.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, dbname=DB_NAME)


def get_producer() -> KafkaProducer:
    """One KafkaProducer for the whole process (it is thread-safe). Created on
    first use, so the API can start before Kafka is reachable; if creating it
    fails, the next call tries again."""
    global _producer
    with _producer_lock:
        if _producer is None:
            _producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks="all",
                retries=5,
            )
        return _producer


def query_api_db(sql: str, params: tuple = ()) -> list[tuple]:
    """Run a read query on the API's one shared connection.

    The API and the consumer each keep their own connection: the consumer's is
    inside a write transaction most of the time, and the API must not see (or
    commit) that half-finished work. The API's connection is autocommit and is
    used by one request at a time. If it breaks it is dropped, and the next
    request reconnects.
    """
    global _api_conn
    with _api_conn_lock:
        if _api_conn is None or _api_conn.closed:
            _api_conn = get_db_connection()
            _api_conn.autocommit = True
        try:
            with _api_conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall()
        except DB_CONNECTION_ERRORS:
            _close_quietly(_api_conn)
            _api_conn = None
            raise


def _close_quietly(conn) -> None:
    """Close a connection that may already be broken; there is nothing more
    to do if that fails."""
    with suppress(Exception):
        conn.close()


# --- fraud rules + storage ---------------------------------------------------


def classify(amount: float) -> str:
    """Rule 1 (amount), as a Python reference. Both rules run in sql/clean.sql
    on payments_clean, with these same thresholds (app/clean.py passes them)."""
    return "flagged" if amount > FLAG_THRESHOLD else "cleared"


# Experiment only (scripts/kill_test.py plain-INSERT run): with DEDUP_DISABLED=1
# the consumer uses a plain INSERT into raw_payments_nodedup, a copy of the raw
# table without the (partition, offset) unique key, to show what re-delivery
# protection prevents. Never set it in normal operation.
DEDUP_DISABLED = os.getenv("DEDUP_DISABLED") == "1"
RAW_TABLE = "raw_payments_nodedup" if DEDUP_DISABLED else "raw_payments"

# The raw layer: every record exactly as received, plus where it came from.
# A Kafka re-delivery has the same partition and offset, so it is stored once.
RAW_INSERT_SQL = f"""
INSERT INTO {RAW_TABLE} (payload, kafka_partition, kafka_offset)
VALUES (%(payload)s::jsonb, %(partition)s, %(offset)s)
{"" if DEDUP_DISABLED else "ON CONFLICT (kafka_partition, kafka_offset) DO NOTHING"}
RETURNING raw_id
"""


def parse_record(raw: bytes) -> dict:
    """The message body as a JSON object; ValueError (not JSON) or TypeError (not an object) otherwise."""
    record = json.loads(raw.decode("utf-8"))  # UnicodeDecodeError / JSONDecodeError are ValueErrors
    if not isinstance(record, dict):
        raise TypeError(f"expected a JSON object, got {type(record).__name__}")
    return record


def store_raw(conn, raw: bytes, partition: int, offset: int) -> str:
    """Store one message in the raw layer: 'stored', or 'duplicate' for a
    Kafka re-delivery of a message that is already there."""
    record = parse_record(raw)
    with conn.cursor() as cur:
        cur.execute(
            RAW_INSERT_SQL,
            {"payload": json.dumps(record, allow_nan=False), "partition": partition, "offset": offset},
        )
        if cur.fetchone() is None:
            conn.rollback()
            log.info("duplicate ignored: %s (%s/%s@%s)", record.get("payment_id"), TOPIC_NAME, partition, offset)
            return "duplicate"
    conn.commit()
    return "stored"


def send_to_dead_letter(raw: bytes, error: Exception, source: str) -> None:
    """Park a record we could not process, with the reason, for a human to
    look at. Waits for Kafka to confirm, so a record is never silently lost."""
    payload = {
        "raw": raw.decode("utf-8", errors="replace"),
        "error": f"{type(error).__name__}: {error}"[:2000],
        "source": source,
        "failed_at": _utcnow().isoformat(),
    }
    get_producer().send(DEAD_LETTER_TOPIC, value=payload).get(timeout=10)


def process_message(conn, raw: bytes, partition: int, offset: int) -> str:
    """Handle one Kafka message: 'stored', 'duplicate' or 'dead_letter'.

    Amount, timestamp or ID problems are not checked here: the record is stored
    raw and sql/clean.sql fixes or quarantines it. Only a message that is not a
    JSON object at all (or that Postgres refuses as JSONB) goes to the
    dead-letter topic, and processing carries on. A lost DB connection is
    re-raised instead, so the caller can reconnect and retry.
    """
    source = f"{TOPIC_NAME}/{partition}@{offset}"
    try:
        return store_raw(conn, raw, partition, offset)
    except DB_CONNECTION_ERRORS:
        raise
    except Exception as exc:  # noqa: BLE001 - any other failure means a bad record
        conn.rollback()  # leave the connection usable for the next record
        send_to_dead_letter(raw, exc, source)
        log.warning("dead-lettered %s: %s: %s", source, type(exc).__name__, exc)
        return "dead_letter"


# --- consumer thread -----------------------------------------------------------


def make_consumer(stop: threading.Event) -> KafkaConsumer | None:
    while not stop.is_set():
        try:
            return KafkaConsumer(
                TOPIC_NAME,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                auto_offset_reset="earliest",
                group_id=CONSUMER_GROUP,
                # Offsets are committed by hand, only after a batch is fully
                # stored or dead-lettered, so a crash re-delivers instead of
                # losing. Re-delivered messages are absorbed by ON CONFLICT.
                enable_auto_commit=False,
            )
        except Exception as exc:  # noqa: BLE001 - Kafka may still be starting up
            log.info("Kafka not reachable yet (%s); retrying", exc)
            stop.wait(RETRY_SECONDS)
    return None


def _process_with_retry(conn, msg, stop: threading.Event):
    """Process one message, retrying while the DB (or Kafka, for the dead-letter
    send) is down. Returns (conn, done); done is False only when stopping."""
    source = f"{msg.topic}/{msg.partition}@{msg.offset}"
    while not stop.is_set():
        try:
            if conn is None or conn.closed:
                conn = get_db_connection()
            outcome = process_message(conn, msg.value, msg.partition, msg.offset)
            log.debug("%s -> %s", source, outcome)
            return conn, True
        except Exception as exc:  # noqa: BLE001 - never drop a message; wait and retry
            log.warning("could not process %s (%s); retrying", source, exc)
            if conn is not None:
                _close_quietly(conn)
                conn = None
            stop.wait(RETRY_SECONDS)
    return conn, False


def consume_loop(stop: threading.Event) -> None:
    """Runs in a background thread: reads payments off Kafka and stores each one
    as received in raw_payments, using one DB connection."""
    consumer = make_consumer(stop)
    if consumer is None:
        return
    conn = None
    try:
        while not stop.is_set():
            try:
                batches = consumer.poll(timeout_ms=1000)
                done = True
                for messages in batches.values():
                    for msg in messages:
                        conn, done = _process_with_retry(conn, msg, stop)
                        if not done:
                            break
                    if not done:
                        break
                if batches and done:
                    consumer.commit()
            except Exception as exc:  # noqa: BLE001 - keep consuming; uncommitted messages come back
                log.warning("consumer loop error (%s); continuing", exc)
                stop.wait(RETRY_SECONDS)
    finally:
        consumer.close()
        if conn is not None:
            _close_quietly(conn)


# --- API -------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _producer, _api_conn
    stop = threading.Event()
    thread = None
    # Skipped during tests (see tests/conftest.py) so pytest doesn't spend
    # time retrying a Kafka connection that isn't there.
    if os.getenv("DISABLE_KAFKA_CONSUMER") != "1":
        thread = threading.Thread(target=consume_loop, args=(stop,), daemon=True)
        thread.start()
    yield
    stop.set()
    if thread is not None:
        thread.join(timeout=5)
    if _producer is not None:
        _producer.close(timeout=5)
        _producer = None
    if _api_conn is not None:
        _close_quietly(_api_conn)
        _api_conn = None


app = FastAPI(title="Fintech Transaction Pipeline", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def dashboard():
    return FileResponse("static/index.html")


@app.post("/transactions", status_code=202)
def create_transaction(record: Annotated[dict[str, Any], Body()]):
    """Publishes the record to Kafka, as sent, and waits for Kafka to confirm.

    Any JSON object is accepted: amount, timestamp and ID problems are fixed or
    quarantined later by the clean step (sql/clean.sql). Only a body that is not
    a JSON object at all is rejected (422). Does NOT write to the DB.
    """
    try:
        json.dumps(record, allow_nan=False)
    except ValueError:  # NaN / Infinity are not JSON
        raise HTTPException(status_code=422, detail="body is not valid JSON (NaN or Infinity)")
    account = record.get("account_id")
    # Keyed by the normalized account, so one account's payments stay in order on one partition.
    key = account.strip().upper().encode("utf-8") if isinstance(account, str) else None
    try:
        get_producer().send(TOPIC_NAME, key=key, value=record).get(timeout=10)
    except Exception as exc:  # noqa: BLE001 - any producer failure -> 503
        raise HTTPException(status_code=503, detail=f"Kafka unavailable: {exc}")
    return {"status": "submitted", "transaction": record}


@app.get("/transactions")
def list_transactions(status: str | None = None):
    """The latest cleaned payments (payments_clean, built by the clean step)."""
    select = f"SELECT {', '.join(COLUMNS)} FROM payments_clean"
    order = " ORDER BY event_time DESC, payment_id DESC LIMIT 100"
    try:
        if status:
            rows = query_api_db(select + " WHERE status = %s" + order, (status,))
        else:
            rows = query_api_db(select + order)
    except DB_CONNECTION_ERRORS as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")
    return [dict(zip(COLUMNS, row)) for row in rows]


@app.get("/health")
def health():
    return {"status": "ok"}
