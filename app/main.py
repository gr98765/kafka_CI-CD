import json
import os
import threading
import time
from contextlib import asynccontextmanager

import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from kafka import KafkaConsumer, KafkaProducer
from pydantic import BaseModel, Field

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "fintech")
DB_PASSWORD = os.getenv("DB_PASSWORD", "fintech")
DB_NAME = os.getenv("DB_NAME", "transactions_db")
TOPIC_NAME = "transactions"
FLAG_THRESHOLD = 1000.00


class Transaction(BaseModel):
    account_id: str = Field(..., examples=["acct_123"])
    merchant: str = Field(..., examples=["Amazon"])
    amount: float = Field(..., gt=0, examples=[49.99])
    location: str = Field(default="unknown", examples=["Tucson, AZ"])


def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST, user=DB_USER, password=DB_PASSWORD, dbname=DB_NAME
    )


def get_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )


def classify(amount: float) -> str:
    """Simple rule-based risk check. This is the whole 'business logic'."""
    return "flagged" if amount > FLAG_THRESHOLD else "cleared"


def consume_loop():
    """Runs in a background thread: reads transactions off Kafka, applies
    the flag/clear rule, and writes the result into Postgres."""
    consumer = None
    while consumer is None:
        try:
            consumer = KafkaConsumer(
                TOPIC_NAME,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                auto_offset_reset="earliest",
                group_id="transaction-consumer-group",
            )
        except Exception:  # noqa: BLE001 - retry until Kafka is up
            time.sleep(3)  # Kafka may still be starting up

    for message in consumer:
        txn = message.value
        status = classify(txn["amount"])
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO transactions (account_id, merchant, amount, location, status)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        txn["account_id"],
                        txn["merchant"],
                        txn["amount"],
                        txn.get("location", "unknown"),
                        status,
                    ),
                )
                conn.commit()
        finally:
            conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Skipped during tests (see tests/conftest.py) so pytest doesn't spend
    # time retrying a Kafka connection that isn't there.
    if os.getenv("DISABLE_KAFKA_CONSUMER") != "1":
        thread = threading.Thread(target=consume_loop, daemon=True)
        thread.start()
    yield


app = FastAPI(title="Fintech Transaction Pipeline", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def dashboard():
    return FileResponse("static/index.html")


@app.post("/transactions", status_code=202)
def create_transaction(txn: Transaction):
    """Publishes a transaction event to Kafka. Does NOT write to the DB
    directly -- that happens asynchronously via the consumer."""
    try:
        producer = get_producer()
        producer.send(TOPIC_NAME, value=txn.model_dump())
        producer.flush()
    except Exception as exc:  # noqa: BLE001 - any producer failure -> 503
        raise HTTPException(status_code=503, detail=f"Kafka unavailable: {exc}")
    return {"status": "submitted", "transaction": txn.model_dump()}


@app.get("/transactions")
def list_transactions(status: str | None = None):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if status:
                cur.execute(
                    "SELECT id, account_id, merchant, amount, location, status, created_at "
                    "FROM transactions WHERE status = %s ORDER BY created_at DESC LIMIT 100",
                    (status,),
                )
            else:
                cur.execute(
                    "SELECT id, account_id, merchant, amount, location, status, created_at "
                    "FROM transactions ORDER BY created_at DESC LIMIT 100"
                )
            rows = cur.fetchall()
    finally:
        conn.close()

    columns = ["id", "account_id", "merchant", "amount", "location", "status", "created_at"]
    return [dict(zip(columns, row)) for row in rows]


@app.get("/health")
def health():
    return {"status": "ok"}
