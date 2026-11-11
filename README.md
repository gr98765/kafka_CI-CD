# Fintech Transaction Pipeline

An event-driven transaction processing service. Incoming transactions are published to
Kafka, stored exactly as received by a background consumer, then cleaned (fixed,
de-duplicated or quarantined) and run through two fraud rules in Postgres. A small dashboard shows the live results. The whole pipeline is tested and
built automatically on every push via GitHub Actions.

## Architecture

```
POST /transactions  →  Kafka topic "transactions"  →  consumer  →  raw_payments (as received)
                                                                         │
                                                         python -m app.clean (sql/clean.sql)
                                                                ┌────────┴─────────┐
                                                                ▼                  ▼
                                                  payments_clean + flags   payments_quarantine
                                                                │              (with reason)
                                                                ▼
                                                GET /transactions  →  dashboard (/)
```

- **FastAPI** (`app/main.py`) accepts **any JSON object** on `POST /transactions` and publishes it
  to Kafka as sent (202). It only rejects a body that is not a JSON object at all (422).
  Amount, timestamp and ID problems are the clean step's job, not the API's.
- **Raw layer**: the consumer stores every message in `raw_payments` (payload as JSONB, arrival
  time, Kafka partition and offset). A Kafka re-delivery (same partition and offset) is stored once;
  a message that is not a JSON object goes to the `payments.dead-letter` topic.
- **Clean layer** (`sql/clean.sql`, run by `python -m app.clean`) rebuilds two tables from raw:
  - `payments_clean`: amounts like `"$12.50"` / `"12.50"` fixed; ISO, epoch and
    `YYYY-MM-DD HH:MM:SS` timestamps normalized to UTC; `account_id` trimmed and uppercased;
    currency uppercased; duplicate `payment_id`s removed (first arrival kept).
  - `payments_quarantine`: records that cannot be fixed, with a `reason`: `missing_payment_id`,
    `missing_account_id`, `bad_amount`, `non_positive_amount`, `currency_not_usd`, `bad_timestamp`.
    Nothing is silently dropped: every raw row is kept, quarantined, or a later copy of a kept
    `payment_id` (the step prints all three counts).
- **Fraud rules** run on `payments_clean` by event time: amount over $1000.00 (`reason = 'amount'`),
  and the 7th-or-later payment by one account within 180 seconds (`reason = 'velocity'`).
  Thresholds live in `app/main.py`. Everything is in US dollars; no currency conversion.
- **Dashboard** (`static/index.html`) polls `GET /transactions` (latest `payments_clean` rows)
  every 3 seconds, so new payments appear after the next clean step.

## Running locally

```bash
docker compose up --build
```

Postgres is published on host port **5433** (not the default 5432), because 5432 is often
already taken by a locally installed Postgres. Inside the compose network the services still
talk to `postgres:5432`; only connections from your machine (e.g. `psql -h localhost -p 5433`,
or the `scripts/` tools, which default to `DB_PORT=5433`) use the new port.

Then:
- Dashboard: http://localhost:8000
- API docs (Swagger): http://localhost:8000/docs
- Submit a test transaction:

```bash
curl -X POST http://localhost:8000/transactions \
  -H "Content-Type: application/json" \
  -d '{"account_id": "acct_123", "merchant": "Amazon", "amount": 1500, "location": "Tucson, AZ"}'
```

Then run the clean step (`docker compose exec consumer python -m app.clean`, or
`DB_PORT=5433 python -m app.clean` from the host). That amount is over $1000, so the dashboard
shows it as `flagged`.

Existing database volume? The init script only runs on a fresh volume; apply the new tables with
`docker exec -i kafka_ci-cd-postgres-1 psql -U fintech -d transactions_db < db-init-script.sql`
(every statement is idempotent).

## Measuring the pipeline

```bash
python -m scripts.replay        # POST data/synthetic/payments_raw.jsonl (mess included)
DB_PORT=5433 python -m app.clean
python -m scripts.clean_score   # cleaning vs. mess_log.csv, accounting, rules comparison
python -m scripts.score         # rules comparison + velocity thresholds -> results/rules_comparison.md
```

## Running tests

```bash
pip install -r requirements.txt
DISABLE_KAFKA_CONSUMER=1 pytest -v
```

Tests use mocks for Kafka. The SQL tests (clean step, raw layer) need a real Postgres: set
`TEST_DB_HOST` (and `TEST_DB_PORT=5433` against the compose database); each test runs in its own
throwaway schema. Without it they are skipped (CI sets `REQUIRE_DB_TESTS=1`, so they must run).
Lint with the same ruff as CI: `pip install ruff==0.17.4 && ruff check .`.

## Crash test

`python -m scripts.kill_test` replays a slice of the synthetic data, SIGKILLs the consumer at
several points (25%, 50%, 75% by default), restarts it, and prints raw rows stored at the kill,
messages re-delivered after the restart and ignored, and final sent / stored / lost / duplicated
in `raw_payments`.

It then does one more run with duplicate protection **off**, using the `DEDUP_DISABLED=1` test
switch on the consumer: plain `INSERT` (no `ON CONFLICT`) into `raw_payments_nodedup`, a copy of
the raw table without the (partition, offset) unique key (it gets a plain, non-unique index on
`payment_id` so the script's counts stay fast), to show how many duplicates re-delivery
would create. The script
sets and resets the switch itself (`DEDUP_DISABLED=1 docker compose up -d consumer`, then back to
`0`). It is for this experiment only: never set it in normal operation. The real `raw_payments`
table is not touched while it is on.

## CI/CD

`.github/workflows/ci.yml` runs on every push/PR to `main`:
1. Installs dependencies
2. Lints with `ruff`
3. Runs the pytest suite
4. Builds the Docker image (proves it packages correctly)

## Known gotcha

`kafka-python` (2.0.2) is broken on Python 3.12 — it throws
`ModuleNotFoundError: No module named 'kafka.vendor.six.moves'` on import. This project
uses the maintained fork `kafka-python-ng` instead, which is a drop-in replacement (same
`from kafka import ...` API) and works fine on 3.12.
