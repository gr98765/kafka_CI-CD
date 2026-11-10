# Fintech Transaction Pipeline

An event-driven transaction processing service. Incoming transactions are published to
Kafka, picked up by a background consumer, run through a simple risk rule, and stored in
Postgres. A small dashboard shows the live results. The whole pipeline is tested and
built automatically on every push via GitHub Actions.

## Architecture

```
POST /transactions  →  Kafka topic "transactions"  →  background consumer
                                                              │
                                                    applies flag/clear rule
                                                              │
                                                              ▼
                                                          PostgreSQL
                                                              │
                                                              ▼
                                              GET /transactions  →  dashboard (/)
```

- **FastAPI** (`app/main.py`) exposes the REST API: submit a transaction, list transactions.
- **Kafka** decouples "a transaction happened" from "processing that transaction" — the API
  responds immediately (202 Accepted) without waiting for the DB write.
- **Postgres** stores the processed transactions.
- **Risk rule**: any transaction over $1000 is marked `flagged`, otherwise `cleared`.
  (Intentionally simple and rule-based — no ML claims here, just a clear, explainable
  business rule, which is what most real fraud-review pipelines start with.)
- **Dashboard** (`static/index.html`) polls `GET /transactions` every 3 seconds.

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

That amount is over $1000, so refresh the dashboard and it should show up as `flagged`.

## Running tests

```bash
pip install -r requirements.txt
DISABLE_KAFKA_CONSUMER=1 pytest -v
```

Tests use mocks for Kafka and Postgres, so you don't need the full stack running just to
test the API logic.

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
