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
python -m scripts.train_model   # models vs. rules on days 26-30 -> results/model_comparison.md
```

`scripts/train_model.py` (needs `pip install -r requirements-dev.txt`) builds features from
`payments_clean` using only each account's earlier payments, trains logistic regression and gradient
boosting on days 1-21, picks each threshold on days 22-25 so its false alarms per legit payment match
the rules' on the test window, and scores days 26-30. The model with the best validation average
precision is saved to `models/fraud_model.joblib` (git-ignored; re-run the script to recreate it).

### Larger evaluation (3 seeds, ~550K payments each)

```bash
python -m generator.generate --out data/large    --accounts 5000 --days 60 --seed 1   # also large_s2/s3, seeds 2/3
python -m scripts.larger_eval load --seed 1        # COPY into schema eval_s1 (dropped and recreated)
python -m scripts.larger_eval evaluate --seed 1    # one process per seed -> results/larger_eval/seed1.json
python -m scripts.larger_eval sql-speed --seed 1   # velocity + daily summary, before/after an index
python -m scripts.larger_eval report               # all seeds -> results/larger_eval.md
```

Each seed gets its own schema (`eval_s1`..`eval_s3`), bulk-loaded with COPY from the generator's clean
file, so the app's tables are untouched. Split by day: train 1-36, validation 37-46, test 47-60. It
compares the fixed rules, hand-tuned rules (velocity limit 5, plus 5+ payments under $5 in 3 minutes),
logistic regression and gradient boosting, with each model's threshold set on validation to match the
fixed rules' validation false alarms. Payments are held as numpy arrays (about 400 MB peak per seed);
a test checks the array version gives the same recall, precision and false alarms as the
one-object-per-payment version.

### Load test and freshness

Needs the whole stack (`docker compose up -d`) and `pip install -r requirements-dev.txt`.

```bash
python -m scripts.report reset       # empty raw_payments, payments_clean, payments_quarantine
locust -f scripts/locustfile.py --headless --host http://localhost:8000 -u 50 -r 10 -t 2m --csv results/locust
python -m scripts.report summary     # req/s, p50/p95/p99, failures; then sent vs stored, lost, duplicated
python -m scripts.freshness          # 20 single payments: seconds from POST until in raw_payments
```

Every Locust request is a record sampled from `data/synthetic/payments_raw.jsonl` with a fresh
`payment_id`; users send back-to-back with no wait. `summary` waits for the consumer to drain before
counting. The API replies after handing the payment to Kafka, so its latency measures the hand-off,
not saving to Postgres. Numbers depend on the machine and Docker's memory limit; both reports print it.

## Running tests

```bash
pip install -r requirements-dev.txt   # requirements.txt + scikit-learn for the model tests
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
3. Validates the Kubernetes manifests in `k8s/` with kubeconform (no cluster needed)
4. Runs the pytest suite
5. Builds the Docker image (proves it packages correctly)

A second job, `k8s-smoke`, runs the pipeline on Kubernetes (next section).

## Kubernetes (CI only)

The cluster is a **throwaway one-node [kind](https://kind.sigs.k8s.io/) cluster on a GitHub
Actions runner**, created and deleted by the `k8s-smoke` job on every run. Nothing here is a
production setup: Kafka and Postgres are single copies with small memory settings and no
persistent disk. It does not run on the laptop (not enough memory).

```
 GitHub runner
 ┌───────────────────────────── kind cluster (1 node) ─────────────────────────────┐
 │                                                                                  │
 │  Job create-topics ──(once: "transactions", 3 partitions)──▶ Deployment kafka    │
 │                                                                  ▲     │         │
 │  Deployment api (1) ──── POST /transactions ─────────────────────┘     │         │
 │        ▲                                                               ▼         │
 │        │                                     Deployment consumer (1, scaled 0-3) │
 │        │                                                               │         │
 │        │                                        Deployment postgres ◀──┘         │
 │        │                                         (raw_payments)  ▲               │
 └────────┼─────────────────────────────────────────────────────────┼───────────────┘
   localhost:8080 (NodePort 30080)                      localhost:5432 (NodePort 30432)
          └──────────────── python -m scripts.k8s_smoke ─────────────┘
```

| File | What it is |
|---|---|
| `k8s/configmap.yaml` | Kafka and Postgres addresses, topic name and partitions, fraud thresholds |
| `k8s/kafka.yaml`, `k8s/postgres.yaml` | Single-copy Kafka (KRaft, 384 MB heap) and Postgres, CI only |
| `k8s/topic-job.yaml` | One-off Job that creates the topic with 3 partitions |
| `k8s/api.yaml` | API Deployment (HTTP probes, limits) and its Service |
| `k8s/consumer.yaml` | Consumer Deployment; probes check a heartbeat file (`app/healthcheck.py`) |
| `.github/kind-config.yaml` | The kind cluster, with the two NodePorts mapped to the runner |

The `k8s-smoke` job creates the cluster, builds and loads the image, starts Kafka and Postgres,
runs the topic Job, starts the API and the consumer, then runs `python -m scripts.k8s_smoke`:

1. **Smoke**: 500 payments with 1 consumer; prints sent / stored / lost / duplicated.
2. **Backlog**: 10,000 payments sent while the consumer is scaled to 0, then the consumer is
   scaled to 1 (and, in a second run, to 3); prints the seconds until all are stored. Tables are
   emptied between runs.
3. **Recovery**: 3 consumers, 10,000 payments, one consumer pod deleted once a third are stored;
   prints how long until the backlog is back where it was, plus lost / duplicated.

The job fails if any accepted payment is lost. On failure it prints `kubectl describe` and logs
for every pod that is not ready. The summary table also appears on the run's summary page.

## Known gotcha

`kafka-python` (2.0.2) is broken on Python 3.12 — it throws
`ModuleNotFoundError: No module named 'kafka.vendor.six.moves'` on import. This project
uses the maintained fork `kafka-python-ng` instead, which is a drop-in replacement (same
`from kafka import ...` API) and works fine on 3.12.
