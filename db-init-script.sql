-- Three layers. Everything is in US dollars; no currency conversion.
--   raw_payments        every Kafka message exactly as received (written by the consumer)
--   payments_clean      fixed, de-duplicated payments with the fraud flags (built by sql/clean.sql)
--   payments_quarantine records that cannot be fixed, with the reason (built by sql/clean.sql)
-- All statements are idempotent, so this file can be re-run on an existing database.

CREATE TABLE IF NOT EXISTS raw_payments (
    raw_id          BIGSERIAL PRIMARY KEY,
    payload         JSONB NOT NULL,                              -- the message body, as received
    received_at     TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),  -- when the consumer stored it
    kafka_partition INTEGER NOT NULL,
    kafka_offset    BIGINT NOT NULL,
    -- A Kafka re-delivery (same partition and offset) is the same record: stored once.
    UNIQUE (kafka_partition, kafka_offset)
);

CREATE TABLE IF NOT EXISTS payments_clean (
    payment_id  TEXT PRIMARY KEY,               -- first arrival wins; later copies are dropped as duplicates
    account_id  TEXT NOT NULL,                  -- trimmed, uppercased
    merchant_id TEXT,
    merchant    TEXT,                           -- display name (dashboard form); generator data uses merchant_id
    amount      NUMERIC(12, 2) NOT NULL,        -- US dollars, > 0
    currency    TEXT NOT NULL,                  -- always 'USD'
    country     TEXT,
    channel     TEXT,
    location    TEXT,
    event_time  TIMESTAMPTZ NOT NULL,           -- when the payment happened, UTC (rules use this)
    status      TEXT NOT NULL,                  -- 'flagged' or 'cleared'
    reason      TEXT,                           -- 'amount' or 'velocity' when flagged, else NULL
    created_at  TIMESTAMPTZ NOT NULL,           -- when the raw record arrived
    raw_id      BIGINT NOT NULL                 -- the raw_payments row it came from
);

-- The velocity rule counts one account's payments in a time window.
CREATE INDEX IF NOT EXISTS idx_payments_clean_account_time
    ON payments_clean (account_id, event_time, payment_id);

CREATE TABLE IF NOT EXISTS payments_quarantine (
    raw_id          BIGINT PRIMARY KEY,         -- the raw_payments row
    payment_id      TEXT,                       -- as received (may be missing)
    reason          TEXT NOT NULL,              -- see sql/clean.sql for the list
    payload         JSONB NOT NULL,
    received_at     TIMESTAMPTZ NOT NULL,
    kafka_partition INTEGER NOT NULL,
    kafka_offset    BIGINT NOT NULL
);
