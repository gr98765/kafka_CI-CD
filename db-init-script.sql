CREATE TABLE IF NOT EXISTS transactions (
    payment_id  TEXT PRIMARY KEY,              -- set by the sender; a redelivered payment hits ON CONFLICT
    account_id  VARCHAR(50) NOT NULL,
    merchant_id TEXT,
    merchant    VARCHAR(100),                  -- display name (dashboard form); generator data uses merchant_id
    amount      NUMERIC(12, 2) NOT NULL,       -- US dollars
    currency    VARCHAR(3) NOT NULL DEFAULT 'USD',
    country     TEXT,
    channel     TEXT,
    location    VARCHAR(100),
    event_time  TIMESTAMPTZ NOT NULL,          -- when the payment happened (rules use this, not arrival time)
    status      VARCHAR(20) NOT NULL,          -- 'flagged' or 'cleared'
    reason      TEXT,                          -- 'amount' or 'velocity' when flagged, else NULL
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()  -- when the consumer stored it
);

-- The velocity rule counts one account's payments in a time window.
CREATE INDEX IF NOT EXISTS idx_transactions_account_time
    ON transactions (account_id, event_time, payment_id);
