-- Clean layer: rebuild payments_clean and payments_quarantine from raw_payments.
--
-- Run by app/clean.py inside one transaction, which passes the rule thresholds
-- from app/main.py as %(flag_threshold)s, %(velocity_limit)s and
-- %(velocity_window)s. (Because of that, a literal percent sign in this file
-- must be written twice.) A full rebuild every run: same raw data in, same
-- tables out.
--
-- Fixes:     amount "$12.50" / "12.50" / 12.5 -> 12.50
--            timestamp ISO string, epoch seconds (number or digit string) or
--              "YYYY-MM-DD HH:MM:SS" -> TIMESTAMPTZ in UTC (no zone given = UTC)
--            account_id trimmed and uppercased; currency trimmed and uppercased
--            duplicate payment_id: the first arrival is kept (by arrival time,
--              then Kafka partition and offset); later copies stay in raw only
-- Quarantine reasons (first that applies):
--            missing_payment_id, missing_account_id,
--            bad_amount           missing, unreadable, or too large for NUMERIC(12,2)
--            non_positive_amount  zero or negative (after rounding to cents)
--            currency_not_usd     anything but USD after trimming/uppercasing, or missing
--            bad_timestamp        missing, unreadable, or outside the years 2000-2099
-- Fraud rules (same as before): amount > flag_threshold -> 'amount'; the
-- velocity_limit-th or later payment by one account within velocity_window
-- seconds, by event time, ties ordered by payment_id -> 'velocity'.

SET LOCAL TIME ZONE 'UTC';

-- Trim all leading/trailing whitespace (btrim alone only removes spaces).
CREATE OR REPLACE FUNCTION pg_temp.trim_ws(s text) RETURNS text
LANGUAGE sql IMMUTABLE AS $fn$
    SELECT regexp_replace(s, '^\s+|\s+$', '', 'g')
$fn$;

-- Amount: a JSON number, or a string like "12.50", "$12.50", "-$3.00", "1,234.50".
CREATE OR REPLACE FUNCTION pg_temp.clean_amount(v jsonb) RETURNS numeric
LANGUAGE plpgsql IMMUTABLE AS $fn$
DECLARE
    s text;
BEGIN
    IF v IS NULL THEN
        RETURN NULL;
    ELSIF jsonb_typeof(v) = 'number' THEN
        RETURN (v #>> '{}')::numeric;
    ELSIF jsonb_typeof(v) <> 'string' THEN
        RETURN NULL;
    END IF;
    s := pg_temp.trim_ws(v #>> '{}');
    s := regexp_replace(s, '^([-+]?)\s*\$\s*', '\1');      -- "$12.50" -> "12.50", "-$3" -> "-3"
    IF s ~ '^[-+]?[0-9]{1,3}(,[0-9]{3})+(\.[0-9]+)?$' THEN  -- thousands separators
        s := replace(s, ',', '');
    END IF;
    IF s ~ '^[-+]?([0-9]+(\.[0-9]*)?|\.[0-9]+)$' THEN
        RETURN s::numeric;
    END IF;
    RETURN NULL;
END
$fn$;

-- Timestamp: epoch seconds (JSON number or digit string), ISO 8601 with an
-- optional zone, or "YYYY-MM-DD HH:MM:SS". No zone means UTC.
CREATE OR REPLACE FUNCTION pg_temp.clean_ts(v jsonb) RETURNS timestamptz
LANGUAGE plpgsql IMMUTABLE AS $fn$
DECLARE
    s text;
    t timestamptz;
BEGIN
    IF v IS NULL OR jsonb_typeof(v) NOT IN ('number', 'string') THEN
        RETURN NULL;
    END IF;
    s := pg_temp.trim_ws(v #>> '{}');
    BEGIN
        IF s ~ '^[0-9]+(\.[0-9]+)?$' THEN
            t := to_timestamp(s::double precision);
        ELSIF jsonb_typeof(v) = 'number' THEN
            RETURN NULL;                                       -- negative or exponent form
        ELSIF s ~* '^[0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?(Z|[+-][0-9]{2}(:?[0-9]{2})?)$' THEN
            t := s::timestamptz;                               -- zone given
        ELSIF s ~* '^[0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?$' THEN
            t := s::timestamp AT TIME ZONE 'UTC';              -- no zone: UTC
        ELSE
            RETURN NULL;
        END IF;
    EXCEPTION WHEN others THEN                                 -- e.g. month 13, out of range
        RETURN NULL;
    END;
    IF t < '2000-01-01T00:00:00Z' OR t >= '2100-01-01T00:00:00Z' THEN
        RETURN NULL;
    END IF;
    RETURN t;
END
$fn$;

TRUNCATE payments_clean, payments_quarantine;

CREATE TEMP TABLE clean_parsed ON COMMIT DROP AS
WITH parsed AS (
    SELECT
        r.raw_id,
        r.payload,
        r.received_at,
        r.kafka_partition,
        r.kafka_offset,
        NULLIF(r.payload ->> 'payment_id', '') AS payment_id_raw,
        NULLIF(upper(pg_temp.trim_ws(r.payload ->> 'account_id')), '') AS account_id,
        pg_temp.clean_amount(r.payload -> 'amount') AS amount,
        upper(pg_temp.trim_ws(r.payload ->> 'currency')) AS currency,
        pg_temp.clean_ts(r.payload -> 'timestamp') AS event_time
    FROM raw_payments AS r
),
ranked AS (
    SELECT
        p.*,
        CASE WHEN pg_temp.trim_ws(coalesce(p.payment_id_raw, '')) = '' THEN NULL ELSE p.payment_id_raw END AS payment_id,
        row_number() OVER (
            PARTITION BY p.payment_id_raw
            ORDER BY p.received_at, p.kafka_partition, p.kafka_offset, p.raw_id
        ) AS arrival
    FROM parsed AS p
)
SELECT
    k.*,
    CASE
        WHEN k.payment_id IS NULL THEN 'missing_payment_id'
        WHEN k.account_id IS NULL THEN 'missing_account_id'
        WHEN k.amount IS NULL OR abs(k.amount) >= 10000000000 THEN 'bad_amount'
        WHEN round(k.amount, 2) <= 0 THEN 'non_positive_amount'
        WHEN k.currency IS DISTINCT FROM 'USD' THEN 'currency_not_usd'
        WHEN k.event_time IS NULL THEN 'bad_timestamp'
    END AS quarantine_reason
FROM ranked AS k
-- One row per payment_id (the first arrival). Records without a payment_id
-- cannot be de-duplicated, so every one of them is kept (and quarantined).
WHERE k.payment_id IS NULL OR k.arrival = 1;

INSERT INTO payments_quarantine (raw_id, payment_id, reason, payload, received_at, kafka_partition, kafka_offset)
SELECT raw_id, payment_id_raw, quarantine_reason, payload, received_at, kafka_partition, kafka_offset
FROM clean_parsed
WHERE quarantine_reason IS NOT NULL;

-- Rule 1 (amount) on insert.
INSERT INTO payments_clean (payment_id, account_id, merchant_id, merchant, amount, currency,
                            country, channel, location, event_time, status, reason, created_at, raw_id)
SELECT
    payment_id,
    account_id,
    payload ->> 'merchant_id',
    payload ->> 'merchant',
    round(amount, 2),
    currency,
    payload ->> 'country',
    payload ->> 'channel',
    coalesce(payload ->> 'location', 'unknown'),
    event_time,
    CASE WHEN round(amount, 2) > %(flag_threshold)s THEN 'flagged' ELSE 'cleared' END,
    CASE WHEN round(amount, 2) > %(flag_threshold)s THEN 'amount' END,
    received_at,
    raw_id
FROM clean_parsed
WHERE quarantine_reason IS NULL;

-- Rule 2 (velocity), by event time over every clean payment of the account.
-- An amount-flagged payment keeps reason 'amount' but still counts.
UPDATE payments_clean AS t
SET status = 'flagged', reason = 'velocity'
WHERE t.status = 'cleared'
  AND (
      SELECT count(*)
      FROM payments_clean AS w
      WHERE w.account_id = t.account_id
        AND w.event_time >= t.event_time - make_interval(secs => %(velocity_window)s)
        AND (w.event_time, w.payment_id) <= (t.event_time, t.payment_id)
  ) >= %(velocity_limit)s;
