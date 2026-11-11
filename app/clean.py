"""Clean step: rebuild payments_clean and payments_quarantine from raw_payments.

    python -m app.clean                      # inside the stack: docker compose exec consumer python -m app.clean
    DB_PORT=5433 python -m app.clean         # from the host (compose maps Postgres to 5433)

Runs sql/clean.sql in one transaction with the fraud-rule thresholds from
app/main.py, then prints what happened to every raw record: kept, quarantined
(by reason) or dropped as a later copy of a payment_id already kept.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.main import (
    FLAG_THRESHOLD,
    VELOCITY_LIMIT,
    VELOCITY_WINDOW_SECONDS,
    get_db_connection,
)

CLEAN_SQL = Path(__file__).resolve().parent.parent / "sql" / "clean.sql"


def run_clean(conn) -> dict:
    """Rebuild the clean layer on `conn` and commit. Returns the counts."""
    params = {
        "flag_threshold": FLAG_THRESHOLD,
        "velocity_limit": VELOCITY_LIMIT,
        "velocity_window": VELOCITY_WINDOW_SECONDS,
    }
    try:
        with conn.cursor() as cur:
            cur.execute(CLEAN_SQL.read_text(encoding="utf-8"), params)
            cur.execute("SELECT count(*) FROM raw_payments")
            raw = cur.fetchone()[0]
            cur.execute("SELECT status, reason, count(*) FROM payments_clean GROUP BY 1, 2 ORDER BY 1, 2")
            clean = {f"{status}/{reason}" if reason else status: n for status, reason, n in cur.fetchall()}
            cur.execute("SELECT reason, count(*) FROM payments_quarantine GROUP BY 1 ORDER BY 1")
            quarantine = dict(cur.fetchall())
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    kept, quarantined = sum(clean.values()), sum(quarantine.values())
    return {
        "raw": raw,
        "clean": kept,
        "clean_by_status": clean,
        "quarantined": quarantined,
        "quarantine_by_reason": quarantine,
        # Every raw record is accounted for: kept, quarantined, or a later copy.
        "duplicates_dropped": raw - kept - quarantined,
    }


def main() -> None:
    conn = get_db_connection()
    try:
        print(json.dumps(run_clean(conn), indent=2))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
