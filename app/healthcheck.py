"""Exec probe for the consumer pod (it has no HTTP server).

    python -m app.healthcheck --max-age 60                      # liveness
    python -m app.healthcheck --max-age 15 --require consuming  # readiness

Exits 0 if HEARTBEAT_FILE (written by app.main.heartbeat about once a second)
was updated within --max-age seconds and, with --require, holds that state.
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def check(path: str, max_age: float, require: str | None = None, now: float | None = None) -> tuple[bool, str]:
    try:
        age = (time.time() if now is None else now) - os.path.getmtime(path)
        with open(path) as f:
            state = f.read().strip()
    except OSError as exc:
        return False, f"no heartbeat: {exc}"
    if age > max_age:
        return False, f"heartbeat {age:.0f}s old (limit {max_age:.0f}s)"
    if require and state != require:
        return False, f"state {state!r}, want {require!r}"
    return True, f"ok: {state}, {age:.1f}s old"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--file", default=os.getenv("HEARTBEAT_FILE", "/tmp/consumer-heartbeat"))
    p.add_argument("--max-age", type=float, required=True)
    p.add_argument("--require")
    args = p.parse_args()
    ok, message = check(args.file, args.max_age, args.require)
    print(message)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
