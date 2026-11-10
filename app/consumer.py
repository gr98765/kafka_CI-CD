"""Run only the Kafka consumer, without the HTTP API.

    python -m app.consumer

docker-compose runs this as its own `consumer` service, so the consumer can be
killed and restarted (scripts/kill_test.py) while the API keeps taking payments.
"""

import logging
import signal
import threading

from app.main import consume_loop

log = logging.getLogger("pipeline")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stop = threading.Event()
    # `docker stop` sends SIGTERM: finish the current message, then exit
    # without committing the unfinished batch (it is re-delivered next start).
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    log.info("consumer starting")
    consume_loop(stop)
    log.info("consumer stopped")


if __name__ == "__main__":
    main()
