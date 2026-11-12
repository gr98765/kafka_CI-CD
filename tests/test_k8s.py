"""Kubernetes pieces that can be checked without a cluster: the consumer's
heartbeat and exec probe, the k8s_smoke helpers, and the manifests themselves."""

import os
import time
from pathlib import Path

import pytest

from app import main
from app.healthcheck import check
from scripts.k8s_smoke import catch_up_seconds, failures, make_records, total_lag

K8S = Path(__file__).resolve().parent.parent / "k8s"
yaml = pytest.importorskip("yaml")  # PyYAML comes with uvicorn[standard]


def manifests() -> list[dict]:
    docs = []
    for path in sorted(K8S.glob("*.yaml")):
        docs += [d for d in yaml.safe_load_all(path.read_text()) if d]
    return docs


def by_kind(kind: str) -> dict[str, dict]:
    return {d["metadata"]["name"]: d for d in manifests() if d["kind"] == kind}


# --- heartbeat + probe ------------------------------------------------------------------


def test_heartbeat_writes_state_when_configured(tmp_path, monkeypatch):
    path = tmp_path / "hb"
    monkeypatch.setattr(main, "HEARTBEAT_FILE", str(path))
    main.heartbeat("starting")
    assert path.read_text() == "starting"
    main.heartbeat("consuming")
    assert path.read_text() == "consuming"


def test_heartbeat_is_a_no_op_without_a_file(monkeypatch):
    monkeypatch.setattr(main, "HEARTBEAT_FILE", None)
    main.heartbeat("consuming")  # must not raise


def test_probe_passes_on_a_fresh_heartbeat_in_the_right_state(tmp_path):
    path = tmp_path / "hb"
    path.write_text("consuming")
    assert check(str(path), 15, "consuming")[0]
    assert check(str(path), 60)[0]


def test_probe_fails_when_stale_missing_or_still_starting(tmp_path):
    path = tmp_path / "hb"
    path.write_text("starting")
    ok, why = check(str(path), 15, "consuming")
    assert not ok and "want 'consuming'" in why
    assert check(str(path), 60)[0]  # alive while starting: liveness passes

    old = time.time() - 120
    os.utime(path, (old, old))
    ok, why = check(str(path), 60)
    assert not ok and "old" in why

    ok, why = check(str(tmp_path / "missing"), 60)
    assert not ok and "no heartbeat" in why


# --- k8s_smoke helpers ----------------------------------------------------------------


def test_make_records_gives_unique_ids_over_many_accounts():
    records = make_records(2000, "t")
    assert len({r["payment_id"] for r in records}) == 2000
    assert records[0]["payment_id"] == "t_000001"
    assert len({r["account_id"] for r in records}) > 500


LAG_OUTPUT = """
Consumer group 'transaction-consumer-group' has no active members.

GROUP                      TOPIC        PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG   CONSUMER-ID  HOST  CLIENT-ID
transaction-consumer-group transactions 0          100             6800            6700  -            -     -
transaction-consumer-group transactions 1          90              6690            6600  -            -     -
transaction-consumer-group transactions 2          -               6700            -     -            -     -
"""


def test_total_lag_sums_known_partitions():
    assert total_lag(LAG_OUTPUT) == 13300
    assert total_lag("Consumer group 'x' does not exist.") is None


def test_catch_up_measures_time_back_to_the_pre_deletion_backlog():
    # backlog 100 before t=20, jumps to 900 after the deletion, back to 120 at t=27
    samples = [(t, 1000 + 50 * t, 900 + 50 * t) for t in range(10, 20)]
    samples += [(21, 2050, 1150), (23, 2150, 1500), (25, 2250, 2000), (27, 2350, 2230), (29, 2450, 2350)]
    assert catch_up_seconds(samples, 20.0) == 7.0
    assert catch_up_seconds(samples, 5.0) is None  # nothing before the event


def test_failures_flag_lost_payments_and_an_incomplete_smoke():
    ok = {"sent": 500, "stored": 500, "lost": 0, "duplicated": 0}
    assert failures(ok, [{"replicas": 1, "lost": 0}], {"lost": 0}) == []
    assert failures({**ok, "stored": 499, "lost": 1}, [], None)[0].startswith("smoke")
    assert failures(ok, [{"replicas": 3, "lost": 2}], {"lost": 1}) == [
        "backlog with 3 replica(s): lost 2", "recovery: lost 1"]  # fmt: skip


# --- manifests ------------------------------------------------------------------------


def test_every_manifest_parses_and_starts_with_a_plain_comment():
    for path in sorted(K8S.glob("*.yaml")):
        assert path.read_text().startswith("# "), f"{path.name} has no comment at the top"
        assert list(yaml.safe_load_all(path.read_text())), path.name
    assert set(by_kind("Deployment")) == {"api", "consumer", "kafka", "postgres"}
    assert set(by_kind("Service")) >= {"api", "kafka", "postgres"}


def test_every_deployment_has_probes_and_resource_limits():
    for name, d in by_kind("Deployment").items():
        assert d["spec"]["replicas"] == 1, name
        for c in d["spec"]["template"]["spec"]["containers"]:
            assert c.get("readinessProbe") and c.get("livenessProbe"), f"{name}: missing probe"
            assert {"cpu", "memory"} <= set(c["resources"]["limits"]), f"{name}: missing limits"
            assert {"cpu", "memory"} <= set(c["resources"]["requests"]), f"{name}: missing requests"


def test_api_and_consumer_take_settings_from_the_configmap():
    for name in ("api", "consumer"):
        [c] = by_kind("Deployment")[name]["spec"]["template"]["spec"]["containers"]
        assert c["envFrom"] == [{"configMapRef": {"name": "pipeline-config"}}], name


def test_configmap_points_at_the_in_cluster_services_and_matches_the_rules():
    data = by_kind("ConfigMap")["pipeline-config"]["data"]
    assert data["KAFKA_BOOTSTRAP_SERVERS"] == "kafka:9092"
    assert data["DB_HOST"] == "postgres"
    assert data["KAFKA_TOPIC"] == main.TOPIC_NAME
    assert data["TOPIC_PARTITIONS"] == "3"
    assert float(data["FLAG_THRESHOLD"]) == main.FLAG_THRESHOLD
    assert int(data["VELOCITY_LIMIT"]) == main.VELOCITY_LIMIT
    assert int(data["VELOCITY_WINDOW_SECONDS"]) == main.VELOCITY_WINDOW_SECONDS


def test_topic_job_creates_the_topic_from_the_configmap():
    job = by_kind("Job")["create-topics"]
    [c] = job["spec"]["template"]["spec"]["containers"]
    script = c["args"][0]
    assert '--topic "$KAFKA_TOPIC" --partitions "$TOPIC_PARTITIONS"' in script
    assert c["envFrom"] == [{"configMapRef": {"name": "pipeline-config"}}]
