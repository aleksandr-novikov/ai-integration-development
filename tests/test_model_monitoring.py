"""ML model monitoring (#16): inputs, model behaviour, inference metrics, quality alert."""

import json
import logging
import math
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient
from prometheus_client import generate_latest

from app import inference_adapter as adapter
from app import metrics_storage as storage
from app import model_monitoring as mm
from app import source_storage
from inference.service import create_app
from ml import model_registry as registry

SOURCE = {"project_id": "project-m", "connection_id": "connection-m",
          "schema_name": "public", "table_name": "events"}
OTHER = {**SOURCE, "connection_id": "connection-other"}
NOW = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)
VERSION = "if-" + "a" * 64
TOKEN = "synthetic-monitoring-token"


def history(values, *, start=NOW - timedelta(minutes=100), step=5):
    """(row_counts, null_rates) from [(row_count, null_rate)]; None omits a metric."""
    stamps = [(start + timedelta(minutes=step * i)).isoformat() for i in range(len(values))]
    rc = [{"ts": ts, "value": r} for ts, (r, _) in zip(stamps, values, strict=True) if r != "skip"]
    nr = [{"ts": ts, "value": n} for ts, (_, n) in zip(stamps, values, strict=True) if n != "skip"]
    return rc, nr


def steady(count, start_rows=1000):
    return [(start_rows + 10 * i, 0.02 + 0.001 * (i % 3)) for i in range(count)]


def summary(values, **kwargs):
    rc, nr = history(values, **kwargs)
    return mm.input_summary(rc, nr, now=NOW, max_age=timedelta(minutes=30))


# ── Criterion 1: inputs ────────────────────────────────────────────────────


def test_input_summary_of_complete_fresh_history():
    result = summary(steady(21), start=NOW - timedelta(minutes=100))
    assert result["conclusive"] is True and result["reasons"] == []
    assert result["valid_vectors"] == result["vectors"] == 20
    assert {name: f["missing"] for name, f in result["features"].items()} == dict.fromkeys(
        ("row_count", "null_rate", "d_row_count", "d_null_rate"), 0)
    rows = result["features"]["row_count"]["distribution"]
    assert (rows["min"], rows["max"], rows["median"]) == (1010, 1200, 1105)
    assert result["features"]["d_row_count"]["distribution"]["min"] == 10
    assert result["features"]["null_rate"]["bounds"] == "0 <= null_rate <= 1"
    assert result["freshness"]["fresh"] is True
    assert result["freshness"]["age_seconds"] == 0


def test_missing_metric_is_counted_not_zero_filled():
    values = steady(23)
    values[5] = (values[5][0], "skip")  # null_rate was not collected
    values[9] = ("skip", values[9][1])
    result = summary(values, start=NOW - timedelta(minutes=110))
    features = result["features"]
    assert features["row_count"]["missing"] == 1 and features["null_rate"]["missing"] == 1
    assert result["aligned_instants"] == 21 and result["valid_vectors"] == 20
    # Deltas span the gap; no zero row or zero delta is invented for it.
    assert features["row_count"]["distribution"]["min"] == 1010
    assert features["d_row_count"]["distribution"]["max"] == 20


@pytest.mark.parametrize(
    ("bad", "reason"),
    [((None, 0.02), "not_a_finite_number"), ((math.nan, 0.02), "not_a_finite_number"),
     ((1000, math.inf), "not_a_finite_number"), ((-5, 0.02), "out_of_bounds"),
     ((1000, 1.5), "out_of_bounds")],
)
def test_invalid_values_make_inputs_inconclusive(bad, reason):
    values = steady(30)
    values[10] = bad
    result = summary(values, start=NOW - timedelta(minutes=145))
    assert result["invalid_vectors"][reason] == 2  # both pairs that use the bad point
    assert result["valid_vectors"] == 27
    assert result["conclusive"] is False and "invalid_values" in result["reasons"]
    for feature in result["features"].values():
        assert feature["distribution"]["min"] >= -1  # nothing invalid entered the statistics


def test_empty_short_and_stale_histories_are_not_conclusive():
    empty = mm.input_summary([], [], now=NOW, max_age=timedelta(minutes=30))
    assert empty["reasons"] == ["no_valid_vectors"]
    assert empty["features"]["row_count"]["distribution"] is None
    assert empty["freshness"]["fresh"] is False
    short = summary(steady(10), start=NOW - timedelta(minutes=45))
    assert short["reasons"] == ["insufficient_valid_vectors"] and short["valid_vectors"] == 9
    stale = summary(steady(21), start=NOW - timedelta(hours=5))
    assert stale["reasons"] == ["stale"] and stale["freshness"]["fresh"] is False


# ── Criterion 2: scores, version and age ───────────────────────────────────


def score_rows(flags, *, demo=True, source=SOURCE, version=VERSION):
    return [
        {"ts": (NOW - timedelta(minutes=5 * (len(flags) - i))).isoformat(),
         "score": -0.05 if flag else 0.1 + 0.001 * i, "is_anomaly": flag,
         "source": dict(source), "model_version": version,
         "feature_schema_version": "1.0.0", "demo": demo}
        for i, flag in enumerate(flags)
    ]


def test_score_summary_keeps_demo_and_regular_results_apart():
    rows = score_rows([False, True, False, False]) + score_rows([True, True], demo=False)
    result = mm.score_summary(rows, demo=True)
    assert (result["results"], result["excluded_other_mode"], result["anomalies"]) == (4, 2, 1)
    assert result["anomaly_rate"] == 0.25
    assert result["score"]["min"] == -0.05
    assert mm.score_summary([], demo=True)["anomaly_rate"] is None


def manifest(training, evaluation=None):
    return {"model_version": VERSION, "feature_schema_version": "1.0.0",
            "origin": {"kind": "local-candidate"}, "training": training,
            "evaluation": evaluation}


def test_version_age_and_unrecorded_training_time():
    recorded = mm.version_info(
        manifest({"trained_at": (NOW - timedelta(days=3)).isoformat(), "time_status": "recorded",
                  "training_data_sha256": "b" * 64}), now=NOW)
    assert recorded["age_days"] == 3 and recorded["quality_admitted"] is False
    assert recorded["labelled_evaluation"] is None
    historical = mm.version_info(
        manifest({"trained_at": None, "time_status": "not_recorded_in_historical_export",
                  "training_data_sha256": "b" * 64},
                 {"split": "test", "data_sha256": "c" * 64, "run_id": "r",
                  "metrics": {"false_positive_rate": 0.1327}}), now=NOW)
    assert historical["age_days"] is None
    assert historical["time_status"] == "not_recorded_in_historical_export"
    # The FPR of the #9 export is above 10%: it is presented, never admitted.
    assert historical["quality_admitted"] is False
    assert historical["labelled_evaluation"]["metrics"]["false_positive_rate"] == 0.1327


# ── Criterion 2: inference errors and computation time ─────────────────────


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("monitoring-registry")
    start = datetime.now(UTC) - timedelta(days=1)
    rc = [{"ts": (start + timedelta(minutes=i)).isoformat(), "value": 100 + i}
          for i in range(205)]
    nr = [{"ts": row["ts"], "value": 0.02 + (i % 3) * 0.001} for i, row in enumerate(rc)]
    candidate = registry.train_candidate(root, source=SOURCE, row_counts=rc, null_rates=nr,
                                         git_sha="a" * 40, parameters={"n_estimators": 3})
    return root, candidate["model_version"]


@pytest.fixture
def service(bundle):
    root, _ = bundle
    return TestClient(create_app({"registry_path": str(root), "demo": True,
                                  "principals": [{"token": TOKEN, "sources": [SOURCE]}]}))


def observe(service, version, monkeypatch, client=None):
    monkeypatch.setenv("TEST_MONITORING_TOKEN", TOKEN)
    config = {"url": "http://testserver", "demo": True, "token_env": "TEST_MONITORING_TOKEN"}
    stamps = [(datetime.now(UTC) - timedelta(minutes=m)).replace(microsecond=0).isoformat()
              for m in (10, 5)]
    rc = [{"ts": stamps[0], "value": 300}, {"ts": stamps[1], "value": 310}]
    nr = [{"ts": stamps[0], "value": 0.02}, {"ts": stamps[1], "value": 0.021}]
    return adapter.predict(SOURCE, version, "1.0.0", client=client or service,
                           _history=(rc, nr), _config=config)


def series_labels(version, source=SOURCE):
    return {**source, "model_version": version, "feature_schema_version": "1.0.0",
            "demo": "true"}


def test_service_reports_computation_time_without_changing_the_body(service, bundle):
    _, version = bundle
    body = {"source": SOURCE, "model_version": version, "feature_schema_version": "1.0.0",
            "observed_at": "2026-11-09T12:00:00+00:00",
            "previous_observed_at": "2026-11-09T11:55:00+00:00",
            "features": {"row_count": 310.0, "null_rate": 0.021, "d_row_count": 10.0,
                         "d_null_rate": 0.001}}
    response = service.post("/v1/predict", json=body,
                            headers={"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 200
    assert set(response.json()) == {"source", "model_version", "feature_schema_version",
                                    "observed_at", "score", "threshold", "is_anomaly",
                                    "demo", "quality_admitted"}
    seconds = adapter.server_timing(response.headers["server-timing"])
    assert seconds is not None and 0 <= seconds < 5
    refused = service.post("/v1/predict", json={**body, "features": {}},
                           headers={"Authorization": f"Bearer {TOKEN}"})
    assert refused.status_code == 422 and "server-timing" not in refused.headers


@pytest.mark.parametrize(
    ("header", "expected"),
    [("predict;dur=1.5", 0.0015), ('cache;dur=9, predict;desc="x";dur="2"', 0.002),
     (None, None), ("", None), ("predict", None), ("predict;dur=abc", None),
     ("predict;dur=-1", None), ("predict;dur=nan", None), ("other;dur=3", None)],
)
def test_server_timing_parsing(header, expected):
    assert adapter.server_timing(header) == expected


def test_inference_calls_errors_and_time_per_version(service, bundle, monkeypatch):
    _, version = bundle
    before = mm.inference_summary(generate_latest().decode(), series_labels(version))
    observe(service, version, monkeypatch)
    observe(service, version, monkeypatch)

    class Down:
        def post(self, *args, **kwargs):
            raise httpx.ConnectError("down")

    with pytest.raises(adapter.AdapterError, match="inference_unavailable"):
        observe(service, version, monkeypatch, client=Down())
    with pytest.raises(adapter.AdapterError) as refused:
        observe(service, "if-" + "f" * 64, monkeypatch)  # unknown version
    assert refused.value.code == "model_unavailable"
    after = mm.inference_summary(generate_latest().decode(), series_labels(version))
    assert after["ok"] - before["ok"] == 2
    assert after["errors"]["inference_unavailable"] - before["errors"].get(
        "inference_unavailable", 0) == 1
    assert "model_unavailable" not in after["errors"]  # another version's series
    timing = after["predict_seconds"]
    assert timing["count"] - before["predict_seconds"]["count"] == 2
    assert timing["mean"] > 0 and timing["p95_bucket_le"] is not None
    other = mm.inference_summary(generate_latest().decode(), series_labels("if-" + "f" * 64))
    assert other["errors"] == {"model_unavailable": 1} and other["ok"] == 0


# ── Criterion 4: alert rule ────────────────────────────────────────────────


BASELINE = {"kind": "anomaly_rate_baseline", "source": SOURCE, "model_version": VERSION,
            "feature_schema_version": "1.0.0", "demo": True, "observations": 1175,
            "anomalies": 10, "origin": {"split": "validation"}}


def test_threshold_is_the_binomial_tail_bound_of_the_baseline():
    rule = mm.AnomalyRateRule()
    k = mm.threshold_count(rule, BASELINE)
    p = mm.wilson_upper(10, 1175)
    assert k == 3 and round(p, 4) == 0.0156
    assert mm.binomial_tail(k, rule.window, p) <= rule.alpha < mm.binomial_tail(
        k - 1, rule.window, p)
    assert mm.binomial_tail(0, 20, p) == 1.0
    assert mm.threshold_count(rule, {**BASELINE, "anomalies": 0, "observations": 1175}) == 2


def alert(baseline=BASELINE, **change):
    series = {"source": SOURCE, "model_version": VERSION, "feature_schema_version": "1.0.0",
              "demo": True, **change}
    return mm.AnomalyRateAlert(baseline, **series)


def test_normal_deviation_alert_recovery_with_single_events(caplog):
    rule = alert()
    flags, events, states = [], [], []
    plan = [False] * 25 + [True] * 4 + [False] * 22
    with caplog.at_level(logging.INFO, logger="app.alerts"):
        for flag in plan:
            flags.append(flag)
            evaluation = rule.observe(score_rows(flags))
            events.append(evaluation.event)
            states.append(evaluation.state)
            # Same data again never produces another transition.
            assert rule.observe(score_rows(flags)).event is None
    assert events.index("firing") == 27  # third anomalous result in the window
    assert [e for e in events if e] == ["firing", "resolved"]
    assert events.index("resolved") == 46  # two of four anomalies left the window
    assert states[-1] == "inactive" and set(states[:27]) == {"inactive"}
    firing, resolved = [r for r in caplog.records if r.name == "app.alerts"]
    assert firing.levelno == logging.WARNING and resolved.levelno == logging.INFO
    assert firing.event == "alert_firing" and firing.source == SOURCE
    assert firing.model_version == VERSION and firing.anomalies == 3


def test_insufficient_window_is_not_normal_and_keeps_firing():
    rule = alert()
    short = rule.observe(score_rows([True] * 19))
    assert (short.status, short.condition, short.anomalies) == (
        "insufficient_observations", None, None)
    assert rule.observe(score_rows([True] * 20)).event == "firing"
    # Fewer results later (shorter retention) cannot resolve the alert.
    again = rule.observe(score_rows([False] * 5))
    assert (again.status, again.state, again.event) == (
        "insufficient_observations", "firing", None)
    assert rule.observe([]).status == "insufficient_observations"


@pytest.mark.parametrize(
    ("baseline", "change", "status"),
    [(None, {}, "baseline_missing"),
     ({**BASELINE, "observations": 499, "anomalies": 0}, {}, "baseline_insufficient"),
     ({**BASELINE, "anomalies": 2000}, {}, "baseline_invalid"),
     ({**BASELINE, "anomalies": True}, {}, "baseline_invalid"),
     ({"observations": 1000}, {}, "baseline_invalid"),
     ({**BASELINE, "observations": 1000, "anomalies": 1000}, {}, "baseline_rate_too_high"),
     (BASELINE, {"model_version": "if-" + "b" * 64}, "baseline_mismatch"),
     (BASELINE, {"source": OTHER}, "baseline_mismatch"),
     (BASELINE, {"feature_schema_version": "2.0.0"}, "baseline_mismatch"),
     (BASELINE, {"demo": False}, "baseline_mismatch")],
)
def test_unusable_baseline_never_evaluates(baseline, change, status):
    evaluation = alert(baseline, **change).observe(score_rows([True] * 25))
    assert (evaluation.status, evaluation.condition, evaluation.event) == (status, None, None)
    assert evaluation.state == "inactive"


def test_alert_ignores_results_of_other_sources_versions_and_modes():
    rule = alert()
    own = score_rows([False] * 20)
    foreign = (score_rows([True] * 20, source=OTHER) + score_rows([True] * 20, demo=False)
               + score_rows([True] * 20, version="if-" + "b" * 64))
    evaluation = rule.observe(sorted(own + foreign, key=lambda row: row["ts"]))
    assert (evaluation.observations, evaluation.anomalies, evaluation.condition) == (20, 0, False)


def test_build_baseline_counts_flags():
    baseline = mm.build_baseline([False, True, False], source=SOURCE, model_version=VERSION,
                                 feature_schema_version="1.0.0", demo=True,
                                 origin={"split": "validation"})
    assert (baseline["observations"], baseline["anomalies"]) == (3, 1)
    json.dumps(baseline)


# ── Report over storage: one source, one version ───────────────────────────


@pytest.fixture
def stored(tmp_path, monkeypatch):
    monkeypatch.setattr(storage.settings, "MONITOR_DB_URL", f"sqlite:///{tmp_path / 'm.db'}")
    monkeypatch.setattr(storage.settings, "DATABASE_URL", "sqlite://")
    monkeypatch.setattr(storage, "_engine", None)
    monkeypatch.setattr(storage, "_initialized", False)
    storage.create_user("u", "u@example.test", "unused")
    storage.create_project(SOURCE["project_id"], "u", "M", "m")
    for source in (SOURCE, OTHER):
        storage.create_connection(connection_id=source["connection_id"],
                                  project_id=source["project_id"], name=source["connection_id"],
                                  dsn_encrypted=b"unused", interval_minutes=5)
    start = datetime.now(UTC).replace(second=0, microsecond=0) - timedelta(minutes=5 * 24)
    for source, rows in ((SOURCE, steady(25)), (OTHER, steady(25, start_rows=90_000))):
        for metric, index in (("row_count", 0), ("null_rate", 1)):
            storage.save_metrics(
                [{"ts": (start + timedelta(minutes=5 * i)).isoformat(),
                  "table_name": source["table_name"], "metric_name": metric,
                  "value": row[index]} for i, row in enumerate(rows)],
                source["project_id"], connection_id=source["connection_id"],
                schema_name=source["schema_name"])
    stamps = [row["ts"] for row in source_storage.metrics(SOURCE, "row_count")][1:]
    for i, ts in enumerate(stamps):
        for source, version, anomaly in ((SOURCE, VERSION, i >= 21), (OTHER, VERSION, True),
                                         (SOURCE, "if-" + "b" * 64, True)):
            source_storage.insert_prediction({
                "source": source, "observed_at": ts, "model_version": version,
                "feature_schema_version": "1.0.0", "score": -0.1 if anomaly else 0.1,
                "is_anomaly": anomaly, "demo": True})
    yield
    storage._engine.dispose()


def test_report_reads_one_source_and_one_version(stored):
    rule = alert()
    result = mm.report(SOURCE, VERSION, "1.0.0", demo=True, alert=rule)
    inputs = result["inputs"]
    assert inputs["observation_instants"] == 25 and inputs["valid_vectors"] == 24
    assert inputs["features"]["row_count"]["distribution"]["max"] == 1240  # not 90k+
    assert inputs["conclusive"] is True
    assert inputs["freshness"]["max_age_seconds"] == 1800  # max(30 min, 2 x 5 min)
    scores = result["model"]["scores"]
    assert (scores["results"], scores["anomalies"]) == (24, 3)
    assert result["model"]["version"] == {"model_version": VERSION, "available": False,
                                          "reason": "registry_not_given"}
    assert result["model"]["inference"]["available"] is False
    assert (result["alert"]["status"], result["alert"]["event"]) == ("evaluated", "firing")
    assert result["alert"]["rule"]["window"] == 20
    assert mm.report(SOURCE, VERSION, "1.0.0", demo=True, alert=rule)["alert"]["event"] is None
    regular = mm.report(SOURCE, VERSION, "1.0.0", demo=False, alert=alert(demo=False))
    assert regular["model"]["scores"]["excluded_other_mode"] == 24
    assert regular["alert"]["status"] == "baseline_mismatch"


def test_report_reads_version_from_the_registry(stored, bundle):
    root, version = bundle
    result = mm.report(SOURCE, version, "1.0.0", demo=True, alert=alert(), registry=root)
    info = result["model"]["version"]
    assert info["available"] is True and info["model_version"] == version
    assert 0 <= info["age_days"] < 1 and info["quality_admitted"] is False
    missing = mm.report(SOURCE, "if-" + "e" * 64, "1.0.0", demo=True, alert=alert(),
                        registry=root)
    assert missing["model"]["version"]["reason"] == "manifest_unavailable"
