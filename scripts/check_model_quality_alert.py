"""Reproducible normal → deviation → alert → recovery run of the model alert (#16).

Run: python -m scripts.check_model_quality_alert > docs/course/monitoring/model/demo.json \
        2> /tmp/model-alert-logs.jsonl

One source, one model version, all in one process and on throwaway SQLite:

- the model is the #13 Compose demo model, rebuilt by the existing procedure
  (``scripts.demo_model`` train split + ``train_candidate``, configuration
  ``contamination_005``, seed 42) and checked against selection.json;
- the baseline is this exact version scored on the normal ticks of the #13
  validation split; test is not used;
- every tick stores row_count/null_rate of the source, the real adapter calls
  the real FastAPI service (ASGI test client) and stores the result, then
  ``app.model_monitoring.report`` evaluates inputs, model and the alert.

The stream replays the #13 row generators and collector arithmetic with a
fixed seed outside the train/validation/test ranges. Deviation: the #13
incident batch (10 x the stand's nominal 200 rows from the incident generator:
40% NULL ip_address, server-3 skew) is loaded and the next collection sees it
rolled back; the regular flow goes on, three load/rollback pairs. The table level stays inside the model's 1-14k-row domain, so recovery
is possible. One inference outage in the normal phase exercises the error
counter. Telegram/Sentry/LLM/SMTP are off;
the signal is the ``app.alerts`` log record, nothing is delivered.
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys
import tempfile
from collections import deque
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "docs/course/containers/demo-model/selection.json"
SOURCE = {"project_id": "demo-project", "connection_id": "demo-connection",
          "schema_name": "public", "table_name": "events"}
# Same table name on another connection: its data must never enter SOURCE.
OTHER_SOURCE = {**SOURCE, "connection_id": "other-connection"}
TOKEN_ENV = "MODEL_ALERT_DEMO_TOKEN"
STREAM_SEED = 4_000_000  # disjoint from #13 train/validation/test seeds (1e6/2e6/3e6)
INITIAL_ROWS = 2000  # Compose stand start
PHASES = (("normal", 30), ("deviation", 6), ("recovery", 25))
NOMINAL_ROWS = 200  # Compose stand rows per collection; the #13 incident is 10 x
OUTAGE_TICK = 10
COLLECTION_MINUTES = 5  # the smallest interval a connection accepts
_OFFLINE_SETTINGS = {"SENTRY_DSN": "", "TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": "",
                     "NIM_API_KEY": "", "SMTP_HOST": "", "LOG_FORMAT": "json"}


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


class _Unavailable:
    """Inference client during the outage: the connection is refused."""

    def post(self, *args, **kwargs):
        import httpx

        raise httpx.ConnectError("inference service unavailable (demo outage)")


def _stream():
    """(phase, row_count, null_rate) per collection, first one before any tick."""
    from scripts import demo_model as dm
    from scripts import live_demo, seed_target_db

    class Table(dm._Table):
        """The #13 collector arithmetic plus a rollback of the newest rows."""

        def __init__(self):
            super().__init__()
            self.rows = deque()

        def insert(self, rows):
            for row in rows:
                super().insert([row])
                self.rows.append(tuple(
                    row[column] is None if column in row else column in dm.NULL_WHEN_OMITTED
                    for column in dm.EVENTS_COLUMNS
                ))

        def rollback(self, count):
            for _ in range(min(count, len(self.rows))):
                for column, null in zip(dm.EVENTS_COLUMNS, self.rows.pop(), strict=True):
                    self.nulls[column] -= null
                self.total -= 1

    batch = NOMINAL_ROWS * live_demo._INCIDENT_ROW_MULTIPLIER
    state = random.getstate()
    try:
        random.seed(STREAM_SEED)
        table = Table()
        now = datetime.now(UTC)
        table.insert(seed_target_db._event_row(dm._Addresses, dm._USER_IDS, now)
                     for _ in range(INITIAL_ROWS))
        collections = [("initial", *table.observe())]
        for phase, ticks in PHASES:
            for index in range(ticks):
                rollback = phase == "deviation" and index % 2 == 1
                if rollback:  # the batch of the previous collection is the newest
                    table.rollback(batch)
                rows = 0 if random.random() < dm.IDLE_PROBABILITY else random.randint(
                    *dm.NORMAL_ROWS)
                table.insert(live_demo._generate_event_row(dm._USER_IDS, False)
                             for _ in range(rows))
                if phase == "deviation" and not rollback:
                    table.insert(live_demo._generate_event_row(dm._USER_IDS, True)
                                 for _ in range(batch))
                collections.append((phase, *table.observe()))
    finally:
        random.setstate(state)
    return collections


def _git_sha() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "0" * 40
    return sha if len(sha) == 40 else "0" * 40


def _model_and_baseline(registry: Path):
    """#13 demo model for SOURCE and its baseline on #13 validation normal ticks."""
    from app.model_monitoring import build_baseline
    from ml.features import FEATURE_NAMES
    from ml.model_registry import load_version, train_candidate
    from scripts import demo_model as dm

    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    _, train_histories, train_sha = dm.generate("train")
    manifest = train_candidate(
        registry, source=SOURCE, git_sha=_git_sha(), seed=dm.MODEL_SEED,
        parameters=dm.CONFIGS[dm.SELECTED],
        histories=[{key: h[key] for key in ("row_counts", "null_rates")} for h in train_histories],
        provenance={"purpose": "model-alert-demo", "regime": dm.REGIME,
                    "configuration": dm.SELECTED, "train_data_sha256": train_sha},
    )
    loaded = load_version(registry, manifest["model_version"], source=SOURCE, demo=True)
    _, histories, validation_sha = dm.generate("validation")
    rows, labels, _ = dm.vectors(histories)
    normal = [row for row, label in zip(rows, labels, strict=True) if label == "normal"]
    scores = loaded.pipeline.decision_function(dm._matrix(normal))
    # The batch path must equal the serving path (validate_vector + predict).
    for index in range(0, len(normal), 97):
        served = loaded.predict(dict(zip(FEATURE_NAMES, normal[index], strict=True)))
        assert served["score"] == float(scores[index])
    baseline = build_baseline(
        [score < 0 for score in scores], source=SOURCE,
        model_version=manifest["model_version"], feature_schema_version="1.0.0", demo=True,
        origin={"split": "validation", "regime": dm.REGIME, "labels": "normal ticks only",
                "data_sha256": validation_sha, "configuration": dm.SELECTED},
    )
    expected = selection["candidates"][dm.SELECTED]["metrics"]
    reproduction = {
        "train_data_sha256_matches_selection": train_sha == selection["train_data_sha256"],
        "validation_data_sha256_matches_selection":
            validation_sha == selection["validation_data_sha256"],
        "baseline_matches_selection_false_alarms": (
            (baseline["anomalies"], baseline["observations"])
            == (expected["false_alarms"], expected["normal_points"])
        ),
    }
    return manifest, baseline, reproduction


def _save(storage, source, ts, row_count, null_rate):
    storage.save_metrics(
        [{"ts": ts, "table_name": source["table_name"], "metric_name": metric, "value": value}
         for metric, value in (("row_count", row_count), ("null_rate", null_rate))],
        source["project_id"], connection_id=source["connection_id"],
        schema_name=source["schema_name"],
    )


def run_demo(work_dir: Path) -> dict:
    from fastapi.testclient import TestClient
    from prometheus_client import generate_latest
    from sqlalchemy import text

    import app.metrics_storage as storage
    from app import inference_adapter as adapter
    from app import source_storage
    from app.config import settings
    from app.logging_setup import JsonFormatter
    from app.model_monitoring import AnomalyRateAlert, report
    from inference.service import create_app

    for key, value in _OFFLINE_SETTINGS.items():
        setattr(settings, key, value)
    settings.MONITOR_DB_URL = f"sqlite:///{work_dir / 'monitor.db'}"
    storage._engine = None
    storage._initialized = False
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    stderr = logging.StreamHandler(sys.stderr)  # stdout carries the JSON report
    stderr.setFormatter(JsonFormatter())
    capture = _Capture()
    capture.setFormatter(JsonFormatter())
    root.addHandler(stderr)
    root.addHandler(capture)

    registry = work_dir / "registry"
    manifest, baseline, reproduction = _model_and_baseline(registry)
    version = manifest["model_version"]
    token = "model-alert-demo-token"
    service = TestClient(create_app({"registry_path": str(registry), "demo": True,
                                     "principals": [{"token": token, "sources": [SOURCE]}]}))
    config = {"url": "http://testserver", "demo": True, "token_env": TOKEN_ENV,
              "bindings": [{"source": SOURCE, "model_version": version,
                            "feature_schema_version": "1.0.0"}]}
    (work_dir / "adapter.json").write_text(json.dumps(config), encoding="utf-8")
    os.environ["INFERENCE_ADAPTER_CONFIG"] = str(work_dir / "adapter.json")
    os.environ[TOKEN_ENV] = token
    storage.create_user("demo-user", "demo@example.test", "unused")
    storage.create_project(SOURCE["project_id"], "demo-user", "Demo", "demo")
    for source in (SOURCE, OTHER_SOURCE):
        storage.create_connection(connection_id=source["connection_id"],
                                  project_id=source["project_id"], name=source["connection_id"],
                                  dsn_encrypted=b"unused",
                                  interval_minutes=COLLECTION_MINUTES)

    collections = _stream()
    start = datetime.now(UTC).replace(second=0, microsecond=0) - timedelta(
        minutes=COLLECTION_MINUTES * len(collections))
    alert = AnomalyRateAlert(baseline, source=SOURCE, model_version=version,
                             feature_schema_version="1.0.0", demo=True)
    timeline, at_firing = [], None
    try:
        for tick, (phase, row_count, null_rate) in enumerate(collections):
            ts = (start + timedelta(minutes=COLLECTION_MINUTES * tick)).isoformat()
            _save(storage, SOURCE, ts, row_count, null_rate)
            _save(storage, OTHER_SOURCE, ts, 50_000 + 900 * tick, 0.3)
            outage = tick == OUTAGE_TICK
            adapter.score_project(SOURCE["project_id"],
                                  client=_Unavailable() if outage else service)
            if phase == "initial":
                continue
            result = report(SOURCE, version, "1.0.0", demo=True, alert=alert,
                            registry=registry, metrics_text=generate_latest().decode())
            scores = source_storage.scores(SOURCE, version, "1.0.0")
            newest = scores[-1] if scores and scores[-1]["ts"] == result["alert"]["observed_at"] \
                else None
            timeline.append({
                "tick": tick, "phase": phase, "row_count": row_count, "null_rate": null_rate,
                "inference_outage": outage, "results": len(scores),
                "score": round(newest["score"], 6) if newest and not outage else None,
                "is_anomaly": newest["is_anomaly"] if newest and not outage else None,
                **{key: result["alert"][key] for key in (
                    "status", "anomalies", "anomaly_rate", "condition", "state", "event")},
            })
            if result["alert"]["event"] == "firing":
                at_firing = scores
        # Same data again: the rule must not fire or resolve a second time.
        repeated = [asdict(alert.observe(source_storage.scores(SOURCE, version, "1.0.0")))
                    for _ in range(2)]
        final = report(SOURCE, version, "1.0.0", demo=True, alert=alert, registry=registry,
                       metrics_text=generate_latest().decode())
    finally:
        root.removeHandler(capture)
        root.removeHandler(stderr)

    def evaluate(candidate_baseline, **change):
        series = {"source": SOURCE, "model_version": version,
                  "feature_schema_version": "1.0.0", "demo": True, **change}
        return asdict(AnomalyRateAlert(candidate_baseline, **series).observe(at_firing or []))

    small = {**baseline, "observations": 400, "anomalies": min(baseline["anomalies"], 400)}
    with storage.get_engine().connect() as conn:
        stored = conn.execute(text(
            "SELECT connection_id, model_version, feature_schema_version, is_demo, COUNT(*) "
            "FROM anomaly_scores GROUP BY 1, 2, 3, 4"
        )).fetchall()
    storage._engine.dispose()
    alerts = [json.loads(line) for line in capture.lines if '"logger": "app.alerts"' in line]
    events = [(e["tick"], e["phase"], e["event"]) for e in timeline if e["event"]]
    by_phase = {phase: [e for e in timeline if e["phase"] == phase] for phase, _ in PHASES}
    inference = final["model"]["inference"]
    checks = {
        **reproduction,
        "insufficient_window_not_treated_as_normal": all(
            e["condition"] is None and e["status"] == "insufficient_observations"
            for e in timeline if e["results"] < alert.rule.window
        ),
        "normal_evaluated_without_signal": (
            any(e["condition"] is False for e in by_phase["normal"])
            and all(e["state"] == "inactive" and not e["event"] for e in by_phase["normal"])
        ),
        "deviation_fired": [event for _, phase, event in events if phase == "deviation"]
        == ["firing"],
        "single_firing_and_single_resolved": [event for *_, event in events]
        == ["firing", "resolved"],
        "recovery_resolved": any(e["event"] == "resolved" for e in by_phase["recovery"]),
        "final_state_inactive": final["alert"]["state"] == "inactive",
        "repeated_evaluation_no_new_event": all(r["event"] is None for r in repeated),
        "signal_logged_with_source_and_version": len(alerts) == 2 and all(
            a["source"] == SOURCE and a["model_version"] == version for a in alerts),
        "small_baseline_rejected": evaluate(small)["status"] == "baseline_insufficient",
        "missing_baseline_rejected": evaluate(None)["status"] == "baseline_missing",
        "other_version_baseline_rejected":
            evaluate({**baseline, "model_version": "if-" + "0" * 64})["status"]
            == "baseline_mismatch",
        "other_source_baseline_rejected":
            evaluate({**baseline, "source": OTHER_SOURCE})["status"] == "baseline_mismatch",
        "scores_bound_to_source_and_version": [tuple(row) for row in stored]
        == [(SOURCE["connection_id"], version, "1.0.0", 1, len(collections) - 1)],
        "inputs_exclude_other_source": (
            final["inputs"]["observation_instants"] == len(collections)
            and final["inputs"]["features"]["row_count"]["distribution"]["max"] < 50_000
        ),
        "inference_errors_counted": inference["errors"] == {"inference_unavailable": 1},
        "inference_time_measured": inference["predict_seconds"]["count"] == inference["ok"]
        == len(collections) - 1,
    }
    return {
        "scope": (
            "One process: Flask storage on throwaway SQLite, real inference adapter, "
            "real FastAPI service through its ASGI test client, #13 demo model rebuilt "
            "by the existing procedure. Only the collection clock is simulated "
            f"(one collection per {COLLECTION_MINUTES} minutes). "
            "The signal is the app.alerts log record; "
            "no notification is delivered and no external service is contacted."
        ),
        "source": SOURCE,
        "model": {"model_version": version, "demo": True, "quality_admitted": False,
                  "configuration": manifest["training"]["provenance"]["configuration"],
                  "train_data_sha256": manifest["training"]["provenance"]["train_data_sha256"]},
        "baseline": {key: baseline[key] for key in ("observations", "anomalies", "origin")},
        "rule": {**asdict(alert.rule), "baseline_rate": alert.baseline_rate,
                 "threshold_count": alert.threshold,
                 "threshold_rate": round(alert.threshold / alert.rule.window, 6)},
        "stream": {"seed": STREAM_SEED, "initial_rows": INITIAL_ROWS, "phases": dict(PHASES),
                   "deviation": "#13 incident batch (10 x 200 incident-generator rows) "
                                "loaded, rolled back at the next collection; 3 pairs", "inference_outage_tick": OUTAGE_TICK},
        "checks": checks,
        "passed": all(checks.values()),
        "events": [{"tick": t, "phase": p, "event": e} for t, p, e in events],
        "alert_log": alerts,
        "final_report": {key: final[key] for key in ("inputs", "model", "alert")},
        "timeline": timeline,
    }


def main() -> None:
    # Settings require DATABASE_URL at import; never point the demo at a real DB.
    with tempfile.TemporaryDirectory(prefix="model-alert-demo-") as tmp:
        os.environ["DATABASE_URL"] = f"sqlite:///{Path(tmp) / 'target.db'}"
        for key, value in _OFFLINE_SETTINGS.items():
            os.environ[key] = value
        result = run_demo(Path(tmp))
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
