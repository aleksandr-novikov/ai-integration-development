"""Real Compose CI verification using the existing no-cache integration image.

Host run/cleanup are restricted to an isolated Linux GitHub runner. Container
probes reuse the documented seed, warmup, registry and live-demo code paths.
Only the allowlisted JSON report is uploaded; raw commands never print env.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from scripts.demo_model import CONFIGS, REGIME

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/compose-demo/ci-report.json"
STATE = ROOT / ".git/compose-ci-state.json"
OVERRIDE = ROOT / ".git/compose-ci.json"
DEMO = ROOT / "models/demo"
# Offline result of the same generator/model, committed with the protocol.
DEMO_MODEL_EVIDENCE = ROOT / "docs/course/containers/demo-model/evaluation.json"
PREPARE_DIAGNOSTICS = ROOT / "models/.compose-diagnostics/prepare.json"
PREPARE_STAGES = ("guard", "imports", "target_seed", "demo_prepare", "summary_check",
                  "source_check", "demo_model", "config_write")
PREPARE_ERRORS = ("import_error", "permission_denied", "file_missing", "database_error",
                  "assertion_failed", "os_error", "invalid_data", "interrupted",
                  "process_exit", "unexpected_error")
DEMO_PREPARE_STAGES = ("workspace", "resolve_project", "seed_metrics", "warmup_ml", "verify_project")
# Fixed PostgreSQL codes only, not arbitrary five-character driver strings.
SQLSTATES = ("08001", "08003", "08006", "28P01", "3D000", "3F000", "42P01", "42703",
             "42883", "42804", "42P10", "42501", "23502", "23503", "23505", "23514",
             "22003", "22007", "22008", "22P02", "25P02", "40001", "40P01", "53300",
             "57014", "0A000", "XX000")
DIAGNOSTICS_LIMIT = 8192
# The CI network is internal, so published host ports are unreachable from the
# runner; HTTP checks run in a one-off app container and use service names.
APP_URL = "http://app:5001"
INFERENCE_URL = "http://inference:8000"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def read(path):
    return json.loads(path.read_text())


def environment():
    return dict(line.split("=", 1) for line in (ROOT / ".env").read_text().splitlines()
                if line and not line.startswith("#"))


def matches(pattern):
    return lambda value: type(value) is str and re.fullmatch(pattern, value) is not None


def choices(*values):
    return lambda value: type(value) is str and value in values


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def count(value):
    return type(value) is int and value >= 0


def boolean(value):
    return type(value) is bool


STAGES = (
    "clean checkout", "checkout SHA", "configuration", "Docker version", "Compose version", "validate configuration",
    "inspect image", "private files absent in image", "database readiness", "seed and warmup",
    "prepare diagnostics preflight", "demo model reproducibility",
    "Flask and FastAPI healthchecks", "app container ID", "app image ID", "inference container ID",
    "inference image ID", "normal", "incident", "stop scheduler before persistence check", "snapshot",
    "restart databases", "restart database readiness", "recreate database readiness",
    "recreate application and inference", "after-recreate", "list isolated volumes",
    "remove only runner test resources", "verification", "readiness", "dashboard",
    "dashboard after recreate", "restart persistence", "recreate persistence",
)
ERROR_TYPES = ("command_failed", "timeout", "assertion_failed", "os_error", "invalid_data", "unexpected_error")
SHA = matches(r"[a-f0-9]{40}")
IMAGE_ID = matches(r"sha256:[a-f0-9]{64}")
VERSION = matches(r"v?\d+\.\d+\.\d+")
STAMP = matches(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|\+00:00)")
BOOL = boolean
SOURCE = {"project_id": matches(r"[a-f0-9]{32}"), "connection_id": matches(r"[a-f0-9]{32}"),
          "schema_name": choices("public"), "table_name": choices("events")}
BINDING = {"source": SOURCE, "model_version": matches(r"if-[a-f0-9]{64}"), "feature_schema_version": VERSION}
DEMO_FLAGS = {"demo": lambda value: value is True, "quality_admitted": lambda value: value is False}
HEX64 = matches(r"[a-f0-9]{64}")
QUALITY = {**{key: count for key in ("normal_points", "false_alarms", "incidents", "detected", "aftermath_points",
                                     "aftermath_alarms", "scenarios", "scenarios_passed")},
           **{key: number for key in ("precision", "recall", "f1", "false_positive_rate", "event_recall",
                                      "abrupt_median_delay_ticks", "scenario_pass_rate")},
           "scenario_pass_rate_wilson95": [number]}
DEMO_MODEL = {
    "regime": choices(REGIME), "configuration": choices(*CONFIGS), "seed": count,
    "parameters": {"contamination": number, "n_estimators": count, "max_samples": count},
    "model_version": BINDING["model_version"], "training_vectors": count, "test_vectors": count,
    **{key: HEX64 for key in ("artifact_sha256", "train_data_sha256", "test_data_sha256", "scores_sha256")},
    "test": {"metrics": QUALITY, "families": {name: QUALITY for name in ("general", "compose", "readme")},
             "targets": {name: BOOL for name in ("event_recall", "false_positive_rate", "abrupt_median_delay_ticks")},
             "targets_passed": BOOL},
    "environment": {name: VERSION for name in ("python", "scikit-learn", "numpy", "skops")}, **DEMO_FLAGS,
}
REPRODUCED = ("train_data_sha256", "test_data_sha256", "scores_sha256", "test")
OBSERVATION = {**BINDING, **DEMO_FLAGS, "ts": STAMP, "score": number, "is_anomaly": BOOL, "exact_row_count": count,
               "features": {key: number for key in ("row_count", "null_rate", "d_row_count", "d_null_rate")}}
# A failed observation phase keeps only these fixed categories and integers:
# no exception text, SQL, feature values, URLs or tokens.
OBSERVE_PHASES = ("normal", "incident", "after-recreate")
OBSERVE_CHECKS = ("state", "baseline", "live_demo", "row_total", "history", "window", "order", "demo_flag",
                  "predecessor", "http_status", "response", "save")
OBSERVE_ERRORS = ("AssertionError", "CalledProcessError", "TimeoutExpired", "KeyError", "ValueError",
                  "InvalidHistoryError", "AmbiguousSourceError", "HTTPError", "OperationalError", "other")
RESPONSE_FIELDS = ("demo", "quality_admitted", "source", "observed_at", "model_version",
                   "feature_schema_version", "score", "is_anomaly")
RUN_STATUSES = ("running", "success", "warning", "failed", "skipped")
OBSERVE_DIAGNOSTICS = {
    "check": choices(*OBSERVE_CHECKS), "error_type": choices(*OBSERVE_ERRORS),
    "response_field": choices(*RESPONSE_FIELDS), "http_status": count,
    "live_demo_exit_code": lambda value: value is None or type(value) is int,
    "counts": {key: count for key in ("expected_ticks", "rows_before", "rows_after",
                                      "observations_in_window", "scores_in_window")},
    "collector_runs": {key: count for key in RUN_STATUSES},
    "unscored_statuses": [count], "inference": {"healthz": count, "model": count},
}
TABLE_DIGEST = {"rows": count, "digest": matches(r"[a-f0-9]{32}")}
SNAPSHOT = {"source": SOURCE, "fernet_decryption": BOOL,
            "target": {name: TABLE_DIGEST for name in ("users", "products", "orders", "events")},
            "store": {name: TABLE_DIGEST for name in ("metrics", "anomaly_scores", "projects", "connections")},
            "artifacts": (r"models/registry/versions/if-[a-f0-9]{64}/pipeline\.skops", matches(r"[a-f0-9]{64}"))}
DASHBOARD = {"status": count, "csrf_login": BOOL, "source": SOURCE, "version": BINDING["model_version"],
             "observed_at": STAMP, "score_visible": BOOL, "scope": choices("HTTP HTML/API, not browser rendering")}
REPORT_SCHEMA = {
    "technical_pass": BOOL, "manual_browser_and_second_participant": choices("NOT VERIFIED"),
    "checkout_sha": SHA, "pr_head_sha": SHA,
    "image_reused_from": choices("test_inference_image nocache=True"), "image_id": IMAGE_ID,
    "docker": matches(r"\d+\.\d+\.\d+ / \d+\.\d+\.\d+"), "compose": VERSION,
    "serving_image_ids": {name: IMAGE_ID for name in ("app", "inference")},
    "steps": [{"step": choices(*STAGES), "exit_code": lambda value: value is None or type(value) is int,
               "seconds": number, "error_type": choices(*ERROR_TYPES)}],
    "failure": {"stage": choices(*STAGES), "exit_code": lambda value: value is None or type(value) is int,
                "error_type": choices(*ERROR_TYPES)},
    "prepared": {"binding": BINDING, **DEMO_FLAGS, "demo_model": DEMO_MODEL, "fernet_key_shared": BOOL,
                 "warmup": {"retail-postgres": {key: count for key in (
                     "metrics", "anomaly_scores", "changepoints", "drift_reports", "notifications", "forecast_models")}}},
    "readiness": {"target_db": choices("ok"), "monitor_db": choices("ok"),
                  "monitor_backend": choices("postgresql"), "unauthorized_model_status": count,
                  "model": {**BINDING, **DEMO_FLAGS}},
    "normal": [OBSERVATION], "incident": [OBSERVATION], "after_recreate": [OBSERVATION],
    "demo_model_reproduced": {key: BOOL for key in REPRODUCED},
    "model_scenario": {"normal_points": count, "false_alarms": count, "incident_points": count,
                       "incident_detected": BOOL, "passed": BOOL},
    "dashboard": DASHBOARD, "dashboard_after_recreate": DASHBOARD,
    "restart_persistence": SNAPSHOT, "recreate_persistence": SNAPSHOT, "cleanup": choices("PASS"),
    "prepare_diagnostics": lambda value: valid_prepare_diagnostics(value),
    "observe_diagnostics": {phase: OBSERVE_DIAGNOSTICS for phase in OBSERVE_PHASES},
}
OMIT = object()


def public_report(value, schema=REPORT_SCHEMA):
    """Project only declared fields and constrained scalar values; never stringify input."""
    if isinstance(schema, dict):
        if type(value) is not dict:
            return OMIT
        return {key: result for key, child in schema.items() if key in value
                if (result := public_report(value[key], child)) is not OMIT}
    if isinstance(schema, list):
        if type(value) is not list:
            return OMIT
        return [result for item in value if (result := public_report(item, schema[0])) is not OMIT]
    if isinstance(schema, tuple):
        if type(value) is not dict:
            return OMIT
        pattern, child = schema
        return {key: result for key, item in value.items() if matches(pattern)(key)
                if (result := public_report(item, child)) is not OMIT}
    return value if schema(value) else OMIT


def save_report(value):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(public_report(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


class CommandFailure(RuntimeError):
    def __init__(self, stage, exit_code, error_type):
        super().__init__("CI command failed")
        self.stage, self.exit_code, self.error_type = stage, exit_code, error_type


def failure_fields(error, stage):
    if isinstance(error, CommandFailure):
        return {"stage": error.stage, "exit_code": error.exit_code, "error_type": error.error_type}
    category = "unexpected_error"
    for kind, name in ((AssertionError, "assertion_failed"), (subprocess.TimeoutExpired, "timeout"),
                       (OSError, "os_error"), (ValueError, "invalid_data")):
        if isinstance(error, kind):
            category = name
            break
    return {"stage": stage, "exit_code": None, "error_type": category}


def prepare_error(error):
    for kind, category in ((ImportError, "import_error"), (PermissionError, "permission_denied"),
                           (FileNotFoundError, "file_missing"), (KeyboardInterrupt, "interrupted"),
                           (SystemExit, "process_exit")):
        if isinstance(error, kind):
            return category
    module = sys.modules.get("sqlalchemy.exc")
    if module is not None and isinstance(error, module.SQLAlchemyError):
        return "database_error"
    category = failure_fields(error, "verification")["error_type"]
    return category if category in PREPARE_ERRORS else "unexpected_error"


def empty_diagnostics(availability, error_type=None):
    return {"availability": availability, "mount_verified": False, "events": [],
            "last_successful_stage": None, "failed_stage": None, "active_stage": None,
            "error_type": error_type, "demo_prepare_stage": None, "sqlstate": None}


def prepare_sqlstate(error):
    kind = getattr(sys.modules.get("sqlalchemy.exc"), "DBAPIError", ())
    if not isinstance(error, kind):
        return None
    for name in ("sqlstate", "pgcode"):
        value = getattr(error.orig, name, None)
        if type(value) is str and value in SQLSTATES:
            return value
    return None


def summarize_checkpoints(value):
    if type(value) is not dict or set(value) != {"version", "mount_verified", "events"}:
        raise ValueError("invalid preparation checkpoints")
    if type(value["version"]) is not int or value["version"] != 1 or type(value["mount_verified"]) is not bool:
        raise ValueError("invalid preparation checkpoints")
    events = value["events"]
    if type(events) is not list or len(events) > 2 * len(PREPARE_STAGES):
        raise ValueError("invalid preparation checkpoints")
    result = empty_diagnostics("recorded" if events else "not_started")
    result.update(mount_verified=value["mount_verified"], events=events)
    if events and not value["mount_verified"]:
        raise ValueError("unverified preparation exchange")
    index, active = 0, False
    for event in events:
        if type(event) is not dict or index >= len(PREPARE_STAGES) or result["failed_stage"] is not None:
            raise ValueError("invalid preparation sequence")
        if event.get("stage") != PREPARE_STAGES[index]:
            raise ValueError("invalid preparation sequence")
        status = event.get("state")
        expected = {"stage", "state", "error_type"} if status == "failed" else {"stage", "state"}
        if "substage" in event:
            if event["stage"] != "demo_prepare" or status != "started" or not choices(*DEMO_PREPARE_STAGES)(event["substage"]):
                raise ValueError("invalid preparation substage")
            expected.add("substage")
            result["demo_prepare_stage"] = event["substage"]
        if "sqlstate" in event:
            if event["stage"] != "demo_prepare" or status != "failed" or event.get("error_type") != "database_error" or not choices(*SQLSTATES)(event["sqlstate"]):
                raise ValueError("invalid preparation SQLSTATE")
            expected.add("sqlstate")
            result["sqlstate"] = event["sqlstate"]
        if set(event) != expected or status not in (("passed", "failed") if active else ("started",)):
            raise ValueError("invalid preparation sequence")
        result["active_stage"] = event["stage"] if status == "started" else None
        if status == "failed":
            if event["error_type"] not in PREPARE_ERRORS:
                raise ValueError("invalid preparation error category")
            result.update(failed_stage=event["stage"], error_type=event["error_type"])
        elif status == "passed":
            result["last_successful_stage"] = event["stage"]
            index += 1
        active = status == "started"
    return result


def valid_prepare_diagnostics(value):
    if type(value) is not dict or set(value) != set(empty_diagnostics("not_started")):
        return False
    if value["availability"] in ("invalid", "unavailable"):
        return value["error_type"] in PREPARE_ERRORS and value == empty_diagnostics(value["availability"], value["error_type"])
    try:
        return value == summarize_checkpoints({"version": 1, "mount_verified": value["mount_verified"], "events": value["events"]})
    except (ValueError, TypeError):
        return False


def unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate preparation field")
        result[key] = value
    return result


def read_checkpoints():
    directory = PREPARE_DIAGNOSTICS.parent
    if directory.is_symlink() or directory.stat().st_uid != os.geteuid() or directory.stat().st_mode & 0o077:
        raise ValueError("unsafe preparation exchange directory")
    fd = os.open(PREPARE_DIAGNOSTICS, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ValueError("unsafe preparation exchange file")
        raw = stream.read(DIAGNOSTICS_LIMIT + 1)
    if len(raw) > DIAGNOSTICS_LIMIT:
        raise ValueError("preparation checkpoints too large")
    value = json.loads(raw, object_pairs_hook=unique_fields)
    summarize_checkpoints(value)
    return value


def write_checkpoints(value):
    summarize_checkpoints(value)
    # Read validates the existing private, parent-created destination first.
    read_checkpoints()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=PREPARE_DIAGNOSTICS.parent,
                                         prefix=".prepare-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(value, allow_nan=False))
        temporary.replace(PREPARE_DIAGNOSTICS)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def create_prepare_exchange():
    models = ROOT / "models"
    if models.is_symlink() or not models.is_dir() or models.stat().st_uid != os.geteuid():
        raise ValueError("unsafe models exchange directory")
    PREPARE_DIAGNOSTICS.parent.mkdir(mode=0o700)
    fd = os.open(PREPARE_DIAGNOSTICS, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump({"version": 1, "mount_verified": False, "events": []}, stream)


def prepare_exchange_preflight():
    if os.environ.get("COMPOSE_DEMO_TEST") != "1" or not Path("/.dockerenv").exists():
        raise RuntimeError("preparation exchange requires the isolated test container")
    value = read_checkpoints()
    if value["events"] or value["mount_verified"]:
        raise ValueError("preparation exchange is not fresh")
    value["mount_verified"] = True
    write_checkpoints(value)


def collect_prepare_diagnostics():
    try:
        return summarize_checkpoints(read_checkpoints())
    except OSError as error:
        return empty_diagnostics("unavailable", prepare_error(error))
    except (ValueError, TypeError, RecursionError):
        return empty_diagnostics("invalid", "invalid_data")


class PrepareProgress:
    def __init__(self):
        self.value = read_checkpoints()
        if not self.value["mount_verified"] or self.value["events"]:
            raise ValueError("preparation exchange must pass preflight first")

    def demo_checkpoint(self, stage):
        if stage not in DEMO_PREPARE_STAGES or not self.value["events"]:
            raise ValueError("invalid demo preparation checkpoint")
        event = self.value["events"][-1]
        if event["stage"] != "demo_prepare" or event["state"] != "started":
            raise ValueError("demo preparation is not active")
        # Overwrite only a fixed phase identifier, never callback arguments/data.
        event["substage"] = stage
        write_checkpoints(self.value)

    @contextmanager
    def step(self, stage):
        self.value["events"].append({"stage": stage, "state": "started"})
        write_checkpoints(self.value)
        try:
            yield
        except BaseException as error:
            failed = {"stage": stage, "state": "failed", "error_type": prepare_error(error)}
            if stage == "demo_prepare" and failed["error_type"] == "database_error":
                sqlstate = prepare_sqlstate(error)
                if sqlstate is not None:
                    failed["sqlstate"] = sqlstate
            self.value["events"].append(failed)
            # Preserve the original failure if the exchange became unwritable.
            with suppress(OSError, ValueError):
                write_checkpoints(self.value)
            raise
        else:
            self.value["events"].append({"stage": stage, "state": "passed"})
            write_checkpoints(self.value)


def runner_only():
    if os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("RUNNER_OS") != "Linux":
        raise RuntimeError("run/cleanup require an isolated Linux GitHub Actions runner")
    if not os.environ.get("GITHUB_RUN_ID"):
        raise RuntimeError("missing runner run identity")


def outcome(normal, incident):
    return {
        "normal_points": len(normal),
        "false_alarms": sum(bool(row["is_anomaly"]) for row in normal),
        "incident_points": len(incident),
        "incident_detected": any(row["is_anomaly"] for row in incident),
        "passed": bool(normal and incident)
        and not any(row["is_anomaly"] for row in normal)
        and any(row["is_anomaly"] for row in incident),
    }


def container_state():
    if os.environ.get("COMPOSE_DEMO_TEST") != "1" or not Path("/.dockerenv").exists():
        raise RuntimeError("container probes require the isolated Compose test override")
    from app import crypto
    from app import metrics_storage as ms
    from app.config import settings

    assert urlparse(settings.DATABASE_URL).hostname == "postgres"
    assert urlparse(settings.MONITOR_DB_URL).hostname == "timescaledb"
    assert ms.get_engine().dialect.name == "postgresql"
    for name in ("NIM_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        assert not os.environ.get(name), "external services must be disabled"
    user = ms.get_user_by_email("demo@dbmonitor.app")
    project = ms.get_project_by_slug(user["id"], "retail-postgres")
    connection = next(c for c in ms.list_connections_for_project(project["id"])
                      if c["name"] == "Local Postgres")
    assert crypto.decrypt_dsn(connection["dsn_encrypted"]) == settings.DATABASE_URL
    return {"project_id": project["id"], "connection_id": connection["id"],
            "schema_name": connection["schema_name"], "table_name": "events"}


def prepare():
    progress = PrepareProgress()
    with progress.step("guard"):
        # Only private diagnostics precede the guard; no DB/model writes do.
        if os.environ.get("COMPOSE_DEMO_TEST") != "1" or not Path("/.dockerenv").exists():
            raise RuntimeError("preparation is restricted to the isolated Compose test")
    with progress.step("imports"):
        from app import metrics_storage as ms
        from scripts import demo_model, demo_prepare, seed_target_db

    with progress.step("target_seed"):
        seed_target_db.main(n_users=500, n_products=50, n_orders=1000, n_events=2000)
    with progress.step("demo_prepare"):
        summary = demo_prepare.main(checkpoint=progress.demo_checkpoint)
    with progress.step("summary_check"):
        assert summary["retail-postgres"]["metrics"] > 0
        assert summary["retail-postgres"]["anomaly_scores"] > 0
        assert summary["retail-postgres"]["forecast_models"] > 0
    with progress.step("source_check"):
        source = container_state()
        for email in ("demo@dbmonitor.app", "lake@dbmonitor.app"):
            for project in ms.list_projects_for_user(ms.get_user_by_email(email)["id"]):
                if project["id"] != source["project_id"]:
                    for connection in ms.list_connections_for_project(project["id"]):
                        ms.set_connection_active(project["id"], connection["id"], False)
    with progress.step("demo_model"):
        # Separate demo-only model of the stand's own regime; #9 stays archived.
        sha = os.environ.get("APP_VERSION", "")
        assert re.fullmatch(r"[a-f0-9]{40}", sha), "image must carry the checked commit SHA"
        manifest, evaluation = demo_model.train_and_evaluate(ROOT / "models/registry", source, sha)
    with progress.step("config_write"):
        save(DEMO / "evaluation.json", evaluation)
        binding = {"source": source, **{k: manifest[k] for k in ("model_version", "feature_schema_version")}}
        save(DEMO / "service.json", {"registry_path": "models/registry", "demo": True,
             "principals": [{"token_env": "INFERENCE_ADAPTER_TOKEN", "sources": [source]}]})
        save(DEMO / "adapter.json", {"url": "http://inference:8000", "demo": True,
             "token_env": "INFERENCE_ADAPTER_TOKEN", "bindings": [binding]})
        save(DEMO / "prepared.json", {"binding": binding, "warmup": summary, "demo": True,
             "quality_admitted": False, "fernet_key_shared": True,
             "demo_model": {key: value for key, value in evaluation.items() if key in DEMO_MODEL}})


def reproduced(actual):
    """Same seeds and code must give the committed data; scores also need equal libraries."""
    expected = read(DEMO_MODEL_EVIDENCE)
    result = {key: actual[key] == expected[key] for key in REPRODUCED}
    assert result["train_data_sha256"] and result["test_data_sha256"], "demo data differ from the recorded generator"
    return result


def table_count(source):
    from sqlalchemy import text

    from app import db

    # Fixed identifiers only: the verified demo source is public.events.
    assert (source["schema_name"], source["table_name"]) == ("public", "events")
    with db.get_engine().connect() as conn:
        return conn.execute(text("SELECT count(*) FROM public.events")).scalar_one()


def observe(phase):
    """Run one phase; on failure keep only fixed categories and counts beside it."""
    diagnostics, context = {"check": "state"}, {}
    try:
        _observe(phase, diagnostics, context)
    except BaseException as error:
        name = type(error).__name__
        diagnostics["error_type"] = name if name in OBSERVE_ERRORS else "other"
        if isinstance(error, subprocess.CalledProcessError):
            diagnostics["live_demo_exit_code"] = error.returncode
        with suppress(Exception):
            observe_evidence(context, diagnostics)
        with suppress(Exception):
            save(DEMO / (phase + "-diagnostics.json"), public_report(diagnostics, OBSERVE_DIAGNOSTICS))
        raise


def observe_evidence(context, diagnostics):
    """Independent best-effort probes after a failure; none may hide the original error."""
    from app import source_storage

    counts = diagnostics.setdefault("counts", {})
    if "batches" in context:
        counts["expected_ticks"] = len(context["batches"])
    if "before" in context:
        counts["rows_before"] = context["before"]
    source, binding, started = (context.get(key) for key in ("source", "binding", "started"))
    if source is None or binding is None or started is None:
        return
    with suppress(Exception):
        counts["rows_after"] = table_count(source)
    history = {}
    with suppress(Exception):
        # Quarantine mode reads past duplicates instead of raising, like score_source.
        for name in ("row_count", "null_rate"):
            history[name] = {row["ts"]: row["value"] for row in source_storage.metrics(source, name, quarantine=True)}
    aligned = sorted(history["row_count"].keys() & history["null_rate"].keys()) if len(history) == 2 else []
    window = [stamp for stamp in aligned if stamp >= started]
    if len(history) == 2:
        counts["observations_in_window"] = len(window)
    scored = None
    with suppress(Exception):
        scored = {row["ts"] for row in source_storage.scores(source, binding["model_version"],
                                                              binding["feature_schema_version"])}
        counts["scores_in_window"] = sum(stamp >= started for stamp in scored)
    with suppress(Exception):
        diagnostics["collector_runs"] = collector_runs(source, started)
    with suppress(Exception):
        inference_evidence(context, diagnostics, aligned, window, scored or set(), history)


def collector_runs(source, started):
    from sqlalchemy import text

    from app import metrics_storage as ms

    with ms.get_engine().connect() as conn:
        rows = conn.execute(text(
            "SELECT status, count(*) FROM collector_runs WHERE project_id = :project_id "
            "AND connection_id = :connection_id AND started_at >= :started GROUP BY status"),
            {"project_id": source["project_id"], "connection_id": source["connection_id"],
             "started": started}).fetchall()
    return {status: int(total) for status, total in rows if status in RUN_STATUSES}


def inference_evidence(context, diagnostics, aligned, window, scored, history):
    """Status codes only: service health, model access and the unscored window pairs.

    0 means no HTTP response (connection refused, DNS or timeout) or no valid pair.
    """
    import httpx

    from ml.features import FEATURE_NAMES, build_vector

    source, binding = context["source"], context["binding"]
    url = read(DEMO / "adapter.json")["url"]
    headers = {"Authorization": "Bearer " + os.environ["INFERENCE_ADAPTER_TOKEN"]}
    params = {**source, **{key: binding[key] for key in ("model_version", "feature_schema_version")}}

    def status(method, path, **kwargs):
        try:
            return getattr(client, method)(url + path, **kwargs).status_code
        except httpx.HTTPError:
            return 0

    with httpx.Client(timeout=20, trust_env=False) as client:
        diagnostics["inference"] = {"healthz": status("get", "/healthz"),
                                    "model": status("get", "/v1/model", params=params, headers=headers)}
        statuses = []
        for stamp in [stamp for stamp in window if stamp not in scored][:10]:
            position = aligned.index(stamp)
            if position == 0:
                statuses.append(0)  # no predecessor: nothing to request
                continue
            previous = aligned[position - 1]
            try:
                vector = build_vector(history["row_count"][stamp], history["null_rate"][stamp],
                                      history["row_count"][previous], history["null_rate"][previous])
            except ValueError:
                statuses.append(0)  # quarantined duplicate: no valid pair
                continue
            payload = {**binding, "features": dict(zip(FEATURE_NAMES, vector, strict=True)),
                       "observed_at": stamp, "previous_observed_at": previous}
            statuses.append(status("post", "/v1/predict", json=payload, headers=headers))
        diagnostics["unscored_statuses"] = statuses


def _observe(phase, diagnostics, context):
    from app import source_storage
    from ml.features import FEATURE_NAMES, build_rows
    from scripts.live_demo import _BASE_ROWS_PER_TICK, _INCIDENT_ROW_MULTIPLIER

    source = context["source"] = container_state()
    binding = context["binding"] = read(DEMO / "prepared.json")["binding"]
    # live_demo inserts each batch, then collects and scores that tick once.
    batches = context["batches"] = ([_BASE_ROWS_PER_TICK * _INCIDENT_ROW_MULTIPLIER] if phase == "incident"
                                    else [_BASE_ROWS_PER_TICK] * 3)
    if phase == "normal":
        diagnostics["check"] = "baseline"
        # Seeded history is stored without this connection's source key, so the
        # first tick would have no predecessor and no score. One existing
        # collector pass over the unchanged table provides that baseline; a
        # single observation has no pair, so it is never scored itself.
        from collectors.per_project import collect_for_connection

        collect_for_connection(source["project_id"], source["connection_id"])
        baseline = [{row["ts"] for row in source_storage.metrics(source, name)} for name in ("row_count", "null_rate")]
        assert len(baseline[0]) == 1 and baseline[0] == baseline[1], "baseline observation missing"
    diagnostics["check"] = "row_total"
    before = context["before"] = table_count(source)
    # Same canonical form as stored score timestamps, so string order is time order.
    started = context["started"] = datetime.now(UTC).isoformat(timespec="microseconds")
    command = [sys.executable, "-m", "scripts.live_demo", "--project-id", source["project_id"],
               "--connection-id", source["connection_id"], "--interval", "1",
               "--ticks", str(len(batches))]
    if phase == "incident":
        command += ["--incident-at", "1"]
    diagnostics["check"] = "live_demo"
    subprocess.run(command, check=True, timeout=180)
    finished = datetime.now(UTC).isoformat(timespec="microseconds")
    exact = list(itertools.accumulate(batches, initial=before))[1:]
    diagnostics["check"] = "row_total"
    assert table_count(source) == exact[-1], "inserted rows differ from the phase batches"
    diagnostics["check"] = "history"
    rc, nr = (source_storage.metrics(source, name) for name in ("row_count", "null_rate"))
    stamps, vectors = build_rows(rc, nr)
    features = {stamp: dict(zip(FEATURE_NAMES, vector, strict=True))
                for stamp, vector in zip(stamps, vectors, strict=True)}
    rows = source_storage.scores(source, binding["model_version"], binding["feature_schema_version"])
    new = [{**row, "features": features[row["ts"]], "quality_admitted": False}
           for row in rows if started <= row["ts"] <= finished]
    # Exactly one score per tick of this run, in tick order. row_count comes from
    # n_live_tup, an estimate, so the exact COUNT(*) after the tick is stored next
    # to it instead of being compared with it or used to choose the phase.
    diagnostics["check"] = "window"
    assert len(new) == len(batches), "phase observations differ from its ticks"
    diagnostics["check"] = "order"
    assert all(a["ts"] < b["ts"] for a, b in itertools.pairwise(new)), "observations out of order"
    for row, count in zip(new, exact, strict=True):
        row["exact_row_count"] = count
    diagnostics["check"] = "demo_flag"
    assert all(row["demo"] for row in new), "new versioned observations missing"
    # Re-run the same selected service on these exact features and compare the
    # response to persisted results; never manufacture is_anomaly from injection.
    import httpx

    config = read(DEMO / "adapter.json")
    aligned = sorted({r["ts"] for r in rc} & {r["ts"] for r in nr})
    with httpx.Client(timeout=20, trust_env=False) as client:
        for row in new:
            diagnostics["check"] = "predecessor"
            position = aligned.index(row["ts"])
            assert position > 0, "observation without a preceding collection"
            payload = {**binding, "features": row["features"], "observed_at": row["ts"],
                       "previous_observed_at": aligned[position - 1]}
            diagnostics["check"] = "http_status"
            response = client.post(config["url"] + "/v1/predict", json=payload,
                                   headers={"Authorization": "Bearer " + os.environ["INFERENCE_ADAPTER_TOKEN"]})
            diagnostics["http_status"] = response.status_code
            assert response.status_code == 200
            diagnostics["check"] = "response"
            actual = response.json()
            expected = {"demo": True, "quality_admitted": False, "source": source, "observed_at": row["ts"],
                        **{key: binding[key] for key in ("model_version", "feature_schema_version")},
                        "score": row["score"], "is_anomaly": row["is_anomaly"]}
            for field in RESPONSE_FIELDS:
                diagnostics["response_field"] = field
                # Identity, not equality, for the two demo flags, as before.
                same = (actual[field] is expected[field] if field in ("demo", "quality_admitted")
                        else actual[field] == expected[field])
                assert same, "service result differs from the stored observation"
            diagnostics.pop("response_field")
    diagnostics["check"] = "save"
    save(DEMO / (phase + ".json"), new)


def snapshot():
    from sqlalchemy import text

    from app import db
    from app import metrics_storage as ms

    source = container_state()
    result = {"source": source, "fernet_decryption": True, "target": {}, "store": {}}
    for engine, tables, target in (
        (db.get_engine(), ("users", "products", "orders", "events"), result["target"]),
        (ms.get_engine(), ("metrics", "anomaly_scores", "projects", "connections"), result["store"]),
    ):
        with engine.connect() as conn:
            for table in tables:
                # Fixed identifiers only; order and hash the complete test rows.
                row = conn.execute(text(f"SELECT count(*), md5(COALESCE(string_agg(row_to_json(t)::text, '' ORDER BY row_to_json(t)::text), '')) FROM {table} t")).one()
                target[table] = {"rows": row[0], "digest": row[1]}
    result["artifacts"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (ROOT / "models/registry").rglob("*.skops")}
    assert result["artifacts"]
    save(DEMO / "snapshot.json", result)


def require_test_container():
    if os.environ.get("COMPOSE_DEMO_TEST") != "1" or not Path("/.dockerenv").exists():
        raise RuntimeError("HTTP probes require the isolated Compose test container")


def readiness():
    require_test_container()
    import httpx

    binding = read(DEMO / "prepared.json")["binding"]
    with httpx.Client(timeout=30, trust_env=False) as client:
        health = client.get(APP_URL + "/healthz")
        assert health.status_code == 200
        checks = health.json()["checks"]
        assert checks["target_db"]["status"] == checks["monitor_db"]["status"] == "ok"
        assert checks["monitor_db"]["backend"] == "postgresql"
        url = INFERENCE_URL + "/v1/model"
        params = {**binding["source"], **{k: binding[k] for k in ("model_version", "feature_schema_version")}}
        assert client.get(url, params=params).status_code == 401
        model = client.get(url, params=params, headers={"Authorization": "Bearer " + os.environ["INFERENCE_ADAPTER_TOKEN"]})
        assert model.status_code == 200
        assert model.json()["demo"] is True and model.json()["quality_admitted"] is False
    body = model.json()
    # The parent compares the version with the checked SHA; only contract fields are kept.
    save(DEMO / "readiness.json", {"version": health.json()["version"], "target_db": "ok", "monitor_db": "ok",
                                   "monitor_backend": "postgresql", "unauthorized_model_status": 401,
                                   "model": {key: body[key] for key in {**BINDING, **DEMO_FLAGS} if key in body}})


def dashboard(phase):
    require_test_container()
    import httpx

    if phase not in ("incident", "after-recreate"):
        raise ValueError("unknown dashboard phase")
    rows = read(DEMO / (phase + ".json"))
    binding = read(DEMO / "prepared.json")["binding"]
    source = binding["source"]
    with httpx.Client(base_url=APP_URL, follow_redirects=True, timeout=30, trust_env=False) as client:
        login = client.get("/auth/login")
        assert login.status_code == 200
        token = re.search(r'name="csrf_token"[^>]*value="([^"]+)"', login.text).group(1)
        logged = client.post("/auth/login", data={"email": "demo@dbmonitor.app", "password": "demo12345", "csrf_token": token})
        assert logged.status_code == 200 and "/auth/login" not in str(logged.url)
        switched = client.post("/projects/retail-postgres/switch", data={"csrf_token": token})
        assert switched.status_code == 200
        page = client.get("/dashboard/inference")
        assert page.status_code == 200
        assert all(value in page.text for value in (source["connection_id"], binding["model_version"], rows[-1]["ts"], str(rows[-1]["score"])))
        params = {**source, **{k: binding[k] for k in ("model_version", "feature_schema_version")}}
        response = client.get("/api/inference/scores", params=params)
        assert response.status_code == 200
        assert any(r["ts"] == rows[-1]["ts"] and r["score"] == rows[-1]["score"] for r in response.json())
    save(DEMO / ("dashboard-" + phase + ".json"),
         {"status": 200, "csrf_login": True, "source": source, "version": binding["model_version"],
          "observed_at": rows[-1]["ts"], "score_visible": True, "scope": "HTTP HTML/API, not browser rendering"})


class ComposeRun:
    def __init__(self):
        runner_only()
        self.values = environment() if (ROOT / ".env").exists() else {}
        self.report = {"technical_pass": False, "steps": [], "manual_browser_and_second_participant": "NOT VERIFIED"}
        self.stage = "verification"

    def command(self, label, args, timeout=180):
        self.stage = label
        started = time.monotonic()
        try:
            process = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            self.report["steps"].append({"step": label, "exit_code": None, "error_type": "timeout",
                                         "seconds": round(time.monotonic() - started, 2)})
            save_report(self.report)
            raise CommandFailure(label, None, "timeout") from None
        self.report["steps"].append({"step": label, "exit_code": process.returncode,
                                     "seconds": round(time.monotonic() - started, 2)})
        save_report(self.report)
        if process.returncode:
            raise CommandFailure(label, process.returncode, "command_failed")
        return process.stdout

    def compose(self, label, *args, timeout=180):
        project = self.values["COMPOSE_PROJECT_NAME"]
        return self.command(label, ["docker", "compose", "-p", project, "-f", "docker-compose.yml",
                                    "-f", str(OVERRIDE.relative_to(ROOT)), "--profile", "inference", *args], timeout)

    def probe(self, mode, label=None):
        self.compose(label or mode, "run", "--rm", "-T", "--no-deps", "app", "python", "-m",
                     "scripts.verify_compose_demo", "inside-" + mode, timeout=600)

    def observe_probe(self, phase):
        try:
            self.probe(phase)
        finally:
            # Written by the failed container phase into the shared models mount.
            path = DEMO / (phase + "-diagnostics.json")
            if path.is_file():
                self.report.setdefault("observe_diagnostics", {})[phase] = read(path)
                save_report(self.report)

    def prepare_in_container(self):
        self.stage = "prepare diagnostics preflight"
        try:
            # Existing ./models:/app/models RW bind, same APP_UID/GID on both sides.
            # A round trip proves visibility and atomic replacement before seeding.
            create_prepare_exchange()
            self.compose("prepare diagnostics preflight", "run", "--rm", "-T", "--no-deps",
                         "app", "python", "-m", "scripts.verify_compose_demo", "inside-prepare-preflight")
            value = read_checkpoints()
            assert value["mount_verified"] and not value["events"]
            # Seed, warmup, demo-model training and its one-time test evaluation.
            self.compose("seed and warmup", "run", "--rm", "-T", "--no-deps",
                         "app", "python", "-m", "scripts.verify_compose_demo", "inside-prepare", timeout=900)
            diagnostics = collect_prepare_diagnostics()
            assert diagnostics["last_successful_stage"] == "config_write"
            assert diagnostics["failed_stage"] is None and diagnostics["active_stage"] is None
        finally:
            # Also runs on command failure/timeout, including before the first checkpoint.
            self.report["prepare_diagnostics"] = collect_prepare_diagnostics()
            save_report(self.report)

    def run(self):
        from scripts.init_compose_demo import create_environment

        assert not (ROOT / ".env").exists(), "CI requires a fresh checkout without .env"
        assert not self.command("clean checkout", ["git", "status", "--porcelain"]).strip()
        sha = self.command("checkout SHA", ["git", "rev-parse", "HEAD"]).strip()
        assert sha == os.environ["COMPOSE_SMOKE_SHA"]
        self.stage = "configuration"
        create_environment(ROOT / ".env", version=sha, uid=os.getuid(), gid=os.getgid())
        self.values = environment()
        image = os.environ["COMPOSE_SMOKE_IMAGE"]
        with (ROOT / ".env").open("a") as stream:
            stream.write(f"\nAPP_IMAGE={image}\nINFERENCE_ADAPTER_CONFIG=/app/models/demo/adapter.json\n")
        self.values = environment()
        save(STATE, {"project": self.values["COMPOSE_PROJECT_NAME"], "run_id": os.environ["GITHUB_RUN_ID"]})
        services = {name: {"cpus": 1, "mem_limit": memory, "restart": "no"} for name, memory in
                    (("app", "2g"), ("inference", "1g"), ("postgres", "512m"), ("timescaledb", "768m"))}
        services["app"]["environment"] = {"COMPOSE_DEMO_TEST": "1"}
        save(OVERRIDE, {"services": services, "networks": {"default": {"internal": True}}})
        self.report.update(checkout_sha=sha, pr_head_sha=os.environ.get("COMPOSE_SMOKE_PR_HEAD"),
                           image_reused_from="test_inference_image nocache=True",
                           docker=self.command("Docker version", ["docker", "version", "--format", "{{.Client.Version}} / {{.Server.Version}}"]).strip(),
                           compose=self.command("Compose version", ["docker", "compose", "version", "--short"]).strip())
        self.compose("validate configuration", "config", "--quiet")
        image_info = json.loads(self.command("inspect image", ["docker", "image", "inspect", image]))[0]
        assert "APP_VERSION=" + sha in image_info["Config"]["Env"]
        assert not any(secret in str(image_info) for key in ("SECRET_KEY", "FERNET_KEY", "INFERENCE_ADAPTER_TOKEN") if (secret := self.values[key]))
        self.report["image_id"] = image_info["Id"]
        check_image = "from pathlib import Path; assert all(not Path('/app', p).exists() for p in ('.env', '.env.local', 'backups', 'models/registry', 'docs/course/experiments')); assert Path('/app/docs/course/api/inference.openapi.yaml').is_file()"
        self.command("private files absent in image", ["docker", "run", "--rm", "--network", "none", image, "python", "-c", check_image])
        self.compose("database readiness", "up", "-d", "--no-build", "--wait", "--wait-timeout", "120", "postgres", "timescaledb")
        self.prepare_in_container()
        self.report["prepared"] = read(DEMO / "prepared.json")
        self.stage = "demo model reproducibility"
        self.report["demo_model_reproduced"] = reproduced(self.report["prepared"]["demo_model"])
        save_report(self.report)
        self.compose("Flask and FastAPI healthchecks", "up", "-d", "--no-build", "--wait", "--wait-timeout", "150", "inference", "app")
        self.report["serving_image_ids"] = {}
        for service in ("app", "inference"):
            cid = self.compose(service + " container ID", "ps", "-q", service).strip()
            actual_image = self.command(service + " image ID", ["docker", "inspect", "--format", "{{.Image}}", cid]).strip()
            assert actual_image == image_info["Id"]
            self.report["serving_image_ids"][service] = actual_image
        self.probe("readiness")
        ready = read(DEMO / "readiness.json")
        assert ready.pop("version") == sha
        self.report["readiness"] = ready
        for phase in ("normal", "incident"):
            self.observe_probe(phase)
            self.report[phase] = read(DEMO / (phase + ".json"))
            save_report(self.report)
        self.report["model_scenario"] = outcome(self.report["normal"], self.report["incident"])
        self.probe("dashboard-incident", "dashboard")
        self.report["dashboard"] = read(DEMO / "dashboard-incident.json")
        self.compose("stop scheduler before persistence check", "stop", "app", "inference")
        self.probe("snapshot")
        before = read(DEMO / "snapshot.json")
        for action in ("restart", "recreate"):
            if action == "restart":
                self.compose("restart databases", "restart", "postgres", "timescaledb")
            args = ["up", "-d", "--no-build", "--wait", "--wait-timeout", "120"]
            if action == "recreate":
                args += ["--force-recreate"]
            self.compose(action + " database readiness", *args, "postgres", "timescaledb")
            self.probe("snapshot")
            after = read(DEMO / "snapshot.json")
            self.stage = action + " persistence"
            assert before == after, action + " changed test data/artifact"
            self.report[action + "_persistence"] = after
        self.compose("recreate application and inference", "up", "-d", "--no-build", "--force-recreate", "--wait", "--wait-timeout", "150", "app", "inference")
        self.observe_probe("after-recreate")
        self.report["after_recreate"] = read(DEMO / "after-recreate.json")
        self.probe("dashboard-after-recreate", "dashboard after recreate")
        self.report["dashboard_after_recreate"] = read(DEMO / "dashboard-after-recreate.json")
        self.report["technical_pass"] = True


def cleanup():
    runner_only()
    if not STATE.exists():
        return
    state = read(STATE)
    assert state["run_id"] == os.environ["GITHUB_RUN_ID"]
    assert re.fullmatch(r"course13-[a-f0-9]{12}", state["project"])
    run = ComposeRun()
    assert run.values["COMPOSE_PROJECT_NAME"] == state["project"]
    report = read(REPORT)
    run.report = report
    volumes = run.command("list isolated volumes", ["docker", "volume", "ls", "--filter",
                          "label=com.docker.compose.project=" + state["project"], "--format", "{{.Name}}"])
    assert set(volumes.split()) <= {state["project"] + "_pg_data", state["project"] + "_timescale_data"}
    try:
        run.compose("remove only runner test resources", "down", "--volumes", "--remove-orphans", timeout=120)
        report["cleanup"] = "PASS"
    finally:
        save_report(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "cleanup", "check-outcome", "inside-prepare-preflight", "inside-prepare",
                                         "inside-readiness", "inside-normal", "inside-incident", "inside-dashboard-incident",
                                         "inside-after-recreate", "inside-dashboard-after-recreate", "inside-snapshot"))
    mode = parser.parse_args().mode
    if mode.startswith("inside-"):
        action = mode.removeprefix("inside-")
        if action == "prepare-preflight":
            prepare_exchange_preflight()
        elif action == "prepare":
            prepare()
        elif action == "snapshot":
            snapshot()
        elif action == "readiness":
            readiness()
        elif action.startswith("dashboard-"):
            dashboard(action.removeprefix("dashboard-"))
        else:
            observe(action)
    elif mode == "cleanup":
        cleanup()
    elif mode == "check-outcome":
        result = public_report(read(REPORT))
        quality = result.get("prepared", {}).get("demo_model", {}).get("test", {})
        print(json.dumps(result.get("model_scenario", {})))
        print(json.dumps({"demo_model_test_targets_passed": quality.get("targets_passed")}))
        if not result.get("technical_pass") or not result.get("model_scenario", {}).get("passed"):
            raise SystemExit("Actual demo model did not confirm both normal and incident states; no calibration performed")
        if quality.get("targets_passed") is not True:
            raise SystemExit("Demo model did not meet the declared targets on its independent test split")
    else:
        run = ComposeRun()
        try:
            run.run()
        except Exception as exc:
            run.report["failure"] = failure_fields(exc, run.stage)
            raise SystemExit("Compose verification failed; see sanitized ci-report.json") from None
        finally:
            save_report(run.report)
        print("Compose technical checks PASS; actual model result:", public_report(run.report)["model_scenario"])


if __name__ == "__main__":
    main()
