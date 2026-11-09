"""HTTP access log and the HTTP 5xx alert rule (#15)."""

from __future__ import annotations

import json
import logging

import pytest

from app.http_alert import HttpErrorAlert, HttpErrorRule, Snapshot, snapshot_from_metrics
from app.logging_setup import AppLoggedRequestHandler, JsonFormatter

METRICS_TEXT = """\
# HELP http_requests_total Total HTTP requests served.
# TYPE http_requests_total counter
http_requests_total{endpoint="health",method="GET",status="200"} 7.0
http_requests_total{endpoint="health",method="GET",status="503"} 2.0
http_requests_total{endpoint="unknown",method="GET",status="404"} 3.0
http_requests_total{endpoint="metrics",method="GET",status="500"} 50.0
# HELP failed_login_attempts_total Failed logins.
# TYPE failed_login_attempts_total counter
failed_login_attempts_total 4.0
"""


def test_snapshot_counts_5xx_and_skips_metrics_endpoint():
    snap = snapshot_from_metrics(METRICS_TEXT, 0, HttpErrorRule())
    assert (snap.requests, snap.errors) == (12, 2)


def _feed(alert, points):
    return [alert.observe(Snapshot(ts, req, err)) for ts, req, err in points]


def test_rule_waits_for_minimum_requests():
    alert = HttpErrorAlert(HttpErrorRule(min_requests=10))
    results = _feed(alert, [(0, 0, 0), (30, 9, 9)])
    assert results[-1].condition is None
    assert alert.state == "inactive"


def test_rule_fires_above_threshold_and_resolves_when_errors_leave_window():
    alert = HttpErrorAlert(HttpErrorRule(threshold=0.1, window_seconds=300, min_requests=10))
    results = _feed(alert, [
        (0, 0, 0),
        (150, 20, 2),    # 10 % — not above the threshold
        (300, 30, 4),    # 4/30 = 13 %
        (450, 40, 4),    # 2/20 within the window: still 10 % → resolves
    ])
    assert [r.event for r in results] == [None, None, "firing", "resolved"]
    assert results[1].condition is False
    assert results[2].ratio == pytest.approx(0.1333)
    assert alert.state == "inactive"


def test_insufficient_data_keeps_firing_state():
    alert = HttpErrorAlert(HttpErrorRule(min_requests=10, window_seconds=60))
    _feed(alert, [(0, 0, 0), (30, 10, 5)])
    assert alert.state == "firing"
    result = alert.observe(Snapshot(120, 12, 5))  # no earlier snapshot in the window
    assert result.condition is None and result.state == "firing"


def test_counter_reset_drops_history():
    alert = HttpErrorAlert(HttpErrorRule(min_requests=1))
    _feed(alert, [(0, 100, 0), (30, 120, 0)])
    result = alert.observe(Snapshot(60, 5, 5))  # process restarted
    assert result.requests == 0 and result.condition is None


def test_error_counter_reset_does_not_resolve_falsely():
    # After a restart the total already exceeds the old value, errors do not.
    alert = HttpErrorAlert(HttpErrorRule(threshold=0.1, window_seconds=300, min_requests=10))
    results = _feed(alert, [(0, 100, 80), (30, 120, 100), (60, 140, 70)])
    assert [r.event for r in results] == [None, "firing", None]
    assert results[-1].condition is None and results[-1].errors == 0
    assert alert.state == "firing"
    # Evaluation resumes on fresh snapshots only.
    resumed = _feed(alert, [(90, 145, 71), (120, 160, 72)])
    assert [r.condition for r in resumed] == [None, False]
    assert resumed[-1].event == "resolved"
    assert (resumed[-1].requests, resumed[-1].errors) == (20, 2)


def test_signal_is_logged_with_structured_fields(caplog):
    alert = HttpErrorAlert(HttpErrorRule(min_requests=1))
    with caplog.at_level(logging.INFO, logger="app.alerts"):
        _feed(alert, [(0, 0, 0), (30, 10, 5), (60, 60, 5)])  # 50 %, then 5/60
    events = [(r.levelname, r.event) for r in caplog.records]
    assert events == [("WARNING", "alert_firing"), ("INFO", "alert_resolved")]
    assert caplog.records[0].errors == 5 and caplog.records[0].requests == 10


@pytest.fixture
def client(tmp_path, monkeypatch):
    import app.metrics_storage as storage
    from app.app import create_app

    monkeypatch.setattr(storage.settings, "MONITOR_DB_URL", f"sqlite:///{tmp_path / 'm.db'}")
    monkeypatch.setattr(storage, "_engine", None)
    monkeypatch.setattr(storage, "_initialized", False)
    return create_app({"TESTING": True}).test_client()


class _JsonLines(logging.Handler):
    """Format at emit time — request_id is read from the live request context."""

    def __init__(self):
        super().__init__()
        self.setFormatter(JsonFormatter())
        self.lines = []

    def emit(self, record):
        self.lines.append(self.format(record))


def test_access_log_uses_endpoint_name_not_raw_path(client):
    http_logger = logging.getLogger("app.http")
    capture = _JsonLines()
    http_logger.addHandler(capture)
    try:
        resp = client.get(
            "/auth/reset-password/secret-reset-token-123?token=secret-query-456",
            headers={"X-Request-Id": "rid-15"},
        )
    finally:
        http_logger.removeHandler(capture)
    [line] = capture.lines
    payload = json.loads(line)
    assert payload["event"] == "http_request"
    assert payload["endpoint"] == "auth.reset_password"
    assert payload["status"] == resp.status_code
    assert payload["request_id"] == "rid-15"
    assert payload["duration_ms"] >= 0
    assert "secret-" not in line


def test_server_errors_are_logged_at_error_level(client, caplog, monkeypatch):
    from sqlalchemy import create_engine

    monkeypatch.setattr("app.db._engine", create_engine("sqlite:////nonexistent-dir/t.db"))
    with caplog.at_level(logging.INFO, logger="app.http"):
        assert client.get("/healthz").status_code == 503
    record = next(r for r in caplog.records if r.name == "app.http")
    assert record.levelname == "ERROR" and record.status == 503


@pytest.mark.parametrize("log_format", ["json", "text"])
def test_unhandled_exception_log_names_endpoint_not_token_path(tmp_path, monkeypatch, log_format):
    import io

    import app.metrics_storage as storage
    from app.app import create_app
    from app.config import settings

    monkeypatch.setattr(settings, "LOG_FORMAT", log_format)
    monkeypatch.setattr(storage.settings, "MONITOR_DB_URL", f"sqlite:///{tmp_path / 'm.db'}")
    monkeypatch.setattr(storage, "_engine", None)
    monkeypatch.setattr(storage, "_initialized", False)

    def broken_lookup(token_hash):
        raise RuntimeError("synthetic storage failure token=fake-exc-token-15")

    monkeypatch.setattr(storage, "get_active_password_reset_token", broken_lookup)
    # PROPAGATE_EXCEPTIONS=False — production path: Flask logs and returns 500.
    app = create_app({"TESTING": True, "PROPAGATE_EXCEPTIONS": False})
    # Keep the real root handler (formatter + scrubbing), only redirect its stream.
    [handler] = [h for h in logging.getLogger().handlers if getattr(h, "_dbmon_managed", False)]
    out = io.StringIO()
    handler.setStream(out)

    resp = app.test_client().get(
        "/auth/reset-password/fake-path-token-15?token=fake-query-token-15",
        headers={"X-Request-Id": "rid-exc-15"},
    )
    log = out.getvalue()

    assert resp.status_code == 500
    assert b"Internal Server Error" in resp.data
    assert "fake-path-token" not in log and "fake-query-token" not in log
    assert "fake-exc-token" not in log  # exception text inside the traceback
    assert "/auth/reset-password" not in log
    assert "Traceback (most recent call last)" in log
    assert "RuntimeError: synthetic storage failure" in log
    if log_format == "json":
        records = [json.loads(line) for line in log.splitlines() if line.startswith("{")]
        [exc] = [r for r in records if r["message"].startswith("Exception on")]
        [http] = [r for r in records if r["logger"] == "app.http"]
        assert exc["level"] == "ERROR"
        assert exc["message"] == "Exception on auth.reset_password [GET]"
        assert "synthetic storage failure" in exc["exc_info"]
        assert (http["level"], http["status"], http["endpoint"]) == (
            "ERROR", 500, "auth.reset_password",
        )
        assert exc["request_id"] == http["request_id"] == "rid-exc-15"
    else:
        assert "ERROR app.app: Exception on auth.reset_password [GET]" in log
        assert "ERROR app.http: GET auth.reset_password -> 500" in log


_TB_SECRETS = {
    "token": "fake-tb-token-15",
    "dsn": "fake-dsn-pw-15",
    "bearer": "fake-bearer-15-abcdef",
}


@pytest.mark.parametrize("log_format", ["json", "text"])
def test_traceback_secrets_are_scrubbed_on_probe_failure(monkeypatch, log_format):
    import io

    import app.connections
    import app.db
    from app.logging_setup import configure_logging
    from app.security import init_logging_filter

    init_logging_filter()
    configure_logging(log_format)
    [handler] = [h for h in logging.getLogger().handlers if getattr(h, "_dbmon_managed", False)]
    out = io.StringIO()
    handler.setStream(out)

    def broken_adapter(dsn, **_):
        raise RuntimeError(
            f"catalog refused token={_TB_SECRETS['token']} "
            f"via postgresql://svc:{_TB_SECRETS['dsn']}@db/x "
            f"Authorization: Bearer {_TB_SECRETS['bearer']}"
        )

    monkeypatch.setattr(app.db, "make_adapter_for_url", broken_adapter)
    # Existing handler: logger.warning("iceberg probe failed: %s", exc, exc_info=True).
    result = app.connections._probe_iceberg("iceberg+rest://localhost:8181")
    log = out.getvalue()

    assert result["code"] == "catalog_error"
    for secret in _TB_SECRETS.values():
        assert secret not in log
    assert "Traceback (most recent call last)" in log
    assert "in broken_adapter" in log
    assert "RuntimeError: catalog refused token=***" in log
    assert "postgresql://svc:***@db/x" in log and "Bearer ***" in log
    if log_format == "json":
        [record] = [json.loads(line) for line in log.splitlines() if line.startswith("{")]
        assert record["level"] == "WARNING" and record["logger"] == "app.connections"
        assert record["message"].startswith("iceberg probe failed: catalog refused token=***")
        assert "RuntimeError: catalog refused token=***" in record["exc_info"]
    else:
        assert "WARNING app.connections: iceberg probe failed: catalog refused token=***" in log


def test_dev_server_handler_drops_werkzeug_access_line(caplog):
    handler = object.__new__(AppLoggedRequestHandler)  # no socket needed
    with caplog.at_level(logging.INFO, logger="werkzeug"):
        handler.log_request(200, 10)
    assert caplog.records == []


def test_demo_scenario_passes(tmp_path, monkeypatch):
    import app.db
    import app.metrics_storage as storage
    from app.config import settings
    from scripts import check_http_error_alert as demo

    for key in [*demo._OFFLINE_SETTINGS, "MONITOR_DB_URL"]:
        monkeypatch.setattr(settings, key, getattr(settings, key))
    monkeypatch.setattr(storage, "_engine", None)
    monkeypatch.setattr(storage, "_initialized", False)
    monkeypatch.setattr(app.db, "_engine", None)

    report = demo.run_demo(tmp_path)
    assert report["checks"] == {name: True for name in report["checks"]}
    events = [(e["phase"], e["event"]) for e in report["evaluations"] if e["event"]]
    assert events == [("failure", "firing"), ("recovery", "resolved")]
