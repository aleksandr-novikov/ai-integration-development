"""Reproducible normal → failure → recovery run of the HTTP 5xx alert (#15).

Run: python -m scripts.check_http_error_alert > docs/course/monitoring/demo.json 2> /tmp/alert-logs.jsonl

Real Flask app (TESTING mode, no scheduler) on throwaway SQLite files, real
request hooks and real /metrics text. Only the scrape clock is simulated:
one /metrics scrape per 30 s tick, so a 5-minute window takes no 5 minutes.
The failure is the target DB becoming unreachable, which turns /healthz
into 503. Telegram/Sentry/LLM/SMTP are forced off; nothing leaves the
process, so the result checks rule + signal, not notification delivery.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

SCRAPE_INTERVAL_S = 30
PHASES = (("normal", 10), ("failure", 6), ("recovery", 12))
# Fake values that must never reach logs or /metrics.
FAKE_SECRETS = {
    "reset_token_in_path": "demo-reset-token-0f1e2d3c",
    "token_in_query": "demo-query-token-4b5a6978",
    "bearer_header": "demo-bearer-value-a1b2c3d4",
}
_OFFLINE_SETTINGS = {
    "SENTRY_DSN": "",
    "TELEGRAM_BOT_TOKEN": "",
    "TELEGRAM_CHAT_ID": "",
    "NIM_API_KEY": "",
    "SMTP_HOST": "",
    "LOG_FORMAT": "json",
}


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


def run_demo(work_dir: Path) -> dict:
    from sqlalchemy import create_engine

    import app.db
    import app.metrics_storage as storage
    from app.app import create_app
    from app.config import settings
    from app.http_alert import HttpErrorAlert, snapshot_from_metrics
    from app.logging_setup import JsonFormatter

    for key, value in _OFFLINE_SETTINGS.items():
        setattr(settings, key, value)
    settings.MONITOR_DB_URL = f"sqlite:///{work_dir / 'monitor.db'}"
    storage._engine = None
    storage._initialized = False
    healthy_target = create_engine(f"sqlite:///{work_dir / 'target.db'}")
    broken_target = create_engine(f"sqlite:///{work_dir / 'missing' / 'target.db'}")
    app.db._engine = healthy_target

    flask_app = create_app({"TESTING": True})
    root = logging.getLogger()
    for handler in root.handlers:
        if getattr(handler, "_dbmon_managed", False):
            handler.setStream(sys.stderr)  # stdout carries the JSON report
    capture = _Capture()
    capture.setFormatter(JsonFormatter())
    root.addHandler(capture)

    client = flask_app.test_client()
    alert = HttpErrorAlert()
    metrics_text = ""

    def scrape(ts: float):
        nonlocal metrics_text
        metrics_text = client.get("/metrics").get_data(as_text=True)
        return alert.observe(snapshot_from_metrics(metrics_text, ts, alert.rule))

    evaluations = []
    try:
        scrape(0)
        tick = 0
        for phase, ticks in PHASES:
            app.db._engine = broken_target if phase == "failure" else healthy_target
            for _ in range(ticks):
                tick += 1
                if tick == 1:
                    client.get(
                        f"/auth/reset-password/{FAKE_SECRETS['reset_token_in_path']}"
                        f"?token={FAKE_SECRETS['token_in_query']}",
                        headers={"Authorization": f"Bearer {FAKE_SECRETS['bearer_header']}"},
                    )
                client.get("/healthz")
                client.get("/")
                client.get("/no-such-page")
                result = scrape(tick * SCRAPE_INTERVAL_S)
                evaluations.append({"tick": tick, "phase": phase, **asdict(result)})
    finally:
        app.db._engine = healthy_target
        root.removeHandler(capture)

    records = [json.loads(line) for line in capture.lines]

    def first(pred):
        return next((r for r in records if pred(r)), None)

    by_phase = {p: [e for e in evaluations if e["phase"] == p] for p, _ in PHASES}
    exposed = sorted(
        name for name, value in FAKE_SECRETS.items()
        if value in metrics_text or any(value in line for line in capture.lines)
    )
    samples = {
        "http_request_ok": first(
            lambda r: r.get("event") == "http_request" and r["status"] < 400
            and r["endpoint"] != "metrics"
        ),
        "http_request_5xx": first(lambda r: r.get("event") == "http_request" and r["status"] >= 500),
        "dependency_failure": first(lambda r: r["logger"] == "app.health"),
        "alert_firing": first(lambda r: r.get("event") == "alert_firing"),
        "alert_resolved": first(lambda r: r.get("event") == "alert_resolved"),
    }
    checks = {
        "normal_never_firing": all(e["state"] == "inactive" for e in by_phase["normal"]),
        "normal_had_enough_data": any(e["condition"] is False for e in by_phase["normal"]),
        "failure_detected": any(e["event"] == "firing" for e in by_phase["failure"]),
        "recovery_resolved": any(e["event"] == "resolved" for e in by_phase["recovery"]),
        "final_state_inactive": evaluations[-1]["state"] == "inactive",
        "signal_logged": bool(samples["alert_firing"] and samples["alert_resolved"]),
        "http_requests_and_errors_logged": bool(
            samples["http_request_ok"] and samples["http_request_5xx"]
        ),
        "fake_secrets_absent_from_logs_and_metrics": not exposed,
    }
    return {
        "scope": (
            "Flask test client, TESTING mode, throwaway SQLite; simulated scrape clock "
            f"({SCRAPE_INTERVAL_S}s per tick). Rule evaluation and log signal only; "
            "no notification is delivered and no external service is contacted."
        ),
        "rule": {**asdict(alert.rule), "excluded_endpoints": sorted(alert.rule.excluded_endpoints)},
        "traffic_per_tick": ["GET /healthz", "GET /", "GET /no-such-page (404)"],
        "failure": "target DB engine points to a non-existent directory → /healthz 503",
        "phases": dict(PHASES),
        "checks": checks,
        "passed": all(checks.values()),
        "exposed_fake_secrets": exposed,
        "log_records": len(records),
        "log_samples": samples,
        "evaluations": evaluations,
    }


def main() -> None:
    # Settings require DATABASE_URL at import; never point the demo at a real DB.
    with tempfile.TemporaryDirectory(prefix="http-alert-demo-") as tmp:
        os.environ["DATABASE_URL"] = f"sqlite:///{Path(tmp) / 'target.db'}"
        for key, value in _OFFLINE_SETTINGS.items():
            os.environ[key] = value
        report = run_demo(Path(tmp))
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
