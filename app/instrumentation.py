"""Prometheus instrumentation (#101).

Exposes ``GET /metrics`` in the standard text exposition format and the
collector primitives the rest of the codebase increments:

- ``http_requests_total{method,endpoint,status}`` — bumped via Flask
  ``before_request`` / ``after_request`` hooks installed in
  :func:`install_http_instrumentation`.
- ``http_request_duration_seconds`` — histogram observed in the same hooks.
- ``collector_runs_total{result}`` — incremented from
  ``collectors/per_project.py::collect_for_connection`` on success / error.
- ``failed_login_attempts_total`` — incremented from
  ``app/auth.py::login`` on wrong password / unknown email.

The same after_request hook writes one ``app.http`` access-log line per
request (#15): method, endpoint name, status and latency. The raw path and
query string are never logged — ``/auth/reset-password/<token>`` would
otherwise put a live reset token into the log stream.

We use the default global ``REGISTRY`` so a future ``prometheus-multiproc``
swap (for gunicorn -w N) only changes the registry, not the call sites.
"""

from __future__ import annotations

import logging
import time

from flask import Response, g, request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

logger = logging.getLogger(__name__)
access_logger = logging.getLogger("app.http")


# ── Counters / histograms ──────────────────────────────────────────────────

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests served, partitioned by method, endpoint and status.",
    labelnames=("method", "endpoint", "status"),
)

# Bucket boundaries cover the realistic range for a Flask app: <10ms (cached
# lookups), 25/50/100ms (typical DB-roundtrip), 250/500ms (slower queries),
# 1/2.5/5/10s (anomalous; aim to alert on these). Default prometheus-client
# buckets stop at 10s which is fine for our SLO ceiling.
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency (seconds), by method + endpoint.",
    labelnames=("method", "endpoint"),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

collector_runs_total = Counter(
    "collector_runs_total",
    "Per-connection collector ticks, partitioned by outcome.",
    labelnames=("result",),  # "ok" | "error"
)

failed_login_attempts_total = Counter(
    "failed_login_attempts_total",
    "Failed /auth/login attempts (wrong password or unknown email).",
)

# Published inference calls of app/inference_adapter.py (#16). The full source
# identity, version and demo mode keep series of different models apart.
INFERENCE_LABELS = (
    "project_id", "connection_id", "schema_name", "table_name",
    "model_version", "feature_schema_version", "demo",
)

inference_predictions_total = Counter(
    "inference_predictions_total",
    "Inference service calls by outcome: ok or the adapter error code.",
    labelnames=(*INFERENCE_LABELS, "outcome"),
)

# Server-Timing of the service: model computation only, without HTTP. One
# IsolationForest vector takes milliseconds; the upper buckets catch overload.
inference_predict_duration_seconds = Histogram(
    "inference_predict_duration_seconds",
    "Prediction computation time reported by the inference service (seconds).",
    labelnames=INFERENCE_LABELS,
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)


# ── HTTP hooks ─────────────────────────────────────────────────────────────


def _start_timer() -> None:
    """Record start time so after_request can compute latency.

    Using ``g`` keeps it request-scoped without polluting ``request``.
    """
    g._prom_start_time = time.perf_counter()


def _record_response(response):
    """Bump http_requests_total / http_request_duration_seconds and log.

    ``request.endpoint`` is None for 404s on unknown routes — we substitute
    a literal so the cardinality doesn't explode with the request path
    (which can be arbitrary user-controlled garbage). The access log uses
    the same endpoint name for the same reason, plus it keeps path-embedded
    secrets (reset tokens, invite tokens) out of the logs.
    """
    endpoint = request.endpoint or "unknown"
    method = request.method
    status = str(response.status_code)
    http_requests_total.labels(
        method=method, endpoint=endpoint, status=status,
    ).inc()
    duration_ms = None
    start = getattr(g, "_prom_start_time", None)
    if start is not None:
        elapsed = time.perf_counter() - start
        http_request_duration_seconds.labels(
            method=method, endpoint=endpoint,
        ).observe(elapsed)
        duration_ms = round(elapsed * 1000, 1)
    # 5xx is a server-side failure worth an ERROR; 4xx is the client's
    # problem and stays INFO so bots probing random URLs don't page anyone.
    access_logger.log(
        logging.ERROR if response.status_code >= 500 else logging.INFO,
        "%s %s -> %s in %s ms", method, endpoint, status, duration_ms,
        extra={
            "event": "http_request",
            "method": method,
            "endpoint": endpoint,
            "status": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    return response


def install_http_instrumentation(app) -> None:
    """Wire the request-level Counter / Histogram into a Flask app."""
    app.before_request(_start_timer)
    app.after_request(_record_response)


# ── /metrics endpoint ──────────────────────────────────────────────────────

def metrics_response() -> Response:
    """Render the current registry in Prometheus text exposition format.

    No CSRF (would be hostile to scrapers), no auth (the convention is that
    the endpoint sits behind a network ACL — exposing /metrics behind login
    breaks every Prometheus deployment). Rate-limited at the route level
    in app.app to discourage someone using it as an infinite-pull DoS.

    Note: ``CONTENT_TYPE_LATEST`` already encodes the charset; we pass it
    via ``content_type=`` (verbatim) rather than ``mimetype=`` to avoid
    Flask appending a second ``charset=utf-8`` that some strict scrapers
    refuse to parse.
    """
    return Response(generate_latest(), content_type=CONTENT_TYPE_LATEST)


__all__ = [
    "collector_runs_total",
    "failed_login_attempts_total",
    "http_request_duration_seconds",
    "http_requests_total",
    "inference_predict_duration_seconds",
    "inference_predictions_total",
    "install_http_instrumentation",
    "metrics_response",
]
