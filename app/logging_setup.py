"""Structured JSON logging (#102).

Default mode is human-readable text — what local dev wants. Set
``LOG_FORMAT=json`` in env and every emitted log line becomes a single
JSON object, parsable by Loki / ELK / Datadog without ingest-side regex.

Why a hand-rolled formatter instead of ``python-json-logger``?
- One library file, ~50 lines, no new dependency.
- We need to pull ``request_id`` from ``flask.g`` — small custom hook
  is cleaner than configuring an external formatter to do the same.
- ``setLogRecordFactory`` (installed in ``app.security``) already scrubs
  DSN passwords before any formatter sees the record, so the JSON
  output is automatically safe — no extra integration.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

from flask import Flask
from werkzeug.serving import WSGIRequestHandler

# Keys present on every ``LogRecord`` by default. We skip them when
# harvesting "extra" fields supplied via logger.info(..., extra={...}),
# otherwise the JSON payload would balloon with framework noise.
_STANDARD_RECORD_ATTRS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
}


class JsonFormatter(logging.Formatter):
    """One JSON object per line — keys are stable, values are scrubbed.

    Required fields: ``timestamp``, ``level``, ``logger``, ``message``,
    ``request_id``. Additional fields:
    - ``exc_info`` (stringified traceback) when present
    - any ``extra={...}`` dict keys passed to the log call
    """

    def format(self, record: logging.LogRecord) -> str:
        # ``record.getMessage`` applies %-formatting using already-scrubbed
        # args (LogRecord factory in app.security runs first), so the
        # rendered string can never resurrect a plaintext DSN here.
        payload: dict = {
            "timestamp": (
                datetime.fromtimestamp(record.created, tz=UTC)
                .isoformat(timespec="milliseconds")
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": _request_id_from_context(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        # Honour anything passed via `extra=...`. Values that can't be
        # JSON-serialised are str()-ed — better than dropping the call.
        for key, value in record.__dict__.items():
            if key in _STANDARD_RECORD_ATTRS or key in payload:
                continue
            try:
                json.dumps(value)
            except TypeError:
                value = str(value)
            payload[key] = value

        return json.dumps(payload, ensure_ascii=False, default=str)


def _request_id_from_context() -> str | None:
    """Best-effort pull of ``g.request_id`` set by the Flask before_request
    hook. Returns None outside an app context (background scheduler ticks,
    CLI invocations) so the field stays present but null."""
    try:
        from flask import g, has_app_context
    except ImportError:  # pragma: no cover - Flask is a hard dep
        return None
    if not has_app_context():
        return None
    return getattr(g, "request_id", None)


def configure_logging(log_format: str, level: str = "INFO") -> None:
    """Install the chosen formatter on the root handler.

    Idempotent: re-installing replaces the previous handler so repeated
    ``create_app()`` calls in tests don't stack StreamHandlers.

    ``log_format``:
      - ``"json"`` — JsonFormatter, stdout
      - anything else — default human-readable text
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Drop our previously-installed handler (if any) so this function can
    # be called more than once per process without accumulating handlers.
    for h in list(root.handlers):
        if getattr(h, "_dbmon_managed", False):
            root.removeHandler(h)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler._dbmon_managed = True  # type: ignore[attr-defined]
    if log_format.lower() == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"
        ))
    root.addHandler(handler)


class AppLoggedRequestHandler(WSGIRequestHandler):
    """Werkzeug dev-server handler without its own access-log line (#15).

    The Docker image runs ``python -m app.app``, i.e. the Werkzeug server.
    Its default access line logs the raw request path (a password-reset
    token lives in the path), has no ``request_id`` and carries ANSI colour
    codes into JSON logs. ``app.instrumentation`` already writes a safe
    ``app.http`` line for every request, so the Werkzeug one is dropped.
    ``log_error`` / server messages are untouched.
    """

    def log_request(self, code="-", size="-") -> None:
        return None


class EndpointLoggingFlask(Flask):
    """Flask whose unhandled-exception line names the endpoint (#15).

    Stock ``Flask.log_exception`` writes ``Exception on <request.path>``,
    so a failure inside ``/auth/reset-password/<token>`` put the live token
    into the log. Same ERROR level, logger and ``exc_info`` (traceback);
    only the path is replaced by the Flask endpoint name, as in ``app.http``.
    """

    def log_exception(self, exc_info) -> None:
        from flask import request

        self.logger.error(
            "Exception on %s [%s]", request.endpoint or "unknown", request.method,
            exc_info=exc_info,
        )


__all__ = [
    "AppLoggedRequestHandler",
    "EndpointLoggingFlask",
    "JsonFormatter",
    "configure_logging",
]
