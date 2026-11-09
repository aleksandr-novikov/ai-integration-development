"""Sentry init + scrubbing tests (#103).

We don't talk to Sentry's network — ``sentry_sdk.init`` accepts a custom
``transport`` callable, so we capture outgoing envelopes locally and
assert on them. That covers the full path: init → exception captured →
before_send scrub → transport.send.
"""

from __future__ import annotations

import logging

import pytest

from app.sentry import _scrub, before_send, init_sentry

# ── before_send / _scrub unit tests ────────────────────────────────────────


def test_scrub_masks_dsn_in_string():
    """Top-level string with a DSN gets the password redacted."""
    out = _scrub("postgresql://user:topsecret@host/db")
    assert "topsecret" not in out
    assert "user:***@host" in out


def test_scrub_recurses_into_dict():
    event = {
        "extra": {"connection": "postgresql://u:hidden@h:5432/d"},
    }
    out = _scrub(event)
    assert "hidden" not in str(out)


def test_scrub_redacts_secret_keyed_dict_values():
    """Keys containing password/token/secret/api_key/authorization
    have their values wholesale replaced — even before recursing."""
    event = {
        "extra": {
            "password": "Pa$$w0rd-1234",
            "api_key": "sk-1234",
            "bearer_token": "Bearer eyJ…",
            "AUTHORIZATION": "Basic dXNlcjo=",
            "username": "alice",  # NOT secret — preserved
        },
    }
    out = _scrub(event)
    assert out["extra"]["password"] == "[REDACTED]"
    assert out["extra"]["api_key"] == "[REDACTED]"
    assert out["extra"]["bearer_token"] == "[REDACTED]"
    assert out["extra"]["AUTHORIZATION"] == "[REDACTED]"
    assert out["extra"]["username"] == "alice"


def test_scrub_handles_list_and_tuple():
    event = {"breadcrumbs": [
        "no secret here",
        "postgresql://u:leaked@h/d",
        ("nested", "postgresql://u:tupled@h/d"),
    ]}
    out = _scrub(event)
    serialised = str(out)
    assert "leaked" not in serialised
    assert "tupled" not in serialised


def test_before_send_returns_dict_not_none():
    """We never drop events — scrub-or-pass-through, never silent drop."""
    out = before_send({"extra": {"trace": "ok"}}, {})
    assert out is not None
    assert isinstance(out, dict)


def test_before_send_swallows_scrub_errors(monkeypatch):
    """Pathological event that breaks _scrub still ships SOMETHING —
    a stand-in without the raw data, never the raw event (#15)."""
    def boom(_):
        raise RuntimeError("scrub broke")

    monkeypatch.setattr("app.sentry._scrub", boom)
    event = {"event_id": "a" * 32, "raw": "data"}
    out = before_send(event, {})
    assert out is not event
    assert out["event_id"] == "a" * 32
    assert out["tags"] == {"scrub_failed": "true"}
    assert "data" not in repr(out)


# ── Password-reset token in the URL path (#15) ─────────────────────────────

_RESET_TOKEN = "fake-secret-token-15"
_RESET_URL = f"https://example.test/auth/reset-password/{_RESET_TOKEN}"
_SAFE_RESET_URL = "https://example.test/auth/reset-password/[REDACTED]"


def _reset_event(url):
    return {
        "event_id": "b" * 32,
        "level": "error",
        "transaction": "auth.reset_password",
        "request": {"url": url, "method": "GET", "query_string": ""},
        "exception": {"values": [{"type": "RuntimeError", "value": "synthetic storage failure"}]},
        "extra": {"attempt": 1},
    }


def test_before_send_masks_reset_token_in_request_url():
    out = before_send(_reset_event(_RESET_URL), {})
    assert _RESET_TOKEN not in repr(out)
    assert out["request"]["url"] == _SAFE_RESET_URL
    # Everything else is untouched.
    assert out == {**_reset_event(_SAFE_RESET_URL)}


@pytest.mark.parametrize("url", [
    "https://example.test/dashboard",
    "https://example.test/auth/login?next=/dashboard",
    "/auth/reset-password/<token>",  # url_rule template, not a token
])
def test_before_send_keeps_urls_without_reset_token(url):
    assert before_send(_reset_event(url), {})["request"]["url"] == url


def test_before_send_masks_reset_token_in_frame_locals_and_encoded_path():
    event = {"exception": {"values": [{"stacktrace": {"frames": [{"vars": {
        "environ": {
            "PATH_INFO": f"'/auth/reset-password/{_RESET_TOKEN}'",
            "RAW_URI": "'/auth/reset-password/fake%2Dsecret%2Dtoken-15?x=1'",
        },
        "req": f"<Request '{_RESET_URL}' [GET]>",
    }}]}}]}}
    out = repr(before_send(event, {}))
    assert _RESET_TOKEN not in out and "fake%2Dsecret" not in out
    assert "'/auth/reset-password/[REDACTED]?x=1'" in out
    assert f"<Request '{_SAFE_RESET_URL}' [GET]>" in out


@pytest.mark.parametrize("request_part", [
    {}, {"url": ""}, {"url": None}, {"url": 12345}, {"url": b"/auth/reset-password/x"},
])
def test_before_send_tolerates_missing_or_odd_url(request_part):
    event = {"event_id": "c" * 32, "request": dict(request_part)}
    out = before_send(event, {})
    assert out == event  # nothing to mask, nothing lost, not withheld


def test_before_send_still_masks_existing_secret_formats_next_to_reset_url():
    event = _reset_event(_RESET_URL)
    event["exception"]["values"][0]["value"] = (
        "postgresql://svc:fake-dsn-pw@db/x Authorization: Bearer fake-bearer-15-abcdef "
        "bot 123456789:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA key AKIAABCDEFGHIJKLMNOP "
        "url ?token=fake-query-tok&password=fake-pass"
    )
    out = repr(before_send(event, {}))
    for secret in ("fake-dsn-pw", "fake-bearer-15", "AAAAAAAAAAAAAAAAAAAA",
                   "AKIAABCDEFGHIJKLMNOP", "fake-query-tok", "fake-pass", _RESET_TOKEN):
        assert secret not in out
    assert "svc:***@db" in out and "Bearer ***" in out and _SAFE_RESET_URL in out


def test_before_send_scrub_failure_does_not_leak_reset_url():
    class Unprintable(Exception):
        def __str__(self):
            raise ValueError("cannot render")

    event = _reset_event(_RESET_URL)
    event["extra"]["error"] = Unprintable()  # breaks the real _scrub
    out = before_send(event, {})
    assert _RESET_TOKEN not in repr(out) and "reset-password" not in repr(out)
    assert out["event_id"] == "b" * 32
    assert out["tags"] == {"scrub_failed": "true"}


# ── init_sentry behaviour ──────────────────────────────────────────────────


def test_init_sentry_noop_when_dsn_empty(monkeypatch):
    """Empty SENTRY_DSN → don't even import sentry_sdk, return False."""
    from app.config import settings
    monkeypatch.setattr(settings, "SENTRY_DSN", "")
    assert init_sentry() is False


def test_init_sentry_initialises_when_dsn_present(monkeypatch):
    """With a DSN configured we actually call sentry_sdk.init."""
    from app.config import settings as cfg
    monkeypatch.setattr(cfg, "SENTRY_DSN",
                        "https://public@sentry.example.com/1")
    monkeypatch.setattr(cfg, "SENTRY_ENVIRONMENT", "test-env")

    captured = {}

    def fake_init(**kwargs):
        captured.update(kwargs)

    import sentry_sdk
    monkeypatch.setattr(sentry_sdk, "init", fake_init)

    assert init_sentry() is True
    assert captured["dsn"] == "https://public@sentry.example.com/1"
    assert captured["environment"] == "test-env"
    assert callable(captured["before_send"])
    assert captured["send_default_pii"] is False


def test_init_sentry_against_real_sdk_does_not_crash(monkeypatch):
    """Regression: ``sentry_sdk.init`` rejects unknown options with a
    TypeError. Tests that monkeypatch ``init`` away never exercise the
    real option whitelist, so SDK 2.x renames (e.g. ``request_bodies`` →
    ``max_request_body_size``) slip through. This test calls the REAL
    init and asserts it returns cleanly — if a future SDK bump removes
    or renames an option we use, this fails immediately.
    """
    from app.config import settings as cfg
    monkeypatch.setattr(cfg, "SENTRY_DSN",
                        "https://public@sentry.example.com/1")
    monkeypatch.setattr(cfg, "SENTRY_ENVIRONMENT", "test-real-init")
    # The DSN is fake — no events will ever be sent (the SDK only
    # validates DSN format, not reachability, during init).
    assert init_sentry() is True


# ── End-to-end with fake transport ─────────────────────────────────────────


def _make_fake_transport(events: list):
    """Build a Transport subclass that captures every envelope's event
    JSON into ``events``. Sentry 2.x dropped the "transport callable"
    shim, so we have to subclass ``sentry_sdk.transport.Transport``.
    """
    from sentry_sdk.transport import Transport

    class _Captor(Transport):
        def __init__(self, options=None):
            super().__init__(options or {})

        def capture_envelope(self, envelope):
            for item in envelope.items:
                payload = item.payload
                # Item payloads can be raw bytes or a JsonPayload object
                # depending on the item type — try both.
                if hasattr(payload, "json"):
                    events.append(payload.json)

        def flush(self, *_a, **_kw):
            pass

        def kill(self, *_a, **_kw):
            pass

    return _Captor


def test_dsn_password_is_scrubbed_in_captured_event():
    """Exception with a DSN in the message → Sentry sees the masked form."""
    import sentry_sdk

    events: list[dict] = []
    sentry_sdk.init(
        dsn="https://public@sentry.example.com/1",
        transport=_make_fake_transport(events),
        before_send=before_send,
        send_default_pii=False,
        traces_sample_rate=0.0,
        # Don't auto-attach the Flask integration — this test is about
        # the scrub itself, not request context capture.
        integrations=[],
    )
    try:
        raise RuntimeError("connect failed: postgresql://u:topsecret@h:5432/d")
    except RuntimeError:
        sentry_sdk.capture_exception()

    sentry_sdk.flush(timeout=2)

    assert events, "expected at least one event captured"
    blob = repr(events[-1])
    assert "topsecret" not in blob
    assert "u:***@h" in blob


def test_password_key_in_extra_is_redacted_end_to_end():
    """capture_message with secret-named scope extras → Sentry event
    has REDACTED placeholders, not the raw values."""
    import sentry_sdk

    events: list[dict] = []
    sentry_sdk.init(
        dsn="https://public@sentry.example.com/1",
        transport=_make_fake_transport(events),
        before_send=before_send,
        send_default_pii=False,
        traces_sample_rate=0.0,
        integrations=[],
    )
    with sentry_sdk.new_scope() as scope:
        scope.set_extra("password", "Pa$$w0rd-1234")
        scope.set_tag("api_key", "sk-secret")
        sentry_sdk.capture_message("test event")

    sentry_sdk.flush(timeout=2)

    assert events
    blob = repr(events[-1])
    assert "Pa$$w0rd-1234" not in blob
    assert "sk-secret" not in blob
    assert "[REDACTED]" in blob


@pytest.fixture
def _reset_sentry():
    import sentry_sdk

    yield
    sentry_sdk.init()  # no DSN: later tests do not record events


def test_reset_token_absent_from_flask_integration_event(tmp_path, monkeypatch, _reset_sentry):
    """Real FlaskIntegration, local transport: unhandled error inside
    ``auth.reset_password`` → no event carries the token (url, frames)."""
    import json

    import sentry_sdk
    from sentry_sdk.integrations.flask import FlaskIntegration

    import app.metrics_storage as storage
    from app.app import create_app

    monkeypatch.setattr(storage.settings, "MONITOR_DB_URL", f"sqlite:///{tmp_path / 'm.db'}")
    monkeypatch.setattr(storage, "_engine", None)
    monkeypatch.setattr(storage, "_initialized", False)

    def broken_lookup(token_hash):
        raise RuntimeError("synthetic storage failure")

    monkeypatch.setattr(storage, "get_active_password_reset_token", broken_lookup)
    events: list[dict] = []
    sentry_sdk.init(
        dsn="https://public@sentry.example.test/1",
        transport=_make_fake_transport(events),
        before_send=before_send,
        send_default_pii=False,
        traces_sample_rate=0.0,
        integrations=[FlaskIntegration()],
        max_request_body_size="never",
    )
    app = create_app({"TESTING": True, "PROPAGATE_EXCEPTIONS": False})
    resp = app.test_client().get(
        f"/auth/reset-password/{_RESET_TOKEN}", base_url="https://example.test",
    )
    sentry_sdk.flush(timeout=2)

    assert resp.status_code == 500
    assert events
    assert all(_RESET_TOKEN not in json.dumps(e, default=str) for e in events)
    [exc_event] = [e for e in events if e.get("exception")]
    assert exc_event["request"]["url"] == _SAFE_RESET_URL
    assert exc_event["transaction"] == "auth.reset_password"
    [exc] = exc_event["exception"]["values"]
    assert (exc["type"], exc["value"]) == ("RuntimeError", "synthetic storage failure")
    assert exc["stacktrace"]["frames"]


# Suppress noisy ``app.sentry`` log lines from polluting unrelated test output.
@pytest.fixture(autouse=True)
def _quiet_sentry_logger():
    logging.getLogger("app.sentry").setLevel(logging.ERROR)
    yield
