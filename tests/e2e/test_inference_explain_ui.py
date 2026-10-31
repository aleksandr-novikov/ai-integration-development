"""Real browser → Flask → scoped SQLite explanation, including CSRF."""

import threading

import pytest
from playwright.sync_api import expect
from werkzeug.serving import make_server

from app import dashboard, source_storage
from tests.test_inference_service import SOURCE, bundles, client  # noqa: F401
from tests.test_inference_service import ui as inference_ui  # noqa: F401

pytestmark = pytest.mark.e2e


def test_collected_anomaly_explained_from_table_page(request, page, monkeypatch):
    app, http, config = request.getfixturevalue("inference_ui")
    app.config["WTF_CSRF_ENABLED"] = True
    monkeypatch.setattr("app.llm.settings.NIM_API_KEY", "")
    monkeypatch.setattr(
        dashboard,
        "_list_tables_for_dashboard",
        lambda *a: [{"table_name": "orders", "schema": "public"}],
    )
    monkeypatch.setattr(dashboard, "_table_schema_for_dashboard", lambda *a: [])
    entry = config["bindings"][0]
    stamp = source_storage.metrics(SOURCE, "row_count")[-1]["ts"]
    source_storage.save_prediction(
        {**entry, "observed_at": stamp, "score": -0.5, "is_anomaly": True, "demo": True}
    )
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    page.context.add_cookies(
        [{"name": "session", "value": http.get_cookie("session").value, "url": url}]
    )
    try:
        page.goto(url + "/dashboard/schema/orders")
        row = page.locator(".anomaly-row").first
        expect(row).to_be_visible()
        with page.expect_response(lambda response: response.url.endswith("/api/explain")) as result:
            row.locator(".anomaly-toggle").click()
        assert result.value.status == 200
        body = result.value.request.post_data_json
        assert body["source"] == SOURCE
        assert result.value.request.headers.get("x-csrftoken")
        expect(row.locator(".anomaly-explain-result")).to_be_visible()
        # Reopening a row reuses the explanation without another HTTP request.
        row.locator(".anomaly-toggle").click()
        row.locator(".anomaly-toggle").click()
        expect(row.locator(".anomaly-explain-result")).to_be_visible()
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
