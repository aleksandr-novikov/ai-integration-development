import os

import httpx
import pytest

os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost/test")


@pytest.fixture(autouse=True)
def offline_httpx(monkeypatch):
    """NIM/Telegram tests must mock transport; local Flask/MLflow stays usable."""
    sync_request = httpx.HTTPTransport.handle_request
    async_request = httpx.AsyncHTTPTransport.handle_async_request

    def check(request):
        if request.url.host not in {"localhost", "127.0.0.1", "::1"}:
            raise AssertionError(f"Unmocked external HTTPX request: {request.url.host}")

    def handle_request(transport, request):
        check(request)
        return sync_request(transport, request)

    async def handle_async_request(transport, request):
        check(request)
        return await async_request(transport, request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle_request)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle_async_request)
