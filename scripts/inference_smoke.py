"""Real localhost HTTP/Swagger evidence; archive baseline is explicit demo only.

python -m scripts.inference_smoke --output /tmp/inference-evidence --swagger
The browser option needs Playwright/Chromium. No production DB or LLM is used.
"""

import argparse
import importlib.metadata
import json
import math
import os
import platform
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from ml.inference_contract import document
from ml.model_registry import stage_export


def memory_bytes():
    try:
        if platform.system() == "Darwin":
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def browser_checks(url, token, payload, output):
    from playwright.sync_api import sync_playwright

    results = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
        )
        page = browser.new_page(viewport={"width": 1440, "height": 1080})
        page.on("pageerror", lambda error: print(f"Swagger browser error: {error}", flush=True))
        page.on(
            "requestfailed",
            lambda request: print(
                f"Swagger asset failed: {request.url}: {request.failure}", flush=True
            ),
        )
        page.goto(url + "/docs")
        page.wait_for_function("typeof ui !== 'undefined' && !!ui.preauthorizeApiKey")
        page.evaluate("token => ui.preauthorizeApiKey('AdapterBearer', token)", token)
        for operation, path, body in [
            ("health", "/healthz", None),
            (
                "inspectModel",
                "/v1/model",
                {
                    key: payload[key]
                    for key in ("source", "model_version", "feature_schema_version")
                },
            ),
            ("predictAnomaly", "/v1/predict", payload),
        ]:
            block = page.locator(f"#operations-default-{operation}")
            block.locator(".opblock-summary-control").click()
            block.get_by_role("button", name="Try it out", exact=True).click()
            if body is not None:
                block.locator("textarea").fill(json.dumps(body, ensure_ascii=False, indent=2))
            with page.expect_response(
                lambda response, path=path: response.url == url + path
                and response.request.method in ("GET", "POST")
            ) as actual:
                block.get_by_role("button", name="Execute", exact=True).click()
            response = actual.value
            assert response.status == 200, response.text()
            results.append(
                {
                    "path": path,
                    "request": body,
                    "status": response.status,
                    "response": response.json(),
                }
            )
            page.wait_for_timeout(200)
            # The disposable token has no use after this process exits; keep
            # even that value out of checked-in screenshots and example curls.
            page.evaluate(
                "token => document.querySelectorAll('.curl-command').forEach(el => el.textContent = el.textContent.replaceAll(token, '<temporary-token>'))",
                token,
            )
            block.screenshot(path=str(output / f"swagger-{operation}.png"))
        browser.close()
    return results


def run(output, swagger=False):
    output.mkdir(parents=True, exist_ok=True)
    payload = document()["paths"]["/v1/predict"]["post"]["requestBody"]["content"][
        "application/json"
    ]["example"]
    source = payload["source"]
    token = secrets.token_urlsafe(32)
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="inference-smoke-") as temporary:
        directory = Path(temporary)
        manifest = stage_export(
            directory / "registry",
            root / "docs/course/experiments/mlflow-export.zip",
            source=source,
        )
        payload["model_version"] = manifest["model_version"]
        config = {
            "registry_path": str(directory / "registry"),
            "demo": True,
            "principals": [{"token_env": "INFERENCE_SMOKE_TOKEN", "sources": [source]}],
        }
        (directory / "config.json").write_text(json.dumps(config))
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        env = {
            **os.environ,
            "INFERENCE_CONFIG": str(directory / "config.json"),
            "INFERENCE_SMOKE_TOKEN": token,
        }
        with (directory / "server.log").open("w+") as log:
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "inference.service:create_app",
                    "--factory",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                cwd=root,
                env=env,
                stdout=log,
                stderr=log,
            )
            try:
                with httpx.Client(base_url=url, timeout=20, trust_env=False) as client:
                    deadline = time.monotonic() + 30
                    while True:
                        try:
                            if client.get("/healthz").status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        if server.poll() is not None or time.monotonic() > deadline:
                            log.seek(0)
                            raise RuntimeError(log.read())
                        time.sleep(0.1)
                    headers = {"Authorization": f"Bearer {token}"}
                    cold_start = time.perf_counter()
                    response = client.post("/v1/predict", json=payload, headers=headers)
                    cold_ms = (time.perf_counter() - cold_start) * 1000
                    response.raise_for_status()
                    sample = response.json()
                    swagger_results = (
                        browser_checks(url, token, payload, output) if swagger else "not run"
                    )
                    samples, failures = [], []
                    for _ in range(100):
                        started = time.perf_counter()
                        response = client.post("/v1/predict", json=payload, headers=headers)
                        elapsed = (time.perf_counter() - started) * 1000
                        if response.status_code == 200:
                            samples.append(elapsed)
                        else:
                            failures.append({"status": response.status_code, "elapsed_ms": elapsed})
                    evidence = {
                        "recorded_at_utc": datetime.now(UTC).isoformat(),
                        "mode": "explicit demo: archived baseline did not pass quality admission",
                        "request": payload,
                        "response": sample,
                        "environment": {
                            "system": platform.system(),
                            "machine": platform.machine(),
                            "python": platform.python_version(),
                            "logical_cpus": os.cpu_count(),
                            "ram_bytes": memory_bytes(),
                            "versions": {
                                name: importlib.metadata.version(name)
                                for name in ("fastapi", "uvicorn", "scikit-learn", "skops")
                            },
                        },
                        "measurement": {
                            "transport": "real localhost HTTP, one uvicorn worker; warm model cached, manifest/artifact bytes revalidated per request; no history/Flask time",
                            "cpu_ram_limit": "unrestricted local host; not the proposed 2-vCPU/4-GiB SLA profile",
                            "concurrency": 1,
                            "request_rate": "sequential, no imposed rate",
                            "cold_request_ms": cold_ms,
                            "success_count": len(samples),
                            "failures": failures,
                            "p95_ms_nearest_rank": sorted(samples)[
                                math.ceil(len(samples) * 0.95) - 1
                            ]
                            if samples
                            else None,
                            "samples_ms": samples,
                        },
                        "swagger": swagger_results,
                    }
                    (output / "smoke.json").write_text(
                        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n"
                    )
                    print(
                        json.dumps(
                            {
                                "output": str(output),
                                "success": len(samples),
                                "failures": len(failures),
                                "p95_ms": evidence["measurement"]["p95_ms_nearest_rank"],
                                "swagger": swagger,
                            }
                        )
                    )
            finally:
                server.terminate()
                server.wait(timeout=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--swagger", action="store_true")
    args = parser.parse_args()
    run(args.output, args.swagger)
