"""Real Compose CI verification using the existing no-cache integration image.

Host run/cleanup are restricted to an isolated Linux GitHub runner. Container
probes reuse the documented seed, warmup, registry and live-demo code paths.
Only the allowlisted JSON report is uploaded; raw commands never print env.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/compose-demo/ci-report.json"
STATE = ROOT / ".git/compose-ci-state.json"
OVERRIDE = ROOT / ".git/compose-ci.json"
DEMO = ROOT / "models/demo"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def read(path):
    return json.loads(path.read_text())


def environment():
    return dict(line.split("=", 1) for line in (ROOT / ".env").read_text().splitlines()
                if line and not line.startswith("#"))


def redact(message, values):
    for key in ("SECRET_KEY", "FERNET_KEY", "INFERENCE_ADAPTER_TOKEN"):
        if secret := values.get(key):
            message = message.replace(secret, "<redacted>")
    return re.sub(r"([a-zA-Z][a-zA-Z0-9+.-]*://)[^\s/@]+:[^\s@]+@", r"\1<redacted>@", message)


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
    from app import crypto, metrics_storage as ms
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
    # The guard precedes all writes. No existing database is a permitted target.
    if os.environ.get("COMPOSE_DEMO_TEST") != "1" or not Path("/.dockerenv").exists():
        raise RuntimeError("preparation is restricted to the isolated Compose test")
    from app import metrics_storage as ms
    from ml.model_registry import stage_export
    from scripts import demo_prepare, seed_target_db

    seed_target_db.main(n_users=500, n_products=50, n_orders=1000, n_events=2000)
    summary = demo_prepare.main()
    assert summary["retail-postgres"]["metrics"] > 0
    assert summary["retail-postgres"]["anomaly_scores"] > 0
    assert summary["retail-postgres"]["forecast_models"] > 0
    source = container_state()
    for email in ("demo@dbmonitor.app", "lake@dbmonitor.app"):
        for project in ms.list_projects_for_user(ms.get_user_by_email(email)["id"]):
            if project["id"] != source["project_id"]:
                for connection in ms.list_connections_for_project(project["id"]):
                    ms.set_connection_active(project["id"], connection["id"], False)
    manifest = stage_export(ROOT / "models/registry", Path("/tmp/mlflow-export.zip"), source=source)
    binding = {"source": source, **{k: manifest[k] for k in ("model_version", "feature_schema_version")}}
    save(DEMO / "service.json", {"registry_path": "models/registry", "demo": True,
         "principals": [{"token_env": "INFERENCE_ADAPTER_TOKEN", "sources": [source]}]})
    save(DEMO / "adapter.json", {"url": "http://inference:8000", "demo": True,
         "token_env": "INFERENCE_ADAPTER_TOKEN", "bindings": [binding]})
    save(DEMO / "prepared.json", {"binding": binding, "warmup": summary, "demo": True,
         "quality_admitted": False, "baseline_fpr": 0.1327, "fernet_key_shared": True})


def observe(phase):
    from app import source_storage
    from ml.features import FEATURE_NAMES, build_rows

    source = container_state()
    binding = read(DEMO / "prepared.json")["binding"]
    started = datetime.now(UTC).isoformat()
    command = [sys.executable, "-m", "scripts.live_demo", "--project-id", source["project_id"],
               "--connection-id", source["connection_id"], "--interval", "1",
               "--ticks", "1" if phase == "incident" else "3"]
    if phase == "incident":
        command += ["--incident-at", "1"]
    subprocess.run(command, check=True, timeout=180)
    rc, nr = (source_storage.metrics(source, name) for name in ("row_count", "null_rate"))
    stamps, vectors = build_rows(rc, nr)
    features = {stamp: dict(zip(FEATURE_NAMES, vector, strict=True))
                for stamp, vector in zip(stamps, vectors, strict=True)}
    rows = source_storage.scores(source, binding["model_version"], binding["feature_schema_version"])
    new = [{**row, "features": features[row["ts"]], "quality_admitted": False}
           for row in rows if row["ts"] >= started]
    assert new and all(row["demo"] for row in new), "new versioned observations missing"
    # Re-run the same selected service on these exact features and compare the
    # response to persisted results; never manufacture is_anomaly from injection.
    import httpx

    config = read(DEMO / "adapter.json")
    aligned = sorted({r["ts"] for r in rc} & {r["ts"] for r in nr})
    with httpx.Client(timeout=20, trust_env=False) as client:
        for row in new:
            payload = {**binding, "features": row["features"], "observed_at": row["ts"],
                       "previous_observed_at": aligned[aligned.index(row["ts"]) - 1]}
            response = client.post(config["url"] + "/v1/predict", json=payload,
                                   headers={"Authorization": "Bearer " + os.environ["INFERENCE_ADAPTER_TOKEN"]})
            assert response.status_code == 200
            actual = response.json()
            assert actual["demo"] is True and actual["quality_admitted"] is False
            assert actual["source"] == source and actual["observed_at"] == row["ts"]
            assert all(actual[key] == binding[key] for key in ("model_version", "feature_schema_version"))
            assert actual["score"] == row["score"] and actual["is_anomaly"] == row["is_anomaly"]
    save(DEMO / (phase + ".json"), new)


def snapshot():
    from sqlalchemy import text

    from app import db, metrics_storage as ms

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


class ComposeRun:
    def __init__(self):
        runner_only()
        self.values = environment() if (ROOT / ".env").exists() else {}
        self.report = {"technical_pass": False, "steps": [], "manual_browser_and_second_participant": "NOT VERIFIED"}

    def command(self, label, args, timeout=180):
        started = time.monotonic()
        process = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        self.report["steps"].append({"step": label, "command": " ".join(args),
                                     "exit_code": process.returncode, "seconds": round(time.monotonic() - started, 2)})
        save(REPORT, self.report)
        if process.returncode:
            raise RuntimeError(label + ": " + redact(process.stderr + process.stdout, self.values)[-5000:])
        return process.stdout

    def compose(self, label, *args, timeout=180):
        project = self.values["COMPOSE_PROJECT_NAME"]
        return self.command(label, ["docker", "compose", "-p", project, "-f", "docker-compose.yml",
                                    "-f", str(OVERRIDE.relative_to(ROOT)), "--profile", "inference", *args], timeout)

    def probe(self, mode):
        self.compose(mode, "run", "--rm", "-T", "--no-deps", "app", "python", "-m",
                     "scripts.verify_compose_demo", "inside-" + mode, timeout=600)

    def dashboard(self, rows):
        import httpx

        binding = read(DEMO / "prepared.json")["binding"]
        source = binding["source"]
        with httpx.Client(base_url="http://127.0.0.1:" + self.values["PORT"],
                          follow_redirects=True, timeout=30, trust_env=False) as client:
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
        return {"status": 200, "csrf_login": True, "source": source, "version": binding["model_version"],
                "observed_at": rows[-1]["ts"], "score_visible": True, "scope": "HTTP HTML/API, not browser rendering"}

    def run(self):
        from scripts.init_compose_demo import create_environment

        assert not (ROOT / ".env").exists(), "CI requires a fresh checkout without .env"
        assert not self.command("clean checkout", ["git", "status", "--porcelain"]).strip()
        sha = self.command("checkout SHA", ["git", "rev-parse", "HEAD"]).strip()
        assert sha == os.environ["COMPOSE_SMOKE_SHA"]
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
        self.report.update(checkout_sha=sha, image_reused_from="test_inference_image nocache=True",
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
        self.compose("seed and warmup", "run", "--rm", "-T", "--no-deps", "-v",
                     str(ROOT / "docs/course/experiments/mlflow-export.zip") + ":/tmp/mlflow-export.zip:ro",
                     "app", "python", "-m", "scripts.verify_compose_demo", "inside-prepare", timeout=600)
        self.report["prepared"] = read(DEMO / "prepared.json")
        self.compose("Flask and FastAPI healthchecks", "up", "-d", "--no-build", "--wait", "--wait-timeout", "150", "inference", "app")
        self.report["serving_image_ids"] = {}
        for service in ("app", "inference"):
            cid = self.compose(service + " container ID", "ps", "-q", service).strip()
            actual_image = self.command(service + " image ID", ["docker", "inspect", "--format", "{{.Image}}", cid]).strip()
            assert actual_image == image_info["Id"]
            self.report["serving_image_ids"][service] = actual_image
        import httpx

        binding = self.report["prepared"]["binding"]
        with httpx.Client(timeout=30, trust_env=False) as client:
            health = client.get("http://127.0.0.1:" + self.values["PORT"] + "/healthz")
            assert health.status_code == 200
            assert health.json()["version"] == sha
            checks = health.json()["checks"]
            assert checks["target_db"]["status"] == checks["monitor_db"]["status"] == "ok"
            assert checks["monitor_db"]["backend"] == "postgresql"
            url = "http://127.0.0.1:" + self.values["INFERENCE_PORT"] + "/v1/model"
            params = {**binding["source"], **{k: binding[k] for k in ("model_version", "feature_schema_version")}}
            assert client.get(url, params=params).status_code == 401
            model = client.get(url, params=params, headers={"Authorization": "Bearer " + self.values["INFERENCE_ADAPTER_TOKEN"]})
            assert model.status_code == 200
            assert model.json()["demo"] is True and model.json()["quality_admitted"] is False
            self.report["readiness"] = {"target_db": "ok", "monitor_db": "ok", "monitor_backend": "postgresql", "model": model.json(), "unauthorized_model_status": 401}
        for phase in ("normal", "incident"):
            self.probe(phase)
            self.report[phase] = read(DEMO / (phase + ".json"))
            save(REPORT, self.report)
        self.report["model_scenario"] = outcome(self.report["normal"], self.report["incident"])
        self.report["dashboard"] = self.dashboard(self.report["incident"])
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
            assert before == after, action + " changed test data/artifact"
            self.report[action + "_persistence"] = after
        self.compose("recreate application and inference", "up", "-d", "--no-build", "--force-recreate", "--wait", "--wait-timeout", "150", "app", "inference")
        self.probe("after-recreate")
        self.report["after_recreate"] = read(DEMO / "after-recreate.json")
        self.report["dashboard_after_recreate"] = self.dashboard(self.report["after_recreate"])
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
        report["cleanup"] = "PASS; only the recorded runner project and its test volumes"
    finally:
        save(REPORT, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "cleanup", "check-outcome", "inside-prepare", "inside-normal", "inside-incident", "inside-after-recreate", "inside-snapshot"))
    mode = parser.parse_args().mode
    if mode.startswith("inside-"):
        action = mode.removeprefix("inside-")
        if action == "prepare":
            prepare()
        elif action == "snapshot":
            snapshot()
        else:
            observe(action)
    elif mode == "cleanup":
        cleanup()
    elif mode == "check-outcome":
        result = read(REPORT)
        print(json.dumps(result.get("model_scenario", {})))
        if not result.get("technical_pass") or not result.get("model_scenario", {}).get("passed"):
            raise SystemExit("Actual demo model did not confirm both normal and incident states; no calibration performed")
    else:
        run = ComposeRun()
        try:
            run.run()
        except Exception as exc:
            run.report["failure"] = {"type": type(exc).__name__, "message": redact(str(exc), run.values)}
            if STATE.exists():
                try:
                    logs = run.compose("failure diagnostics", "logs", "--no-color", "--tail", "60",
                                       "app", "inference", "postgres", "timescaledb", timeout=30)
                    run.report["diagnostic_tail"] = redact(logs, run.values)[-8000:]
                except Exception:
                    run.report["diagnostic_tail"] = "unavailable; inspect recorded step exit codes"
            raise SystemExit("Compose verification failed; see sanitized ci-report.json") from None
        finally:
            save(REPORT, run.report)
        print("Compose technical checks PASS; actual model result:", run.report["model_scenario"])


if __name__ == "__main__":
    main()
