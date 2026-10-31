"""Measure adapter/storage overhead on temporary SQLite with a deterministic HTTP stub.

The stub excludes model execution and network latency; this is not an end-to-end SLA.
The 2880-point fixture crosses the 30-day cutoff by five points. Both implementations
use the same production history reader, schema and response values to isolate adapter
work. No LLM, Telegram, production DB or model weights are used.
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from app import inference_adapter as current
from app import metrics_storage as storage


def baseline_adapter(ref):
    """Run the repository's earlier adapter with its earlier wire validator."""
    root = Path(__file__).resolve().parents[1]
    modules = []
    for name, filename in (
        ("baseline_contract", "ml/inference_contract.py"),
        ("baseline_adapter", "app/inference_adapter.py"),
    ):
        source = subprocess.check_output(["git", "show", f"{ref}:{filename}"], cwd=root, text=True)
        module = types.ModuleType(name)
        module.__file__ = str(root / filename)
        exec(compile(source, module.__file__, "exec"), module.__dict__)
        modules.append(module)
    modules[1].validator = modules[0].validator
    return modules[1]


class Service:
    def __init__(self):
        self.calls = 0

    def post(self, url, json=None, **kwargs):
        self.calls += 1
        return httpx.Response(
            200,
            json={k: v for k, v in json.items() if k != "features" and k != "previous_observed_at"}
            | {"score": 0.1, "threshold": 0.0, "is_anomaly": False, "demo": True},
        )


def run(adapter, count, sources):
    with tempfile.TemporaryDirectory() as tmp:
        storage.settings.MONITOR_DB_URL = f"sqlite:///{tmp}/metrics.db"
        storage._engine = None
        storage._initialized = False
        storage.create_user("u", "bench@example.test", "unused")
        storage.create_project("p", "u", "bench", "bench")
        entries = []
        start = datetime.now(UTC) - timedelta(minutes=15 * (count + 4))
        for i in range(sources):
            source = {
                "project_id": "p",
                "connection_id": f"c{i}",
                "schema_name": "public",
                "table_name": "orders",
            }
            storage.create_connection(
                connection_id=f"c{i}", project_id="p", name="bench", dsn_encrypted=b"unused"
            )
            storage.save_metrics(
                [
                    {
                        "ts": (start + timedelta(minutes=j * 15)).isoformat(),
                        "table_name": "orders",
                        "metric_name": m,
                        "value": (100 + j if m == "row_count" else 0.02),
                    }
                    for j in range(count)
                    for m in ("row_count", "null_rate")
                ],
                "p",
                connection_id=f"c{i}",
                schema_name="public",
            )
            entries.append(
                {"source": source, "model_version": "v1", "feature_schema_version": "1.0.0"}
            )
        config = {"bindings": entries, "demo": True}
        path = Path(tmp) / "adapter.json"
        path.write_text(json.dumps(config))
        os.environ["INFERENCE_ADAPTER_CONFIG"] = str(path)
        os.environ["INFERENCE_ADAPTER_TOKEN"] = "synthetic"
        client = Service()
        result = {}
        for tick in ("initial", "repeat", "two_new"):
            if tick == "two_new":
                for entry in entries:
                    source = entry["source"]
                    storage.save_metrics(
                        [
                            {
                                "ts": (start + timedelta(minutes=j * 15)).isoformat(),
                                "table_name": "orders",
                                "metric_name": m,
                                "value": (100 + j if m == "row_count" else 0.02),
                            }
                            for j in range(count, count + 2)
                            for m in ("row_count", "null_rate")
                        ],
                        "p",
                        connection_id=source["connection_id"],
                        schema_name="public",
                    )
            calls = client.calls
            before = time.perf_counter()
            adapter.score_project("p", client=client)
            result[tick] = {
                "seconds": round(time.perf_counter() - before, 6),
                "http_calls": client.calls - calls,
            }
        storage._engine.dispose()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", help="Optional earlier Git commit to compare")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    adapters = [("after", current)]
    if args.baseline_ref:
        adapters.insert(0, ("before", baseline_adapter(args.baseline_ref)))
    output = {}
    for name, adapter in adapters:
        output[name] = {}
        for count, sources in [(96, 1), (384, 1), (2880, 1), (96, 3)]:
            result = run(adapter, count, sources)
            output[name][f"{count}x{sources}"] = result
            print(name, count, sources, result, flush=True)
            args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
