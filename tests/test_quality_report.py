"""Quality policy boundaries and audit of the immutable MLflow test export."""

import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from scripts.check_model_quality import quality_gate, read_report

ARCHIVE = Path(__file__).resolve().parents[1] / "docs/course/experiments/mlflow-export.zip"


@pytest.mark.parametrize(
    "field,value",
    [
        ("event_recall", 0.799),
        ("false_positive_rate", 0.1001),
        ("abrupt_median_delay_ticks", 1.01),
        ("event_recall", None),
        ("false_positive_rate", float("nan")),
        ("abrupt_median_delay_ticks", float("inf")),
    ],
)
def test_quality_policy_rejects_failed_or_missing_evidence(field, value):
    metrics = {"event_recall": 0.8, "false_positive_rate": 0.1, "abrupt_median_delay_ticks": 1}
    assert quality_gate(metrics)["passed"]
    metrics[field] = value
    gate = quality_gate(metrics)
    assert not gate["passed"]
    assert not gate["checks"][field]["passed"]


def test_frozen_baseline_is_rejected_and_slices_preserve_denominators():
    report = read_report(ARCHIVE)
    assert not report["quality_gate"]["passed"]
    assert report["overall"]["false_positives"] == 240
    assert report["overall"]["false_positive_rate"] == pytest.approx(240 / 1809)
    assert report["overall"]["events_detected"] == report["overall"]["events_total"] == 9
    for count in (
        "true_positives",
        "false_positives",
        "false_negatives",
        "true_negatives",
        "events_total",
    ):
        assert sum(m[count] for m in report["by_table"].values()) == report["overall"][count]
    assert set(report["by_incident_kind"]) == {"null_spike", "volume_jump", "regime_change"}
    for metrics in report["by_incident_kind"].values():
        assert metrics["events_total"] == 3
        assert metrics["false_positives"] + metrics["true_negatives"] == 1809
    assert max(e["delay_ticks"] for e in report["events"]) == 5


@pytest.mark.parametrize("damage", ["metric", "label", "threshold"])
def test_audit_detects_inconsistent_archived_evidence(tmp_path, damage):
    output = tmp_path / "changed.zip"
    with zipfile.ZipFile(ARCHIVE) as original, zipfile.ZipFile(output, "w") as changed:
        # Models are deliberately not copied: the audit must never load them.
        for name in (
            "summary.json",
            "final_test/run.json",
            "final_test/artifacts/manifest.json",
            "final_test/artifacts/events.json",
            "final_test/artifacts/predictions.csv",
        ):
            payload = original.read(name)
            if damage == "metric" and name == "final_test/run.json":
                data = json.loads(payload)
                data["metrics"]["false_positive_rate"] = 0
                payload = json.dumps(data).encode()
            if damage in ("label", "threshold") and name.endswith("predictions.csv"):
                frame = pd.read_csv(io.BytesIO(payload), keep_default_na=False)
                field = "is_anomaly" if damage == "label" else "predicted_anomaly"
                frame.loc[0, field] = 1 - frame.loc[0, field]
                payload = frame.to_csv(index=False).encode()
            changed.writestr(name, payload)
    with pytest.raises(ValueError, match="mismatch|inconsistent"):
        read_report(output)


def test_quality_cli_writes_failure_and_returns_nonzero(tmp_path):
    output = tmp_path / "quality.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.check_model_quality",
            "--archive",
            str(ARCHIVE),
            "--report",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1, completed.stderr
    assert "FAIL" in completed.stdout
    assert not json.loads(output.read_text())["quality_gate"]["passed"]


def test_quality_cli_preserves_source_archive():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.check_model_quality",
            "--archive",
            str(ARCHIVE),
            "--report",
            str(ARCHIVE),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "overwrite" in completed.stderr
