"""Quality policy boundaries and audit of the immutable MLflow test export."""

import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from scripts import check_model_quality as audit
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
def test_audit_detects_inconsistent_archived_evidence(tmp_path, monkeypatch, damage):
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
    # Exercise semantic validation behind the independent identity gate.
    # CLI users cannot override the pinned digest; that gate is tested below.
    monkeypatch.setattr(audit, "EXPECTED_ARCHIVE_SHA256",
                        hashlib.sha256(output.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="mismatch|inconsistent"):
        read_report(output)


def test_archive_pin_matches_recorded_experiment_checksum():
    assert audit.EXPECTED_ARCHIVE_SHA256 == ARCHIVE.with_suffix(".sha256").read_text().split()[0]


def test_relocated_identical_archive_is_accepted(tmp_path):
    relocated = tmp_path / "export.zip"
    relocated.write_bytes(ARCHIVE.read_bytes())
    assert read_report(relocated)["source_archive_sha256"] == audit.EXPECTED_ARCHIVE_SHA256


def test_internally_consistent_other_run_is_rejected(tmp_path):
    output = tmp_path / "other-experiment.zip"
    with zipfile.ZipFile(ARCHIVE) as original, zipfile.ZipFile(output, "w") as changed:
        for name in original.namelist():
            payload = original.read(name)
            if name == "summary.json":
                data = json.loads(payload)
                data["test"]["run_id"] = "other-consistent-run"
                payload = json.dumps(data).encode()
            elif name == "final_test/run.json":
                data = json.loads(payload)
                data["run_id"] = "other-consistent-run"
                payload = json.dumps(data).encode()
            changed.writestr(name, payload)
    with pytest.raises(ValueError, match="archive SHA-256 mismatch"):
        read_report(output)


@pytest.mark.parametrize("failure", ["parent_is_file", "report_is_directory"])
def test_quality_cli_output_errors_return_technical_error(tmp_path, failure):
    blocked = tmp_path / "blocked"
    if failure == "parent_is_file":
        blocked.write_text("keep this file")
        output = blocked / "quality.json"
    else:
        blocked.mkdir()
        output = blocked
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.check_model_quality", "--archive", str(ARCHIVE),
         "--report", str(output)], capture_output=True, text=True,
    )
    assert completed.returncode == 2, completed.stderr
    assert "error:" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not completed.stdout
    if failure == "parent_is_file":
        assert blocked.read_text() == "keep this file"


def test_quality_cli_serialization_error_returns_technical_error(tmp_path, monkeypatch, capsys):
    output = tmp_path / "quality.json"
    monkeypatch.setattr(sys, "argv", ["check_model_quality", "--report", str(output)])
    monkeypatch.setattr(audit, "read_report", lambda _: {"bad": float("nan")})
    with pytest.raises(SystemExit) as exc:
        audit.main()
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err
    assert not output.exists()


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
