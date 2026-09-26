"""Metric arithmetic, no test-based selection, validation gate and portable models."""

import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from ml.course_experiments import (
    CANDIDATES,
    Candidate,
    evaluate,
    fit_candidate,
    predict,
    select_candidate,
)
from scripts.generate_ml_dataset import FEATURES, Config, write_dataset


def test_event_metrics_match_intervals_objects_and_first_signal():
    frame = pd.DataFrame(
        {
            "project_id": ["p"] * 6,
            "table_name": ["users"] * 5 + ["orders"],
            "ts": [f"2026-01-01T00:{m}:00+00:00" for m in ("00", "15", "30", "45")]
            + ["2026-01-01T01:00:00+00:00", "2026-01-01T00:15:00+00:00"],
            "is_anomaly": [0, 1, 1, 1, 0, 1],
            "predicted_anomaly": [1, 0, 1, 1, 0, 0],
        }
    )
    events = [
        {
            "incident_id": "a",
            "project_id": "p",
            "table_name": "users",
            "kind": "null_spike",
            "split": "validation",
            "start": frame.ts[1],
            "end": frame.ts[3],
        },
        {
            "incident_id": "b",
            "project_id": "p",
            "table_name": "orders",
            "kind": "volume_jump",
            "split": "validation",
            "start": frame.ts[5],
            "end": frame.ts[5],
        },
    ]
    metrics, details = evaluate(frame, events, "validation", 15)
    assert metrics["point_precision"] == pytest.approx(2 / 3)
    assert metrics["point_recall"] == 0.5
    assert metrics["point_f1"] == pytest.approx(4 / 7)
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["event_recall"] == 0.5
    assert metrics["events_detected"] == 1
    assert metrics["median_delay_ticks"] == 1
    assert details[1]["delay_ticks"] is None
    assert not details[1]["detected"]
    frame["predicted_anomaly"] = 0
    empty, _ = evaluate(frame, events, "validation", 15)
    assert empty["point_precision"] == 0
    assert empty["median_delay_ticks"] is None
    assert empty["event_recall"] == 0


def result(name, fpr, f1, recall=1, delay=0):
    return {
        "name": name,
        "parameters": {"n_estimators": 100},
        "metrics": {
            "point_f1": f1,
            "false_positive_rate": fpr,
            "event_recall": recall,
            "abrupt_median_delay_ticks": delay,
        },
    }


def test_selection_prioritizes_case_constraints_over_point_f1():
    good = result("good", 0.05, 0.7)
    assert select_candidate([result("high_f1", 0.11, 0.95), good]) == good
    assert (
        select_candidate([good, result("better_feasible", 0.08, 0.8)])["name"] == "better_feasible"
    )


def test_diagnostic_selection_when_no_candidate_passes():
    assert select_candidate([result("x", 0.4, 0.8), result("y", 0.2, 0.5)])["name"] == "y"
    assert select_candidate([result("x", 0.2, 0.8, 0.4), result("y", 0.2, 0.5, 0.9)])["name"] == "y"


def test_one_factor_plan_and_baseline():
    baseline = CANDIDATES[0].parameters()
    assert baseline == {"contamination": 0.01, "n_estimators": 100, "max_samples": 256}
    assert len(CANDIDATES) == 5
    for candidate in CANDIDATES[1:]:
        assert sum(value != baseline[key] for key, value in candidate.parameters().items()) == 1


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "dataset"
    write_dataset(path, Config(seed=999, train_ticks=257, validation_ticks=96, test_ticks=96))
    return path


def test_fit_never_needs_validation_or_test(dataset):
    for split in ("validation", "test"):
        (dataset / f"{split}.csv").unlink()
    models = fit_candidate(dataset, Candidate("tiny", n_estimators=5))
    train = pd.read_csv(dataset / "train.csv")
    for table, pipeline in models.items():
        expected = train.loc[train.table_name.eq(table), list(FEATURES)].to_numpy().mean(axis=0)
        assert np.allclose(pipeline["scaler"].mean_, expected)
        assert pipeline["scaler"].n_samples_seen_ == 256


def test_invalid_dataset_blocks_training_and_tracking(dataset, tmp_path, monkeypatch):
    from scripts import run_ml_experiments as runner

    (dataset / "test.csv").unlink()
    monkeypatch.setattr(
        runner, "fit_candidate", lambda *args: pytest.fail("training must not start")
    )
    output = tmp_path / "blocked"
    with pytest.raises(ValueError, match="dataset validation failed"):
        runner.run_suite(dataset, output, require_clean=False)
    assert not (output / "mlflow.db").exists()
    assert not json.loads((output / "data-validation.json").read_text())["valid"]


def test_complete_suite_and_export_roundtrip(dataset, tmp_path, monkeypatch):
    import mlflow.sklearn

    from scripts import run_ml_experiments as runner
    from scripts.import_ml_experiments import import_export

    output = tmp_path / "suite"
    original = runner.record_run
    phases = []

    def observe(*args, **kwargs):
        split = args[6]
        phases.append(split)
        if split == "test":
            decision = json.loads((output / "selection.json").read_text())
            assert args[2].name == decision["name"]
            assert len(phases) == 6
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "record_run", observe)
    summary = runner.run_suite(dataset, output, require_clean=False)
    assert phases == ["validation"] * 5 + ["test"]
    assert len(summary["validation"]) == 5
    descriptors = list((output / "export").glob("*/run.json"))
    assert len(descriptors) == 6
    assert all(json.loads(p.read_text())["status"] == "FINISHED" for p in descriptors)
    assert summary["selection"]["name"] == select_candidate(summary["validation"])["name"]
    mapping = import_export(output / "mlflow-export.zip", tmp_path / "imported")
    assert len(mapping) == 6
    selected_run = summary["test"]["run_id"]
    model = mlflow.sklearn.load_model(f"runs:/{mapping[selected_run]}/models/users")
    frame = pd.read_csv(dataset / "test.csv")
    users = frame[frame.table_name.eq("users")]
    loaded = predict({"users": model}, users)
    stored = pd.read_csv(output / "export/final_test/artifacts/predictions.csv")
    expected = stored[stored.table_name.eq("users")]
    np.testing.assert_allclose(loaded.score, expected.score, atol=1e-12)
    np.testing.assert_array_equal(loaded.predicted_anomaly, expected.predicted_anomaly)
    with pytest.raises(ValueError, match="empty"):
        runner.run_suite(dataset, output, require_clean=False)


def test_import_rejects_path_traversal(tmp_path):
    from scripts.import_ml_experiments import import_export

    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../../escape", "no")
    with pytest.raises(ValueError, match="unsafe path"):
        import_export(archive, tmp_path / "output")
    assert not (tmp_path / "output").exists()
