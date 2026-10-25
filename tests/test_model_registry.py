"""Traceable bundles, safe restore, explicit admission and legacy compatibility."""

import copy
import csv
import io
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pytest
import skops.io as sio
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml import anomaly_detector as legacy
from ml import model_registry as registry
from ml.features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_SHA256,
    build_rows,
    build_vector,
    validate_vector,
)

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/course/experiments/mlflow-export.zip"
SOURCE = {
    "project_id": "project-a",
    "connection_id": "connection-a",
    "schema_name": "public",
    "table_name": "orders",
}
GIT_SHA = "a" * 40  # Synthetic provenance for isolated unit-test training only.
FEATURES = {"row_count": 110.0, "null_rate": 0.03, "d_row_count": 10.0, "d_null_rate": 0.01}


def _history(n=205):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rc = [
        {"ts": (start + timedelta(minutes=15 * i)).isoformat(), "value": 100 + i} for i in range(n)
    ]
    nr = [{"ts": point["ts"], "value": 0.02 + (i % 3) * 0.001} for i, point in enumerate(rc)]
    return rc, nr


@pytest.fixture(scope="module")
def local_candidate(tmp_path_factory):
    root = tmp_path_factory.mktemp("candidate")
    rc, nr = _history()
    manifest = registry.train_candidate(
        root,
        source=SOURCE,
        row_counts=rc,
        null_rates=nr,
        git_sha=GIT_SHA,
        parameters={"n_estimators": 5},
    )
    return root, manifest


@pytest.fixture(scope="module")
def archived_candidate(tmp_path_factory):
    root = tmp_path_factory.mktemp("archived-candidate")
    return root, registry.stage_export(root, ARCHIVE, source=SOURCE)


def _copy_bundle(tmp_path, candidate):
    root, manifest = candidate
    shutil.copytree(root, tmp_path / "registry")
    return tmp_path / "registry", copy.deepcopy(manifest)


def _qualified_fixture(root, original, *, seed=None, source=None):
    """Synthetic PASS evidence exercises admission mechanics, not real ML quality."""
    source = source or SOURCE
    data = (root / "versions" / original["model_version"] / "pipeline.skops").read_bytes()
    training = copy.deepcopy(original["training"])
    if seed is not None:
        pipeline = sio.loads(data, trusted=registry.TRUSTED_SKOPS_TYPES)
        matrix = np.asarray(build_rows(*_history())[1])
        pipeline.set_params(model__random_state=seed)
        pipeline.fit(matrix)
        data = sio.dumps(pipeline)
        training["seed"] = seed
    evaluation = {
        "split": "test",
        "data_sha256": "b" * 64,
        "metrics": {
            "event_recall": 1.0,
            "false_positive_rate": 0.01,
            "abrupt_median_delay_ticks": 0,
        },
    }
    return registry.stage_bytes(
        root,
        data,
        source=source,
        training=training,
        evaluation=evaluation,
        origin={"kind": "synthetic-unit-test-evidence"},
    )


def test_new_training_records_provenance_schema_seed_and_actual_time(local_candidate):
    root, manifest = local_candidate
    training = manifest["training"]
    assert training["git_sha"] == GIT_SHA
    assert training["dataset_id"] == training["training_data_sha256"]
    assert len(training["dataset_id"]) == 64
    assert training["seed"] == 42
    assert training["parameters"]["n_estimators"] == 5
    assert training["run_id"] is None  # No MLflow run is invented by unit tests.
    assert datetime.fromisoformat(training["training_started_at"]) <= datetime.fromisoformat(
        training["trained_at"]
    )
    assert manifest["feature_schema_sha256"] == FEATURE_SCHEMA_SHA256
    assert manifest["feature_schema_version"] == "1.0.0"
    assert not (root / "active").exists()


def test_training_and_single_point_serving_use_identical_feature_math(local_candidate):
    root, manifest = local_candidate
    loaded = registry.load_version(root, manifest["model_version"], source=SOURCE, demo=True)
    rc, nr = _history()
    _, rows = build_rows(rc, nr)
    current = {"row_count": rc[-1]["value"], "null_rate": nr[-1]["value"]}
    previous = {"row_count": rc[-2]["value"], "null_rate": nr[-2]["value"]}
    expected = loaded.pipeline.decision_function([rows[-1]])[0]
    result = loaded.predict_observations(current, previous)
    assert result["score"] == expected
    assert result["model_version"] == manifest["model_version"]
    assert result["feature_schema_version"] == manifest["feature_schema_version"]
    assert result["source"] == SOURCE
    assert result["is_anomaly"] == (result["score"] < 0)


def test_model_and_scaler_are_restored_without_fit(local_candidate, monkeypatch):
    root, manifest = local_candidate

    def forbidden(*args, **kwargs):
        pytest.fail("Restore/inference must never fit")

    monkeypatch.setattr(IsolationForest, "fit", forbidden)
    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    monkeypatch.setattr(Pipeline, "fit", forbidden)
    loaded = registry.load_version(root, manifest["model_version"], source=SOURCE, demo=True)
    assert loaded.predict(FEATURES) == loaded.predict(FEATURES)


def test_frozen_mlflow_model_restores_predictions_and_run_identity(archived_candidate):
    root, manifest = archived_candidate
    assert manifest["training"]["run_id"] == "269a806ac22140419c89186ee059eddb"
    assert manifest["origin"]["evaluation_run_id"] == "36db7cd846c94d05897732aa3eec448d"
    assert manifest["training"]["git_sha"] == "f94947c6c36e4f93be46dce5c861d434b672a894"
    assert manifest["training"]["trained_at"] is None
    assert manifest["training"]["time_status"] == "not_recorded_in_historical_export"
    loaded = registry.load_version(root, manifest["model_version"], source=SOURCE, demo=True)
    with zipfile.ZipFile(ARCHIVE) as bundle:
        rows = list(
            csv.DictReader(
                io.StringIO(bundle.read("final_test/artifacts/predictions.csv").decode())
            )
        )
    selected = [r for r in rows if r["table_name"] == "orders"]
    values = np.asarray([[float(r[key]) for key in FEATURE_NAMES] for r in selected])
    actual = loaded.pipeline.decision_function(values)
    np.testing.assert_allclose(actual, [float(r["score"]) for r in selected], rtol=0, atol=1e-12)


def test_failed_baseline_cannot_be_normal_active_but_demo_is_separate(tmp_path, archived_candidate):
    root, manifest = _copy_bundle(tmp_path, archived_candidate)
    version = manifest["model_version"]
    assert manifest["evaluation"]["overall_metrics"]["false_positive_rate"] > 0.1
    with pytest.raises(registry.RegistryError, match="quality admission"):
        registry.activate(root, version, source=SOURCE)
    assert not (root / "active").exists()
    registry.activate(root, version, source=SOURCE, demo=True)
    assert registry.load_active(root, source=SOURCE, demo=True).manifest["model_version"] == version
    with pytest.raises(registry.RegistryError, match="no valid active"):
        registry.load_active(root, source=SOURCE)


def test_explicit_switch_and_rollback_preserve_both_versions(tmp_path, local_candidate):
    root, original = _copy_bundle(tmp_path, local_candidate)
    first = _qualified_fixture(root, original)
    second = _qualified_fixture(root, original, seed=43)
    registry.activate(root, first["model_version"], source=SOURCE)
    before = registry.load_active(root, source=SOURCE).predict(FEATURES)
    registry.activate(root, second["model_version"], source=SOURCE)
    assert (
        registry.load_active(root, source=SOURCE).manifest["model_version"]
        == second["model_version"]
    )
    registry.activate(root, first["model_version"], source=SOURCE)
    assert registry.load_active(root, source=SOURCE).predict(FEATURES) == before
    assert (root / "versions" / second["model_version"] / "pipeline.skops").exists()


def test_staging_a_new_candidate_does_not_switch_active(tmp_path, local_candidate):
    root, original = _copy_bundle(tmp_path, local_candidate)
    active = _qualified_fixture(root, original)
    registry.activate(root, active["model_version"], source=SOURCE)
    _qualified_fixture(root, original, seed=43)
    assert (
        registry.load_active(root, source=SOURCE).manifest["model_version"]
        == active["model_version"]
    )


def test_same_table_in_another_connection_cannot_load_this_version(local_candidate):
    root, manifest = local_candidate
    other = {**SOURCE, "connection_id": "connection-b"}
    with pytest.raises(registry.RegistryError, match="source"):
        registry.load_version(root, manifest["model_version"], source=other, demo=True)


@pytest.mark.parametrize("field", ["connection_id", "schema_name", "project_id", "table_name"])
def test_source_identity_must_be_complete(field):
    source = {k: v for k, v in SOURCE.items() if k != field}
    with pytest.raises(registry.RegistryError, match="source"):
        registry.validate_source(source)


def test_corrupted_artifact_is_rejected_before_deserialization(
    tmp_path, local_candidate, monkeypatch
):
    root, manifest = _copy_bundle(tmp_path, local_candidate)
    path = root / "versions" / manifest["model_version"] / "pipeline.skops"
    path.write_bytes(path.read_bytes() + b"corrupt")
    monkeypatch.setattr(
        registry, "_load_pipeline", lambda *a: pytest.fail("checksum must be checked first")
    )
    with pytest.raises(registry.RegistryError, match="checksum"):
        registry.load_version(root, manifest["model_version"], source=SOURCE, demo=True)


def test_edited_manifest_is_rejected_before_loading(tmp_path, local_candidate):
    root, manifest = _copy_bundle(tmp_path, local_candidate)
    path = root / "versions" / manifest["model_version"] / "manifest.json"
    manifest["training"]["seed"] = 43
    path.write_text(json.dumps(manifest))
    with pytest.raises(registry.RegistryError, match="manifest checksum"):
        registry.load_version(root, manifest["model_version"], source=SOURCE, demo=True)


def test_bad_switch_keeps_previous_active_pointer(tmp_path, local_candidate):
    root, original = _copy_bundle(tmp_path, local_candidate)
    active = _qualified_fixture(root, original)
    registry.activate(root, active["model_version"], source=SOURCE)
    pointer = registry._pointer(root, SOURCE, False)
    before = pointer.read_bytes()
    with pytest.raises(registry.RegistryError):
        registry.activate(root, "if-" + "0" * 64, source=SOURCE)
    assert pointer.read_bytes() == before


@pytest.mark.parametrize("version", ["legacy-unknown", "../../outside", "latest", "if-bad"])
def test_invalid_version_has_no_path_or_latest_fallback(version, tmp_path):
    with pytest.raises(registry.RegistryError, match="model_version"):
        registry.load_version(tmp_path, version, source=SOURCE, demo=True)


@pytest.mark.parametrize(
    "value",
    [True, None, "0.1", float("nan"), float("inf"), pytest.param(10**400, id="float-overflow")],
)
def test_invalid_inference_number_is_rejected(value, local_candidate):
    root, manifest = local_candidate
    loaded = registry.load_version(root, manifest["model_version"], source=SOURCE, demo=True)
    with pytest.raises(ValueError, match="finite number"):
        loaded.predict({**FEATURES, "row_count": value})


def test_cannot_claim_admission_from_passed_flag_or_training_metrics(local_candidate):
    _, manifest = local_candidate
    training = manifest["training"]
    assert not registry.admitted(training, {"passed": True})
    same_data = {"split": "test", "data_sha256": training["training_data_sha256"], "metrics": {}}
    assert not registry.admitted(training, same_data)
    bad = {
        "split": "test",
        "data_sha256": "b" * 64,
        "metrics": {"event_recall": True, "false_positive_rate": 0, "abrupt_median_delay_ticks": 0},
    }
    assert not registry.admitted(training, bad)


def test_old_joblib_payload_still_loads_and_scores_after_feature_refactor(tmp_path, monkeypatch):
    rc, nr = _history()
    matrix = np.asarray(build_rows(rc, nr)[1])
    scaler = StandardScaler().fit(matrix)
    model = IsolationForest(n_estimators=5, random_state=42).fit(scaler.transform(matrix))
    monkeypatch.setattr(legacy, "MODELS_DIR", tmp_path)
    joblib.dump(
        {"scaler": scaler, "model": model, "trained_at": "2026-01-01", "n_points": len(matrix)},
        tmp_path / "orders__anomaly.joblib",
    )
    monkeypatch.setattr(
        legacy, "get_metrics", lambda table, metric, *a, **kw: rc if metric == "row_count" else nr
    )
    monkeypatch.setattr(
        legacy, "train", lambda *a, **kw: pytest.fail("old artifact must be reused")
    )
    results = legacy.score_table("orders")
    assert len(results) == len(matrix)
    np.testing.assert_allclose(
        [r["score"] for r in results], model.decision_function(scaler.transform(matrix)), atol=1e-6
    )


def test_new_training_and_legacy_loader_produce_same_vectors(monkeypatch):
    rc, nr = _history(3)
    nr.pop(1)
    monkeypatch.setattr(
        legacy, "get_metrics", lambda table, metric, *a, **kw: rc if metric == "row_count" else nr
    )
    _, matrix = legacy._load_features("orders", timedelta(days=1))
    expected = build_vector(rc[-1]["value"], nr[-1]["value"], rc[0]["value"], nr[0]["value"])
    np.testing.assert_array_equal(matrix[0], expected)
    assert validate_vector(dict(zip(FEATURE_NAMES, expected, strict=True))) == expected


def test_observation_conversion_reports_numeric_overflow_as_value_error():
    with pytest.raises(ValueError, match="representable"):
        build_vector(10**400, 0.02, 100, 0.02)


def test_cli_missing_version_is_controlled_error_with_no_training(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps(SOURCE))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.model_versions",
            "--registry",
            str(tmp_path / "registry"),
            "inspect",
            "--source",
            str(source),
            "--version",
            "if-" + "0" * 64,
            "--demo",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "registry").exists()


def test_invalid_code_identity_is_rejected_before_fit(tmp_path, monkeypatch):
    rc, nr = _history()
    monkeypatch.setattr(
        Pipeline, "fit", lambda *a, **kw: pytest.fail("validate provenance before fit")
    )
    with pytest.raises(registry.RegistryError, match="git_sha"):
        registry.train_candidate(
            tmp_path, source=SOURCE, row_counts=rc, null_rates=nr, git_sha="unknown"
        )


def test_pickle_input_is_a_controlled_error_and_never_uses_joblib(
    tmp_path, local_candidate, monkeypatch
):
    _, manifest = local_candidate
    monkeypatch.setattr(
        joblib, "load", lambda *a, **kw: pytest.fail("new registry must not unpickle")
    )
    with pytest.raises(registry.RegistryError):
        registry.stage_bytes(
            tmp_path, b"not-a-skops-zip", source=SOURCE, training=manifest["training"]
        )


def test_valid_json_with_wrong_manifest_type_is_controlled_error(tmp_path, local_candidate):
    root, manifest = _copy_bundle(tmp_path, local_candidate)
    (root / "versions" / manifest["model_version"] / "manifest.json").write_text("[]")
    with pytest.raises(registry.RegistryError):
        registry.load_version(root, manifest["model_version"], source=SOURCE, demo=True)


def test_inspection_metadata_cannot_change_prediction_version(local_candidate):
    root, manifest = local_candidate
    loaded = registry.load_version(root, manifest["model_version"], source=SOURCE, demo=True)
    view = loaded.manifest
    view["model_version"] = "pretend-version"
    view["source"]["connection_id"] = "other"
    prediction = loaded.predict(FEATURES)
    assert prediction["model_version"] == manifest["model_version"]
    assert prediction["source"] == SOURCE


def test_staging_does_not_expose_mutable_shared_schema(tmp_path, local_candidate, monkeypatch):
    root, original = local_candidate
    schema = copy.deepcopy(registry.FEATURE_SCHEMA)
    monkeypatch.setattr(registry, "FEATURE_SCHEMA", schema)
    data = (root / "versions" / original["model_version"] / "pipeline.skops").read_bytes()
    manifest = registry.stage_bytes(tmp_path, data, source=SOURCE, training=original["training"])
    manifest["feature_schema"]["names"][0] = "changed"
    assert schema["names"][0] == "row_count"


@pytest.mark.parametrize(
    "change", ["seed", "parameters", "environment", "unknown-parameter", "parameter-type"]
)
def test_artifact_must_match_declared_training_metadata(tmp_path, local_candidate, change):
    root, original = local_candidate
    data = (root / "versions" / original["model_version"] / "pipeline.skops").read_bytes()
    training = copy.deepcopy(original["training"])
    if change == "seed":
        training["seed"] = 999
    elif change == "parameters":
        training["parameters"]["n_estimators"] = 999
    elif change == "unknown-parameter":
        training["parameters"]["not_a_parameter"] = None
    elif change == "parameter-type":
        training["parameters"]["n_estimators"] = 5.0
    else:
        training["environment"]["scikit-learn"] = "0.0.0"
    with pytest.raises(registry.RegistryError):
        registry.stage_bytes(tmp_path, data, source=SOURCE, training=training)
    assert not (tmp_path / "versions").exists()


def test_unknown_skops_types_are_not_automatically_trusted(tmp_path, local_candidate, monkeypatch):
    root, original = local_candidate
    data = (root / "versions" / original["model_version"] / "pipeline.skops").read_bytes()
    monkeypatch.setattr(sio, "get_untrusted_types", lambda **kw: ["unapproved.CustomObject"])
    monkeypatch.setattr(
        sio, "loads", lambda *a, **kw: pytest.fail("unapproved type must not be loaded")
    )
    with pytest.raises(registry.RegistryError, match="unapproved"):
        registry.stage_bytes(tmp_path, data, source=SOURCE, training=original["training"])


def test_even_consistent_manifest_cannot_load_unknown_feature_schema(tmp_path, local_candidate):
    root, manifest = _copy_bundle(tmp_path, local_candidate)
    old_directory = root / "versions" / manifest["model_version"]
    del manifest["model_version"]
    manifest["feature_schema_version"] = "2.0.0"
    version = "if-" + registry.sha256(registry.canonical(manifest))
    manifest["model_version"] = version
    destination = root / "versions" / version
    old_directory.rename(destination)
    (destination / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(registry.RegistryError, match="feature schema"):
        registry.load_version(root, version, source=SOURCE, demo=True)


def test_reimporting_same_frozen_version_preserves_bytes(tmp_path, archived_candidate):
    root, manifest = _copy_bundle(tmp_path, archived_candidate)
    directory = root / "versions" / manifest["model_version"]
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    assert (
        registry.stage_export(root, ARCHIVE, source=SOURCE)["model_version"]
        == manifest["model_version"]
    )
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before
