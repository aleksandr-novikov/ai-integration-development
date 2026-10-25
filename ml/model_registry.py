"""Local immutable model/scaler bundles and explicit per-source activation.

No Flask, scheduler, MLflow server or training is invoked while loading/scoring.
The registry directory and evaluation evidence are maintained by a trusted operator.
Checksums detect corruption; they are not signatures from an external authority.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from ml.features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    FEATURE_SCHEMA_SHA256,
    FEATURE_SCHEMA_VERSION,
    build_rows,
    build_vector,
    validate_vector,
)

VERSION_PATTERN = re.compile(r"if-[a-f0-9]{64}\Z")
SOURCE_FIELDS = ("project_id", "connection_id", "schema_name", "table_name")
TRUSTED_SKOPS_TYPES = ["sklearn.tree._tree.Tree"]
TARGETS = {"event_recall": 0.8, "false_positive_rate": 0.1, "abrupt_median_delay_ticks": 1.0}


class RegistryError(ValueError):
    """Missing, incompatible or inadmissible model; never trigger a fallback fit."""


def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def validate_source(source):
    if not isinstance(source, dict) or set(source) != set(SOURCE_FIELDS):
        raise RegistryError("source requires project_id, connection_id, schema_name and table_name")
    if any(not isinstance(source[key], str) or not source[key].strip() for key in SOURCE_FIELDS):
        raise RegistryError("source values must be nonempty strings")
    if any(len(source[key]) > (128 if key.endswith("_id") else 255) for key in SOURCE_FIELDS):
        raise RegistryError("source value exceeds the contract length limit")
    return dict(source)


def _digest(value, size, name):
    if not isinstance(value, str) or not re.fullmatch(f"[a-f0-9]{{{size}}}", value):
        raise RegistryError(f"invalid {name}")


def _validate_training(training):
    if not isinstance(training, dict):
        raise RegistryError("training metadata must be an object")
    for key, size in [("git_sha", 40), ("dataset_id", 64), ("training_data_sha256", 64)]:
        _digest(training.get(key), size, key)
    if (
        type(training.get("seed")) is not int
        or not 0 <= training["seed"] < 2**32
        or not isinstance(training.get("parameters"), dict)
    ):
        raise RegistryError("training seed and parameters are required")
    if not {"n_estimators", "contamination", "max_samples"} <= training["parameters"].keys():
        raise RegistryError("training parameters must identify the fitted configuration")
    if training.get("run_id") is not None:
        _digest(training["run_id"], 32, "run_id")
    if training.get("trained_at") is None:
        if training.get("time_status") != "not_recorded_in_historical_export":
            raise RegistryError("training time is required for new candidates")
    else:
        try:
            stamp = datetime.fromisoformat(training["trained_at"])
            if stamp.utcoffset() is None:
                raise ValueError("timezone required")
        except (ValueError, TypeError) as exc:
            raise RegistryError("invalid training timestamp") from exc
    if not isinstance(training.get("environment"), dict) or not training["environment"].get(
        "scikit-learn"
    ):
        raise RegistryError("training environment must identify scikit-learn")


def admitted(training, evaluation):
    """Do not trust a supplied passed=true flag or evaluate on the training file."""
    if not isinstance(evaluation, dict) or evaluation.get("split") != "test":
        return False
    data_hash = evaluation.get("data_sha256")
    if not isinstance(data_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", data_hash):
        return False
    if data_hash == training["training_data_sha256"]:
        return False
    groups = [evaluation.get("metrics")]
    if "overall_metrics" in evaluation:
        groups.append(evaluation["overall_metrics"])
    for metrics in groups:
        if not isinstance(metrics, dict):
            return False
        for name, target in TARGETS.items():
            value = metrics.get(name)
            if type(value) not in (int, float) or not math.isfinite(value):
                return False
            if name == "event_recall" and not target <= value <= 1:
                return False
            if name == "false_positive_rate" and not 0 <= value <= target:
                return False
            if name == "abrupt_median_delay_ticks" and not 0 <= value <= target:
                return False
    return True


def _load_pipeline(data, training):
    import skops.io as sio

    if training["environment"]["scikit-learn"] != importlib.metadata.version("scikit-learn"):
        raise RegistryError("restore the recorded scikit-learn version before loading")
    try:
        unknown = set(sio.get_untrusted_types(data=data)) - set(TRUSTED_SKOPS_TYPES)
    except Exception as exc:
        raise RegistryError("invalid skops artifact; pickle/joblib input is not accepted") from exc
    if unknown:
        raise RegistryError("unapproved types in model artifact")
    try:
        pipeline = sio.loads(data, trusted=TRUSTED_SKOPS_TYPES)
    except Exception as exc:
        raise RegistryError("cannot deserialize the approved skops artifact") from exc
    if type(pipeline) is not Pipeline or [name for name, _ in pipeline.steps] != [
        "scaler",
        "model",
    ]:
        raise RegistryError("expected a StandardScaler + IsolationForest pipeline")
    scaler, model = pipeline.named_steps["scaler"], pipeline.named_steps["model"]
    if type(scaler) is not StandardScaler or type(model) is not IsolationForest:
        raise RegistryError("unexpected pipeline component")
    try:
        check_is_fitted(scaler)
        check_is_fitted(model)
    except ValueError as exc:
        raise RegistryError("pipeline components must already be fitted") from exc
    if scaler.n_features_in_ != 4 or model.n_features_in_ != 4:
        raise RegistryError("artifact has an incompatible feature count")
    if model.random_state != training["seed"]:
        raise RegistryError("artifact seed does not match training metadata")
    parameters = model.get_params(deep=False)
    if any(
        key not in parameters
        or type(parameters[key]) is not type(value)
        or parameters[key] != value
        for key, value in training["parameters"].items()
    ):
        raise RegistryError("artifact parameters do not match training metadata")
    return pipeline


def stage_bytes(root, data, *, source, training, evaluation=None, origin=None):
    """Stage trusted local training/export output, without activating it.

    A single skops artifact covers the model AND its fitted scaler atomically.
    Version identity covers artifact bytes, schema, source and all provenance.
    """
    source = validate_source(source)
    _validate_training(training)
    _load_pipeline(data, training)
    contents = {
        "format_version": 1,
        "source": source,
        "feature_schema": FEATURE_SCHEMA,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "training": training,
        "evaluation": evaluation,
        "origin": origin,
        "artifact": {
            "file": "pipeline.skops",
            "sha256": sha256(data),
            "contains": ["scaler", "model"],
        },
    }
    contents = json.loads(canonical(contents))  # Detach mutable caller/schema dictionaries.
    version = "if-" + sha256(canonical(contents))
    manifest = {**contents, "model_version": version}
    versions = Path(root) / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    destination = versions / version
    if destination.exists():
        # Reimporting the same immutable bytes is safe, but does not repair corruption.
        load_version(root, version, source=source, demo=True)
        return manifest
    with tempfile.TemporaryDirectory(prefix=".stage-", dir=versions) as temporary:
        pending = Path(temporary) / "bundle"
        pending.mkdir()
        (pending / "pipeline.skops").write_bytes(data)
        (pending / "manifest.json").write_bytes(canonical(manifest) + b"\n")
        try:
            pending.rename(destination)
        except OSError:
            if not destination.is_dir():
                raise
            load_version(root, version, source=source, demo=True)
    return manifest


@dataclass(frozen=True)
class LoadedVersion:
    _manifest: dict
    pipeline: Pipeline

    @property
    def manifest(self):
        return json.loads(canonical(self._manifest))

    def predict(self, features):
        vector = validate_vector(features)
        score = float(self.pipeline.decision_function(np.asarray([vector], dtype=float))[0])
        if not math.isfinite(score):
            raise RegistryError("model returned a nonfinite score")
        return {
            "source": dict(self._manifest["source"]),
            "model_version": self._manifest["model_version"],
            "feature_schema_version": self._manifest["feature_schema_version"],
            "score": score,
            "threshold": 0,
            "is_anomaly": score < 0,
        }

    def predict_observations(self, current, previous):
        vector = build_vector(
            current["row_count"], current["null_rate"], previous["row_count"], previous["null_rate"]
        )
        return self.predict(dict(zip(FEATURE_NAMES, vector, strict=True)))


def load_version(root, version, *, source, demo=False):
    """Verify manifest and artifact bytes BEFORE deserialization; no fallback."""
    validate_source(source)
    if not isinstance(version, str) or not VERSION_PATTERN.fullmatch(version):
        raise RegistryError("invalid model_version")
    directory = Path(root) / "versions" / version
    try:
        manifest = json.loads((directory / "manifest.json").read_bytes())
        if not isinstance(manifest, dict):
            raise RegistryError("version manifest must be an object")
        contents = {key: value for key, value in manifest.items() if key != "model_version"}
        if manifest["model_version"] != version or "if-" + sha256(canonical(contents)) != version:
            raise RegistryError("manifest checksum mismatch")
        if manifest["format_version"] != 1 or manifest["source"] != source:
            raise RegistryError("source or manifest format mismatch")
        if (
            manifest["feature_schema_version"] != FEATURE_SCHEMA_VERSION
            or manifest["feature_schema_sha256"] != FEATURE_SCHEMA_SHA256
            or manifest["feature_schema"] != FEATURE_SCHEMA
        ):
            raise RegistryError("incompatible feature schema")
        _validate_training(manifest["training"])
        if not demo and not admitted(manifest["training"], manifest["evaluation"]):
            raise RegistryError("model has no positive quality admission; use explicit demo mode")
        artifact = manifest["artifact"]
        if artifact["file"] != "pipeline.skops" or artifact["contains"] != ["scaler", "model"]:
            raise RegistryError("invalid artifact descriptor")
        data = (directory / "pipeline.skops").read_bytes()
        if sha256(data) != artifact["sha256"]:
            raise RegistryError("model/scaler checksum mismatch")
        pipeline = _load_pipeline(data, manifest["training"])
    except RegistryError:
        raise
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as exc:
        raise RegistryError("missing or malformed version bundle") from exc
    return LoadedVersion(manifest, pipeline)


def _pointer(root, source, demo):
    validate_source(source)
    return (
        Path(root) / ("demo-active" if demo else "active") / (sha256(canonical(source)) + ".json")
    )


def activate(root, version, *, source, demo=False):
    """Explicit switch or rollback; a failed validation leaves the old pointer intact."""
    loaded = load_version(root, version, source=source, demo=demo)
    path = _pointer(root, source, demo)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical({"model_version": version, "source": source, "demo": demo}) + b"\n"
    fd, temporary = tempfile.mkstemp(prefix=".activate-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return loaded.manifest


def load_active(root, *, source, demo=False):
    path = _pointer(root, source, demo)
    try:
        pointer = json.loads(path.read_bytes())
        if pointer["source"] != source or pointer["demo"] is not demo:
            raise RegistryError("active pointer scope mismatch")
        version = pointer["model_version"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RegistryError("no valid active version for this source and mode") from exc
    return load_version(root, version, source=source, demo=demo)


def train_candidate(
    root, *, source, row_counts, null_rates, git_sha, seed=42, run_id=None, parameters=None
):
    """New offline training uses the same build_rows/build_vector as serving.

    Training never changes an active pointer. Quality is unevaluated by default.
    Dataset ID hashes the exact input snapshot, scoped to the complete source.
    """
    import skops.io as sio

    source = validate_source(source)
    _digest(git_sha, 40, "git_sha")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise RegistryError("seed must be an unsigned 32-bit integer")
    if run_id is not None:
        _digest(run_id, 32, "run_id")
    if parameters is not None and not isinstance(parameters, dict):
        raise RegistryError("parameters must be an object")
    snapshot = {"source": source, "row_counts": row_counts, "null_rates": null_rates}
    data_hash = sha256(canonical(snapshot))
    _, rows = build_rows(row_counts, null_rates)
    if len(rows) < 200:
        raise RegistryError("at least 200 aligned training vectors are required")
    matrix = np.asarray(
        [validate_vector(dict(zip(FEATURE_NAMES, row, strict=True))) for row in rows]
    )
    started = datetime.now(UTC).isoformat(timespec="microseconds")
    params = {
        "n_estimators": 100,
        "contamination": 0.01,
        "max_samples": "auto",
        **(parameters or {}),
    }
    if "random_state" in params:
        raise RegistryError("use seed instead of overriding random_state")
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("model", IsolationForest(**params, random_state=seed))]
    )
    pipeline.fit(matrix)
    trained_at = datetime.now(UTC).isoformat(timespec="microseconds")
    training = {
        "git_sha": git_sha,
        "dataset_id": data_hash,
        "training_data_sha256": data_hash,
        "seed": seed,
        "parameters": params,
        "run_id": run_id,
        "training_started_at": started,
        "trained_at": trained_at,
        "time_status": "recorded",
        "n_points": len(rows),
        "python_version": platform.python_version(),
        "source_sha256": {
            name: sha256((Path(__file__).parent / name).read_bytes())
            for name in ["features.py", "model_registry.py"]
        },
        "environment": {
            name: importlib.metadata.version(name) for name in ["scikit-learn", "numpy", "skops"]
        },
    }
    return stage_bytes(
        root,
        sio.dumps(pipeline),
        source=source,
        training=training,
        origin={"kind": "local-candidate", "automatic_activation": False},
    )


def stage_export(root, archive, *, source):
    """Reuse the frozen MLflow final_test artifact; never train or alter #9."""
    import io
    import zipfile

    import yaml

    from scripts.check_model_quality import EXPECTED_ARCHIVE_SHA256, read_report

    source = validate_source(source)
    archive = Path(archive)
    raw = archive.read_bytes()
    if sha256(raw) != EXPECTED_ARCHIVE_SHA256:
        raise RegistryError("expected the immutable #9 MLflow export")
    report = read_report(archive)
    table = source["table_name"]
    if table not in report["by_table"]:
        raise RegistryError("table is not present in the exported benchmark")
    with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
        summary = json.loads(bundle.read("summary.json"))
        manifest = json.loads(bundle.read("final_test/artifacts/manifest.json"))
        prefix = f"final_test/artifacts/models/{table}/"
        metadata = yaml.safe_load(bundle.read(prefix + "MLmodel"))
        data = bundle.read(prefix + "model.skops")
    training = {
        "git_sha": report["source_git_sha"],
        "dataset_id": report["dataset_id"],
        "training_data_sha256": manifest["files"]["train.csv"]["sha256"],
        "seed": summary["seed"],
        "parameters": summary["selection"]["parameters"],
        "run_id": summary["selection"]["validation_run_id"],
        # MLmodel time is serialization time, not proof of the time fit completed.
        "trained_at": None,
        "time_status": "not_recorded_in_historical_export",
        "artifact_created_at": metadata["utc_time_created"],
        "environment": summary["provenance"]["packages"],
        "python_version": summary["provenance"]["python"],
        "source_sha256": summary["provenance"]["source_sha256"],
    }
    evaluation = {
        "split": "test",
        "data_sha256": manifest["files"]["test.csv"]["sha256"],
        "run_id": report["run_id"],
        "metrics": report["by_table"][table],
        "overall_metrics": report["overall"],
    }
    return stage_bytes(
        root,
        data,
        source=source,
        training=training,
        evaluation=evaluation,
        origin={
            "kind": "mlflow-export",
            "archive_sha256": sha256(raw),
            "evaluation_run_id": report["run_id"],
            "original_pipeline_sha256": sha256(data),
        },
    )
