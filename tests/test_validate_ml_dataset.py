"""Dataset validation must reject independently introduced faults, not just bad hashes."""

import csv
import hashlib
import json
import subprocess
import sys

import pytest

from scripts.demo_dataset_validation import corrupt_dataset
from scripts.generate_ml_dataset import FIELDS, Config, write_dataset
from scripts.validate_ml_dataset import validate_dataset


@pytest.fixture
def dataset(tmp_path):
    directory = tmp_path / "dataset"
    write_dataset(directory, Config(train_ticks=201, validation_ticks=96, test_ticks=96))
    return directory


def edit_rows(directory, edit, split="validation"):
    path = directory / f"{split}.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    edit(rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def failures(report):
    return {c["check"] for c in report["checks"] if c["status"] == "fail"}


def test_valid_dataset_read_only(dataset):
    before = {p.name: p.read_bytes() for p in dataset.iterdir()}
    report = validate_dataset(dataset)
    assert report["valid"], report
    assert len(report["checks"]) >= 20
    assert before == {p.name: p.read_bytes() for p in dataset.iterdir()}


@pytest.mark.parametrize(
    ("column", "value", "check"),
    [
        ("project_id", "", "required_identifiers"),
        ("table_name", "unknown", "required_identifiers"),
        ("row_count", "text", "numeric_types"),
        ("row_count", "", "numeric_types"),
        ("row_count", "-1", "row_count_domain"),
        ("row_count", "1.5", "row_count_domain"),
        ("null_rate", "1.01", "null_rate_range"),
        ("null_rate", "-0.01", "null_rate_range"),
        ("d_row_count", "inf", "finite_features"),
        ("d_null_rate", "NaN", "finite_features"),
        ("ts", "2026-01-03T02:15:00", "utc_timestamps"),
        ("ts", "not-a-date", "utc_timestamps"),
        ("ts", "2026-01-03T02:15:00+01:00", "utc_timestamps"),
        ("d_row_count", "999999", "delta_alignment_row_count"),
        ("d_null_rate", "0.9", "delta_alignment_null_rate"),
        ("is_anomaly", "1", "incident_labels"),
        ("is_anomaly", "2", "label_domain"),
        ("incident_id", "fake", "incident_labels"),
    ],
)
def test_individual_faults(dataset, column, value, check):
    edit_rows(dataset, lambda rows: rows[0].__setitem__(column, value))
    report = validate_dataset(dataset)
    assert check in failures(report)
    assert "manifest_integrity" in failures(report)
    assert not report["valid"]


def test_duplicate_and_split_overlap(dataset):
    with (dataset / "train.csv").open(newline="", encoding="utf-8") as stream:
        train_row = next(csv.DictReader(stream))
    edit_rows(dataset, lambda rows: rows.append(train_row))
    assert {"unique_observation_key", "split_separation"} <= failures(validate_dataset(dataset))


def test_out_of_order(dataset):
    def swap(rows):
        rows[0], rows[3] = rows[3], rows[0]

    edit_rows(dataset, swap)
    assert "observation_order" in failures(validate_dataset(dataset))


def test_missing_tick(dataset):
    edit_rows(dataset, lambda rows: rows.pop(0))
    assert "feature_time_grid" in failures(validate_dataset(dataset))


@pytest.mark.parametrize("contents", ["", "other,columns\n1,2\n", ",".join(FIELDS) + "\n"])
def test_bad_csv(dataset, contents):
    (dataset / "train.csv").write_text(contents)
    assert "schema_train" in failures(validate_dataset(dataset))


@pytest.mark.parametrize("contents", ["{", "[]", "null", '{"config": {"train_ticks": "bad"}}'])
def test_bad_manifest(dataset, contents):
    (dataset / "manifest.json").write_text(contents)
    assert "manifest_integrity" in failures(validate_dataset(dataset))


def test_missing_dataset(tmp_path):
    report = validate_dataset(tmp_path / "missing")
    assert not report["valid"]
    assert {"schema_train", "schema_validation", "schema_test"} <= failures(report)


@pytest.mark.parametrize("contents", ["{", "{}", "[]", '[{"incident_id": "x"}]'])
def test_bad_incidents(dataset, contents):
    (dataset / "incidents.json").write_text(contents)
    assert "incident_intervals" in failures(validate_dataset(dataset))


def test_corrupted_demo_bounded_and_reproducible(dataset):
    corrupt_dataset(dataset)
    first = validate_dataset(dataset)
    assert first == validate_dataset(dataset)
    assert first["failed_checks"] >= 12
    assert all(len(c["examples"]) <= 5 for c in first["checks"])


def test_semantic_checks_survive_updated_hashes(dataset):
    edit_rows(dataset, lambda rows: rows[0].__setitem__("null_rate", "2"))
    path = dataset / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["validation.csv"]["sha256"] = hashlib.sha256(
        (dataset / "validation.csv").read_bytes()
    ).hexdigest()
    del manifest["dataset_id"]
    manifest["dataset_id"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    path.write_text(json.dumps(manifest))
    report = validate_dataset(dataset)
    assert "manifest_integrity" not in failures(report)
    assert "null_rate_range" in failures(report)


def test_cli_exit_codes_and_no_overwrite(dataset, tmp_path):
    command = [
        sys.executable,
        "-m",
        "scripts.validate_ml_dataset",
        "--dataset",
        str(dataset),
        "--report",
    ]
    report_path = tmp_path / "report.json"
    good = subprocess.run([*command, str(report_path)], capture_output=True, text=True)
    assert good.returncode == 0, good.stderr
    assert json.loads(report_path.read_text())["valid"]
    corrupt_dataset(dataset)
    bad = subprocess.run([*command, str(report_path)], capture_output=True, text=True)
    assert bad.returncode == 1, bad.stderr
    assert not json.loads(report_path.read_text())["valid"]
    original = (dataset / "train.csv").read_bytes()
    blocked = subprocess.run([*command, str(dataset / "train.csv")], capture_output=True)
    assert blocked.returncode == 2
    assert (dataset / "train.csv").read_bytes() == original


def test_custom_sampling_interval(tmp_path):
    dataset = tmp_path / "custom"
    write_dataset(
        dataset,
        Config(
            train_ticks=201,
            validation_ticks=96,
            test_ticks=96,
            interval_minutes=5,
            project_id="custom",
        ),
    )
    assert validate_dataset(dataset)["valid"]


@pytest.mark.parametrize(
    "field,value",
    [("train_ticks", 201.5), ("project_id", 5), ("start", None), ("interval_minutes", True)],
)
def test_invalid_config_types(dataset, field, value):
    path = dataset / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["config"][field] = value
    path.write_text(json.dumps(manifest))
    assert "manifest_integrity" in failures(validate_dataset(dataset))
