import csv
import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pytest

from scripts.generate_ml_dataset import (
    FEATURES,
    PROFILES,
    Config,
    fit_training_scaler,
    generate,
    load_features,
    write_dataset,
)


@pytest.fixture
def config():
    return Config(train_ticks=240, validation_ticks=120, test_ticks=120)


def test_identical_config_is_byte_reproducible_in_different_directories(tmp_path, config):
    first, second = tmp_path / "first", tmp_path / "second"
    assert write_dataset(first, config) == write_dataset(second, config)
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {
        p.name: p.read_bytes() for p in second.iterdir()
    }
    third = write_dataset(tmp_path / "third", replace(config, seed=43))
    assert (
        third["files"]["train.csv"]["sha256"]
        != json.loads((first / "manifest.json").read_text())["files"]["train.csv"]["sha256"]
    )


def test_splits_are_chronological_disjoint_and_have_unique_keys(config):
    data, _ = generate(config)
    assert max(r["ts"] for r in data["train"]) < min(r["ts"] for r in data["validation"])
    assert max(r["ts"] for r in data["validation"]) < min(r["ts"] for r in data["test"])
    rows = [row for rows in data.values() for row in rows]
    keys = {(r["project_id"], r["table_name"], r["ts"]) for r in rows}
    assert len(keys) == len(rows)
    assert len(data["train"]) == (config.train_ticks - 1) * len(PROFILES)
    assert len(data["validation"]) == config.validation_ticks * len(PROFILES)
    assert len(data["test"]) == config.test_ticks * len(PROFILES)


def test_labels_match_injected_intervals_and_train_is_normal(config):
    data, events = generate(config)
    assert all(r["is_anomaly"] == 0 and not r["incident_id"] for r in data["train"])
    assert len(events) == 18
    for split in ("validation", "test"):
        for table in PROFILES:
            expected = [e for e in events if e["split"] == split and e["table_name"] == table]
            assert {e["kind"] for e in expected} == {"null_spike", "volume_jump", "regime_change"}
            rows = [r for r in data[split] if r["table_name"] == table]
            assert any(not r["is_anomaly"] for r in rows)
            for row in rows:
                matches = [e for e in expected if e["start"] <= row["ts"] <= e["end"]]
                assert len(matches) <= 1
                assert row["is_anomaly"] == int(bool(matches))
                assert row["incident_id"] == (matches[0]["incident_id"] if matches else "")
            for event in expected:
                labelled = [r for r in rows if r["incident_id"] == event["incident_id"]]
                assert labelled[0]["ts"] == event["start"]
                assert labelled[-1]["ts"] == event["end"]
                if event["kind"] in ("null_spike", "regime_change"):
                    assert labelled[0]["null_rate"] > 0.09
                    assert labelled[-1]["null_rate"] < 0.03  # recovery is labelled too
                if event["kind"] == "volume_jump":
                    assert labelled[0]["d_row_count"] > 0
                    assert labelled[-1]["d_row_count"] < 0


def test_deltas_use_only_previous_observation_in_same_series(config):
    data, _ = generate(config)
    for table in PROFILES:
        previous = None
        for split in ("train", "validation", "test"):
            rows = [r for r in data[split] if r["table_name"] == table]
            for row in rows:
                if previous:
                    assert datetime.fromisoformat(row["ts"]) - datetime.fromisoformat(
                        previous["ts"]
                    ) == timedelta(minutes=config.interval_minutes)
                    assert row["d_row_count"] == row["row_count"] - previous["row_count"]
                    assert row["d_null_rate"] == pytest.approx(
                        row["null_rate"] - previous["null_rate"]
                    )
                previous = row  # retained across split boundaries, not across tables


def test_extending_future_does_not_change_train_or_validation(config):
    original, _ = generate(config)
    extended, _ = generate(replace(config, test_ticks=240))
    assert original["train"] == extended["train"]
    assert original["validation"] == extended["validation"]
    larger_validation, _ = generate(replace(config, validation_ticks=240))
    assert original["train"] == larger_validation["train"]


def test_feature_allowlist_and_scaler_fit_only_train(tmp_path, config):
    write_dataset(tmp_path, config)
    train = np.array(load_features(tmp_path, "train", "users"))
    scaler = fit_training_scaler(tmp_path, "users")
    assert FEATURES == ("row_count", "null_rate", "d_row_count", "d_null_rate")
    assert train.shape == (config.train_ticks - 1, 4)
    np.testing.assert_allclose(scaler.mean_, train.mean(axis=0))
    np.testing.assert_allclose(scaler.var_, train.var(axis=0))
    assert scaler.n_samples_seen_ == config.train_ticks - 1
    # Prove no hidden fit/reads on future files, even if they are unusable.
    (tmp_path / "validation.csv").write_text("invalid future data")
    (tmp_path / "test.csv").unlink()
    other = fit_training_scaler(tmp_path, "users")
    np.testing.assert_array_equal(other.mean_, scaler.mean_)
    with pytest.raises(ValueError, match="no rows"):
        fit_training_scaler(tmp_path, "missing")


def test_raw_values_are_finite_and_in_physical_range(config):
    data, _ = generate(config)
    for rows in data.values():
        for row in rows:
            assert np.isfinite([row[f] for f in FEATURES]).all()
            assert row["row_count"] >= 0
            assert 0 <= row["null_rate"] <= 1


def test_refuses_overwriting_an_existing_dataset(tmp_path, config):
    write_dataset(tmp_path, config)
    before = (tmp_path / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="not empty"):
        write_dataset(tmp_path, replace(config, seed=43))
    assert (tmp_path / "manifest.json").read_bytes() == before


@pytest.mark.parametrize(
    "changes",
    [
        {"start": "2026-01-01"},
        {"start": "2026-01-01T00:00:00+03:00"},
        {"train_ticks": 200},
        {"validation_ticks": 95},
        {"test_ticks": 0},
        {"interval_minutes": 0},
        {"project_id": " "},
    ],
)
def test_invalid_config_creates_no_output(tmp_path, config, changes):
    output = tmp_path / "dataset"
    with pytest.raises(ValueError):
        write_dataset(output, replace(config, **changes))
    assert not output.exists()


def test_csv_roundtrip_preserves_feature_values(tmp_path, config):
    expected, _ = generate(config)
    write_dataset(tmp_path, config)
    with (tmp_path / "validation.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(expected["validation"])
    for actual, source in zip(rows, expected["validation"], strict=True):
        assert actual["incident_id"] == source["incident_id"]
        for field in FEATURES:
            assert float(actual[field]) == source[field]


def test_cli_generates_without_site_packages_or_database_config(tmp_path):
    env = dict(os.environ)
    env.pop("DATABASE_URL", None)
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "scripts.generate_ml_dataset",
            "--output",
            str(tmp_path),
            "--train-ticks",
            "201",
            "--validation-ticks",
            "96",
            "--test-ticks",
            "96",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Dataset " in result.stdout
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["files"]["train.csv"]["rows"] == 600
