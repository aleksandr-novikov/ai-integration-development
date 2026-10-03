"""Production detector behaviour on unseen stationary/synthetic incident windows.

Unlike the older smoke tests, training and scoring never use the same observations.
The archived #9 test is audited separately; these are contract/behaviour scenarios.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from ml import anomaly_detector as detector


def metric_window(seed, size, start, noise=1.0):
    rng = np.random.default_rng(seed)
    values = {
        "row_count": 10_000 + rng.normal(0, 20 * noise, size),
        "null_rate": np.clip(0.02 + rng.normal(0, 0.0005 * noise, size), 0, 1),
    }
    return {
        metric: [
            {
                "ts": (start + timedelta(minutes=15 * i)).isoformat(),
                "value": float(value),
                "tags": None,
            }
            for i, value in enumerate(series)
        ]
        for metric, series in values.items()
    }


def source(window):
    def get_metrics(table, metric, project_id=None, window=None):
        return data[metric]

    data = window
    return get_metrics


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    directory = tmp_path_factory.mktemp("independent-quality-model")
    train = metric_window(103, 640, datetime(2026, 2, 1, tzinfo=UTC))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(detector, "MODELS_DIR", directory)
        patch.setattr(detector, "get_metrics", source(train))
        metadata = detector.train("quality", project_id="course-quality")
    assert metadata["n_points"] == 639
    return directory, train


@pytest.fixture
def score(fitted, monkeypatch):
    directory, train = fitted
    monkeypatch.setattr(detector, "MODELS_DIR", directory)
    monkeypatch.setattr(detector, "train", lambda *a, **kw: pytest.fail("test must not retrain"))

    def run(window):
        assert train["row_count"][-1]["ts"] < window["row_count"][0]["ts"]
        monkeypatch.setattr(detector, "get_metrics", source(window))
        return detector.score_table("quality", project_id="course-quality")

    return run


def held_out(noise=1):
    return metric_window(211, 257, datetime(2026, 3, 1, tzinfo=UTC), noise=noise)


@pytest.mark.parametrize("size", [0, 1, 200])
def test_empty_or_short_history_cannot_create_model(size, tmp_path, monkeypatch):
    window = metric_window(103, size, datetime(2026, 2, 1, tzinfo=UTC))
    monkeypatch.setattr(detector, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(detector, "get_metrics", source(window))
    with pytest.raises(detector.InsufficientDataError):
        detector.train("quality", project_id="course-quality")
    assert not list(tmp_path.glob("*.joblib"))


def test_unseen_stationary_window_keeps_false_alarms_below_case_limit(score):
    predictions = score(held_out())
    assert len(predictions) == 256
    assert sum(p["is_anomaly"] for p in predictions) / len(predictions) <= 0.1
    assert all(np.isfinite(p["score"]) for p in predictions)


@pytest.mark.parametrize("kind", ["null_spike", "volume_jump", "regime_change"])
def test_unseen_incidents_detected_inside_their_intervals(score, kind):
    window = held_out()
    first, last = (90, 98) if kind != "regime_change" else (120, 180)
    for i in range(first, last):
        if kind in ("null_spike", "regime_change"):
            window["null_rate"][i]["value"] += 0.2 if kind == "null_spike" else 0.08
        if kind in ("volume_jump", "regime_change"):
            window["row_count"][i]["value"] *= 1.5 if kind == "volume_jump" else 1.15
    predictions = score(window)
    start, end = window["row_count"][first]["ts"], window["row_count"][last]["ts"]
    detected = [p for p in predictions if start <= p["ts"] <= end and p["is_anomaly"]]
    assert detected, f"missed {kind} on independent observations"


def test_higher_measurement_noise_increases_false_alarms(score):
    normal = score(held_out())
    noisy = score(held_out(noise=5))
    assert all(np.isfinite(p["score"]) for p in noisy)
    assert sum(p["is_anomaly"] for p in noisy) > sum(p["is_anomaly"] for p in normal)


def test_missing_metric_ticks_are_aligned_without_zero_imputation(score, monkeypatch):
    window = held_out()
    removed = {window["null_rate"][i]["ts"] for i in (40, 41, 42)}
    window["null_rate"] = [r for r in window["null_rate"] if r["ts"] not in removed]
    predictions = score(window)
    assert len(predictions) == 253
    assert not removed.intersection(p["ts"] for p in predictions)
    assert all(np.isfinite(p["score"]) for p in predictions)
    monkeypatch.setattr(detector, "get_metrics", source(window))
    timestamps, features = detector._load_features("quality", timedelta(days=14), "course-quality")
    # Production behaviour: deltas bridge the gap between matched observations.
    index = timestamps.index(datetime.fromisoformat(window["row_count"][43]["ts"]))
    assert features[index, 2] == pytest.approx(
        window["row_count"][43]["value"] - window["row_count"][39]["value"]
    )


def test_repeated_scoring_is_identical_and_does_not_retrain(score):
    assert score(held_out()) == score(held_out())
