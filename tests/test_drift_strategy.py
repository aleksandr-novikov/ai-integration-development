"""Regression checks for the offline #17 drift-policy scenario."""

import numpy as np
import pytest

from scripts import check_drift_strategy as policy

FIRST = policy.REFERENCE_VECTORS + 1
LAST = policy.REFERENCE_VECTORS + policy.WINDOW_VECTORS


@pytest.fixture
def normal():
    return policy.scenario(np.random.default_rng(7))


def window_from(future):
    return policy.select(policy.features(future), FIRST, LAST)


def test_normal_growth_is_expected_not_drift(normal):
    reference, windows, _ = normal
    result = policy.compare(reference, windows[0])
    # The fixed reference flags the growing level itself; the policy does not.
    assert result["features"]["row_count"]["ks_d"] == 1.0
    assert result["features"]["row_count"]["signal"] is False
    assert result["status"] == "expected_growth"
    assert result["row_count_trend"]["trend_break"] is False


def test_level_step_is_caught_by_trend_not_by_increments():
    results, common = policy.run(np.random.default_rng(7), step=2000)
    assert [r["status"] for r in results] == ["signal", "signal"]
    assert common == ["row_count"]


def test_growth_rate_change_agrees_between_level_and_increments():
    _, common = policy.run(np.random.default_rng(7), growth=15)
    assert common == ["d_row_count", "row_count"]


@pytest.mark.parametrize("feature, value", [
    ("row_count", np.nan), ("row_count", np.inf), ("row_count", -np.inf),
    ("null_rate", np.nan), ("null_rate", np.inf), ("null_rate", -np.inf),
])
def test_nonfinite_current_value_is_never_no_signal(normal, feature, value):
    reference, _, future = normal
    damaged = window_from(policy.corrupted(future, FIRST + 10, feature, value))
    result = policy.compare(reference, damaged)
    assert result["status"] == "invalid_data"
    assert result["invalid_vectors"]["nonfinite"] == 2


def test_nonfinite_reference_value_blocks_comparison(normal):
    reference, windows, _ = normal
    damaged = {name: values.copy() for name, values in reference.items()}
    damaged["d_null_rate"][5] = np.nan
    result = policy.compare(damaged, windows[0])
    assert (result["status"], result["window"]) == ("invalid_data", "reference")


def test_out_of_bounds_value_is_invalid(normal):
    reference, _, future = normal
    damaged = window_from(policy.corrupted(future, FIRST + 10, "null_rate", 1.5))
    assert policy.compare(reference, damaged)["invalid_vectors"]["out_of_bounds"] == 2


def test_sufficiency_is_counted_after_validation(normal):
    reference, _, future = normal
    damaged = {name: np.array(values) for name, values in future.items()}
    damaged["row_count"] = damaged["row_count"].astype(float)
    # 88 corrupted ticks invalidate 89 vectors: 288 present, only 199 valid.
    damaged["row_count"][(future["tick"] >= FIRST) & (future["tick"] < FIRST + 88)] = np.nan
    current = window_from(damaged)
    assert len(current["row_count"]) == policy.WINDOW_VECTORS
    result = policy.compare(reference, current)
    assert result["status"] == "insufficient_data"
    assert result["valid_vectors"] == 199


def test_delta_across_collection_gap_is_dropped(normal):
    _, _, future = normal
    kept = (future["tick"] < FIRST + 10) | (future["tick"] > FIRST + 12)
    window = window_from({name: values[kept] for name, values in future.items()})
    # Three missing ticks remove three vectors plus the delta spanning the gap.
    assert len(window["d_row_count"]) == policy.WINDOW_VECTORS - 4
    assert FIRST + 13 not in window["tick"]


def test_trend_is_not_extrapolated_beyond_reference_length(normal):
    reference, windows, _ = normal
    late = windows[0] | {"tick": windows[0]["tick"] + policy.REFERENCE_VECTORS}
    assert policy.compare(reference, late)["status"] == "baseline_expired"
