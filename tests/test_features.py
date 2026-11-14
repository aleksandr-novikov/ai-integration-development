"""Unit tests for ml.features.validate_vector boundaries (#44).

Types, NaN/inf, overflow, a negative row_count and an impossible d_row_count
are already checked through the model registry and the inference service;
these tests cover the rest of the function directly.
"""

import pytest

from ml.features import FEATURE_NAMES, validate_vector


def features(row_count, null_rate, d_row_count, d_null_rate):
    return dict(zip(FEATURE_NAMES, (row_count, null_rate, d_row_count, d_null_rate), strict=True))


@pytest.mark.parametrize(
    "values",
    [
        pytest.param((0, 0, 0, 0), id="empty-table"),
        pytest.param((10, 1.0, 10, 1.0), id="null-rate-0-to-1"),
        pytest.param((10, 0.0, -5, -1.0), id="null-rate-1-to-0"),
    ],
)
def test_exact_physical_boundaries_are_accepted(values):
    vector = validate_vector(features(*values))
    assert vector == [float(value) for value in values]
    assert all(type(value) is float for value in vector)


@pytest.mark.parametrize(
    ("null_rate", "d_null_rate"),
    [pytest.param(0.2, 0.3, id="previous-below-0"), pytest.param(0.8, -0.3, id="previous-above-1")],
)
def test_previous_null_rate_outside_unit_interval_is_rejected(null_rate, d_null_rate):
    with pytest.raises(ValueError, match="impossible previous observation"):
        validate_vector(features(100, null_rate, 0, d_null_rate))


def test_subtraction_roundoff_at_zero_previous_null_rate_is_tolerated():
    # 0.3 - (0.1 + 0.2) == -5.55e-17: the previous null rate is 0 up to roundoff.
    assert validate_vector(features(100, 0.3, 0, 0.1 + 0.2))[3] == 0.1 + 0.2
    with pytest.raises(ValueError, match="impossible previous observation"):
        validate_vector(features(100, 0.3, 0, 0.3 + 1e-9))


@pytest.mark.parametrize(
    "values",
    [
        pytest.param({name: 0.0 for name in FEATURE_NAMES[:3]}, id="missing"),
        pytest.param({**features(0, 0, 0, 0), "extra": 0.0}, id="extra"),
        pytest.param([0.0, 0.0, 0.0, 0.0], id="positional-list"),
    ],
)
def test_feature_set_must_match_names_exactly(values):
    with pytest.raises(ValueError, match="exactly four named features"):
        validate_vector(values)


def test_vector_order_follows_feature_names_not_input_order():
    reversed_input = dict(reversed(features(110, 0.03, 10, 0.01).items()))
    assert list(reversed_input) != list(FEATURE_NAMES)
    assert validate_vector(reversed_input) == [110.0, 0.03, 10.0, 0.01]
