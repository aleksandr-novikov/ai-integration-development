"""Shared four-feature construction, independent of Flask and storage."""

import hashlib
import json
import math
from itertools import pairwise

FEATURE_NAMES = ("row_count", "null_rate", "d_row_count", "d_null_rate")
FEATURE_SCHEMA_VERSION = "1.0.0"
FEATURE_SCHEMA = {
    "version": FEATURE_SCHEMA_VERSION,
    "names": list(FEATURE_NAMES),
    "null_rate": "table-level mean column null rate",
    "alignment": "inner join on canonical timestamp, first aligned point omitted",
    "deltas": "current minus previous available aligned observation; no time normalization",
    "scaling": "raw features; use the scaler stored with the model, never fit at inference",
}
FEATURE_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(FEATURE_SCHEMA, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def build_vector(row_count, null_rate, previous_row_count, previous_null_rate):
    """Same arithmetic for batch training and the future online adapter."""
    try:
        current_rc, current_nr, previous_rc, previous_nr = map(
            float, (row_count, null_rate, previous_row_count, previous_null_rate)
        )
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("observations must be representable as float numbers") from exc
    return [
        current_rc,
        current_nr,
        current_rc - previous_rc,
        current_nr - previous_nr,
    ]


def build_rows(row_counts, null_rates):
    """Build vectors from aligned history; no interpolation or zero filling.

    Storage/caller supplies canonical timestamps within exactly one source.
    Kept identical to the existing detector's alignment and delta arithmetic.
    """
    rc = {row["ts"]: row["value"] for row in row_counts}
    nr = {row["ts"]: row["value"] for row in null_rates}
    timestamps = sorted(rc.keys() & nr.keys())
    vectors = [
        build_vector(rc[now], nr[now], rc[before], nr[before])
        for before, now in pairwise(timestamps)
    ]
    return timestamps[1:], vectors


def validate_vector(values):
    """Validate a named serving input before applying the saved scaler."""
    if not isinstance(values, dict) or set(values) != set(FEATURE_NAMES):
        raise ValueError("exactly four named features are required")
    vector = []
    for name in FEATURE_NAMES:
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{name} must be a finite number")
        try:
            numeric = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite number") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"{name} must be a finite number")
        vector.append(numeric)
    rc, nr, drc, dnr = vector
    # Tolerance only compensates subtraction roundoff at physical boundaries.
    if rc < 0 or not 0 <= nr <= 1 or not -1 <= dnr <= 1:
        raise ValueError("features are outside physical bounds")
    if rc - drc < -1e-12 or not -1e-12 <= nr - dnr <= 1 + 1e-12:
        raise ValueError("features imply an impossible previous observation")
    return vector
