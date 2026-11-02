"""Offline synthetic evidence for #17; not a production monitoring/retraining job.

Run: python -m scripts.check_drift_strategy > docs/course/drift/demo.json
Uses the existing PSI/KS math, no DB access, model fitting or external service.
"""

from __future__ import annotations

import json

import numpy as np

from ml.drift import ks_two_sample, psi

FEATURES = ("row_count", "null_rate", "d_row_count", "d_null_rate")
SEED = 42


def samples(rng, size, shifted=False):
    """Independent artificial feature marginals, not a coherent DB time series."""
    return {
        "row_count": rng.normal(10000 + (2000 if shifted else 0), 100, size),
        "null_rate": rng.uniform(0.015, 0.025, size),
        "d_row_count": rng.normal(5, 2, size),
        "d_null_rate": rng.uniform(-0.002, 0.002, size),
    }


def proportions(values, reference):
    if np.min(reference) == np.max(reference):
        point = reference[0]
        counts = [sum(values < point), sum(values == point), sum(values > point)]
    else:
        cuts = np.unique(np.quantile(reference, np.arange(1, 10) / 10))
        counts, _ = np.histogram(values, bins=np.r_[-np.inf, cuts, np.inf])
    return {str(i): float(n / len(values)) for i, n in enumerate(counts)}


def compare(reference, current, expected_reference=1344, expected_current=288):
    sizes = [len(reference[f]) for f in FEATURES] + [len(current[f]) for f in FEATURES]
    if min(sizes) < 200 or any(
        len(window[f]) / expected < 0.9
        for window, expected in [(reference, expected_reference), (current, expected_current)]
        for f in FEATURES
    ):
        return {"status": "insufficient_data", "signalled_features": [], "features": {}}
    metrics = {}
    signalled = []
    for feature in FEATURES:
        base, now = reference[feature], current[feature]
        d, p = ks_two_sample([(float(v), 1) for v in base], [(float(v), 1) for v in now])
        value = psi(proportions(base, base), proportions(now, base))
        signal = (d >= 0.1 and p < 0.0125) or value > 0.2
        if signal:
            signalled.append(feature)
        # PSI sums a set of bins: discard platform/hash-order roundoff in the report.
        metrics[feature] = {
            "ks_d": float(f"{d:.12g}"), "ks_p_approx": float(f"{p:.12g}"),
            "psi": float(f"{value:.12g}"), "signal": signal,
        }
    return {"status": "signal" if signalled else "no_signal", "signalled_features": signalled,
            "features": metrics}


def main():
    rng = np.random.default_rng(SEED)
    reference = samples(rng, 1344)
    stable = compare(reference, samples(rng, 288))
    shifted = [compare(reference, samples(rng, 288, shifted=True)) for _ in range(2)]
    sparse = compare(reference, samples(rng, 199))
    # Enough points for the absolute minimum, but less than 90% coverage.
    incomplete = compare(reference, samples(rng, 220))
    persistent = sorted(set(shifted[0]["signalled_features"]) &
                        set(shifted[1]["signalled_features"]))
    assert stable["status"] == "no_signal"
    assert persistent == ["row_count"]
    assert sparse["status"] == incomplete["status"] == "insufficient_data"
    result = {
        "kind": "synthetic-policy-demonstration", "seed": SEED,
        "scope": "Independent synthetic marginals; not measured production quality or concept drift",
        "baseline_points": 1344, "current_window_points": 288,
        "stable": stable, "shifted_nonoverlapping_windows": shifted,
        "persistent_features": persistent, "below_minimum": sparse,
        "below_coverage": incomplete,
        "decision": "investigate_and_request_verified_labels; no automatic retraining",
        "model_trained": False, "active_model_changed": False,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
