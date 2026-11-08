"""Offline synthetic evidence for #17; not a production monitoring/retraining job.

Run: python -m scripts.check_drift_strategy > docs/course/drift/demo.json
Uses the existing PSI/KS math, no DB access, model fitting or external service.
"""

from __future__ import annotations

import json

import numpy as np

from ml.drift import ks_two_sample, psi

FEATURES = ("row_count", "null_rate", "d_row_count", "d_null_rate")
# The row_count level grows with normal accumulation; it is checked against the
# reference trend instead of the fixed reference distribution.
DISTRIBUTION_FEATURES = ("null_rate", "d_row_count", "d_null_rate")
SEED = 42
REFERENCE_VECTORS = 1344  # 14 days at a 15-minute step
WINDOW_VECTORS = 288  # 72 hours
MIN_VECTORS = 200
MIN_COVERAGE = 0.9
KS_D, KS_P = 0.1, 0.05 / 4
PSI_SIGNAL = 0.2
TREND_SIGMAS = 4
MIN_SHARE = 0.1  # share of window vectors outside the trend band / reference range


def series(rng, start_tick, points, start_rows, growth=5.0, step_at=None, step=0.0):
    """Coherent table history: row_count accumulates per-tick increments."""
    increments = rng.normal(growth, 2, points - 1)
    rows = start_rows + np.r_[0, np.cumsum(increments)]
    ticks = np.arange(start_tick, start_tick + points)
    if step_at is not None:
        rows[ticks >= step_at] += step
    return {"tick": ticks, "row_count": rows, "null_rate": rng.uniform(0.015, 0.025, points)}


def features(history):
    """Same four inputs as the detector; deltas across a collection gap are dropped."""
    ticks = np.asarray(history["tick"])
    rows = np.asarray(history["row_count"], dtype=float)
    nulls = np.asarray(history["null_rate"], dtype=float)
    adjacent = np.diff(ticks) == 1
    with np.errstate(invalid="ignore", over="ignore"):
        window = {
            "tick": ticks[1:], "row_count": rows[1:], "null_rate": nulls[1:],
            "d_row_count": np.diff(rows), "d_null_rate": np.diff(nulls),
        }
    return {name: values[adjacent] for name, values in window.items()}


def select(window, first_tick, last_tick):
    keep = (window["tick"] >= first_tick) & (window["tick"] <= last_tick)
    return {name: values[keep] for name, values in window.items()}


def validate(window):
    """Mask of finite vectors that respect physical bounds, plus reasons."""
    values = np.column_stack([np.asarray(window[f], dtype=float) for f in FEATURES])
    finite = np.isfinite(values).all(axis=1)
    rc, nr, drc, dnr = np.where(np.isfinite(values), values, 0.0).T
    # Tolerance only compensates subtraction roundoff at physical boundaries.
    in_bounds = ((rc >= 0) & (nr >= 0) & (nr <= 1) & (rc - drc >= -1e-12)
                 & (nr - dnr >= -1e-12) & (nr - dnr <= 1 + 1e-12))
    valid = finite & in_bounds
    return valid, {"nonfinite": int((~finite).sum()), "out_of_bounds": int((finite & ~in_bounds).sum())}


def proportions(values, reference):
    if np.min(reference) == np.max(reference):
        point = reference[0]
        counts = [sum(values < point), sum(values == point), sum(values > point)]
    else:
        cuts = np.unique(np.quantile(reference, np.arange(1, 10) / 10))
        counts, _ = np.histogram(values, bins=np.r_[-np.inf, cuts, np.inf])
    return {str(i): float(n / len(values)) for i, n in enumerate(counts)}


def rounded(value):
    # PSI sums a set of bins: discard platform/hash-order roundoff in the report.
    return float(f"{value:.12g}")


def compare(reference, current, expected_reference=REFERENCE_VECTORS,
            expected_current=WINDOW_VECTORS):
    """Validate first; KS/PSI run only on complete, finite, plausible windows."""
    windows = {"reference": (reference, expected_reference),
               "current": (current, expected_current)}
    checked = {}
    for name, (window, expected) in windows.items():
        valid, invalid = validate(window)
        count = int(valid.sum())
        checked[name] = invalid
        if count < MIN_VECTORS or count / expected < MIN_COVERAGE:
            return {"status": "insufficient_data", "window": name, "valid_vectors": count,
                    "expected_vectors": expected, "invalid_vectors": invalid}
    for name, invalid in checked.items():
        if any(invalid.values()):
            # Enough valid points do not make corrupted collection trustworthy.
            return {"status": "invalid_data", "window": name, "invalid_vectors": invalid}

    anchor_tick = reference["tick"][-1]
    steps = current["tick"] - anchor_tick
    if steps.min() < 1 or steps.max() > expected_reference:
        return {"status": "baseline_expired", "max_steps_after_reference": int(steps.max())}

    metrics = {}
    signalled = []
    for feature in FEATURES:
        base, now = reference[feature], current[feature]
        d, p = ks_two_sample([(float(v), 1) for v in base], [(float(v), 1) for v in now])
        value = psi(proportions(base, base), proportions(now, base))
        signal = feature in DISTRIBUTION_FEATURES and (
            (d >= KS_D and p < KS_P) or value > PSI_SIGNAL)
        if signal:
            signalled.append(feature)
        metrics[feature] = {"ks_d": rounded(d), "ks_p_approx": rounded(p),
                            "psi": rounded(value), "signal": signal}
    metrics["row_count"]["role"] = "descriptive; growing level is checked against the trend"

    # Random walk with drift: level after n steps ~ anchor + mu*n +- sigma*sqrt(n).
    mu = float(np.mean(reference["d_row_count"]))
    sigma = float(np.std(reference["d_row_count"], ddof=1))
    expected_rows = reference["row_count"][-1] + mu * steps
    outside_band = np.abs(current["row_count"] - expected_rows) > TREND_SIGMAS * sigma * np.sqrt(steps)
    outside_range = ((current["row_count"] < reference["row_count"].min())
                     | (current["row_count"] > reference["row_count"].max()))
    trend_break = float(outside_band.mean()) >= MIN_SHARE
    if trend_break:
        signalled.insert(0, "row_count")
    range_exceeded = float(outside_range.mean()) >= MIN_SHARE
    status = ("signal" if signalled else "expected_growth" if range_exceeded else "no_signal")
    return {
        "status": status, "signalled_features": signalled, "features": metrics,
        "row_count_trend": {
            "reference_mean_increment": rounded(mu), "reference_sd_increment": rounded(sigma),
            "share_outside_band": rounded(float(outside_band.mean())),
            "share_outside_reference_range": rounded(float(outside_range.mean())),
            "trend_break": trend_break, "range_exceeded": range_exceeded,
        },
    }


def persistent(first, second):
    return sorted(set(first.get("signalled_features", ())) & set(second.get("signalled_features", ())))


def scenario(rng, *, growth=5.0, step=0.0):
    """Reference plus two following non-overlapping 72-hour windows."""
    history = series(rng, 0, REFERENCE_VECTORS + 1, 10000)
    reference = features(history)
    start = REFERENCE_VECTORS
    future = series(rng, start, 2 * WINDOW_VECTORS + 1, history["row_count"][-1], growth,
                    step_at=start + WINDOW_VECTORS // 2, step=step)
    future_window = features(future)
    windows = [select(future_window, start + 1 + i * WINDOW_VECTORS,
                      start + (i + 1) * WINDOW_VECTORS) for i in range(2)]
    return reference, windows, future


def run(rng, **change):
    reference, windows, _ = scenario(rng, **change)
    results = [compare(reference, window) for window in windows]
    return results, persistent(*results)


def corrupted(future, tick, feature, value):
    damaged = {name: np.array(values, dtype=float if name != "tick" else int)
               for name, values in future.items()}
    damaged[feature][damaged["tick"] == tick] = value
    return damaged


def sweep(seeds, **change):
    """Share of seeds with at least one / two consecutive windows signalling."""
    any_signal = both = 0
    for seed in seeds:
        results, common = run(np.random.default_rng(seed), **change)
        any_signal += any(r["status"] == "signal" for r in results)
        both += bool(common)
    return {"seeds": len(seeds), "any_window_signal": any_signal, "persistent_signal": both}


def build_report():
    rng = np.random.default_rng(SEED)
    normal, normal_persistent = run(rng)
    step, step_persistent = run(rng, step=2000)
    rate, rate_persistent = run(rng, growth=15)

    reference, windows, future = scenario(rng)
    first = REFERENCE_VECTORS + 1
    last = REFERENCE_VECTORS + WINDOW_VECTORS
    below_minimum = compare(reference, windows[0] | {
        name: values[:199] for name, values in windows[0].items()})
    gap = {name: values[(future["tick"] < first + 100) | (future["tick"] >= first + 168)]
           for name, values in future.items()}
    collection_gap = compare(reference, select(features(gap), first, last))
    damaged = {
        "nan_row_count": compare(reference, select(
            features(corrupted(future, first + 10, "row_count", np.nan)), first, last)),
        "positive_infinity_null_rate": compare(reference, select(
            features(corrupted(future, first + 10, "null_rate", np.inf)), first, last)),
        "negative_infinity_row_count": compare(reference, select(
            features(corrupted(future, first + 10, "row_count", -np.inf)), first, last)),
    }
    many_nan = future | {"null_rate": np.array(future["null_rate"], dtype=float)}
    many_nan["null_rate"][(future["tick"] >= first + 20) & (future["tick"] < first + 80)] = np.nan
    damaged["nan_leaves_too_few_valid"] = compare(reference, select(features(many_nan), first, last))

    assert [r["status"] for r in normal] == ["expected_growth"] * 2
    assert normal_persistent == []
    assert step_persistent == ["row_count"]
    assert "d_row_count" not in step_persistent
    assert rate_persistent == ["d_row_count", "row_count"]
    assert below_minimum["status"] == collection_gap["status"] == "insufficient_data"
    assert all(damaged[k]["status"] == "invalid_data" for k in list(damaged)[:3])
    assert damaged["nan_leaves_too_few_valid"]["status"] == "insufficient_data"

    seeds = range(1000, 1200)
    return {
        "kind": "synthetic-policy-demonstration", "seed": SEED,
        "scope": "Synthetic coherent table histories; not measured production quality or concept drift",
        "reference_vectors": REFERENCE_VECTORS, "current_window_vectors": WINDOW_VECTORS,
        "history": "row_count = 10000 + cumulative N(5, 2) rows per 15-minute tick; null_rate U(0.015, 0.025)",
        "normal_growth": {"windows": normal, "persistent_features": normal_persistent},
        "level_step_2000_rows": {"windows": step, "persistent_features": step_persistent},
        "growth_rate_change_to_15": {"windows": rate, "persistent_features": rate_persistent},
        "below_minimum": below_minimum, "collection_gap_68_ticks": collection_gap,
        "invalid_values": damaged,
        "seed_sweep_1000_1199": {
            "normal_growth": sweep(seeds), "level_step_2000_rows": sweep(seeds, step=2000),
            "growth_rate_change_to_15": sweep(seeds, growth=15),
        },
        "decision": "expected_growth -> verified candidate refresh; signal -> investigation; "
                    "no automatic retraining",
        "model_trained": False, "active_model_changed": False,
    }


def main():
    print(json.dumps(build_report(), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
