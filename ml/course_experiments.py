"""Offline benchmark protocol; never import the application or connect to a DB."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from scripts.generate_ml_dataset import FEATURES, PROFILES, fit_training_scaler, load_features


@dataclass(frozen=True)
class Candidate:
    name: str
    contamination: float = 0.01
    n_estimators: int = 100
    max_samples: int = 256

    def parameters(self):
        return {key: value for key, value in asdict(self).items() if key != "name"}


# Fixed before examining validation or test. One-factor changes from baseline.
CANDIDATES = (
    Candidate("baseline"),
    Candidate("contamination_005", contamination=0.005),
    Candidate("contamination_050", contamination=0.05),
    Candidate("trees_300", n_estimators=300),
    Candidate("samples_128", max_samples=128),
)


def fit_candidate(dataset, candidate, seed=42):
    """Reuse #7's train-only scaler and feature allowlist, one model per table."""
    models = {}
    for table in PROFILES:
        scaler = fit_training_scaler(dataset, table)
        train = np.asarray(load_features(dataset, "train", table))
        model = IsolationForest(**candidate.parameters(), random_state=seed, n_jobs=1)
        model.fit(scaler.transform(train))
        models[table] = Pipeline([("scaler", scaler), ("model", model)])
    return models


def predict(models, frame):
    """score < 0 follows the production detector's fixed decision convention."""
    result = frame.copy()
    result["score"] = np.nan
    for table, model in models.items():
        mask = result["table_name"].eq(table)
        if mask.any():
            result.loc[mask, "score"] = model.decision_function(
                result.loc[mask, list(FEATURES)].to_numpy(dtype=float)
            )
    if not np.isfinite(result["score"]).all():
        raise ValueError("every observation must have a finite score from its table model")
    result["predicted_anomaly"] = result["score"].lt(0).astype(int)
    return result


def evaluate(predictions, events, split, interval_minutes):
    """Micro point metrics; event recall and delays per inclusive labelled interval."""
    truth = predictions["is_anomaly"].astype(int).eq(1)
    signals = predictions["predicted_anomaly"].astype(int).eq(1)
    tp, fp = int((truth & signals).sum()), int((~truth & signals).sum())
    fn, tn = int((truth & ~signals).sum()), int((~truth & ~signals).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    times = pd.to_datetime(predictions["ts"], utc=True)
    details = []
    for event in events:
        if event["split"] != split:
            continue
        start, end = pd.Timestamp(event["start"]), pd.Timestamp(event["end"])
        mask = (
            predictions["project_id"].eq(event["project_id"])
            & predictions["table_name"].eq(event["table_name"])
            & times.between(start, end)
            & signals
        )
        hits = times[mask]
        delay = (
            (hits.min() - start).total_seconds() / (interval_minutes * 60) if len(hits) else None
        )
        details.append(
            {
                "incident_id": event["incident_id"],
                "kind": event["kind"],
                "detected": bool(len(hits)),
                "delay_ticks": delay,
            }
        )
    delays = [e["delay_ticks"] for e in details if e["detected"]]
    abrupt = [
        e["delay_ticks"]
        for e in details
        if e["detected"] and e["kind"] in ("null_spike", "volume_jump")
    ]
    metrics = {
        "point_precision": precision,
        "point_recall": recall,
        "point_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "event_recall": len(delays) / len(details) if details else None,
        "events_total": len(details),
        "events_detected": len(delays),
        "median_delay_ticks": float(np.median(delays)) if delays else None,
        "abrupt_median_delay_ticks": float(np.median(abrupt)) if abrupt else None,
    }
    return metrics, details


def meets_targets(metrics):
    return (
        metrics["event_recall"] is not None
        and metrics["event_recall"] >= 0.8
        and metrics["false_positive_rate"] is not None
        and metrics["false_positive_rate"] <= 0.1
        and metrics["abrupt_median_delay_ticks"] is not None
        and metrics["abrupt_median_delay_ticks"] <= 1
    )


def select_candidate(results):
    """Validation only. Feasible: max point F1. Otherwise min FPR, then max recall."""
    feasible = [r for r in results if meets_targets(r["metrics"])]
    if feasible:
        return min(
            feasible,
            key=lambda r: (
                -r["metrics"]["point_f1"],
                r["metrics"]["false_positive_rate"],
                r["parameters"]["n_estimators"],
                r["name"],
            ),
        )
    return min(
        results,
        key=lambda r: (
            r["metrics"]["false_positive_rate"]
            if r["metrics"]["false_positive_rate"] is not None
            else 1,
            -(r["metrics"]["event_recall"] or 0),
            -r["metrics"]["point_f1"],
            r["name"],
        ),
    )
