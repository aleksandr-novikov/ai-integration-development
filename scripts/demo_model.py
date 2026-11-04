"""Separate versioned course model for the Compose live demo (#13).

The archived #9 ``events`` model was fitted on 56-68k-row tables observed every
15 minutes; the Compose stand starts at 2 000 rows and inserts ~200 rows per
tick, so every live vector lies outside its training domain. This module builds
synthetic observation histories of the stand itself: it replays the real row
generators (``seed_target_db._event_row``, ``live_demo._generate_event_row``)
through the collector's arithmetic (exact row count, per-column null rate
rounded to 4 digits, table mean rounded to 4 digits) and pairs observations with
``ml.features.build_rows``. Training uses ``ml.model_registry.train_candidate``.

Splits are disjoint seed ranges: train (normal only), validation (choosing one
of the five #9 configurations by the #9 rule) and test (evaluated once for the
selected configuration). Labels come from the generator schedule, never from
scores. The model stays demo-only: ``evaluation`` is not written into the
registry manifest, so registry admission is not granted.

    python -m scripts.demo_model select --out selection.json
    python -m scripts.demo_model train --registry models/registry \\
        --source models/demo/source.json --git-sha <40-hex> \\
        --evaluation-out models/demo/evaluation.json > models/demo/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGIME = "compose-demo-normal/1"
# scripts/schema.sql order; collectors average null rates over all columns.
EVENTS_COLUMNS = (
    "id", "user_id", "session_id", "event_type", "prev_event_type", "prev_event_gap_s",
    "duration_ms", "events_in_session", "ip_address", "ip_events_last_1h", "server_id",
    "device_type", "is_bot_suspected", "created_at",
)
# Nullable columns without DEFAULT: an omitted value is stored as NULL.
NULL_WHEN_OMITTED = frozenset({"prev_event_type", "prev_event_gap_s", "duration_ms", "ip_address"})
INITIAL_ROWS = (1000, 5000)
NORMAL_ROWS = (100, 300)
# Scheduler collections between live-demo batches observe an unchanged table.
IDLE_PROBABILITY = 0.1
TRAIN_SEED = 1_000_000
VALIDATION_SEED = 2_000_000
TEST_SEED = 3_000_000
MODEL_SEED = 42
# The #9 protocol grid, unchanged; max_samples=256 equals "auto" above 256 rows.
CONFIGS = {
    "baseline": {"contamination": 0.01, "n_estimators": 100, "max_samples": 256},
    "contamination_005": {"contamination": 0.005, "n_estimators": 100, "max_samples": 256},
    "contamination_050": {"contamination": 0.05, "n_estimators": 100, "max_samples": 256},
    "trees_300": {"contamination": 0.01, "n_estimators": 300, "max_samples": 256},
    "samples_128": {"contamination": 0.01, "n_estimators": 100, "max_samples": 128},
}
# Declared before the test split was generated; see docs/course/containers/demo-model.
SCENARIO_TARGET = 0.95
SELECTED = "contamination_005"
SOURCE_FILES = ("scripts/demo_model.py", "scripts/live_demo.py", "scripts/seed_target_db.py",
                "ml/features.py", "ml/model_registry.py")
_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)
_USER_IDS = ["00000000-0000-0000-0000-000000000000"]


class _Addresses:
    """Seeder rows only need nullness; Faker's own RNG never decides it."""

    @staticmethod
    def ipv4():
        return "192.0.2.1"


@dataclass(frozen=True)
class Episode:
    family: str
    seed: int
    initial_rows: int
    ticks: int
    rows_per_tick: int
    fixed_rows: bool
    incident_at: int | None


def family(name, count, seed, *, ticks=30, incident=True, compose=None):
    """Episode specs. ``compose`` = (ticks, incident_at) for the fixed 2 000/200 route."""
    episodes = []
    for index in range(count):
        rng = random.Random(seed + index)
        if compose is not None:
            episodes.append(Episode(name, seed + index, 2000, compose[0], 200, True, compose[1]))
            continue
        initial = rng.randint(*INITIAL_ROWS)
        nominal = rng.randint(*NORMAL_ROWS)
        incident_at = rng.randint(5, ticks - 5) if incident and index % 2 else None
        episodes.append(Episode(name, seed + index, initial, ticks, nominal, False, incident_at))
    return episodes


def splits():
    return {
        "train": family("train", 160, TRAIN_SEED, ticks=25, incident=False),
        "validation": family("general", 40, VALIDATION_SEED)
        + family("compose", 100, VALIDATION_SEED + 100_000, compose=(4, 4)),
        "test": family("general", 80, TEST_SEED)
        + family("compose", 200, TEST_SEED + 100_000, compose=(4, 4))
        + family("readme", 200, TEST_SEED + 200_000, compose=(6, 5)),
    }


class _Table:
    def __init__(self):
        self.total = 0
        self.nulls = dict.fromkeys(EVENTS_COLUMNS, 0)

    def insert(self, rows):
        for row in rows:
            self.total += 1
            for column in EVENTS_COLUMNS:
                if column in row:
                    self.nulls[column] += row[column] is None
                else:
                    self.nulls[column] += column in NULL_WHEN_OMITTED

    def observe(self):
        # PostgresAdapter.column_nulls + MetricsCollector._collect_null_rate_full.
        rates = [round(self.nulls[column] / self.total, 4) for column in EVENTS_COLUMNS]
        return self.total, round(sum(rates) / len(rates), 4)


def simulate(episode):
    """Baseline collection, then one collection after each live-demo batch."""
    from scripts import live_demo, seed_target_db

    state = random.getstate()
    try:
        random.seed(episode.seed)
        table = _Table()
        now = datetime.now(UTC)
        table.insert(seed_target_db._event_row(_Addresses, _USER_IDS, now)
                     for _ in range(episode.initial_rows))
        observations, labels = [table.observe()], []
        for tick in range(1, episode.ticks + 1):
            incident = tick == episode.incident_at
            if incident:
                rows = episode.rows_per_tick * live_demo._INCIDENT_ROW_MULTIPLIER
            elif episode.fixed_rows:
                rows = episode.rows_per_tick
            else:
                rows = 0 if random.random() < IDLE_PROBABILITY else random.randint(*NORMAL_ROWS)
            table.insert(live_demo._generate_event_row(_USER_IDS, incident) for _ in range(rows))
            observations.append(table.observe())
            labels.append("incident" if incident else "normal"
                          if episode.incident_at is None or tick < episode.incident_at else "aftermath")
    finally:
        random.setstate(state)
    stamps = [(_EPOCH + timedelta(seconds=i)).isoformat() for i in range(len(observations))]
    return {
        "row_counts": [{"ts": ts, "value": rc} for ts, (rc, _) in zip(stamps, observations, strict=True)],
        "null_rates": [{"ts": ts, "value": nr} for ts, (_, nr) in zip(stamps, observations, strict=True)],
        "labels": labels,
    }


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def generate(name):
    episodes = splits()[name]
    histories = [simulate(episode) for episode in episodes]
    digest = canonical_sha256({"regime": REGIME, "split": name,
                               "episodes": [asdict(e) for e in episodes], "histories": histories})
    return episodes, histories, digest


def vectors(histories):
    from ml.features import build_rows

    rows, labels, owners = [], [], []
    for index, history in enumerate(histories):
        _, matrix = build_rows(history["row_counts"], history["null_rates"])
        assert len(matrix) == len(history["labels"])
        rows += matrix
        labels += history["labels"]
        owners += [index] * len(matrix)
    return rows, labels, owners


def wilson(successes, total, z=1.96):
    if not total:
        return None
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [round(centre - margin, 4), round(centre + margin, 4)]


def evaluate(episodes, labels, owners, flags):
    """#9 definitions: point FPR on normal ticks; one-tick abrupt events."""
    from ml.model_registry import TARGETS

    def counts(selected):
        tp = sum(f for f, label, owner in selected if label == "incident")
        fn = sum(not f for f, label, owner in selected if label == "incident")
        fp = sum(f for f, label, owner in selected if label == "normal")
        tn = sum(not f for f, label, owner in selected if label == "normal")
        after = [f for f, label, owner in selected if label == "aftermath"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return {
            "normal_points": fp + tn, "false_alarms": fp, "incidents": tp + fn, "detected": tp,
            "precision": round(precision, 6), "recall": round(recall, 6),
            "f1": round(2 * precision * recall / (precision + recall), 6) if precision + recall else 0.0,
            "false_positive_rate": round(fp / (fp + tn), 6) if fp + tn else None,
            "event_recall": round(tp / (tp + fn), 6) if tp + fn else None,
            # Events last one tick, so a detected event has zero delay.
            "abrupt_median_delay_ticks": 0.0 if tp else None,
            "aftermath_points": len(after), "aftermath_alarms": sum(after),
        }

    points = list(zip(flags, labels, owners, strict=True))
    overall = counts(points)
    targets = {name: overall[name] is not None and (
        overall[name] >= target if name == "event_recall" else overall[name] <= target)
        for name, target in TARGETS.items()}
    families = {}
    for name in sorted({e.family for e in episodes}):
        selected = [p for p in points if episodes[p[2]].family == name]
        families[name] = counts(selected)
        if name in ("compose", "readme"):
            runs = {}
            for f, label, owner in selected:
                ok = (not f) if label == "normal" else bool(f) if label == "incident" else True
                runs[owner] = runs.get(owner, True) and ok
            passed = sum(runs.values())
            families[name].update(scenarios=len(runs), scenarios_passed=passed,
                                  scenario_pass_rate=round(passed / len(runs), 6),
                                  scenario_pass_rate_wilson95=wilson(passed, len(runs)))
    scenario_ok = all(families[name]["scenario_pass_rate"] >= SCENARIO_TARGET
                      for name in ("compose", "readme") if name in families)
    return {"metrics": overall, "families": families, "targets": targets,
            "targets_passed": all(targets.values()) and scenario_ok}


def _pipeline(parameters):
    from sklearn.ensemble import IsolationForest
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([("scaler", StandardScaler()),
                     ("model", IsolationForest(**parameters, random_state=MODEL_SEED))])


def _matrix(rows):
    import numpy as np

    from ml.features import FEATURE_NAMES, validate_vector

    return np.asarray([validate_vector(dict(zip(FEATURE_NAMES, row, strict=True))) for row in rows])


def select():
    """Fit the five #9 configurations on train, choose on validation only."""
    _, train_histories, train_sha = generate("train")
    episodes, histories, validation_sha = generate("validation")
    train_matrix = _matrix(vectors(train_histories)[0])
    rows, labels, owners = vectors(histories)
    matrix = _matrix(rows)
    results = {}
    for name, parameters in CONFIGS.items():
        pipeline = _pipeline(parameters).fit(train_matrix)
        flags = [bool(score < 0) for score in pipeline.decision_function(matrix)]
        results[name] = {"parameters": parameters, **evaluate(episodes, labels, owners, flags)}
    passing = [name for name, result in results.items() if result["targets_passed"]]
    if passing:
        winner = min(passing, key=lambda n: (-results[n]["metrics"]["f1"],
                                             results[n]["metrics"]["false_positive_rate"],
                                             CONFIGS[n]["n_estimators"], n))
    else:
        winner = min(results, key=lambda n: (results[n]["metrics"]["false_positive_rate"],
                                             -results[n]["metrics"]["event_recall"],
                                             -results[n]["metrics"]["f1"], n))
    return {"regime": REGIME, "train_data_sha256": train_sha, "validation_data_sha256": validation_sha,
            "training_vectors": len(train_matrix), "validation_vectors": len(matrix),
            "rule": "targets on validation, then max F1, lower FPR, fewer trees, name",
            "selected": winner, "targets_met": bool(passing), "candidates": results,
            "environment": _environment()}


def _environment():
    return {"python": sys.version.split()[0], **{
        name: importlib.metadata.version(name) for name in ("scikit-learn", "numpy", "skops")}}


def train_and_evaluate(registry, source, git_sha, *, reference_archive=None):
    """Train the selected configuration once, then score the test split once."""
    from ml.features import FEATURE_NAMES
    from ml.model_registry import load_version, train_candidate

    _, train_histories, train_sha = generate("train")
    provenance = {
        "purpose": "compose-demo", "regime": REGIME, "configuration": SELECTED,
        "train_data_sha256": train_sha, "train_seeds": [TRAIN_SEED, TRAIN_SEED + 159],
        "generator_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                             for name in SOURCE_FILES},
    }
    manifest = train_candidate(
        registry, source=source, git_sha=git_sha, seed=MODEL_SEED, parameters=CONFIGS[SELECTED],
        histories=[{key: h[key] for key in ("row_counts", "null_rates")} for h in train_histories],
        provenance=provenance,
    )
    loaded = load_version(registry, manifest["model_version"], source=source, demo=True)
    episodes, histories, test_sha = generate("test")
    rows, labels, owners = vectors(histories)
    scores = [float(s) for s in loaded.pipeline.decision_function(_matrix(rows))]
    # The batch path must equal the serving path (validate_vector + predict).
    for index in range(0, len(rows), max(1, len(rows) // 25)):
        assert loaded.predict(dict(zip(FEATURE_NAMES, rows[index], strict=True)))["score"] == scores[index]
    result = evaluate(episodes, labels, owners, [score < 0 for score in scores])
    artifact = manifest["artifact"]["sha256"]
    evaluation = {
        "regime": REGIME, "configuration": SELECTED, "parameters": CONFIGS[SELECTED], "seed": MODEL_SEED,
        "model_version": manifest["model_version"], "artifact_sha256": artifact,
        "train_data_sha256": train_sha, "test_data_sha256": test_sha,
        "scores_sha256": canonical_sha256(scores), "training_vectors": manifest["training"]["n_points"],
        "test_vectors": len(rows), "test": result, "environment": _environment(),
        "demo": True, "quality_admitted": False,
    }
    if reference_archive is not None:
        evaluation["reference_archive_9"] = _reference(reference_archive, source, rows, episodes, labels, owners)
    return manifest, evaluation


def _reference(archive, source, rows, episodes, labels, owners):
    """Same test vectors through the frozen #9 export, for comparison only."""
    from ml.model_registry import load_version, stage_export

    with tempfile.TemporaryDirectory() as directory:
        manifest = stage_export(directory, archive, source=source)
        loaded = load_version(directory, manifest["model_version"], source=source, demo=True)
        scores = loaded.pipeline.decision_function(_matrix(rows))
    result = evaluate(episodes, labels, owners, [bool(score < 0) for score in scores])
    return {"artifact_sha256": manifest["artifact"]["sha256"], "test": result}


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    chooser = commands.add_parser("select", help="Choose a #9 configuration on validation")
    chooser.add_argument("--out", type=Path, required=True)
    train = commands.add_parser("train", help="Train the selected configuration and test it once")
    train.add_argument("--registry", type=Path, default=Path("models/registry"))
    train.add_argument("--source", type=Path, required=True)
    train.add_argument("--git-sha", required=True)
    train.add_argument("--evaluation-out", type=Path, required=True)
    train.add_argument("--reference-archive", type=Path)
    args = parser.parse_args(argv)
    if args.command == "select":
        _write(args.out, select())
        return 0
    if not re.fullmatch(r"[a-f0-9]{40}", args.git_sha):
        parser.error("--git-sha must be a full commit SHA")
    source = json.loads(args.source.read_text(encoding="utf-8"))
    manifest, evaluation = train_and_evaluate(args.registry, source, args.git_sha,
                                              reference_archive=args.reference_archive)
    _write(args.evaluation_out, evaluation)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
