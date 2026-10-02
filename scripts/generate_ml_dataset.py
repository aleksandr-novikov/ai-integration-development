"""Generate a labelled, chronological benchmark without a database or network.

Run: python -m scripts.generate_ml_dataset --output data/course
Only fit_training_scaler() needs scikit-learn; generation uses the standard library.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from scripts import synthetic_metrics
from scripts.synthetic_metrics import TableProfile, _row_count_at

GENERATOR_VERSION = "1.0.0"
FEATURES = ("row_count", "null_rate", "d_row_count", "d_null_rate")
SPLITS = ("train", "validation", "test")
PROFILES = {"users": 5_000, "orders": 10_000, "events": 80_000}
FIELDS = ("project_id", "table_name", "ts", *FEATURES, "is_anomaly", "incident_id")


@dataclass(frozen=True)
class Config:
    seed: int = 42
    start: str = "2026-01-01T00:00:00+00:00"
    train_ticks: int = 1_344
    validation_ticks: int = 672
    test_ticks: int = 672
    interval_minutes: int = 15
    project_id: str = "course-benchmark"

    def validate(self) -> datetime:
        start = datetime.fromisoformat(self.start)
        if start.tzinfo is None or start.utcoffset() != timedelta(0):
            raise ValueError("start must include an explicit UTC offset")
        if self.train_ticks < 201:
            raise ValueError("train_ticks must be >= 201 (200 rows after the first delta)")
        if min(self.validation_ticks, self.test_ticks) < 96:
            raise ValueError("validation_ticks and test_ticks must be >= 96")
        if self.interval_minutes < 1:
            raise ValueError("interval_minutes must be positive")
        if not self.project_id.strip():
            raise ValueError("project_id must not be empty")
        return start


def _timestamp(start: datetime, tick: int, interval: int) -> str:
    return (start + timedelta(minutes=tick * interval)).isoformat(timespec="seconds")


def _events(config: Config, table: str, start: datetime) -> list[dict]:
    """Intervals include the first recovery tick, whose delta is still affected."""
    rng = random.Random(f"{config.seed}:{table}:events")
    offset = config.train_ticks
    events = []
    for split, size in zip(SPLITS[1:], (config.validation_ticks, config.test_ticks), strict=True):
        for kind, fraction in (("null_spike", 0.15), ("volume_jump", 0.45), ("regime_change", 0.7)):
            first = offset + int(size * fraction) + rng.randrange(max(1, size // 30))
            duration = max(2, size // (15 if kind == "regime_change" else 60))
            recovery = first + duration
            events.append(
                {
                    "incident_id": f"{config.project_id}:{table}:{split}:{kind}",
                    "project_id": config.project_id,
                    "table_name": table,
                    "split": split,
                    "kind": kind,
                    "first_tick": first,
                    "recovery_tick": recovery,
                    "start": _timestamp(start, first, config.interval_minutes),
                    "end": _timestamp(start, recovery, config.interval_minutes),
                    "recovery_included": True,
                    "null_delta": round(rng.uniform(0.12, 0.3), 6),
                    "volume_multiplier": round(rng.uniform(1.3, 1.7), 6),
                }
            )
        offset += size
    return events


def generate(config: Config) -> tuple[dict[str, list[dict]], list[dict]]:
    start = config.validate()
    total = config.train_ticks + config.validation_ticks + config.test_ticks
    result: dict[str, list[dict]] = {split: [] for split in SPLITS}
    events = []
    # Reuse the existing dashboard generator with noisy, mildly seasonal profiles.
    # Do not reuse its hand-tuned incidents or infer labels from a model's output.
    profile = TableProfile(
        start_fraction=0.7,
        weekly_amplitude=0.002,
        noise_amplitude=0.001,
        backfill_fraction=0.0,
        anomalies=(),
        growth_steps=(),
    )
    for table, current in PROFILES.items():
        rng = random.Random(f"{config.seed}:{table}:baseline")
        table_events = _events(config, table, start)
        events.extend(table_events)
        previous = None
        for tick in range(total):
            ts = start + timedelta(minutes=tick * config.interval_minutes)
            # Fixed rate independent of the requested end of validation/test.
            # _row_count_at evaluates one timestamp; it never reads future samples.
            progress = tick / 2_688
            row_count = _row_count_at(progress, current, ts, rng, profile)
            null_rate = round(0.02 + rng.uniform(-0.002, 0.002), 6)
            event = next(
                (e for e in table_events if e["first_tick"] <= tick <= e["recovery_tick"]), None
            )
            if event and tick < event["recovery_tick"]:
                if event["kind"] == "null_spike":
                    null_rate = round(null_rate + event["null_delta"], 6)
                elif event["kind"] == "volume_jump":
                    row_count = round(row_count * event["volume_multiplier"])
                else:
                    # Sustained change of the normal operating regime, then recovery.
                    row_count = round(row_count * 1.1)
                    null_rate = round(null_rate + 0.08, 6)
            values = (row_count, null_rate)
            if previous is not None:
                split = (
                    "train"
                    if tick < config.train_ticks
                    else "validation"
                    if tick < config.train_ticks + config.validation_ticks
                    else "test"
                )
                result[split].append(
                    {
                        "project_id": config.project_id,
                        "table_name": table,
                        "ts": ts.isoformat(timespec="seconds"),
                        "row_count": row_count,
                        "null_rate": null_rate,
                        "d_row_count": row_count - previous[0],
                        "d_null_rate": round(null_rate - previous[1], 6),
                        "is_anomaly": int(event is not None),
                        "incident_id": event["incident_id"] if event else "",
                    }
                )
            previous = values
    for rows in result.values():
        rows.sort(key=lambda r: (r["ts"], r["project_id"], r["table_name"]))
    return result, events


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_dataset(output: Path, config: Config) -> dict:
    data, events = generate(config)
    # A version is immutable: no accidental replacement of an earlier benchmark.
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, rows in data.items():
        path = output / f"{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        files[path.name] = {
            "sha256": _sha256(path),
            "rows": len(rows),
            "first_ts": rows[0]["ts"],
            "last_ts": rows[-1]["ts"],
            "anomalous_rows": sum(r["is_anomaly"] for r in rows),
        }
    (output / "incidents.json").write_text(
        json.dumps(events, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    files["incidents.json"] = {"sha256": _sha256(output / "incidents.json"), "events": len(events)}
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "python_version": platform.python_version(),
        "generator_sources": {
            "scripts/generate_ml_dataset.py": _sha256(Path(__file__)),
            "scripts/synthetic_metrics.py": _sha256(Path(synthetic_metrics.__file__)),
        },
        "config": asdict(config),
        "tables": PROFILES,
        "features": list(FEATURES),
        "null_rate_aggregation": "synthetic table-level mean column null rate",
        "labels": "known injected intervals, inclusive of one recovery tick",
        "delta_policy": "previous observed tick in the same project/table, including split boundaries; first tick omitted",
        "selection_protocol": "fit on train; select hyperparameters and threshold on validation; evaluate selected model once on test",
        "files": files,
    }
    payload = json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()
    manifest["dataset_id"] = hashlib.sha256(payload).hexdigest()
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_features(output: Path, split: str, table: str) -> list[list[float]]:
    """Explicit allowlist excludes labels, incident IDs, timestamps and split names."""
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    with (output / f"{split}.csv").open(newline="", encoding="utf-8") as stream:
        rows = [
            [float(row[name]) for name in FEATURES]
            for row in csv.DictReader(stream)
            if row["table_name"] == table
        ]
    if not rows:
        raise ValueError(f"no rows for table {table!r} in {split}")
    return rows


def fit_training_scaler(output: Path, table: str):
    """Return a per-table scaler fitted only on train, never validation/test."""
    from sklearn.preprocessing import StandardScaler

    return StandardScaler().fit(load_features(output, "train", table))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/course"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", default=Config.start)
    parser.add_argument("--train-ticks", type=int, default=Config.train_ticks)
    parser.add_argument("--validation-ticks", type=int, default=Config.validation_ticks)
    parser.add_argument("--test-ticks", type=int, default=Config.test_ticks)
    parser.add_argument("--interval-minutes", type=int, default=15)
    args = vars(parser.parse_args())
    output = args.pop("output")
    try:
        manifest = write_dataset(output, Config(**args))
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Dataset {manifest['dataset_id']} written to {output}")


if __name__ == "__main__":
    main()
