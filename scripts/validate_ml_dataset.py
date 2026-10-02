"""Validate the course benchmark without changing it; exit 1 blocks training."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.generate_ml_dataset import FEATURES, FIELDS, PROFILES, SPLITS, Config


class Report:
    def __init__(self):
        self.checks = []

    def add(self, name, errors=()):
        errors = list(errors)
        self.checks.append(
            {
                "check": name,
                "status": "fail" if errors else "pass",
                "error_count": len(errors),
                "examples": errors[:5],
            }
        )

    def rows(self, name, frame, mask):
        self.add(
            name, (f"{row['_split']}.csv:{row['_line']}" for _, row in frame.loc[mask].iterrows())
        )

    def result(self, dataset_id):
        failed = sum(c["status"] == "fail" for c in self.checks)
        return {
            "report_version": 1,
            "dataset_id": dataset_id,
            "valid": failed == 0,
            "failed_checks": failed,
            "checks": self.checks,
        }


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8")), []
    except (OSError, ValueError) as exc:
        return None, [f"{path.name}: {type(exc).__name__}"]


def _manifest(directory, report):
    manifest, errors = _read_json(directory / "manifest.json")
    config = None
    if not errors:
        try:
            if not isinstance(manifest, dict):
                raise ValueError("manifest must be an object")
            config = Config(**manifest["config"])
            for field in (
                "seed",
                "train_ticks",
                "validation_ticks",
                "test_ticks",
                "interval_minutes",
            ):
                if type(getattr(config, field)) is not int:
                    raise ValueError(f"{field} must be an integer")
            if not isinstance(config.start, str) or not isinstance(config.project_id, str):
                raise ValueError("start and project_id must be strings")
            config.validate()
            if manifest["features"] != list(FEATURES) or manifest["tables"] != PROFILES:
                raise ValueError("unexpected feature or table schema")
            payload = {k: v for k, v in manifest.items() if k != "dataset_id"}
            expected = hashlib.sha256(
                json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            if manifest["dataset_id"] != expected:
                errors.append("manifest.json: dataset_id does not match content")
            for name in (*[f"{s}.csv" for s in SPLITS], "incidents.json"):
                digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
                if manifest["files"][name]["sha256"] != digest:
                    errors.append(f"{name}: SHA-256 mismatch")
        except (KeyError, TypeError, ValueError, OSError, OverflowError) as exc:
            errors.append(f"manifest.json: invalid metadata or missing file ({type(exc).__name__})")
            config = None
    report.add("manifest_integrity", errors)
    return manifest if isinstance(manifest, dict) else {}, config


def _load_csv(directory, report):
    frames = []
    for split in SPLITS:
        errors = []
        try:
            frame = pd.read_csv(directory / f"{split}.csv", dtype=str, keep_default_na=False)
            if list(frame.columns) != list(FIELDS):
                errors.append(f"{split}.csv: expected columns {list(FIELDS)}")
            if frame.empty:
                errors.append(f"{split}.csv: empty dataset")
            if not errors:
                frame["_split"] = split
                frame["_line"] = np.arange(len(frame)) + 2
                frames.append(frame)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            errors.append(f"{split}.csv: cannot read ({type(exc).__name__})")
        report.add(f"schema_{split}", errors)
    return pd.concat(frames, ignore_index=True) if frames else None


def _temporal_checks(frame, config, report):
    time = frame["_time"]
    report.rows(
        "unique_observation_key",
        frame,
        frame.duplicated(["project_id", "table_name", "_time"], keep=False),
    )
    previous = frame.groupby(["project_id", "table_name"], sort=False)["_time"].shift()
    report.rows("observation_order", frame, time.notna() & previous.notna() & (time <= previous))
    overlap = []
    for left, right in pairwise(SPLITS):
        a = frame.loc[frame["_split"] == left, "_time"].dropna()
        b = frame.loc[frame["_split"] == right, "_time"].dropna()
        if a.empty or b.empty or a.max() >= b.min():
            overlap.append(f"{left}/{right}: missing timestamps or overlapping time ranges")
    report.add("split_separation", overlap)
    errors = []
    if config is None:
        errors.append("cannot verify feature timestamps without valid manifest configuration")
    else:
        start = pd.Timestamp(config.start)
        offset = 0
        for split in SPLITS:
            ticks = getattr(config, f"{split}_ticks")
            first = offset + (1 if split == "train" else 0)
            expected = pd.date_range(
                start + pd.Timedelta(first * config.interval_minutes, unit="min"),
                periods=ticks - (1 if split == "train" else 0),
                freq=pd.Timedelta(config.interval_minutes, unit="min"),
            )
            part = frame.loc[frame["_split"] == split]
            for table in PROFILES:
                observed = part.loc[part["table_name"] == table, "_time"]
                if len(observed) != len(expected) or set(observed) != set(expected):
                    errors.append(f"{split}/{table}: incomplete or misaligned observation grid")
            offset += ticks
    report.add("feature_time_grid", errors)

    ordered = frame.sort_values(["project_id", "table_name", "_time"])
    groups = ordered.groupby(["project_id", "table_name"], sort=False)
    for feature in ("row_count", "null_rate"):
        expected = groups[feature].diff()
        # First exported observation has no exported predecessor. Its delta can
        # only be checked for type/finiteness, not reconstructed from this CSV.
        actual = ordered[f"d_{feature}"]
        comparable = expected.notna() & np.isfinite(expected) & np.isfinite(actual)
        mismatch = comparable & ~np.isclose(actual, expected, rtol=0, atol=1e-6)
        report.rows(f"delta_alignment_{feature}", ordered, mismatch)


def _labels(directory, frame, report):
    events, errors = _read_json(directory / "incidents.json")
    expected = pd.Series("", index=frame.index)
    if not errors:
        try:
            if not isinstance(events, list) or not events:
                raise ValueError("events must be a nonempty list")
            seen = set()
            for event in events:
                identifier = event["incident_id"]
                start, end = pd.Timestamp(event["start"]), pd.Timestamp(event["end"])
                if (
                    not isinstance(identifier, str)
                    or not identifier
                    or identifier in seen
                    or start.tzinfo is None
                    or end.tzinfo is None
                    or pd.isna(start)
                    or pd.isna(end)
                    or start > end
                    or event["split"] not in SPLITS[1:]
                ):
                    raise ValueError("invalid incident interval or identifier")
                seen.add(identifier)
                mask = (
                    (frame["project_id"] == event["project_id"])
                    & (frame["table_name"] == event["table_name"])
                    & (frame["_split"] == event["split"])
                    & frame["_time"].between(start, end)
                )
                if not mask.any() or expected[mask].ne("").any():
                    errors.append(f"{identifier}: empty or overlapping incident interval")
                expected.loc[mask] = identifier
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            errors.append(f"incidents.json: invalid event schema ({type(exc).__name__})")
    report.add("incident_intervals", errors)
    if not errors:
        report.rows(
            "incident_labels",
            frame,
            frame["incident_id"].ne(expected)
            | frame["is_anomaly"].ne(expected.ne("").astype(int).astype(str)),
        )
    report.rows("label_domain", frame, ~frame["is_anomaly"].isin(["0", "1"]))
    report.rows(
        "normal_training_set", frame, frame["_split"].eq("train") & frame["is_anomaly"].ne("0")
    )


def validate_dataset(directory: Path) -> dict:
    """Return bounded diagnostic examples; input files are only read."""
    report = Report()
    manifest, config = _manifest(directory, report)
    frame = _load_csv(directory, report)
    if frame is None:
        return report.result(manifest.get("dataset_id"))
    identifiers = frame["project_id"].str.strip().eq("") | ~frame["table_name"].isin(PROFILES)
    if config:
        identifiers |= frame["project_id"].ne(config.project_id)
    report.rows("required_identifiers", frame, identifiers)
    numeric = frame[list(FEATURES)].apply(pd.to_numeric, errors="coerce")
    report.rows("numeric_types", frame, numeric.isna().any(axis=1))
    report.rows("finite_features", frame, ~np.isfinite(numeric).all(axis=1))
    frame[list(FEATURES)] = numeric
    report.rows("null_rate_range", frame, ~frame["null_rate"].between(0, 1))
    report.rows(
        "row_count_domain",
        frame,
        ~np.isfinite(frame["row_count"])
        | frame["row_count"].lt(0)
        | frame["row_count"].mod(1).ne(0),
    )
    frame["_time"] = pd.to_datetime(frame["ts"], format="ISO8601", utc=True, errors="coerce")
    report.rows(
        "utc_timestamps",
        frame,
        frame["_time"].isna() | ~frame["ts"].str.contains(r"T.*(?:Z|\+00:00)$"),
    )
    _temporal_checks(frame, config, report)
    _labels(directory, frame, report)
    return report.result(manifest.get("dataset_id"))


def save_report(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/course"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    # Prevent overwriting dataset files with a report, including via symlinks.
    if args.report.resolve().is_relative_to(args.dataset.resolve()):
        parser.error("report must be outside the dataset directory")
    report = validate_dataset(args.dataset)
    save_report(report, args.report)
    print(
        f"{'PASS' if report['valid'] else 'FAIL'}: {report['failed_checks']} failed checks; {args.report}"
    )
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
