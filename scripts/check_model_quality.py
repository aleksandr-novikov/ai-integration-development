"""Audit frozen #9 predictions, without fitting a model or selecting a new threshold.

Exit 0: CASE quality targets passed; exit 1: targets failed; exit 2: invalid input.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ml.course_experiments import evaluate

TARGETS = {"event_recall": 0.8, "false_positive_rate": 0.1, "abrupt_median_delay_ticks": 1.0}


def quality_gate(metrics):
    result = {}
    for name, target in TARGETS.items():
        value = metrics.get(name)
        finite = isinstance(value, int | float) and math.isfinite(value)
        result[name] = {
            "actual": value,
            "target": target,
            "passed": bool(
                finite and (value >= target if name == "event_recall" else value <= target)
            ),
        }
    return {"passed": all(item["passed"] for item in result.values()), "checks": result}


def audit_predictions(predictions, events, manifest):
    """Point metrics within slices; per-kind FPR shares the same normal rows."""
    split = "test"
    interval = manifest["config"]["interval_minutes"]
    overall, details = evaluate(predictions, events, split, interval)
    by_table, by_kind = {}, {}
    for table in sorted(predictions["table_name"].unique()):
        rows = predictions[predictions["table_name"].eq(table)]
        subset = [e for e in events if e["table_name"] == table]
        by_table[table], _ = evaluate(rows, subset, split, interval)
    test_events = [e for e in events if e["split"] == split]
    for kind in sorted({e["kind"] for e in test_events}):
        subset = [e for e in test_events if e["kind"] == kind]
        ids = {e["incident_id"] for e in subset}
        rows = predictions[predictions["is_anomaly"].eq(0) | predictions["incident_id"].isin(ids)]
        by_kind[kind], _ = evaluate(rows, subset, split, interval)
    return {
        "overall": overall,
        "by_table": by_table,
        "by_incident_kind": by_kind,
        "events": details,
        "quality_gate": quality_gate(overall),
    }


def read_report(archive):
    with zipfile.ZipFile(archive) as bundle:

        def read_json(name):
            return json.loads(bundle.read(name))

        summary = read_json("summary.json")
        run = read_json("final_test/run.json")
        manifest = read_json("final_test/artifacts/manifest.json")
        events = read_json("final_test/artifacts/events.json")
        # Full intervals are in incidents.json's metadata source, not events.json,
        # which contains only detection outcomes. Rebuild the fixed generator's
        # intervals from the recorded configuration and verify their exact hash.
        from scripts.generate_ml_dataset import FEATURES, FIELDS, Config, generate

        generated, intervals = generate(Config(**manifest["config"]))
        payload = (json.dumps(intervals, indent=2, ensure_ascii=False) + "\n").encode()
        if hashlib.sha256(payload).hexdigest() != manifest["files"]["incidents.json"]["sha256"]:
            raise ValueError("generator does not reproduce the archived incident intervals")
        frame = pd.read_csv(
            io.BytesIO(bundle.read("final_test/artifacts/predictions.csv")), keep_default_na=False
        )
    if (
        summary["test"]["run_id"] != run["run_id"]
        or summary["dataset_id"] != manifest["dataset_id"]
        or summary["selection"]["name"] != summary["test"]["name"]
        or run["status"] != "FINISHED"
    ):
        raise ValueError("inconsistent run identity or dataset identity")
    if len(frame) != manifest["files"]["test.csv"]["rows"] or frame.empty:
        raise ValueError("incorrect observation count")
    if (
        not frame["is_anomaly"].isin([0, 1]).all()
        or not frame["predicted_anomaly"].isin([0, 1]).all()
    ):
        raise ValueError("labels must be binary")
    if (
        not np.isfinite(frame["score"]).all()
        or not frame["predicted_anomaly"].eq(frame["score"].lt(0).astype(int)).all()
    ):
        raise ValueError("invalid scores or inconsistent decision threshold")
    original = io.StringIO(newline="")
    writer = csv.DictWriter(original, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(generated["test"])
    if (
        hashlib.sha256(original.getvalue().encode()).hexdigest()
        != manifest["files"]["test.csv"]["sha256"]
    ):
        raise ValueError("generator does not reproduce the archived test data")
    expected = pd.DataFrame(generated["test"])
    for field in FIELDS:
        if field in FEATURES:
            matched = np.allclose(frame[field], expected[field], rtol=0, atol=1e-12)
        else:
            matched = frame[field].eq(expected[field]).all()
        if not matched:
            raise ValueError(f"prediction input mismatch: {field}")
    report = audit_predictions(frame, intervals, manifest)
    for key, value in report["overall"].items():
        stored = run["metrics"][key]
        if (value is None) != (stored is None) or (
            value is not None and not math.isclose(value, stored, abs_tol=1e-12)
        ):
            raise ValueError(f"metric mismatch: {key}")
    if report["events"] != events:
        raise ValueError("event detection outcomes do not match the archived predictions")
    return {
        "report_version": 1,
        "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "source_git_sha": summary["provenance"]["git_sha"],
        "dataset_id": manifest["dataset_id"],
        "run_id": run["run_id"],
        "selected_candidate": summary["selection"]["name"],
        "evaluation": "audit_of_frozen_test_predictions_no_refit_or_reselection",
        **report,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=Path("docs/course/experiments/mlflow-export.zip")
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.resolve() == args.archive.resolve():
        parser.error("report must not overwrite its source archive")
    try:
        report = read_report(args.archive)
    except (OSError, KeyError, ValueError, TypeError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        f"{'PASS' if report['quality_gate']['passed'] else 'FAIL'}: model quality targets; {args.report}"
    )
    raise SystemExit(0 if report["quality_gate"]["passed"] else 1)


if __name__ == "__main__":
    main()
