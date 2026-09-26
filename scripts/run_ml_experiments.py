"""Run the fixed five-candidate protocol, then evaluate the winner once on test."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

# All tracking is local. No remote tracking server or telemetry is needed.
os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/course-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from ml.course_experiments import (
    CANDIDATES,
    evaluate,
    fit_candidate,
    meets_targets,
    predict,
    select_candidate,
)
from scripts.generate_ml_dataset import FEATURES
from scripts.validate_ml_dataset import save_report, validate_dataset

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "course-isolation-forest"


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def provenance():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    sources = (
        "ml/course_experiments.py",
        "scripts/run_ml_experiments.py",
        "scripts/import_ml_experiments.py",
        "scripts/generate_ml_dataset.py",
        "scripts/synthetic_metrics.py",
        "scripts/validate_ml_dataset.py",
        "docs/course/experiments/PROTOCOL.md",
    )
    return {
        "git_sha": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources
        },
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "mlflow",
                "scikit-learn",
                "numpy",
                "scipy",
                "pandas",
                "matplotlib",
                "skops",
                "joblib",
                "threadpoolctl",
            )
        },
    }


def init_tracking(output):
    uri = f"sqlite:///{output.resolve() / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    experiment_id = mlflow.create_experiment(
        EXPERIMENT, artifact_location=(output.resolve() / "mlartifacts").as_uri()
    )
    return experiment_id


def plot_metrics(rows, destination):
    names = [r["name"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
    for ax, metric, title, target in zip(
        axes,
        ("point_f1", "false_positive_rate", "event_recall"),
        ("Point F1", "False positive rate", "Event recall"),
        (None, 0.1, 0.8),
        strict=True,
    ):
        values = [r["metrics"][metric] or 0 for r in rows]
        bars = ax.barh(names, values, color=["#667085"] + ["#247d91"] * (len(rows) - 1))
        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
        ax.set_xlim(0, max(1.15, max(values) * 1.15))
        ax.invert_yaxis()
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
        if target is not None:
            ax.axvline(target, color="#d85b35", linestyle="--", linewidth=1)
    fig.suptitle("Isolation Forest — validation only (fixed seed 42)")
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def record_run(
    output,
    experiment_id,
    candidate,
    models,
    frame,
    events,
    split,
    manifest,
    source,
    validation,
    seed,
    *,
    comparison=None,
    selection=None,
):
    name = candidate.name if split == "validation" else "final_test"
    folder = output / "export" / name
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True)
    predictions = predict(models, frame)
    metrics, event_details = evaluate(
        predictions, events, split, manifest["config"]["interval_minutes"]
    )
    parameters = {
        **candidate.parameters(),
        "seed": seed,
        "dataset_id": manifest["dataset_id"],
        "features": ",".join(FEATURES),
        "threshold": 0,
        "fit_split": "train",
        "evaluation_split": split,
        "selected_candidate": candidate.name,
    }
    tags = {
        "git_sha": source["git_sha"],
        "git_dirty": str(source["git_dirty"]).lower(),
        "mlflow.source.git.commit": source["git_sha"],
        "dataset_id": manifest["dataset_id"],
        "phase": split,
        "course_issue": "9",
    }
    with mlflow.start_run(experiment_id=experiment_id, run_name=name, tags=tags) as active:
        mlflow.log_params(parameters)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
        mlflow.set_tag("targets_met", str(meets_targets(metrics)).lower())
        write_json(artifacts / "metrics.json", metrics)
        write_json(artifacts / "events.json", event_details)
        write_json(artifacts / "manifest.json", manifest)
        write_json(artifacts / "provenance.json", source)
        write_json(artifacts / "data-validation.json", validation)
        predictions.to_csv(artifacts / "predictions.csv", index=False)
        shutil.copyfile(ROOT / "docs/course/experiments/PROTOCOL.md", artifacts / "PROTOCOL.md")
        for table, model in models.items():
            sample = (
                frame.loc[frame["table_name"].eq(table), list(FEATURES)]
                .head(2)
                .to_numpy(dtype=float)
            )
            mlflow.sklearn.save_model(
                model,
                str(artifacts / "models" / table),
                serialization_format="skops",
                # These trees were fitted above from validated local data.
                skops_trusted_types=["sklearn.tree._tree.Tree"],
                input_example=sample,
                signature=infer_signature(sample, model.predict(sample)),
                pip_requirements=[f"{k}=={v}" for k, v in source["packages"].items()],
            )
        if selection:
            write_json(artifacts / "selection.json", selection)
        if comparison:
            pd.DataFrame(
                [
                    {**r["parameters"], **r["metrics"], "name": r["name"], "run_id": r["run_id"]}
                    for r in comparison
                ]
            ).to_csv(artifacts / "comparison.csv", index=False)
            plot_metrics(comparison, artifacts / "validation.png")
        mlflow.log_artifacts(str(artifacts))
        run_id = active.info.run_id
    stored = mlflow.get_run(run_id)
    descriptor = {
        "name": name,
        "run_id": run_id,
        "status": stored.info.status,
        "start_time": stored.info.start_time,
        "end_time": stored.info.end_time,
        "parameters": stored.data.params,
        "metrics": metrics,
        "tags": stored.data.tags,
    }
    write_json(folder / "run.json", descriptor)
    return {
        "name": candidate.name,
        "run_id": run_id,
        "parameters": candidate.parameters(),
        "metrics": metrics,
        "targets_met": meets_targets(metrics),
    }


def run_suite(dataset, output, seed=42, *, require_clean=True):
    dataset, output = dataset.resolve(), output.resolve()
    if output.is_relative_to(dataset) or dataset.is_relative_to(output):
        raise ValueError("dataset and output must be separate directories")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty; choose a new directory")
    source = provenance()
    if require_clean and source["git_dirty"]:
        raise ValueError("commit changes before a recorded experiment (Git worktree is dirty)")
    output.mkdir(parents=True, exist_ok=True)
    validation = validate_dataset(dataset)
    save_report(validation, output / "data-validation.json")
    if not validation["valid"]:
        raise ValueError(
            "dataset validation failed; see data-validation.json; training not started"
        )
    manifest = json.loads((dataset / "manifest.json").read_text())
    events = json.loads((dataset / "incidents.json").read_text())
    experiment_id = init_tracking(output)
    frame = pd.read_csv(dataset / "validation.csv", keep_default_na=False)
    models_by_name, results = {}, []
    for candidate in CANDIDATES:
        models = fit_candidate(dataset, candidate, seed)
        models_by_name[candidate.name] = models
        result = record_run(
            output,
            experiment_id,
            candidate,
            models,
            frame,
            events,
            "validation",
            manifest,
            source,
            validation,
            seed,
        )
        results.append(result)
        print(
            f"validation {candidate.name}: F1={result['metrics']['point_f1']:.4f}, "
            f"FPR={result['metrics']['false_positive_rate']:.4f}",
            flush=True,
        )
    winner = select_candidate(results)
    selection = {
        "name": winner["name"],
        "validation_run_id": winner["run_id"],
        "validation_targets_met": winner["targets_met"],
        "selected_using": "validation_only",
        "parameters": winner["parameters"],
    }
    # Persist the decision BEFORE loading/scoring the test period.
    write_json(output / "selection.json", selection)
    candidate = next(c for c in CANDIDATES if c.name == winner["name"])
    test = pd.read_csv(dataset / "test.csv", keep_default_na=False)
    final = record_run(
        output,
        experiment_id,
        candidate,
        models_by_name[winner["name"]],
        test,
        events,
        "test",
        manifest,
        source,
        validation,
        seed,
        comparison=results,
        selection=selection,
    )
    summary = {
        "dataset_id": manifest["dataset_id"],
        "provenance": source,
        "seed": seed,
        "validation": results,
        "selection": selection,
        "test": final,
    }
    write_json(output / "export" / "summary.json", summary)
    shutil.make_archive(str(output / "mlflow-export"), "zip", output / "export")
    print(f"Selected {winner['name']}; final test targets_met={final['targets_met']}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/course"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/course-experiments/run-01"))
    args = parser.parse_args()
    try:
        run_suite(args.dataset, args.output)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
