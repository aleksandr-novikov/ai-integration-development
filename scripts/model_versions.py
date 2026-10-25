"""Stage, inspect, activate, restore and score local versioned IF artifacts (#5)."""

import argparse
import json
import subprocess
from pathlib import Path

from ml.model_registry import (
    activate,
    load_active,
    load_version,
    stage_bytes,
    stage_export,
    train_candidate,
)

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("models/registry"))
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser(
        "stage-export", help="Reuse the immutable #9 export without training"
    )
    stage.add_argument("--source", type=Path, required=True)
    stage.add_argument(
        "--archive", type=Path, default=Path("docs/course/experiments/mlflow-export.zip")
    )
    train = commands.add_parser(
        "train", help="Train a new unevaluated candidate from a metric snapshot"
    )
    train.add_argument("--snapshot", type=Path, required=True)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--run-id", default=None)
    local = commands.add_parser(
        "stage-local", help="Stage a trusted local pipeline with training/evaluation evidence"
    )
    local.add_argument("--source", type=Path, required=True)
    local.add_argument("--pipeline", type=Path, required=True)
    local.add_argument("--training", type=Path, required=True)
    local.add_argument("--evaluation", type=Path)
    for name in ("inspect", "activate", "predict"):
        command = commands.add_parser(name)
        command.add_argument("--source", type=Path, required=True)
        command.add_argument("--version", required=name != "predict")
        command.add_argument(
            "--demo", action="store_true", help="Use the separate rejected/unevaluated demo scope"
        )
        if name == "predict":
            command.add_argument("--features", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
            if dirty.strip():
                raise ValueError("commit source changes before recorded candidate training")
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip()
            snapshot = read_json(args.snapshot)
            result = train_candidate(
                args.registry,
                source=snapshot["source"],
                row_counts=snapshot["row_counts"],
                null_rates=snapshot["null_rates"],
                git_sha=git_sha,
                seed=args.seed,
                run_id=args.run_id,
            )
        else:
            source = read_json(args.source)
            if args.command == "stage-export":
                result = stage_export(args.registry, args.archive, source=source)
            elif args.command == "stage-local":
                result = stage_bytes(
                    args.registry,
                    args.pipeline.read_bytes(),
                    source=source,
                    training=read_json(args.training),
                    evaluation=read_json(args.evaluation) if args.evaluation else None,
                    origin={"kind": "trusted-local-evidence"},
                )
            elif args.command == "activate":
                result = activate(args.registry, args.version, source=source, demo=args.demo)
            else:
                loaded = (
                    load_version(args.registry, args.version, source=source, demo=args.demo)
                    if args.version
                    else load_active(args.registry, source=source, demo=args.demo)
                )
                result = (
                    loaded.manifest
                    if args.command == "inspect"
                    else loaded.predict(read_json(args.features))
                )
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
