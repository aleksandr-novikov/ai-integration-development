"""Import the portable #9 export into a fresh local MLflow store (no retraining)."""

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

from scripts.run_ml_experiments import init_tracking, mlflow, write_json


def import_export(archive, output):
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty")
    with tempfile.TemporaryDirectory(prefix="mlflow-import-") as temporary:
        root = Path(temporary).resolve()
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                if not (root / member.filename).resolve().is_relative_to(root):
                    raise ValueError("unsafe path in archive")
            bundle.extractall(root)
        descriptors = sorted(root.glob("*/run.json"))
        if not descriptors:
            raise ValueError("archive contains no runs")
        output.mkdir(parents=True, exist_ok=True)
        experiment_id = init_tracking(output)
        mapping = {}
        for descriptor in descriptors:
            data = json.loads(descriptor.read_text())
            if data["status"] != "FINISHED":
                raise ValueError("only completed experiment runs can be imported")
            tags = {k: v for k, v in data["tags"].items() if not k.startswith("mlflow.")}
            tags["original_run_id"] = data["run_id"]
            tags["original_start_time"] = str(data["start_time"])
            with mlflow.start_run(
                experiment_id=experiment_id, run_name=data["name"], tags=tags
            ) as run:
                mlflow.log_params(data["parameters"])
                mlflow.log_metrics({k: v for k, v in data["metrics"].items() if v is not None})
                mlflow.log_artifacts(str(descriptor.parent / "artifacts"))
                mlflow.log_artifact(str(descriptor), artifact_path="original")
                mapping[data["run_id"]] = run.info.run_id
        write_json(output / "run-id-map.json", mapping)
        return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        mapping = import_export(args.archive, args.output)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Imported {len(mapping)} runs into {args.output / 'mlflow.db'}")


if __name__ == "__main__":
    main()
