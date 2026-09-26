"""Regenerate validation evidence using a temporary valid dataset and a damaged copy."""

import argparse
import csv
import shutil
import tempfile
from pathlib import Path

from scripts.generate_ml_dataset import FIELDS, Config, write_dataset
from scripts.validate_ml_dataset import save_report, validate_dataset


def corrupt_dataset(directory: Path) -> None:
    path = directory / "validation.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["project_id"] = ""
    rows[1]["row_count"] = "not-a-number"
    rows[2]["null_rate"] = "1.5"
    rows[3]["row_count"] = "-1"
    rows[4]["d_null_rate"] = "inf"
    rows[5]["ts"] = "not-a-timestamp"
    rows[6]["ts"] = rows[6]["ts"].replace("+00:00", "")
    rows[7]["d_row_count"] = "123456"
    rows[8]["is_anomaly"] = "1"
    rows[12], rows[15] = rows[15], rows[12]
    rows.append(rows[20].copy())
    with (directory / "train.csv").open(newline="", encoding="utf-8") as stream:
        rows.append(next(csv.DictReader(stream)))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/course/validation"))
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="course-validation-") as temp:
        valid, damaged = Path(temp) / "valid", Path(temp) / "corrupted"
        write_dataset(valid, Config())
        shutil.copytree(valid, damaged)
        corrupt_dataset(damaged)
        clean_report = validate_dataset(valid)
        damaged_report = validate_dataset(damaged)
        if not clean_report["valid"] or damaged_report["valid"]:
            raise RuntimeError("demonstration did not produce expected validation outcomes")
        save_report(clean_report, args.output / "valid.json")
        save_report(damaged_report, args.output / "corrupted.json")
        print(
            f"Saved evidence: valid PASS, corrupted FAIL ({damaged_report['failed_checks']} checks)"
        )


if __name__ == "__main__":
    main()
