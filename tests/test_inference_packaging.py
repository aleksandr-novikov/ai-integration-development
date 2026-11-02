"""Static build-context regression; an actual Docker build is a separate check."""

from pathlib import Path

from docker.utils.build import exclude_paths


def test_contract_survives_dockerignore(tmp_path):
    root = Path(__file__).resolve().parents[1]
    contract = Path("docs/course/api/inference.openapi.yaml")
    target = tmp_path / contract
    target.parent.mkdir(parents=True)
    target.write_bytes((root / contract).read_bytes())
    unrelated = tmp_path / "docs/course/experiments/model.zip"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("synthetic fixture, no model weights")
    patterns = (root / ".dockerignore").read_text().splitlines()
    paths = exclude_paths(str(tmp_path), patterns)
    assert str(contract) in paths
    assert str(unrelated.relative_to(tmp_path)) not in paths
