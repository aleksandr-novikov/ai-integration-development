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


def test_private_runtime_files_are_excluded_from_build_context(tmp_path):
    root = Path(__file__).resolve().parents[1]
    private = (
        ".env", ".env.local", ".env.production", "config/.env.private",
        ".mcp.json", "env.txt", "credentials/client.key", "credentials/client.pem",
        "monitor.db", "monitor.db-wal", "monitor.db-shm", "backups/dump.sql.gz",
        "models/demo/service.json", "models/registry/versions/fixture/pipeline.skops",
        "artifacts/report.json", "data/private.csv", "mlruns/token.txt",
    )
    public = (".env.example", "models/.gitkeep", "app/app.py", "scripts/schema.sql",
              "docs/course/api/inference.openapi.yaml", "requirements-course.txt")
    for name in (*private, *public):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic packaging fixture; no secrets or model weights")
    paths = exclude_paths(str(tmp_path), (root / ".dockerignore").read_text().splitlines())
    assert not set(private).intersection(paths)
    assert set(public).issubset(paths)
