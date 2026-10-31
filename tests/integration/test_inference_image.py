"""Build the actual application image and exercise its packaged ASGI routes."""

import uuid
from pathlib import Path

import docker
import pytest

pytestmark = pytest.mark.integration


def test_runtime_image_contains_contract_and_serves_routes():
    root = Path(__file__).resolve().parents[2]
    tag = f"db-monitoring-contract-test:{uuid.uuid4().hex}"
    client = docker.from_env(timeout=1200)
    try:
        client.images.build(path=str(root), tag=tag, rm=True)
        output = client.containers.run(
            tag,
            [
                "python",
                "-c",
                """
from pathlib import Path
from fastapi.testclient import TestClient
from inference.service import create_app
from ml.inference_contract import CONTRACT_PATH, validator
assert CONTRACT_PATH.is_file()
assert not Path('/app/docs/course/experiments').exists()
validator('PredictionRequest')
with TestClient(create_app()) as client:
    assert client.get('/healthz').status_code == 200
    assert client.get('/docs').status_code == 200
    assert client.get('/openapi.json').status_code == 200
    assert client.post('/v1/model', json={}).status_code == 401
    assert client.post('/v1/predict', json={}).status_code == 401
print('packaged-routes-ok')
""",
            ],
            remove=True,
            network_disabled=True,
        )
        assert b"packaged-routes-ok" in output
    finally:
        try:
            client.images.remove(tag)
        except docker.errors.ImageNotFound:
            pass
        client.close()
