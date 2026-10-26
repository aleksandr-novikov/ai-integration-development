"""ASGI factory: uvicorn inference.service:create_app --factory."""

import copy
import json
import logging
import os
import secrets
import threading
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from jsonschema import ValidationError
from starlette.concurrency import run_in_threadpool

from ml.features import FEATURE_SCHEMA_VERSION
from ml.inference_contract import document, strict_json, validate_semantics, validator
from ml.model_registry import RegistryError, inspect_version, load_version, validate_source

logger = logging.getLogger(__name__)


class ApiError(Exception):
    def __init__(self, status, code):
        self.status, self.code = status, code


def read_config():
    filename = os.environ.get("INFERENCE_CONFIG")
    if not filename:
        return {"registry_path": "models/registry", "principals": [], "demo": False}
    config = json.loads(Path(filename).read_text(encoding="utf-8"))
    principals = []
    for entry in config["principals"]:
        token = os.environ.get(entry["token_env"], "")
        if not token:
            raise ValueError("configured inference token environment variable is empty")
        principals.append({"token": token, "sources": entry["sources"]})
    return {**config, "principals": principals}


def create_app(config=None):
    config = copy.deepcopy(read_config() if config is None else config)
    if type(config.get("demo", False)) is not bool:
        raise ValueError("demo must be a boolean")
    principals = config.get("principals", [])
    for entry in principals:
        if not isinstance(entry["token"], str) or not entry["token"]:
            raise ValueError("empty service token")
        entry["token"] = entry["token"].encode("utf-8")
        entry["sources"] = [validate_source(source) for source in entry["sources"]]
    app = FastAPI(title="DB Monitoring inference")
    spec = document()
    spec["servers"] = [{"url": "/", "description": "Current service origin"}]
    request_validator = validator("PredictionRequest")
    response_validator = validator("PredictionResponse")
    model_validator = validator("ModelRequest")
    cache = OrderedDict()
    cache_lock = threading.Lock()

    @app.exception_handler(ApiError)
    async def api_error(request, exc):
        headers = {"WWW-Authenticate": "Bearer"} if exc.status == 401 else None
        return JSONResponse(
            status_code=exc.status,
            content={"error": {"code": exc.code, "message": exc.code}},
            headers=headers,
        )

    @app.exception_handler(Exception)
    async def internal_error(request, exc):
        logger.exception("Inference request failed")
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "Inference failed"}},
        )

    async def authorized_payload(request, schema):
        authorization = request.headers.get("authorization", "")
        scheme, _, token = authorization.partition(" ")
        principal = None
        if scheme.lower() == "bearer" and token:
            for entry in principals:
                if secrets.compare_digest(token.encode("utf-8"), entry["token"]):
                    principal = entry
        if principal is None:
            raise ApiError(401, "authentication_required")
        if (
            request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            != "application/json"
        ):
            raise ApiError(415, "unsupported_media_type")
        try:
            payload = strict_json(await request.body())
        except (ValueError, UnicodeError):
            raise ApiError(400, "invalid_json") from None
        try:
            schema.validate(payload)
        except ValidationError:
            raise ApiError(422, "invalid_input") from None
        if payload["source"] not in principal["sources"]:
            raise ApiError(403, "source_forbidden")
        if payload["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
            raise ApiError(409, "feature_schema_mismatch")
        return payload

    def resolve(payload):
        try:
            manifest = inspect_version(
                config["registry_path"], payload["model_version"], source=payload["source"]
            )
        except RegistryError:
            raise ApiError(503, "model_unavailable") from None
        if manifest["feature_schema_version"] != payload["feature_schema_version"]:
            raise ApiError(409, "feature_schema_mismatch")
        try:
            key = (payload["model_version"], tuple(sorted(payload["source"].items())))
            with cache_lock:
                # Even cache hits recheck source, manifest, admission and the
                # current artifact bytes. Corruption never serves a stale copy.
                loaded = load_version(
                    config["registry_path"],
                    payload["model_version"],
                    source=payload["source"],
                    demo=config.get("demo", False),
                    cached=cache.get(key),
                )
                cache[key] = loaded
                cache.move_to_end(key)
                if len(cache) > 16:
                    cache.popitem(last=False)
                return loaded
        except RegistryError:
            raise ApiError(503, "model_unavailable") from None

    @app.get("/healthz")
    def health():
        return {"status": "ok", "demo": config.get("demo", False)}

    @app.post("/v1/model")
    async def model(request: Request):
        payload = await authorized_payload(request, model_validator)
        loaded = await run_in_threadpool(resolve, payload)
        manifest = loaded.manifest
        return {
            "source": manifest["source"],
            "model_version": manifest["model_version"],
            "feature_schema_version": manifest["feature_schema_version"],
            "demo": config.get("demo", False),
            "quality_admitted": not config.get("demo", False),
        }

    @app.post("/v1/predict")
    async def predict(request: Request):
        payload = await authorized_payload(request, request_validator)
        try:
            validate_semantics(payload)
        except (ValueError, ValidationError):
            raise ApiError(422, "invalid_input") from None

        def score():
            loaded = resolve(payload)
            try:
                prediction = loaded.predict(payload["features"])
            except RegistryError:
                raise ApiError(503, "model_unavailable") from None
            except ValueError:
                raise ApiError(422, "invalid_input") from None
            result = {
                **prediction,
                "observed_at": payload["observed_at"],
                "demo": config.get("demo", False),
                "quality_admitted": not config.get("demo", False),
            }
            response_validator.validate(result)
            return result

        return await run_in_threadpool(score)

    # Serve the reviewed contract in Swagger, including the manual strict JSON
    # boundary. No generated permissive/coercing body schema replaces it.
    app.openapi = lambda: copy.deepcopy(spec)
    return app
