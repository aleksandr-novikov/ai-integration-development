"""Shared wire validation. No Flask, storage, scheduler or model imports."""

import copy
import json
import math
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator, FormatChecker, ValidationError
from jsonschema.validators import extend
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

CONTRACT_PATH = Path(__file__).resolve().parents[1] / "docs/course/api/inference.openapi.yaml"
MAX_JSON_BYTES = 64 * 1024
MAX_JSON_DEPTH = 32


def _number(checker, value):
    if not Draft202012Validator.TYPE_CHECKER.is_type(value, "number"):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


FiniteValidator = extend(
    Draft202012Validator,
    type_checker=Draft202012Validator.TYPE_CHECKER.redefine("number", _number),
)


def json_too_deep(raw):
    """Bound nesting before invoking the recursive JSON decoder."""
    depth, quoted, escaped = 0, False, False
    for byte in raw:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (91, 123):
            depth += 1
            if depth > MAX_JSON_DEPTH:
                return True
        elif byte in (93, 125):
            depth -= 1
    return False


def strict_json(raw):
    def reject(token):
        raise ValueError("non-standard JSON constant")

    def numeric(token):
        value = float(token)
        if not math.isfinite(value):
            raise ValueError("number overflows binary64")
        return value

    return json.loads(raw, parse_constant=reject, parse_float=numeric, parse_int=numeric)


@lru_cache(maxsize=1)
def _document():
    return yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))


def document():
    # Callers may adapt servers for Swagger without changing shared validators.
    return copy.deepcopy(_document())


@lru_cache(maxsize=16)
def validator(name):
    spec = _document()
    uri = "urn:dbmonitor:inference"
    registry = Registry().with_resource(
        uri, Resource.from_contents(spec, default_specification=DRAFT202012)
    )
    return FiniteValidator(
        {"$ref": f"{uri}#/components/schemas/{name}"},
        registry=registry,
        format_checker=FormatChecker(),
    )


def canonical_ts(value):
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.utcoffset() is None:
        raise ValueError("timezone required")
    from datetime import UTC

    try:
        return stamp.astimezone(UTC).isoformat(timespec="microseconds")
    except OverflowError as exc:
        raise ValueError("timestamp is outside the supported UTC range") from exc


def validate_semantics(payload):
    if canonical_ts(payload["previous_observed_at"]) >= canonical_ts(payload["observed_at"]):
        raise ValidationError("previous observation must precede the current one")
    from ml.features import validate_vector

    validate_vector(payload["features"])
