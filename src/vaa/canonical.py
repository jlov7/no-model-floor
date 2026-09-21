from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

PROFILE = "VAA-CANON-JSON-2"
VERSION = "2.0.0"
MAX_SAFE_INTEGER = 2**53 - 1


class CanonicalError(ValueError):
    pass


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        if unicodedata.normalize("NFC", value) != value:
            raise CanonicalError("strings must be NFC normalized")
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise CanonicalError("surrogate code points are forbidden")
        return value
    if isinstance(value, int):
        if abs(value) > MAX_SAFE_INTEGER:
            raise CanonicalError("integer is outside the interoperable safe range")
        return value
    if isinstance(value, float):
        raise CanonicalError("floating point values are forbidden")
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalError("object keys must be strings")
            normalized_key = _normalize(key)
            if normalized_key in normalized:
                raise CanonicalError("duplicate object key after normalization")
            normalized[normalized_key] = _normalize(item)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [_normalize(item) for item in value]
    raise CanonicalError(f"unsupported canonical value: {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _normalize(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def envelope(
    artifact_kind: str,
    payload: Mapping[str, Any],
    *,
    inputs: Sequence[Mapping[str, str]] = (),
    producer: str = "VAA-CORE",
    claim_ceiling: str = "LOCAL_SYNTHETIC_DEVELOPMENT_ONLY",
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "artifact_kind": artifact_kind,
        "artifact_version": VERSION,
        "claim_ceiling": claim_ceiling,
        "inputs": [dict(item) for item in inputs],
        "payload": dict(payload),
        "producer": producer,
        "profile": PROFILE,
    }
    value["sha256"] = digest(value)
    return value


def verify_envelope(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    expected = {
        "artifact_kind",
        "artifact_version",
        "claim_ceiling",
        "inputs",
        "payload",
        "producer",
        "profile",
        "sha256",
    }
    if set(value) != expected:
        return False
    if value.get("artifact_version") != VERSION or value.get("profile") != PROFILE:
        return False
    if not isinstance(value.get("artifact_kind"), str) or not value["artifact_kind"]:
        return False
    if not isinstance(value.get("payload"), dict) or not isinstance(value.get("inputs"), list):
        return False
    supplied = value.get("sha256")
    if not isinstance(supplied, str) or len(supplied) != 64:
        return False
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    try:
        return supplied == digest(unsigned)
    except CanonicalError:
        return False


def ref(value: Mapping[str, Any]) -> dict[str, str]:
    if not verify_envelope(dict(value)):
        raise CanonicalError("cannot reference an invalid envelope")
    return {"artifact_kind": str(value["artifact_kind"]), "sha256": str(value["sha256"])}
