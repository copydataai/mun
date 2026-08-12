from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from typing import Any, Mapping

_VOLATILE_FIELDS = frozenset({"created_at", "result_digest"})


class ArtifactValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable UTF-8 JSON bytes for machine-result identity."""
    normalized = _normalize(asdict(value) if is_dataclass(value) else value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def machine_result_digest(value: Any) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def validate_machine_result(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    claimed = payload.get("result_digest")
    if claimed is None:
        return payload
    if not isinstance(claimed, str) or claimed != machine_result_digest(payload):
        raise ArtifactValidationError(
            "result_digest_mismatch",
            "Machine result digest does not match its contents",
        )
    return payload


def _normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize(item)
            for key, item in value.items()
            if key not in _VOLATILE_FIELDS
        }
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, str):
        return value.replace("\r\n", "\n").replace("\r", "\n")
    return value
