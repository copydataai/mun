from __future__ import annotations

import hashlib
import json
import statistics
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import MunError


SCHEMA_VERSION = 1
CAPABILITY_OUTCOMES = {"passed", "failed", "unsupported"}
TUPLE_SECTIONS = ("mun", "runtime", "model", "platform", "device")


class QualificationError(MunError):
    pass


def create_qualification_record(manifest: Mapping[str, Any], *, base_dir: Path) -> dict[str, Any]:
    generated_at = _parse_time(_required(manifest, "generated_at"), "generated_at")
    expiry_days = _positive_int(manifest.get("expiry_days", 90), "expiry_days")
    exact_tuple = {name: deepcopy(_mapping(_required(manifest, name), name)) for name in TUPLE_SECTIONS}
    _validate_tuple(exact_tuple)

    fixtures = _fixture_records(_list(_required(manifest, "fixtures"), "fixtures"), base_dir)
    capabilities = _capability_records(_list(_required(manifest, "capabilities"), "capabilities"))
    advertised = sorted(set(_string_list(manifest.get("advertised_capabilities", []), "advertised_capabilities")))
    missing = sorted(set(advertised) - {row["name"] for row in capabilities})
    if missing:
        raise QualificationError(f"missing capability outcomes: {', '.join(missing)}")

    run_outcome = manifest.get("run_outcome", "passed")
    if run_outcome not in {"passed", "failed", "unsupported"}:
        raise QualificationError("run_outcome must be passed, failed, or unsupported")
    status, reason = _status(run_outcome, capabilities, advertised)

    timing = _timing_record(manifest.get("timing"), required=False)
    peak_memory = _peak_memory_record(manifest.get("peak_memory"))
    tuple_digest = _digest(exact_tuple)
    record = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": "mun-exact-tuple-qualification",
        "unsigned": True,
        "local_record": True,
        "generated_at": _format_time(generated_at),
        "expires_at": _format_time(generated_at + timedelta(days=expiry_days)),
        "status": status,
        "status_reason": reason,
        "tuple": exact_tuple,
        "tuple_digest": tuple_digest,
        "fixtures": fixtures,
        "timing": timing,
        "peak_memory": peak_memory,
        "advertised_capabilities": advertised,
        "capabilities": capabilities,
        "diagnostic": manifest.get("diagnostic"),
        "claim_scope": "runtime compatibility and observed execution only; not model quality",
    }
    return record


def qualification_is_expired(
    record: Mapping[str, Any],
    *,
    current_tuple: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> bool:
    if record.get("schema_version") != SCHEMA_VERSION:
        return True
    try:
        expires_at = _parse_time(record["expires_at"], "expires_at")
    except (KeyError, QualificationError):
        return True
    instant = now or datetime.now(UTC)
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=UTC)
    if instant >= expires_at:
        return True
    if current_tuple is not None and _digest(current_tuple) != record.get("tuple_digest"):
        return True
    return _digest(record.get("tuple")) != record.get("tuple_digest")


def missing_tested_claims(
    records: Iterable[Mapping[str, Any]],
    advertised_claims: Iterable[Mapping[str, str]],
    *,
    now: datetime | None = None,
) -> list[Mapping[str, str]]:
    passing: set[tuple[str, str]] = set()
    for record in records:
        if record.get("status") != "tested" or qualification_is_expired(record, now=now):
            continue
        digest = record.get("tuple_digest")
        for capability in record.get("capabilities", []):
            if isinstance(capability, Mapping) and capability.get("outcome") == "passed":
                passing.add((str(digest), str(capability.get("name"))))
    return [
        claim
        for claim in advertised_claims
        if (claim.get("tuple_digest", ""), claim.get("capability", "")) not in passing
    ]


def _status(
    run_outcome: str,
    capabilities: list[dict[str, Any]],
    advertised: list[str],
) -> tuple[str, str]:
    if run_outcome == "unsupported":
        return "unsupported", "outside_supported_matrix"
    if run_outcome == "failed" or any(row["outcome"] == "failed" for row in capabilities):
        return "failed", "physical_execution_failed"
    if any(row["outcome"] != "passed" for row in capabilities if row["name"] in advertised):
        return "failed", "advertised_capability_did_not_pass"
    return "eligible", "observed_execution_unavailable"


def _validate_tuple(exact_tuple: Mapping[str, Mapping[str, Any]]) -> None:
    required = {
        "mun": ("version", "revision"),
        "runtime": ("name", "version", "manifest_sha256"),
        "model": ("repository", "revision", "artifact_sha256", "manifest_sha256"),
        "platform": ("os", "os_version", "kernel", "architecture", "python"),
        "device": (
            "name", "backend", "driver_version", "accelerator_target", "physical_memory_bytes",
            "requested", "effective", "precision",
        ),
    }
    for section, fields in required.items():
        missing = [field for field in fields if exact_tuple[section].get(field) in (None, "")]
        if missing:
            raise QualificationError(f"{section} missing exact tuple fields: {', '.join(missing)}")


def _fixture_records(rows: list[Any], base_dir: Path) -> list[dict[str, Any]]:
    if not rows:
        raise QualificationError("fixtures must contain at least one content-addressed fixture")
    records = []
    for index, value in enumerate(rows):
        row = _mapping(value, f"fixtures[{index}]")
        relative = Path(str(_required(row, "path")))
        if relative.is_absolute() or ".." in relative.parts:
            raise QualificationError(f"fixture path must be relative: {relative}")
        path = base_dir / relative
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise QualificationError(f"cannot read fixture {relative}: {exc}") from exc
        records.append({
            "path": relative.as_posix(),
            "role": row.get("role"),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    return sorted(records, key=lambda row: (row["role"] or "", row["path"]))


def _capability_records(rows: list[Any]) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(rows):
        row = _mapping(value, f"capabilities[{index}]")
        name = str(_required(row, "name"))
        outcome = _required(row, "outcome")
        if outcome not in CAPABILITY_OUTCOMES:
            raise QualificationError(f"invalid capability outcome for {name}: {outcome}")
        if name in records:
            raise QualificationError(f"duplicate capability outcome: {name}")
        records[name] = {"name": name, "outcome": outcome, "diagnostic": row.get("diagnostic")}
    return [records[name] for name in sorted(records)]


def _timing_record(value: Any, *, required: bool) -> dict[str, Any] | None:
    if value is None:
        if required:
            raise QualificationError("tested records require cold and warm timing")
        return None
    timing = _mapping(value, "timing")
    cold = _mapping(_required(timing, "cold"), "timing.cold")
    warm = _mapping(_required(timing, "warm"), "timing.warm")
    samples = [float(sample) for sample in _list(_required(warm, "samples_seconds"), "timing.warm.samples_seconds")]
    if not samples or any(sample < 0 for sample in samples):
        raise QualificationError("warm timing samples must contain non-negative values")
    audio_seconds = float(_required(warm, "audio_seconds"))
    if audio_seconds <= 0:
        raise QualificationError("timing.warm.audio_seconds must be positive")
    median = statistics.median(samples)
    return {
        "cold": {
            "load_seconds": float(_required(cold, "load_seconds")),
            "end_to_end_seconds": float(_required(cold, "end_to_end_seconds")),
        },
        "warm": {
            "samples_seconds": samples,
            "median_seconds": median,
            "audio_seconds": audio_seconds,
            "median_rtf": median / audio_seconds,
        },
    }


def _peak_memory_record(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    record = deepcopy(_mapping(value, "peak_memory"))
    if record.get("process_rss_bytes") is None and not record.get("not_measurable_reason"):
        raise QualificationError("peak_memory requires process_rss_bytes or not_measurable_reason")
    return record


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_time(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise QualificationError(f"{field} must be an ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise QualificationError(f"{field} must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise QualificationError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


def _format_time(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _required(mapping: Mapping[str, Any], field: str) -> Any:
    if field not in mapping or mapping[field] is None:
        raise QualificationError(f"missing required field: {field}")
    return mapping[field]


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QualificationError(f"{field} must be an object")
    return value


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise QualificationError(f"{field} must be a list")
    return value


def _string_list(value: Any, field: str) -> list[str]:
    return [str(item) for item in _list(value, field)]


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise QualificationError(f"{field} must be a positive integer")
    return value
