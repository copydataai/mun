from __future__ import annotations

import resource
import time
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Protocol

from .artifacts import canonical_json_bytes
from .errors import MunError


class QualityError(MunError):
    pass


class QualificationRuntime(Protocol):
    physical_execution: bool

    def transcribe(self, source: Path, exact_tuple: dict[str, Any]) -> Mapping[str, Any]: ...


class DeterministicFixtureRuntime:
    """Normal-test adapter that exercises the runner without claiming hardware."""

    physical_execution = False

    def transcribe(self, source: Path, exact_tuple: dict[str, Any]) -> Mapping[str, Any]:
        return {"text": source.read_text(encoding="utf-8").strip(), "segments": [], "capabilities": {"transcription": "passed"}}


def _distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for index, item in enumerate(left, 1):
        current = [index]
        for other_index, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[other_index] + 1, previous[other_index - 1] + (item != other)))
        previous = current
    return previous[-1]


def _metric(reference: str, hypothesis: str, unit: str) -> float:
    left = reference.split() if unit == "wer" else list(reference)
    right = hypothesis.split() if unit == "wer" else list(hypothesis)
    return _distance(left, right) / max(1, len(left))


def run_quality_qualification(
    manifest: Mapping[str, Any],
    *,
    base_dir: Path,
    runtime: QualificationRuntime,
) -> dict[str, Any]:
    if manifest.get("schema_version") != 1:
        raise QualityError("Unsupported fixture-manifest schema version")
    exact_tuple = manifest.get("tuple")
    fixtures = manifest.get("fixtures")
    policy = manifest.get("policy")
    if not isinstance(exact_tuple, Mapping) or not isinstance(fixtures, list) or not fixtures or not isinstance(policy, Mapping):
        raise QualityError("Fixture manifest requires tuple, fixtures, and explicit policy")
    try:
        generated = datetime.fromisoformat(str(manifest["generated_at"]).replace("Z", "+00:00"))
        expiry_days = int(manifest.get("expiry_days", 90))
    except (KeyError, ValueError) as exc:
        raise QualityError("Fixture manifest requires valid evidence timestamps") from exc

    rows = []
    for fixture in fixtures:
        if not isinstance(fixture, Mapping) or not isinstance(fixture.get("reference_text"), str):
            raise QualityError("Every fixture requires an independently supplied reference text")
        for field in ("id", "source", "source_sha256", "language", "domain", "license", "consent", "allowed_metrics"):
            if field not in fixture:
                raise QualityError(f"Fixture is missing {field}")
        source = (base_dir / str(fixture["source"])).resolve()
        if not source.is_relative_to(base_dir.resolve()) or not source.is_file():
            raise QualityError("Fixture source is missing or escapes the manifest root")
        source_bytes = source.read_bytes()
        if sha256(source_bytes).hexdigest() != fixture["source_sha256"]:
            raise QualityError("Fixture source digest mismatch")
        allowed = fixture["allowed_metrics"]
        if not isinstance(allowed, list) or not set(allowed).issubset({"wer", "cer"}) or not allowed:
            raise QualityError("Fixture allowed_metrics must explicitly select wer and/or cer")
        before_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        started = time.monotonic()
        try:
            result = runtime.transcribe(source, dict(exact_tuple))
            hypothesis = str(result["text"])
            failure = None
        except Exception:
            hypothesis = ""
            result = {"capabilities": {"transcription": "failed"}}
            failure = "transcription_failed"
        elapsed = time.monotonic() - started
        after_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        row = {
            "id": fixture["id"], "source_sha256": fixture["source_sha256"],
            "language": fixture["language"], "domain": fixture["domain"],
            "license": fixture["license"], "consent": fixture["consent"],
            "reference_sha256": sha256(fixture["reference_text"].encode()).hexdigest(),
            "artifact_sha256": sha256(canonical_json_bytes(result)).hexdigest(),
            "elapsed_seconds": round(elapsed, 6),
            "peak_memory_delta": max(0, after_memory - before_memory),
            "capabilities": result.get("capabilities", {}), "failure": failure,
        }
        for name in allowed:
            row[name] = _metric(fixture["reference_text"], hypothesis, name)
        rows.append(row)

    strata = []
    mandatory_failed = False
    mandatory = policy.get("mandatory_strata")
    if not isinstance(mandatory, list) or not mandatory:
        raise QualityError("Qualification policy requires explicit mandatory strata and thresholds")
    for rule in mandatory:
        matched = [row for row in rows if row["language"] == rule.get("language") and row["domain"] == rule.get("domain")]
        passed = bool(matched)
        measurements: dict[str, float] = {}
        for metric in ("wer", "cer"):
            threshold = rule.get(f"max_{metric}")
            values = [row[metric] for row in matched if metric in row]
            if threshold is not None:
                if not values:
                    passed = False
                else:
                    measurements[metric] = sum(values) / len(values)
                    passed = passed and measurements[metric] <= float(threshold)
        passed = passed and all(row["failure"] is None for row in matched)
        mandatory_failed = mandatory_failed or not passed
        strata.append({"language": rule.get("language"), "domain": rule.get("domain"), "outcome": "passed" if passed else "failed", "measurements": measurements, "fixture_count": len(matched)})

    physical = getattr(runtime, "physical_execution", False) is True and runtime.__class__.__module__.startswith("mun.")
    status = "failed" if mandatory_failed else ("tested" if physical else "eligible")
    reason = "mandatory_stratum_failed" if mandatory_failed else ("observed_physical_execution" if physical else "requires_physical_qualification")
    return {
        "schema_version": 1, "record_kind": "mun-transcript-quality-qualification",
        "generated_at": generated.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "expires_at": (generated + timedelta(days=expiry_days)).astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "tuple": dict(exact_tuple), "tuple_digest": sha256(canonical_json_bytes(exact_tuple)).hexdigest(),
        "status": status, "status_reason": reason, "fixtures": rows, "strata": strata,
        "claim_scope": "per-fixture and stratified transcript error evidence; no universal quality score or determinism claim",
    }
