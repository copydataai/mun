from __future__ import annotations

import resource
import time
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Protocol

from .artifacts import canonical_json_bytes
from .core import SourceMedia, TranscriptionOptions, run_transcription_workflow
from .errors import MunError
from .models import InstalledModel


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

    def observed_tuple(self, requested: dict[str, Any]) -> dict[str, Any]:
        if not all(isinstance(requested.get(section), Mapping) for section in ("model", "runtime", "device")):
            return requested
        return {
            "model": {"repository": "example/model", "revision": "abc", "artifact_sha256": "a" * 64},
            "runtime": {"name": "test", "version": "0"},
            "device": {"requested": "cpu", "effective": "cpu", "precision": "test"},
        }


class PublicTranscriptionAdapter:
    physical_execution = True

    def __init__(self, model: InstalledModel, runtime: Any, *, requested_device: str) -> None:
        self.model = model
        self.runtime = runtime
        self.options = TranscriptionOptions(device=requested_device)

    def observed_tuple(self, requested: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": {
                "repository": self.model.id,
                "revision": self.model.revision,
                "artifact_sha256": getattr(self.runtime, "model_artifact_sha256", None),
            },
            "runtime": {"name": self.runtime.info.name, "version": self.runtime.info.version},
            "device": {
                "requested": self.options.device,
                "effective": self.runtime.info.effective_device,
                "precision": self.runtime.info.precision,
            },
        }

    def transcribe(self, source: Path, exact_tuple: dict[str, Any]) -> Mapping[str, Any]:
        result = run_transcription_workflow(
            [SourceMedia(source, Path(source.name))],
            self.model,
            self.options,
            runtime=self.runtime,
        )[0]
        if result.status != "completed" or not result.transcripts:
            raise QualityError("Public transcription workflow did not complete")
        variant = result.transcripts[0]
        return {
            "text": variant.text,
            "segments": [segment.__dict__ for segment in variant.segments],
            "artifact_sha256": result.result_digest,
            "capabilities": {"transcription": "passed"},
        }


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
    now: datetime | None = None,
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
    if generated.tzinfo is None or expiry_days < 1:
        raise QualityError("Fixture manifest requires timezone-aware positive evidence timestamps")
    requested_tuple = dict(exact_tuple)
    observer = getattr(runtime, "observed_tuple", None)
    observed_tuple = observer(requested_tuple) if callable(observer) else requested_tuple
    if not isinstance(observed_tuple, Mapping):
        raise QualityError("Runtime adapter did not provide an exact observed tuple")
    observed_tuple = dict(observed_tuple)
    if all(isinstance(observed_tuple.get(section), Mapping) for section in ("model", "runtime", "device")):
        required = {
            "model": ("repository", "revision", "artifact_sha256"),
            "runtime": ("name", "version"),
            "device": ("requested", "effective", "precision"),
        }
        if any(
            observed_tuple[section].get(field) in (None, "")
            for section, fields in required.items()
            for field in fields
        ):
            raise QualityError("Runtime adapter did not bind a complete runtime, model, and artifact tuple")
    tuple_drift = observed_tuple != requested_tuple
    expires_at = generated + timedelta(days=expiry_days)
    instant = now or datetime.now(UTC)
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=UTC)
    expired = instant >= expires_at

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
            result = runtime.transcribe(source, observed_tuple)
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
        if not isinstance(rule, Mapping):
            raise QualityError("Every mandatory stratum must be an explicit mapping")
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
    if mandatory_failed:
        status, reason = "failed", "mandatory_stratum_failed"
    elif expired:
        status, reason = "failed", "expired_evidence"
    elif tuple_drift:
        status, reason = "failed", "tuple_drift"
    else:
        status = "tested" if physical else "eligible"
        reason = "observed_physical_execution" if physical else "requires_physical_qualification"
    return {
        "schema_version": 1, "record_kind": "mun-transcript-quality-qualification",
        "generated_at": generated.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "expires_at": expires_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "tuple": observed_tuple, "tuple_digest": sha256(canonical_json_bytes(observed_tuple)).hexdigest(),
        "requested_tuple_digest": sha256(canonical_json_bytes(requested_tuple)).hexdigest(),
        "status": status, "status_reason": reason, "fixtures": rows, "strata": strata,
        "claim_scope": "per-fixture and stratified transcript error evidence; no universal quality score or determinism claim",
    }
