from __future__ import annotations

import difflib
import hashlib
import json
import platform
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .artifacts import canonical_json_bytes
from .core import SourceMedia, TranscriptionOptions, render_output, run_transcription_workflow
from .errors import MunError
from .models import InstalledModel
from .transcript import TranscriptResult

ReplayKind = Literal[
    "exact_match",
    "projection_match",
    "within_tolerance",
    "semantic_drift",
    "environment_mismatch",
    "artifact_unavailable",
    "unsupported_replay",
]


@dataclass(frozen=True)
class ReplayOutcome:
    kind: ReplayKind
    reason: str | None = None
    canonical_bytes_match: bool | None = None
    projection_matches: dict[str, bool] | None = None
    normalized_diff: str | None = None
    tolerance: dict[str, Any] | None = None
    tolerance_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outcome"] = payload.pop("kind")
        return {key: value for key, value in payload.items() if value is not None}


def replay_result(
    result_path: Path,
    *,
    source: Path,
    model: InstalledModel,
    runtime: Any,
    tolerance_file: Path | None = None,
) -> ReplayOutcome:
    try:
        expected = TranscriptResult.from_json(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return ReplayOutcome("unsupported_replay", reason=f"invalid_result:{type(exc).__name__}")
    if expected.schema_version != 1 or expected.status != "completed" or expected.operation is None:
        return ReplayOutcome("unsupported_replay", reason="missing_supported_completed_operation")
    if not source.is_file():
        return ReplayOutcome("artifact_unavailable", reason="source_missing")
    if expected.source.sha256 and _sha256(source) != expected.source.sha256:
        return ReplayOutcome("artifact_unavailable", reason="source_digest_mismatch")

    mismatch = _tuple_mismatch(expected, model, runtime)
    if mismatch is not None:
        return ReplayOutcome("environment_mismatch", reason=mismatch)

    parameters = expected.operation.parameters
    options = TranscriptionOptions(
        language=parameters.language,
        timestamps=parameters.timestamps,
        translate=parameters.translate,
        chunk_length=parameters.chunk_length,
        stride_length=parameters.stride_length,
        device=parameters.requested_device,
    )
    media = SourceMedia(source.resolve(), Path(expected.source.relative_path))
    actual = run_transcription_workflow([media], model, options, runtime=runtime)[0]
    if actual.status != "completed":
        return ReplayOutcome("semantic_drift", reason="replay_inference_failed")

    canonical_match = canonical_json_bytes(expected.to_dict()) == canonical_json_bytes(actual.to_dict())
    projections = _projection_matches(expected, actual, media, model)
    if canonical_match and all(projections.values()):
        return ReplayOutcome("exact_match", canonical_bytes_match=True, projection_matches=projections)
    if all(projections[name] for name in ("srt", "txt", "vtt")):
        return ReplayOutcome("projection_match", canonical_bytes_match=False, projection_matches=projections)

    normalized_diff = _normalized_diff(expected, actual)
    if tolerance_file is not None:
        try:
            tolerance = _load_tolerance(tolerance_file)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            return ReplayOutcome("unsupported_replay", reason=f"invalid_tolerance:{type(exc).__name__}")
        if _within_tolerance(expected, actual, tolerance):
            return ReplayOutcome(
                "within_tolerance",
                canonical_bytes_match=False,
                projection_matches=projections,
                normalized_diff=normalized_diff,
                tolerance=tolerance,
                tolerance_file=str(tolerance_file),
            )
    return ReplayOutcome(
        "semantic_drift",
        canonical_bytes_match=False,
        projection_matches=projections,
        normalized_diff=normalized_diff,
    )


def _tuple_mismatch(expected: TranscriptResult, model: InstalledModel, runtime: Any) -> str | None:
    provenance = expected.provenance
    info = runtime.info
    expected_tuple = (
        provenance.model.repository,
        provenance.model.revision,
        provenance.model.artifact_sha256,
        provenance.runtime.name,
        provenance.runtime.version,
        provenance.effective_device,
        provenance.precision,
    )
    actual_tuple = (
        model.id,
        model.revision,
        getattr(runtime, "model_artifact_sha256", None),
        info.name,
        info.version,
        info.effective_device,
        info.precision,
    )
    if expected_tuple != actual_tuple:
        return "model_runtime_tuple_mismatch"
    environment = provenance.runtime.environment
    if environment is not None and (
        environment.python_version,
        environment.python_implementation,
        environment.operating_system,
        environment.machine,
    ) != (platform.python_version(), platform.python_implementation(), sys.platform, platform.machine()):
        return "runtime_environment_mismatch"
    return None


def _projection_matches(
    expected: TranscriptResult, actual: TranscriptResult, media: SourceMedia, model: InstalledModel
) -> dict[str, bool]:
    matches: dict[str, bool] = {}
    for format_name in ("json", "srt", "txt", "vtt"):
        expected_bytes = _render_projection(format_name, expected, media, model)
        actual_bytes = _render_projection(format_name, actual, media, model)
        if expected_bytes is None or actual_bytes is None:
            matches[format_name] = expected_bytes is actual_bytes
        elif format_name == "json":
            matches[format_name] = canonical_json_bytes(json.loads(expected_bytes)) == canonical_json_bytes(json.loads(actual_bytes))
        else:
            matches[format_name] = expected_bytes == actual_bytes
    return matches


def _render_projection(
    format_name: str, result: TranscriptResult, media: SourceMedia, model: InstalledModel
) -> bytes | None:
    try:
        return render_output(format_name, result, media, model, result.provenance.effective_device).encode()
    except (MunError, ValueError):
        return None


def _normalized_diff(expected: TranscriptResult, actual: TranscriptResult) -> str:
    expected_lines = [variant.text.strip() for variant in expected.transcripts]
    actual_lines = [variant.text.strip() for variant in actual.transcripts]
    return "\n".join(
        difflib.unified_diff(expected_lines, actual_lines, fromfile="expected", tofile="replay", lineterm="")
    )


def _load_tolerance(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported tolerance schema")
    if not isinstance(value["timestamp_ms"], int) or value["timestamp_ms"] < 0:
        raise ValueError("timestamp_ms must be a non-negative integer")
    max_wer = value["text"]["max_word_error_rate"]
    if not isinstance(max_wer, (int, float)) or not 0 <= max_wer <= 1:
        raise ValueError("max_word_error_rate must be between zero and one")
    return value


def _within_tolerance(expected: TranscriptResult, actual: TranscriptResult, tolerance: dict[str, Any]) -> bool:
    if len(expected.transcripts) != len(actual.transcripts):
        return False
    normalization = tolerance["normalization"]
    max_wer = float(tolerance["text"]["max_word_error_rate"])
    timing = tolerance["timestamp_ms"]
    for expected_variant, actual_variant in zip(expected.transcripts, actual.transcripts, strict=True):
        if expected_variant.kind != actual_variant.kind or expected_variant.language != actual_variant.language:
            return False
        expected_text = _normalize_text(expected_variant.text, normalization)
        actual_text = _normalize_text(actual_variant.text, normalization)
        if _word_error_rate(expected_text, actual_text) > max_wer:
            return False
        if len(expected_variant.segments) != len(actual_variant.segments):
            return False
        for expected_segment, actual_segment in zip(expected_variant.segments, actual_variant.segments, strict=True):
            if _normalize_text(expected_segment.text, normalization) != _normalize_text(actual_segment.text, normalization):
                return False
            if expected_segment.id != actual_segment.id or expected_segment.speaker_id != actual_segment.speaker_id:
                return False
            if not _time_close(expected_segment.start_ms, actual_segment.start_ms, timing):
                return False
            if not _time_close(expected_segment.end_ms, actual_segment.end_ms, timing):
                return False
    return True


def _normalize_text(value: str, rules: dict[str, Any]) -> str:
    value = unicodedata.normalize(rules.get("unicode_form", "NFC"), value)
    if rules.get("casefold", False):
        value = value.casefold()
    if rules.get("collapse_whitespace", False):
        value = re.sub(r"\s+", " ", value).strip()
    return value


def _word_error_rate(expected: str, actual: str) -> float:
    reference = expected.split()
    hypothesis = actual.split()
    if not reference:
        return 0.0 if not hypothesis else 1.0
    previous = list(range(len(hypothesis) + 1))
    for index, reference_word in enumerate(reference, start=1):
        current = [index]
        for offset, hypothesis_word in enumerate(hypothesis, start=1):
            current.append(min(current[-1] + 1, previous[offset] + 1, previous[offset - 1] + (reference_word != hypothesis_word)))
        previous = current
    return previous[-1] / len(reference)


def _time_close(expected: int | None, actual: int | None, tolerance: int) -> bool:
    if expected is None or actual is None:
        return expected is actual
    return abs(expected - actual) <= tolerance


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
