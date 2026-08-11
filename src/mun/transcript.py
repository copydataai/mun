from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

SCHEMA_VERSION = 1
Status = Literal["completed", "partial", "failed", "cancelled"]
TranscriptKind = Literal["original", "english_translation"]
LanguageSource = Literal["detected", "forced", "model", "unknown"]


@dataclass(frozen=True)
class Language:
    tag: str | None
    source: LanguageSource = "unknown"
    confidence: float | None = None


@dataclass(frozen=True)
class Word:
    text: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class TranscriptSegment:
    id: str
    start_ms: int | None
    end_ms: int | None
    text: str
    speaker_id: str | None = None
    words: list[Word] | None = None


@dataclass(frozen=True)
class TranscriptVariant:
    kind: TranscriptKind
    language: Language
    text: str
    segments: list[TranscriptSegment]


@dataclass(frozen=True)
class Speaker:
    id: str
    assigned_speech_ms: int
    speaking_ms: int


@dataclass(frozen=True)
class Diagnostic:
    severity: Literal["warning", "error"]
    code: str
    message: str
    stage: str
    recoverable: bool


@dataclass(frozen=True)
class SourceRecord:
    name: str
    relative_path: str
    duration_ms: int | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class ModelProvenance:
    repository: str
    revision: str | None
    artifact_sha256: str | None = None


@dataclass(frozen=True)
class RuntimeProvenance:
    name: str
    version: str | None


@dataclass(frozen=True)
class Provenance:
    mun_version: str
    created_at: str
    model: ModelProvenance
    runtime: RuntimeProvenance
    requested_device: str
    effective_device: str
    precision: str | None


@dataclass(frozen=True)
class TranscriptResult:
    schema_version: int
    status: Status
    source: SourceRecord
    transcripts: list[TranscriptVariant]
    speakers: list[Speaker]
    diagnostics: list[Diagnostic]
    provenance: Provenance
    overlap_ms: int = 0

    def to_dict(self) -> dict:
        return _drop_none(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"


@dataclass(frozen=True)
class BatchResult:
    schema_version: int
    files: list[TranscriptResult]
    counts: dict[str, int]

    def to_dict(self) -> dict:
        return {"schema_version": self.schema_version, "files": [f.to_dict() for f in self.files], "counts": self.counts}


def make_batch_result(files: list[TranscriptResult]) -> BatchResult:
    counts = {status: 0 for status in ("completed", "partial", "failed", "cancelled")}
    for result in files:
        counts[result.status] = counts.get(result.status, 0) + 1
    return BatchResult(SCHEMA_VERSION, files, counts)


def make_provenance(*, mun_version: str, model_id: str, revision: str | None, runtime_name: str, runtime_version: str | None, requested_device: str, effective_device: str, precision: str | None) -> Provenance:
    return Provenance(
        mun_version=mun_version,
        created_at=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        model=ModelProvenance(model_id, revision),
        runtime=RuntimeProvenance(runtime_name, runtime_version),
        requested_device=requested_device,
        effective_device=effective_device,
        precision=precision,
    )


def render_txt(result: TranscriptResult, kind: TranscriptKind | None = None) -> str:
    variant = select_variant(result, kind)
    return variant.text.strip() + "\n"


def render_json(result: TranscriptResult) -> str:
    return result.to_json()


def render_srt(result: TranscriptResult, kind: TranscriptKind | None = None) -> str:
    return _render_subtitles(select_variant(result, kind).segments, "srt")


def render_vtt(result: TranscriptResult, kind: TranscriptKind | None = None) -> str:
    return _render_subtitles(select_variant(result, kind).segments, "vtt")


def select_variant(result: TranscriptResult, kind: TranscriptKind | None = None) -> TranscriptVariant:
    if kind:
        for variant in result.transcripts:
            if variant.kind == kind:
                return variant
    if result.transcripts:
        return result.transcripts[0]
    raise ValueError("Transcript result has no transcript variants")


def _render_subtitles(segments: list[TranscriptSegment], format_name: str) -> str:
    if not segments or any(segment.start_ms is None or segment.end_ms is None for segment in segments):
        raise ValueError(f"{format_name.upper()} output requires timestamps")
    lines = ["WEBVTT", ""] if format_name == "vtt" else []
    for index, segment in enumerate(segments, start=1):
        if format_name == "srt":
            lines.append(str(index))
        text = f"[{segment.speaker_id}] {segment.text}" if segment.speaker_id else segment.text
        lines.append(f"{_timestamp(segment.start_ms, format_name)} --> {_timestamp(segment.end_ms, format_name)}")
        lines.extend([text, ""])
    return "\n".join(lines).rstrip() + "\n"


def _timestamp(milliseconds: int, format_name: str) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    separator = "," if format_name == "srt" else "."
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{millis:03d}"


def _drop_none(value):
    if isinstance(value, dict):
        return {k: _drop_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_none(v) for v in value]
    return value
