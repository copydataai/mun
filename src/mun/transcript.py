from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from .artifacts import machine_result_digest, validate_machine_result

SCHEMA_VERSION = 1
Status = Literal["completed", "partial", "failed", "cancelled"]
ExportReceiptState = Literal["completed", "cancelled", "failed_before_commit", "partial_commit"]
ReuseStatus = Literal["reused_verified", "conflict", "incomplete_output_set", "overwrite_required", "queued"]
TranscriptKind = Literal["original", "english_translation"]
LanguageSource = Literal["detected", "forced", "model", "unknown"]
MediaTrust = Literal["untrusted_bytes"]
ModelTrust = Literal["verified_artifact", "unsafe_remote_code"]
ContentTrust = Literal["untrusted_model_output", "untrusted_content"]
AgentEligibilityStatus = Literal["ineligible"]


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
class ExportArtifact:
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ExportReceipt:
    schema_version: int
    state: ExportReceiptState
    source: str
    result_digest: str | None
    artifacts: list[ExportArtifact]
    committed_paths: list[str]
    uncommitted_paths: list[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n"


@dataclass(frozen=True)
class ModelProvenance:
    repository: str
    revision: str | None
    artifact_sha256: str | None = None


@dataclass(frozen=True)
class RuntimeProvenance:
    name: str
    version: str | None
    environment: RuntimeEnvironment | None = None


@dataclass(frozen=True)
class RuntimeEnvironment:
    python_version: str
    python_implementation: str
    operating_system: str
    machine: str


@dataclass(frozen=True)
class ConverterIdentity:
    name: str
    version: str | None


@dataclass(frozen=True)
class PreparedMediaRecord:
    used: bool
    sha256: str | None
    media_format: str | None
    sample_rate_hz: int | None
    channels: int | None
    converter: ConverterIdentity | None


@dataclass(frozen=True)
class OperationParameters:
    language: str | None
    timestamps: bool
    translate: bool
    chunk_length: int
    stride_length: int
    requested_device: str
    effective_device: str
    precision: str | None


@dataclass(frozen=True)
class OperationRecord:
    parameters: OperationParameters
    prepared_media: PreparedMediaRecord
    source_hash_policy: str


@dataclass(frozen=True)
class ReviewMetadata:
    state: Literal["reviewed", "unreviewed"]
    correction_set_id: str
    correction_set_digest: str


@dataclass(frozen=True)
class TrustRecord:
    media: MediaTrust = "untrusted_bytes"
    model: ModelTrust = "verified_artifact"
    content: ContentTrust = "untrusted_model_output"
    review: ReviewMetadata | None = None


@dataclass(frozen=True)
class AgentEligibility:
    status: AgentEligibilityStatus = "ineligible"
    reason: str = "Transcript content requires human judgment before agent use."


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
    operation: OperationRecord | None = None
    overlap_ms: int = 0
    result_digest: str | None = None
    reuse_status: ReuseStatus = "queued"
    trust: TrustRecord = TrustRecord()
    agent_eligibility: AgentEligibility = AgentEligibility()

    def __post_init__(self) -> None:
        if self.trust.media != "untrusted_bytes" or self.trust.content not in {
            "untrusted_model_output", "untrusted_content"
        }:
            raise ValueError("Transcript trust cannot remove source or content taint")
        if self.agent_eligibility.status != "ineligible":
            raise ValueError("Transcript results are not eligible for autonomous agent use")
        if self.result_digest is None:
            object.__setattr__(self, "result_digest", machine_result_digest(self.to_dict()))

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload.pop("reuse_status")
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"

    @classmethod
    def from_json(cls, value: str) -> TranscriptResult:
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise TypeError("Transcript result JSON must contain an object")
        validate_machine_result(payload)
        result = _transcript_result_from_dict(payload)
        if "result_digest" not in payload:
            object.__setattr__(result, "result_digest", None)
        return result


@dataclass(frozen=True)
class BatchResult:
    schema_version: int
    files: list[TranscriptResult]
    counts: dict[str, int]

    def to_dict(self) -> dict:
        return {"schema_version": self.schema_version, "files": [f.to_dict() for f in self.files], "counts": self.counts}


def make_batch_result(files: list[TranscriptResult]) -> BatchResult:
    counts = {status: 0 for status in ("completed", "partial", "failed", "cancelled")}
    counts.update({status: 0 for status in ("processed", "reused_verified", "conflict", "incomplete_output_set", "overwrite_required")})
    for result in files:
        counts[result.status] = counts.get(result.status, 0) + 1
        if result.reuse_status == "reused_verified":
            counts["reused_verified"] += 1
        elif result.reuse_status in {"conflict", "incomplete_output_set"}:
            counts[result.reuse_status] += 1
        else:
            counts["processed"] += 1
            if result.reuse_status == "overwrite_required":
                counts["overwrite_required"] += 1
    return BatchResult(SCHEMA_VERSION, files, counts)


def make_provenance(*, mun_version: str, model_id: str, revision: str | None, artifact_sha256: str | None = None, runtime_name: str, runtime_version: str | None, requested_device: str, effective_device: str, precision: str | None) -> Provenance:
    return Provenance(
        mun_version=mun_version,
        created_at=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        model=ModelProvenance(model_id, revision, artifact_sha256),
        runtime=RuntimeProvenance(
            runtime_name,
            runtime_version,
            RuntimeEnvironment(
                python_version=platform.python_version(),
                python_implementation=platform.python_implementation(),
                operating_system=sys.platform,
                machine=platform.machine(),
            ),
        ),
        requested_device=requested_device,
        effective_device=effective_device,
        precision=precision,
    )


def make_trust(trust_remote_code: bool = False) -> TrustRecord:
    return TrustRecord(model="unsafe_remote_code" if trust_remote_code else "verified_artifact")


def _transcript_result_from_dict(payload: dict[str, Any]) -> TranscriptResult:
    source = payload["source"]
    provenance = payload["provenance"]
    model = provenance["model"]
    runtime = provenance["runtime"]
    environment = runtime.get("environment")
    operation = payload.get("operation")
    trust = payload.get("trust") or {}
    review = trust.get("review")
    eligibility = payload.get("agent_eligibility") or {}
    return TranscriptResult(
        schema_version=payload["schema_version"],
        status=payload["status"],
        source=SourceRecord(
            source["name"], source["relative_path"], source.get("duration_ms"), source.get("sha256")
        ),
        transcripts=[_transcript_variant_from_dict(item) for item in payload["transcripts"]],
        speakers=[Speaker(**item) for item in payload["speakers"]],
        diagnostics=[Diagnostic(**item) for item in payload["diagnostics"]],
        provenance=Provenance(
            mun_version=provenance["mun_version"],
            created_at=provenance["created_at"],
            model=ModelProvenance(model["repository"], model.get("revision"), model.get("artifact_sha256")),
            runtime=RuntimeProvenance(
                runtime["name"],
                runtime.get("version"),
                RuntimeEnvironment(**environment) if environment is not None else None,
            ),
            requested_device=provenance["requested_device"],
            effective_device=provenance["effective_device"],
            precision=provenance.get("precision"),
        ),
        operation=_operation_from_dict(operation) if operation is not None else None,
        overlap_ms=payload.get("overlap_ms", 0),
        result_digest=payload.get("result_digest"),
        trust=TrustRecord(
            media=trust.get("media", "untrusted_bytes"),
            model=trust.get("model", "verified_artifact"),
            content=trust.get("content", "untrusted_model_output"),
            review=ReviewMetadata(**review) if review is not None else None,
        ),
        agent_eligibility=AgentEligibility(
            status=eligibility.get("status", "ineligible"),
            reason=eligibility.get("reason", "Transcript content requires human judgment before agent use."),
        ),
    )


def _transcript_variant_from_dict(payload: dict[str, Any]) -> TranscriptVariant:
    language = payload["language"]
    return TranscriptVariant(
        kind=payload["kind"],
        language=Language(language.get("tag"), language.get("source", "unknown"), language.get("confidence")),
        text=payload["text"],
        segments=[
            TranscriptSegment(
                id=segment["id"],
                start_ms=segment.get("start_ms"),
                end_ms=segment.get("end_ms"),
                text=segment["text"],
                speaker_id=segment.get("speaker_id"),
                words=[Word(**word) for word in segment["words"]] if segment.get("words") is not None else None,
            )
            for segment in payload["segments"]
        ],
    )


def _operation_from_dict(payload: dict[str, Any]) -> OperationRecord:
    prepared = payload["prepared_media"]
    converter = prepared.get("converter")
    return OperationRecord(
        parameters=OperationParameters(**payload["parameters"]),
        prepared_media=PreparedMediaRecord(
            used=prepared["used"],
            sha256=prepared.get("sha256"),
            media_format=prepared.get("media_format"),
            sample_rate_hz=prepared.get("sample_rate_hz"),
            channels=prepared.get("channels"),
            converter=ConverterIdentity(**converter) if converter is not None else None,
        ),
        source_hash_policy=payload["source_hash_policy"],
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
