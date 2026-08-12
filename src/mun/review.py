from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
from typing import Any, Literal

from .artifacts import canonical_json_bytes
from .errors import MunError
from .transcript import ReviewMetadata, TranscriptKind, TranscriptResult, TrustRecord, render_json, render_srt, render_txt, render_vtt

ReviewState = Literal["reviewed", "unreviewed"]
ReviewView = Literal["machine", "corrected"]
_MAX_NOTE_LENGTH = 500


class CorrectionError(MunError):
    pass


@dataclass(frozen=True)
class SegmentCorrection:
    transcript_kind: TranscriptKind
    segment_id: str
    original_text_digest: str
    replacement: str
    note: str | None = None


@dataclass(frozen=True)
class CorrectionSet:
    schema_version: int
    correction_set_id: str
    created_at: str
    parent_result_digest: str
    review_state: ReviewState
    corrections: list[SegmentCorrection]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"

    @property
    def digest(self) -> str:
        return sha256(_correction_json_bytes(self.to_dict())).hexdigest()

    @classmethod
    def from_json(cls, value: str) -> CorrectionSet:
        try:
            payload = json.loads(value)
            if not isinstance(payload, dict):
                raise TypeError
            corrections = [SegmentCorrection(**item) for item in payload["corrections"]]
            result = cls(
                schema_version=payload["schema_version"],
                correction_set_id=payload["correction_set_id"],
                created_at=payload["created_at"],
                parent_result_digest=payload["parent_result_digest"],
                review_state=payload["review_state"],
                corrections=corrections,
            )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise CorrectionError("Invalid correction-set JSON") from exc
        result._validate()
        return result

    def _validate(self) -> None:
        if self.schema_version != 1:
            raise CorrectionError("Unsupported correction-set schema version")
        if not isinstance(self.correction_set_id, str) or not self.correction_set_id:
            raise CorrectionError("Correction-set ID and timestamp are required")
        if not isinstance(self.created_at, str) or not self.created_at:
            raise CorrectionError("Correction-set ID and timestamp are required")
        try:
            timestamp = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise CorrectionError("Correction-set timestamp must be ISO 8601") from exc
        if timestamp.tzinfo is None:
            raise CorrectionError("Correction-set timestamp must include a UTC offset")
        if self.review_state not in {"reviewed", "unreviewed"}:
            raise CorrectionError("Review state must be reviewed or unreviewed")
        if not _is_digest(self.parent_result_digest):
            raise CorrectionError("Parent machine-result digest must be SHA-256")
        targets: set[tuple[str, str]] = set()
        for correction in self.corrections:
            target = (correction.transcript_kind, correction.segment_id)
            if correction.transcript_kind not in {"original", "english_translation"}:
                raise CorrectionError("Unknown transcript kind in correction target")
            if not correction.segment_id or target in targets:
                raise CorrectionError("Correction targets must be unique segments")
            if not _is_digest(correction.original_text_digest):
                raise CorrectionError("Original segment-text digest must be SHA-256")
            if not isinstance(correction.replacement, str):
                raise CorrectionError("Correction replacement must be text")
            if correction.note is not None and not isinstance(correction.note, str):
                raise CorrectionError("Correction note must be text")
            if correction.note is not None and len(correction.note) > _MAX_NOTE_LENGTH:
                raise CorrectionError(f"Correction note must be at most {_MAX_NOTE_LENGTH} characters")
            targets.add(target)


@dataclass(frozen=True)
class CorrectedTranscript:
    parent_result_digest: str
    correction_set_id: str
    correction_set_digest: str
    review_state: ReviewState
    transcript: TranscriptResult

    def identity_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "view": "corrected",
            "review_state": self.review_state,
            "parent_result_digest": self.parent_result_digest,
            "correction_set_id": self.correction_set_id,
            "correction_set_digest": self.correction_set_digest,
            "transcript": self.transcript.to_dict(),
        }

    @property
    def export_digest(self) -> str:
        return sha256(canonical_json_bytes(self.identity_dict())).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity_dict(), "export_digest": self.export_digest}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"


def apply_corrections(machine: TranscriptResult, correction_set: CorrectionSet) -> CorrectedTranscript:
    if machine.result_digest is None or correction_set.parent_result_digest != machine.result_digest:
        raise CorrectionError("Correction set does not match the exact parent machine result")

    payload = machine.to_dict()
    payload.pop("result_digest", None)
    targets = {
        (correction.transcript_kind, correction.segment_id): correction
        for correction in correction_set.corrections
    }
    matched: set[tuple[str, str]] = set()
    for variant in payload["transcripts"]:
        for segment in variant["segments"]:
            target = (variant["kind"], segment["id"])
            correction = targets.get(target)
            if correction is None:
                continue
            actual_digest = sha256(segment["text"].encode("utf-8")).hexdigest()
            if actual_digest != correction.original_text_digest:
                raise CorrectionError(f"Correction target {segment['id']} does not match the original segment text")
            segment["text"] = correction.replacement
            matched.add(target)

    missing = set(targets) - matched
    if missing:
        kind, segment_id = sorted(missing)[0]
        raise CorrectionError(f"Correction target {kind}/{segment_id} does not exist in the parent machine result")
    for variant in payload["transcripts"]:
        if variant["segments"]:
            variant["text"] = " ".join(segment["text"] for segment in variant["segments"])
    payload["trust"] = asdict(TrustRecord(
        media="untrusted_bytes",
        model=machine.trust.model,
        content="untrusted_content",
        review=ReviewMetadata(
            state=correction_set.review_state,
            correction_set_id=correction_set.correction_set_id,
            correction_set_digest=correction_set.digest,
        ),
    ))

    corrected = TranscriptResult.from_json(json.dumps(payload))
    return CorrectedTranscript(
        parent_result_digest=machine.result_digest,
        correction_set_id=correction_set.correction_set_id,
        correction_set_digest=correction_set.digest,
        review_state=correction_set.review_state,
        transcript=corrected,
    )


def render_reviewed(
    format_name: str,
    machine: TranscriptResult,
    corrected: CorrectedTranscript | None,
    view: ReviewView,
) -> str:
    if view == "machine":
        if format_name == "json":
            return json.dumps(
                {
                    "schema_version": 1,
                    "view": "machine",
                    "review_state": "unreviewed",
                    "export_digest": machine.result_digest,
                    "transcript": machine.to_dict(),
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n"
        selected = machine
    elif view == "corrected" and corrected is not None:
        if format_name == "json":
            return corrected.to_json()
        selected = corrected.transcript
    else:
        raise CorrectionError("Corrected rendering requires a valid correction set")
    renderers = {"txt": render_txt, "json": render_json, "srt": render_srt, "vtt": render_vtt}
    try:
        renderer = renderers[format_name]
    except KeyError as exc:
        raise CorrectionError(f"Unknown review render format: {format_name}") from exc
    return renderer(selected)


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _correction_json_bytes(value: Any) -> bytes:
    def normalize(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): normalize(child) for key, child in item.items()}
        if isinstance(item, list):
            return [normalize(child) for child in item]
        if isinstance(item, str):
            return item.replace("\r\n", "\n").replace("\r", "\n")
        return item

    return json.dumps(
        normalize(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
