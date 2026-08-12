from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping

from .artifacts import canonical_json_bytes
from .errors import MunError
from .transcript import TranscriptResult


class AcceptanceError(MunError):
    pass


def create_acceptance(
    machine: TranscriptResult,
    overlay: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    projections: Mapping[str, bytes] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if machine.source.sha256 is None:
        raise AcceptanceError("Machine result lacks an immutable source digest")
    if overlay.get("schema_version") != 1:
        raise AcceptanceError("Unsupported acceptance-overlay schema version")
    if overlay.get("machine_result_digest") != machine.result_digest or overlay.get("source_sha256") != machine.source.sha256:
        raise AcceptanceError("Acceptance overlay is stale for this machine result")

    indexed: dict[tuple[str, str], Any] = {}
    ordered: list[tuple[str, Any]] = []
    for variant in machine.transcripts:
        previous_end = -1
        for segment in variant.segments:
            if segment.start_ms is None or segment.end_ms is None or segment.start_ms < 0 or segment.end_ms <= segment.start_ms:
                raise AcceptanceError("Every accepted segment requires a valid bounded media interval")
            if segment.start_ms < previous_end:
                raise AcceptanceError("Machine-result segment intervals overlap")
            previous_end = segment.end_ms
            key = (variant.kind, segment.id)
            if key in indexed:
                raise AcceptanceError("Machine result has duplicate segment identities")
            indexed[key] = segment
            ordered.append((variant.kind, segment))

    decisions: dict[tuple[str, str], Mapping[str, Any]] = {}
    for decision in overlay.get("decisions", []):
        if not isinstance(decision, Mapping):
            raise AcceptanceError("Invalid segment decision")
        key = (str(decision.get("transcript_kind")), str(decision.get("segment_id")))
        if key in decisions:
            raise AcceptanceError("Acceptance overlay contains a duplicate decision")
        if key not in indexed:
            raise AcceptanceError("Acceptance overlay claims an absent segment")
        disposition = decision.get("disposition")
        if disposition not in {"accepted", "corrected", "exception", "unreviewed"}:
            raise AcceptanceError("Unknown segment disposition")
        if disposition in {"corrected", "exception"} and not str(decision.get("reason", "")).strip():
            raise AcceptanceError(f"{disposition} decisions require a reason")
        if disposition == "corrected" and not isinstance(decision.get("replacement"), str):
            raise AcceptanceError("Corrected decisions require replacement text")
        decisions[key] = decision

    waivers: dict[tuple[str | None, str], Mapping[str, Any]] = {}
    for waiver in policy.get("waivers", []):
        if not isinstance(waiver, Mapping) or not str(waiver.get("reason", "")).strip():
            raise AcceptanceError("Every waiver requires a bounded segment and reason")
        key = (str(waiver["transcript_kind"]) if waiver.get("transcript_kind") else None, str(waiver.get("segment_id")))
        if key in waivers:
            raise AcceptanceError("Duplicate waiver")
        waivers[key] = waiver

    segment_rows = []
    blocked = []
    used_waivers = []
    for kind, segment in ordered:
        key = (kind, segment.id)
        decision = decisions.get(key, {"disposition": "unreviewed"})
        disposition = str(decision["disposition"])
        waiver = waivers.get(key) or waivers.get((None, segment.id))
        if disposition in {"unreviewed", "exception"}:
            if waiver is None:
                blocked.append((disposition, segment.id))
            else:
                used_waivers.append({"transcript_kind": kind, "segment_id": segment.id, "reason": waiver["reason"]})
        segment_rows.append({
            "transcript_kind": kind,
            "segment_id": segment.id,
            "source_sha256": machine.source.sha256,
            "interval": {"start_ms": segment.start_ms, "end_ms": segment.end_ms},
            "machine_text_sha256": sha256(segment.text.encode("utf-8")).hexdigest(),
            "disposition": disposition,
            "replacement": decision.get("replacement") if disposition == "corrected" else None,
            "reason": decision.get("reason"),
        })
    if blocked:
        state, segment_id = blocked[0]
        phrase = "unresolved exception" if state == "exception" else "unreviewed segment"
        raise AcceptanceError(f"Final acceptance blocked by {phrase}: {segment_id}")

    projection_rows = [
        {"name": name, "sha256": sha256(value).hexdigest(), "size_bytes": len(value)}
        for name, value in sorted((projections or {}).items())
    ]
    artifact = {
        "schema_version": 1,
        "artifact_kind": "mun-source-grounded-transcript-acceptance",
        "status": "accepted_with_waiver" if used_waivers else "accepted",
        "source_sha256": machine.source.sha256,
        "machine_result_digest": machine.result_digest,
        "overlay_sha256": sha256(canonical_json_bytes(overlay)).hexdigest(),
        "policy": dict(policy),
        "segments": segment_rows,
        "waivers": used_waivers,
        "trust": machine.to_dict()["trust"],
        "agent_eligibility": machine.to_dict()["agent_eligibility"],
        "claims": {"truth": False, "consent": False, "custody": False, "reviewer_identity": False},
        "projections": projection_rows,
    }
    receipt = {
        "schema_version": 1,
        "receipt_kind": "mun-acceptance-receipt",
        "source_sha256": machine.source.sha256,
        "machine_result_digest": machine.result_digest,
        "overlay_sha256": artifact["overlay_sha256"],
        "policy_sha256": sha256(canonical_json_bytes(policy)).hexdigest(),
        "artifact_sha256": sha256(canonical_json_bytes(artifact)).hexdigest(),
        "projections": projection_rows,
    }
    return artifact, receipt
