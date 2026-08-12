from __future__ import annotations

import json
import os
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping

from .artifacts import canonical_json_bytes
from .errors import MunError


class JournalError(MunError):
    pass


class OperationJournal:
    def __init__(self, path: Path, payload: dict[str, Any]) -> None:
        self.path = path
        self.payload = payload

    @classmethod
    def create(cls, path: Path, bindings: list[Mapping[str, Any]]) -> OperationJournal:
        if path.exists():
            raise JournalError("Refusing to overwrite an operation journal")
        rows = []
        seen = set()
        for binding in bindings:
            source = binding.get("source_sha256")
            if not isinstance(source, str) or len(source) != 64 or source in seen:
                raise JournalError("Journal sources require unique SHA-256 identities")
            seen.add(source)
            bound = deepcopy(dict(binding))
            rows.append({
                "source_sha256": source,
                "binding": bound,
                "binding_digest": sha256(canonical_json_bytes(bound)).hexdigest(),
                "state": "prepared",
                "evidence": {},
            })
        journal = cls(path, {"schema_version": 1, "journal_kind": "mun-transcription-operation", "sources": rows})
        journal._persist()
        return journal

    @classmethod
    def load(cls, path: Path) -> OperationJournal:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise JournalError("Cannot read operation journal") from exc
        if payload.get("schema_version") != 1 or not isinstance(payload.get("sources"), list):
            raise JournalError("Unsupported or malformed operation journal")
        return cls(path, payload)

    def transition(self, source_sha256: str, state: str, *, evidence: Mapping[str, Any] | None = None) -> None:
        allowed = {
            "prepared", "inference_started", "inference_completed", "render_staged",
            "committed", "partial_commit", "failed", "indeterminate",
        }
        if state not in allowed:
            raise JournalError("Unknown journal transition")
        row = next((item for item in self.payload["sources"] if item.get("source_sha256") == source_sha256), None)
        if row is None:
            raise JournalError("Journal transition claims an absent source")
        row["state"] = state
        row["evidence"] = deepcopy(dict(evidence or {}))
        self._persist()

    def classify(self) -> list[dict[str, Any]]:
        outcomes = []
        for row in self.payload["sources"]:
            actual = sha256(canonical_json_bytes(row.get("binding", {}))).hexdigest()
            state = row.get("state")
            evidence = deepcopy(row.get("evidence", {}))
            if actual != row.get("binding_digest"):
                classification = "indeterminate"
            elif state == "committed" and evidence.get("verified") is True:
                classification = "verified-complete"
            elif state in {"prepared", "inference_started", "inference_completed"}:
                classification = "safely-resumable"
            elif state in {"render_staged", "failed"}:
                classification = "must-recompute"
            elif state == "partial_commit":
                classification = "conflict"
            else:
                classification = "indeterminate"
            outcomes.append({
                "source_sha256": row.get("source_sha256"),
                "classification": classification,
                "state": state,
                "binding_digest": row.get("binding_digest"),
                "evidence": evidence,
            })
        return outcomes

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as stream:
                stream.write(canonical_json_bytes(self.payload) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)


def resume_journal(
    path: Path,
    runner: Callable[[Mapping[str, Any], str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    journal = OperationJournal.load(path)
    if runner is not None:
        for row, outcome in zip(journal.payload["sources"], journal.classify()):
            if outcome["classification"] in {"safely-resumable", "must-recompute"}:
                result = runner(deepcopy(row["binding"]), outcome["classification"])
                journal.transition(row["source_sha256"], str(result["state"]), evidence=result.get("evidence", {}))
    return {"schema_version": 1, "journal": path.name, "sources": journal.classify(), "idempotent": True}
