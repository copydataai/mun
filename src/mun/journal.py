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
                stream.write(json.dumps(
                    self.payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
                ).encode("utf-8") + b"\n")
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


def resume_batch_journal(
    path: Path,
    runtime_loader: Callable[[Any, str], Any] | None = None,
) -> dict[str, Any]:
    from .core import SourceMedia, TranscriptionOptions, transcribe_source, write_result_outputs
    from .models import InstalledModel
    from .runtime import create_transformers_runtime
    from .transcript import TranscriptResult

    journal = OperationJournal.load(path)
    for row, outcome in zip(journal.payload["sources"], journal.classify()):
        if outcome["classification"] == "verified-complete":
            continue
        if outcome["classification"] == "indeterminate":
            raise JournalError("Journal binding is indeterminate")

        binding = deepcopy(row["binding"])
        if binding.get("resumable") is not True:
            raise JournalError("Journal source lacks an exact resumable binding")
        source = Path(str(binding.get("source_path", "")))
        if not source.is_file() or _sha256_path(source) != binding.get("source_sha256"):
            raise JournalError("Source media no longer matches the journal binding")
        try:
            model = InstalledModel(**binding["model"])
            options = TranscriptionOptions(**binding["options"])
            formats = list(binding["formats"])
            base = Path(binding["base"])
            expected_runtime = binding["runtime"]
        except (KeyError, TypeError, ValueError) as exc:
            raise JournalError("Journal binding is malformed") from exc

        loader = runtime_loader or (lambda installed, device: create_transformers_runtime(installed, device))
        runtime = loader(model, options.device)
        observed_runtime = {
            "name": runtime.info.name,
            "version": runtime.info.version,
            "requested_device": runtime.info.requested_device,
            "effective_device": runtime.info.effective_device,
            "precision": runtime.info.precision,
            "model_artifact_sha256": getattr(runtime, "model_artifact_sha256", None),
        }
        if observed_runtime != expected_runtime:
            raise JournalError("Runtime and model artifact no longer match the journal binding")

        evidence = outcome.get("evidence", {})
        result_payload = evidence.get("result")
        if outcome["state"] in {"prepared", "inference_started"} or not isinstance(result_payload, Mapping):
            journal.transition(row["source_sha256"], "inference_started")
            result = transcribe_source(
                runtime,
                SourceMedia(source, Path(binding["relative_path"])),
                model,
                options,
            )
            journal.transition(
                row["source_sha256"],
                "inference_completed",
                evidence={"result": result.to_dict(), "result_digest": result.result_digest},
            )
            canonical_result = result
        else:
            canonical_result = TranscriptResult.from_json(json.dumps(result_payload))
            result = canonical_result

        if result.status != "completed":
            journal.transition(row["source_sha256"], "failed", evidence={"result": result.to_dict()})
            continue

        committed = evidence.get("committed", [])
        artifacts = evidence.get("artifacts", [])
        if outcome["state"] == "partial_commit" and (
            not isinstance(committed, list)
            or not committed
            or not isinstance(artifacts, list)
        ):
            raise JournalError("Partial commit evidence cannot verify committed projections")
        artifact_digests = {
            artifact["path"]: artifact["sha256"]
            for artifact in artifacts
            if isinstance(artifact, Mapping)
            and isinstance(artifact.get("path"), str)
            and isinstance(artifact.get("sha256"), str)
        }
        resumable = {
            committed_path: artifact_digests[committed_path]
            for committed_path in committed
            if isinstance(committed_path, str) and committed_path in artifact_digests
        }
        if outcome["state"] == "partial_commit" and len(resumable) != len(committed):
            raise JournalError("Partial commit evidence cannot verify committed projections")

        def transition(state: str, update: dict[str, Any]) -> None:
            journal.transition(row["source_sha256"], state, evidence=update)

        try:
            write_result_outputs(
                base,
                formats,
                result,
                options.translate,
                overwrite=bool(binding.get("overwrite")),
                transition=transition,
                resume_artifacts=resumable,
                canonical_result=canonical_result,
            )
        except MunError as exc:
            raise JournalError("Resumed projections conflict with the journal-bound result") from exc
        journal.transition(row["source_sha256"], "committed", evidence={"verified": True})

    return {"schema_version": 1, "journal": path.name, "sources": journal.classify(), "idempotent": True}


def _sha256_path(path: Path) -> str | None:
    try:
        digest = sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None
