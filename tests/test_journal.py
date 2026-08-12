from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mun.core import Segment, SourceMedia, Transcript, TranscriptionOptions, run_batch
from mun.journal import JournalError, OperationJournal, resume_batch_journal, resume_journal
from mun.models import InstalledModel
from mun.runtime import FakeSpeechRuntime


class JournalTests(unittest.TestCase):
    def test_public_batch_recovers_from_every_durable_boundary_and_repeated_resume(self) -> None:
        class ProcessDeath(BaseException):
            pass

        model = InstalledModel("example/model", "abc123", "/models/example", "2026-08-09T00:00:00Z")
        boundaries = ("pre_inference", "post_inference_pre_render", "render_staged", "partial_commit")

        for boundary in boundaries:
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / "source.wav"
                source.write_bytes(b"journal source")
                media = SourceMedia(source, Path("source.wav"))
                runtime = FakeSpeechRuntime(Transcript("Recovered", [Segment("Recovered", 0.0, 1.0)], "en"))
                runtime.model_artifact_sha256 = "a" * 64
                runtime.test_installed_model = model
                calls = 0

                def kill_at_boundary(name: str) -> None:
                    nonlocal calls
                    if name == boundary:
                        calls += 1
                        raise ProcessDeath

                with self.assertRaises(ProcessDeath):
                    run_batch(
                        [media], model, root / "out", ["json", "txt"], TranscriptionOptions(),
                        False, lambda _: None, runtime=runtime, fault_injector=kill_at_boundary,
                    )

                journal_path = root / "out" / "mun-batch.journal.json"
                self.assertTrue(journal_path.is_file())
                resumed = resume_batch_journal(journal_path, runtime_loader=lambda _model, _device: runtime)
                repeated = resume_batch_journal(journal_path, runtime_loader=lambda _model, _device: runtime)

                self.assertEqual(calls, 1)
                self.assertEqual(resumed["sources"][0]["classification"], "verified-complete")
                self.assertEqual(repeated, resumed)
                self.assertEqual((root / "out" / "source.txt").read_text(encoding="utf-8"), "Recovered\n")
                self.assertEqual(OperationJournal.load(journal_path).payload["sources"][0]["state"], "committed")

    def test_resume_rejects_source_binding_drift_before_running_inference(self) -> None:
        model = InstalledModel("example/model", "abc123", "/models/example", "2026-08-09T00:00:00Z")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            source.write_bytes(b"original")
            runtime = FakeSpeechRuntime(Transcript("text", [], "en"))
            runtime.model_artifact_sha256 = "a" * 64
            runtime.test_installed_model = model

            with patch.object(runtime, "transcribe", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
                run_batch(
                    [SourceMedia(source, Path("source.wav"))], model, root / "out", ["txt"],
                    TranscriptionOptions(), False, lambda _: None, runtime=runtime,
                )
            source.write_bytes(b"changed")

            with self.assertRaises(JournalError):
                resume_batch_journal(root / "out" / "mun-batch.journal.json", runtime_loader=lambda _model, _device: runtime)
    def test_fault_boundaries_classify_fail_closed_and_resume_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "batch.journal.json"
            binding = {
                "source_sha256": "a" * 64,
                "tuple": {"model": "m", "runtime": "r", "artifact": "b" * 64},
                "options": {"timestamps": True},
                "projections": ["json", "txt"],
                "destination": {"root": "transcripts", "overwrite": False},
            }
            journal = OperationJournal.create(path, [binding])
            self.assertEqual(journal.classify()[0]["classification"], "safely-resumable")
            journal.transition("a" * 64, "inference_completed", evidence={"result_digest": "c" * 64})
            self.assertEqual(journal.classify()[0]["classification"], "safely-resumable")
            journal.transition("a" * 64, "render_staged", evidence={"staged": ["x.json", "x.txt"]})
            self.assertEqual(journal.classify()[0]["classification"], "must-recompute")
            journal.transition("a" * 64, "partial_commit", evidence={"committed": ["x.json"], "uncommitted": ["x.txt"]})
            self.assertEqual(journal.classify()[0]["classification"], "conflict")

            first = resume_journal(path)
            second = resume_journal(path)
            self.assertEqual(first, second)
            self.assertEqual(first["sources"][0]["evidence"]["committed"], ["x.json"])

    def test_binding_change_is_indeterminate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "batch.journal.json"
            journal = OperationJournal.create(path, [{"source_sha256": "a" * 64, "tuple": {}, "options": {}, "projections": [], "destination": {}}])
            payload = json.loads(path.read_text())
            payload["sources"][0]["binding_digest"] = "0" * 64
            path.write_text(json.dumps(payload))
            self.assertEqual(OperationJournal.load(path).classify()[0]["classification"], "indeterminate")


if __name__ == "__main__":
    unittest.main()
