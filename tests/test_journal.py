from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

from mun.core import Segment, SourceMedia, Transcript, TranscriptionOptions, run_batch
from mun.journal import JournalError, OperationJournal, resume_batch_journal, resume_journal
from mun.models import InstalledModel
from mun.runtime import FakeSpeechRuntime


class JournalTests(unittest.TestCase):
    def _partial_commit(self, root: Path):
        class ProcessDeath(BaseException):
            pass

        model = InstalledModel("example/model", "abc123", "/models/example", "2026-08-09T00:00:00Z")
        source = root / "source.wav"
        source.write_bytes(b"journal source")
        runtime = FakeSpeechRuntime(Transcript("Recovered", [Segment("Recovered", 0.0, 1.0)], "en"))
        runtime.model_artifact_sha256 = "a" * 64
        runtime.test_installed_model = model

        def kill_at_partial_commit(name: str) -> None:
            if name == "partial_commit":
                raise ProcessDeath

        with self.assertRaises(ProcessDeath):
            run_batch(
                [SourceMedia(source, Path("source.wav"))], model, root / "out", ["json", "txt"],
                TranscriptionOptions(), False, lambda _: None, runtime=runtime,
                fault_injector=kill_at_partial_commit,
            )
        return root / "out" / "mun-batch.journal.json", runtime

    def test_partial_commit_resume_preserves_verified_projection_writes_missing_remainder_and_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            journal_path, runtime = self._partial_commit(root)
            json_path = root / "out" / "source.json"
            txt_path = root / "out" / "source.txt"
            committed = json_path.read_bytes()

            first = resume_batch_journal(journal_path, runtime_loader=lambda _model, _device: runtime)
            second = resume_batch_journal(journal_path, runtime_loader=lambda _model, _device: runtime)

            self.assertEqual(json_path.read_bytes(), committed)
            self.assertEqual(txt_path.read_text(encoding="utf-8"), "Recovered\n")
            self.assertEqual(first, second)
            self.assertEqual(first["sources"][0]["classification"], "verified-complete")

    @unittest.skipUnless(Path("/var").resolve() == Path("/private/var"), "requires the macOS /var alias")
    def test_partial_commit_resume_matches_var_and_private_var_paths(self) -> None:
        with tempfile.TemporaryDirectory(dir="/var/tmp") as temporary:
            alias_root = Path(temporary)
            journal_path, runtime = self._partial_commit(alias_root)
            payload = json.loads(journal_path.read_text(encoding="utf-8"))
            evidence = payload["sources"][0]["evidence"]

            self.assertTrue(evidence["committed"][0].startswith("/private/var/"))
            resumed = resume_batch_journal(journal_path, runtime_loader=lambda _model, _device: runtime)

            self.assertEqual(resumed["sources"][0]["classification"], "verified-complete")

    def test_partial_commit_resume_normalizes_legacy_symlink_alias_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            container = Path(temporary)
            real_root = container / "real"
            real_root.mkdir()
            alias_root = container / "alias"
            os.symlink(real_root, alias_root)
            journal_path, runtime = self._partial_commit(real_root)
            payload = json.loads(journal_path.read_text(encoding="utf-8"))
            evidence = payload["sources"][0]["evidence"]
            real_prefix = str(real_root.resolve())
            alias_prefix = str(alias_root)
            for artifact in evidence["artifacts"]:
                artifact["path"] = artifact["path"].replace(real_prefix, alias_prefix, 1)
            evidence["committed"] = [path.replace(real_prefix, alias_prefix, 1) for path in evidence["committed"]]
            journal_path.write_text(json.dumps(payload), encoding="utf-8")

            resumed = resume_batch_journal(journal_path, runtime_loader=lambda _model, _device: runtime)

            self.assertEqual(resumed["sources"][0]["classification"], "verified-complete")
            self.assertEqual((real_root / "out" / "source.txt").read_text(encoding="utf-8"), "Recovered\n")

    def test_partial_commit_resume_rejects_conflicting_bytes_through_legacy_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            container = Path(temporary)
            real_root = container / "real"
            real_root.mkdir()
            alias_root = container / "alias"
            os.symlink(real_root, alias_root)
            journal_path, runtime = self._partial_commit(real_root)
            payload = json.loads(journal_path.read_text(encoding="utf-8"))
            evidence = payload["sources"][0]["evidence"]
            real_prefix = str(real_root.resolve())
            alias_prefix = str(alias_root)
            for artifact in evidence["artifacts"]:
                artifact["path"] = artifact["path"].replace(real_prefix, alias_prefix, 1)
            evidence["committed"] = [path.replace(real_prefix, alias_prefix, 1) for path in evidence["committed"]]
            journal_path.write_text(json.dumps(payload), encoding="utf-8")
            json_path = real_root / "out" / "source.json"
            json_path.write_bytes(b'{"conflict":true}\n')

            with self.assertRaises(JournalError):
                resume_batch_journal(journal_path, runtime_loader=lambda _model, _device: runtime)

            self.assertFalse((real_root / "out" / "source.txt").exists())

    def test_partial_commit_resume_rejects_output_symlink_even_when_target_bytes_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            journal_path, runtime = self._partial_commit(root)
            json_path = root / "out" / "source.json"
            outside = root / "outside.json"
            outside.write_bytes(json_path.read_bytes())
            json_path.unlink()
            os.symlink(outside, json_path)

            with self.assertRaises(JournalError):
                resume_batch_journal(journal_path, runtime_loader=lambda _model, _device: runtime)

            self.assertTrue(json_path.is_symlink())
            self.assertFalse((root / "out" / "source.txt").exists())

    def test_partial_commit_resume_verifies_json_from_journal_payload_despite_restored_runtime_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            journal_path, runtime = self._partial_commit(root)
            json_path = root / "out" / "source.json"
            committed = json_path.read_bytes()

            from mun import core

            write_outputs = core.write_result_outputs

            def write_after_time_and_identity_drift(*args, **kwargs):
                result = args[2]
                provenance = replace(
                    result.provenance,
                    created_at="2099-12-31T23:59:59Z",
                    mun_version="different-process-version",
                )
                drifted = replace(result, provenance=provenance)
                return write_outputs(*args[:2], drifted, *args[3:], **kwargs)

            with patch("mun.core.write_result_outputs", side_effect=write_after_time_and_identity_drift):
                resumed = resume_batch_journal(journal_path, runtime_loader=lambda _model, _device: runtime)

            self.assertEqual(json_path.read_bytes(), committed)
            self.assertEqual((root / "out" / "source.txt").read_text(encoding="utf-8"), "Recovered\n")
            self.assertEqual(resumed["sources"][0]["classification"], "verified-complete")

    def test_partial_commit_resume_rejects_conflicting_committed_projection_without_writing_remainder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            journal_path, runtime = self._partial_commit(root)
            json_path = root / "out" / "source.json"
            txt_path = root / "out" / "source.txt"
            conflicting = b'{"unrelated": true}\n'
            json_path.write_bytes(conflicting)
            payload = json.loads(journal_path.read_text(encoding="utf-8"))
            payload["sources"][0]["evidence"]["artifacts"][0]["sha256"] = sha256(conflicting).hexdigest()
            journal_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(JournalError):
                resume_batch_journal(journal_path, runtime_loader=lambda _model, _device: runtime)

            self.assertEqual(json_path.read_bytes(), conflicting)
            self.assertFalse(txt_path.exists())

    def test_partial_commit_resume_rejects_unverifiable_committed_projection_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            journal_path, runtime = self._partial_commit(root)
            payload = json.loads(journal_path.read_text(encoding="utf-8"))
            payload["sources"][0]["evidence"].pop("artifacts")
            journal_path.write_text(json.dumps(payload), encoding="utf-8")
            json_path = root / "out" / "source.json"
            committed = json_path.read_bytes()

            with self.assertRaises(JournalError):
                resume_batch_journal(journal_path, runtime_loader=lambda _model, _device: runtime)

            self.assertEqual(json_path.read_bytes(), committed)
            self.assertFalse((root / "out" / "source.txt").exists())

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
