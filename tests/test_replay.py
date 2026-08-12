from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from mun.cli import main
from mun.core import Segment, SourceMedia, Transcript, TranscriptionOptions, run_transcription_workflow
from mun.models import InstalledModel
from mun.replay import ReplayOutcome, replay_result
from mun.runtime import FakeSpeechRuntime, create_transformers_runtime


class CountingRuntime(FakeSpeechRuntime):
    def __init__(self, transcript: Transcript) -> None:
        super().__init__(transcript)
        self.calls = 0

    def transcribe(self, source: Path, options: TranscriptionOptions):
        self.calls += 1
        return super().transcribe(source, options)


class ReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = InstalledModel(
            "example/model", "abc123", "/models/example", "2026-08-12T00:00:00+00:00",
        )
        self.transcript = Transcript("Hello world", [Segment("Hello world", 0.0, 1.25)], "en")

    def _fixture(self, root: Path):
        source = root / "source.wav"
        source.write_bytes(b"bounded replay source")
        runtime = FakeSpeechRuntime(self.transcript)
        runtime.model_artifact_sha256 = "artifact-digest"
        expected = run_transcription_workflow(
            [SourceMedia(source, Path("source.wav"))], self.model, TranscriptionOptions(timestamps=True), runtime=runtime
        )[0]
        result_path = root / "result.json"
        result_path.write_text(expected.to_json(), encoding="utf-8")
        return source, result_path, expected

    def test_fake_runtime_replay_is_exact_for_canonical_and_export_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, result_path, _ = self._fixture(Path(temporary))
            runtime = FakeSpeechRuntime(self.transcript)
            runtime.model_artifact_sha256 = "artifact-digest"

            outcome = replay_result(result_path, source=source, model=self.model, runtime=runtime)

        self.assertEqual(outcome.kind, "exact_match")
        self.assertTrue(outcome.canonical_bytes_match)
        self.assertEqual(outcome.projection_matches, {"json": True, "srt": True, "txt": True, "vtt": True})
        self.assertIsNone(outcome.tolerance)

    def test_source_mismatch_is_reported_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, result_path, _ = self._fixture(Path(temporary))
            source.write_bytes(b"changed source")
            runtime = CountingRuntime(self.transcript)
            runtime.model_artifact_sha256 = "artifact-digest"

            outcome = replay_result(result_path, source=source, model=self.model, runtime=runtime)

        self.assertEqual(outcome.kind, "artifact_unavailable")
        self.assertEqual(outcome.reason, "source_digest_mismatch")
        self.assertEqual(runtime.calls, 0)

    def test_model_or_runtime_tuple_change_cannot_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, result_path, _ = self._fixture(Path(temporary))
            changed_model = replace(self.model, revision="different")
            runtime = CountingRuntime(self.transcript)
            runtime.model_artifact_sha256 = "artifact-digest"

            model_outcome = replay_result(
                result_path,
                source=source,
                model=changed_model,
                runtime=runtime,
                tolerance_file=Path("fixtures/replay/live-tolerances.json"),
            )
            runtime.info = replace(runtime.info, version="different")
            runtime_outcome = replay_result(result_path, source=source, model=self.model, runtime=runtime)

        self.assertEqual(model_outcome.kind, "environment_mismatch")
        self.assertEqual(runtime_outcome.kind, "environment_mismatch")
        self.assertEqual(runtime.calls, 0)

    def test_exact_normalized_diff_reports_semantic_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, result_path, _ = self._fixture(Path(temporary))
            runtime = FakeSpeechRuntime(Transcript("Goodbye world", [Segment("Goodbye world", 0.0, 1.25)], "en"))
            runtime.model_artifact_sha256 = "artifact-digest"

            outcome = replay_result(result_path, source=source, model=self.model, runtime=runtime)

        self.assertEqual(outcome.kind, "semantic_drift")
        self.assertIn("-Hello world", outcome.normalized_diff)
        self.assertIn("+Goodbye world", outcome.normalized_diff)

    def test_live_tolerance_is_explicit_and_allows_only_text_and_timing_bounds(self) -> None:
        tolerances = Path("fixtures/replay/live-tolerances.json")
        with tempfile.TemporaryDirectory() as temporary:
            source, result_path, _ = self._fixture(Path(temporary))
            runtime = FakeSpeechRuntime(Transcript(" hello   world ", [Segment(" hello world ", 0.04, 1.29)], "en"))
            runtime.model_artifact_sha256 = "artifact-digest"

            outcome = replay_result(
                result_path, source=source, model=self.model, runtime=runtime, tolerance_file=tolerances
            )

        self.assertEqual(outcome.kind, "within_tolerance")
        self.assertEqual(outcome.tolerance["timestamp_ms"], 50)
        self.assertEqual(outcome.tolerance_file, str(tolerances))
        self.assertFalse(outcome.canonical_bytes_match)

    def test_matching_exports_with_nonsemantic_record_drift_is_projection_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, result_path, expected = self._fixture(Path(temporary))
            payload = expected.to_dict()
            payload["provenance"]["mun_version"] = "older-mun"
            payload.pop("result_digest")
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            runtime = FakeSpeechRuntime(self.transcript)
            runtime.model_artifact_sha256 = "artifact-digest"

            outcome = replay_result(result_path, source=source, model=self.model, runtime=runtime)

        self.assertEqual(outcome.kind, "projection_match")
        self.assertFalse(outcome.projection_matches["json"])
        self.assertTrue(all(outcome.projection_matches[name] for name in ("srt", "txt", "vtt")))

    def test_cli_prints_typed_json_outcome(self) -> None:
        outcome = ReplayOutcome(kind="unsupported_replay", reason="missing_operation")
        output = io.StringIO()
        with patch("mun.cli.replay_result", return_value=outcome), contextlib.redirect_stdout(output):
            status = main(["replay", "result.json"])

        self.assertEqual(status, 2)
        self.assertEqual(json.loads(output.getvalue())["outcome"], "unsupported_replay")

    def test_live_qualification_is_opt_in_and_skips_without_pinned_model(self) -> None:
        configured = os.environ.get("MUN_REPLAY_LIVE_MODEL")
        if not configured:
            self.skipTest("MUN_REPLAY_LIVE_MODEL is not set to a pinned installed model")
        model_path = Path(configured)
        metadata_path = model_path / "mun-model.json"
        if not metadata_path.is_file():
            self.skipTest("MUN_REPLAY_LIVE_MODEL has no pinned Mun model metadata")
        result_path = Path(os.environ.get("MUN_REPLAY_LIVE_RESULT", "fixtures/replay/live/result.json"))
        source_path = Path(os.environ.get("MUN_REPLAY_LIVE_SOURCE", "fixtures/replay/live/source.wav"))
        self.assertTrue(result_path.is_file(), f"missing live replay result: {result_path}")
        self.assertTrue(source_path.is_file(), f"missing live replay source: {source_path}")
        model = InstalledModel(**json.loads(metadata_path.read_text(encoding="utf-8")))
        expected = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(model.id, expected["provenance"]["model"]["repository"])
        self.assertEqual(model.revision, expected["provenance"]["model"]["revision"])

        runtime = create_transformers_runtime(model, expected["provenance"]["requested_device"])
        outcome = replay_result(
            result_path,
            source=source_path,
            model=model,
            runtime=runtime,
            tolerance_file=Path("fixtures/replay/live-tolerances.json"),
        )

        self.assertIn(outcome.kind, {"exact_match", "projection_match", "within_tolerance"})


if __name__ == "__main__":
    unittest.main()
