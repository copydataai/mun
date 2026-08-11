from __future__ import annotations

import json
import unittest
from pathlib import Path

from mun.core import Segment, SourceMedia, Transcript, TranscriptionOptions, render_output, run_transcription_workflow
from mun.models import InstalledModel
from mun.runtime import FakeSpeechRuntime
from mun.transcript import make_batch_result


class TranscriptContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = InstalledModel("example/model", "abc123", "/models/example", "2026-08-09T00:00:00+00:00")
        self.media = SourceMedia(Path("/private/source.wav"), Path("batch/source.wav"))
        self.runtime = FakeSpeechRuntime(Transcript("Hello world", [Segment("Hello", 0.0, 1.25)], "en"))

    def test_workflow_returns_canonical_result(self) -> None:
        result = run_transcription_workflow([self.media], self.model, TranscriptionOptions(), runtime=self.runtime)[0]
        payload = result.to_dict()

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["source"]["relative_path"], "batch/source.wav")
        self.assertNotIn("/private", json.dumps(payload))
        self.assertEqual(payload["transcripts"][0]["segments"][0]["start_ms"], 0)
        self.assertEqual(payload["provenance"]["runtime"]["name"], "test")

    def test_json_txt_srt_vtt_are_deterministic_projections(self) -> None:
        result = run_transcription_workflow([self.media], self.model, TranscriptionOptions(), runtime=self.runtime)[0]

        self.assertEqual(render_output("txt", result, self.media, self.model, "cpu"), "Hello world\n")
        self.assertIn('"transcripts"', render_output("json", result, self.media, self.model, "cpu"))
        self.assertIn("00:00:00,000 --> 00:00:01,250", render_output("srt", result, self.media, self.model, "cpu"))
        self.assertTrue(render_output("vtt", result, self.media, self.model, "cpu").startswith("WEBVTT\n"))

    def test_batch_result_contains_ordered_records_and_counts(self) -> None:
        results = run_transcription_workflow([self.media, SourceMedia(Path("/x/second.wav"), Path("second.wav"))], self.model, TranscriptionOptions(), runtime=self.runtime)
        batch = make_batch_result(results).to_dict()

        self.assertEqual([item["source"]["relative_path"] for item in batch["files"]], ["batch/source.wav", "second.wav"])
        self.assertEqual(batch["counts"]["completed"], 2)


if __name__ == "__main__":
    unittest.main()
