from __future__ import annotations

import json
import unittest
import time
from pathlib import Path
from unittest.mock import patch

from mun.errors import MunError

from mun.core import (
    Segment,
    SourceMedia,
    Transcript,
    TranscriptionOptions,
    output_paths,
    render_output,
    run_batch,
    run_transcription_workflow,
    write_result_outputs,
)
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
        self.assertIsNone(payload["source"]["sha256"])
        self.assertIn("precision", payload["provenance"])

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

    def test_translation_writes_one_canonical_json_and_variant_text_files(self) -> None:
        runtime = FakeSpeechRuntime(
            Transcript("Hola", [Segment("Hola", 0.0, 1.0)], "es"),
            Transcript("Hello", [Segment("Hello", 0.0, 1.0)], "en"),
        )
        result = run_transcription_workflow(
            [self.media], self.model, TranscriptionOptions(translate=True), runtime=runtime
        )[0]

        self.assertEqual(
            output_paths(Path("out/source"), ["json", "txt"], translated=True),
            [Path("out/source.json"), Path("out/source.original.txt"), Path("out/source.en.txt")],
        )

        with self.subTest("canonical JSON retains every variant"):
            import tempfile

            with tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary) / "source"
                written = write_result_outputs(base, ["json", "txt"], result, translated=True, overwrite=False)
                payload = json.loads(Path(f"{base}.json").read_text(encoding="utf-8"))
                self.assertEqual([variant["kind"] for variant in payload["transcripts"]], ["original", "english_translation"])
                self.assertEqual(
                    written,
                    [Path(f"{base}.json"), Path(f"{base}.original.txt"), Path(f"{base}.en.txt")],
                )

    def test_failed_result_does_not_expose_exception_details(self) -> None:
        class FailingRuntime:
            info = self.runtime.info

            def transcribe(self, source, options):
                raise RuntimeError("secret /Users/private/source.wav")

        result = run_transcription_workflow(
            [self.media], self.model, TranscriptionOptions(), runtime=FailingRuntime()
        )[0]

        payload = json.dumps(result.to_dict())
        self.assertEqual(result.status, "failed")
        self.assertNotIn("secret", payload)
        self.assertNotIn("/Users/private", payload)

    def test_batch_reuses_the_supplied_runtime_and_workflow(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            results, failures = run_batch(
                [self.media],
                self.model,
                Path(temporary),
                ["json"],
                TranscriptionOptions(),
                False,
                lambda _: None,
                runtime=self.runtime,
            )

        self.assertEqual([result.status for result in results], ["completed"])
        self.assertEqual(failures, [])

    def test_run_batch_rejects_non_positive_jobs(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "file.wav"
            source.write_bytes(b"audio")
            media = SourceMedia(source, Path(source.name))
            with self.assertRaises(MunError):
                run_batch(
                    [media],
                    self.model,
                    Path(temporary),
                    ["txt"],
                    TranscriptionOptions(),
                    False,
                    lambda _: None,
                    runtime=self.runtime,
                    jobs=0,
                )

    def test_run_batch_failure_does_not_leak_internal_details(self) -> None:
        import tempfile

        class FailingRuntime:
            info = self.runtime.info

            def transcribe(self, source, options):
                raise RuntimeError("secret /private/source.wav")

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "file.wav"
            source.write_bytes(b"audio")
            media = SourceMedia(source, Path(source.name))
            summaries, failures = run_batch(
                [media],
                self.model,
                Path(temporary),
                ["txt"],
                TranscriptionOptions(),
                False,
                lambda _: None,
                runtime=FailingRuntime(),
            )
            self.assertEqual(summaries[0].status, "failed")
            self.assertFalse(any("private/source.wav" in failure["error"] for failure in failures))

    def test_run_batch_reports_a_failed_source_as_failed(self) -> None:
        import tempfile

        class FailingRuntime:
            info = self.runtime.info

            def transcribe(self, source, options):
                raise RuntimeError("broken media")

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "file.wav"
            source.write_bytes(b"audio")
            progress: list[str] = []
            run_batch(
                [SourceMedia(source, Path(source.name))],
                self.model,
                Path(temporary),
                ["txt"],
                TranscriptionOptions(),
                False,
                progress.append,
                runtime=FailingRuntime(),
            )

        self.assertTrue(any("] failed " in message for message in progress), progress)
        self.assertFalse(any("] complete " in message for message in progress), progress)

    def test_run_batch_parallel_skips_without_loading_model(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "file.wav"
            source.write_bytes(b"audio")
            (Path(temporary) / "file.txt").write_text("cached", encoding="utf-8")
            media = [SourceMedia(source, Path(source.name))]
            with patch("mun.core.load_pipeline") as load_pipeline:
                summaries, failures = run_batch(
                    media,
                    self.model,
                    Path(temporary),
                    ["txt"],
                    TranscriptionOptions(),
                    False,
                    lambda _: None,
                    jobs=4,
                )
            self.assertEqual(load_pipeline.call_count, 0)
            self.assertEqual(summaries[0].status, "partial")
            self.assertEqual(failures, [])

    def test_run_batch_parallel_skips_fast_path_is_not_inflated_by_runtime_setup(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            media = []
            for index in range(20):
                source = Path(temporary) / f"file-{index}.wav"
                source.write_bytes(b"audio")
                (Path(temporary) / f"file-{index}.txt").write_text("cached", encoding="utf-8")
                media.append(SourceMedia(source, Path(source.name)))

            def blocking_load_pipeline(*_args, **_kwargs):
                time.sleep(0.05)
                raise RuntimeError("runtime load should be skipped")

            start = time.perf_counter()
            with patch("mun.core.load_pipeline", side_effect=blocking_load_pipeline) as load_pipeline:
                summaries, failures = run_batch(
                    media,
                    self.model,
                    Path(temporary),
                    ["txt"],
                    TranscriptionOptions(),
                    False,
                    lambda _: None,
                    jobs=4,
                )
            elapsed = time.perf_counter() - start

            self.assertEqual(load_pipeline.call_count, 0)
            self.assertLess(elapsed, 0.20)
            self.assertEqual([item.status for item in summaries], ["partial"] * len(media))
            self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
