from __future__ import annotations

import json
import hashlib
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from mun.errors import MunError

from mun.core import (
    ExportCommitError,
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
from mun.artifacts import ArtifactValidationError
from mun.transcript import TranscriptResult, make_batch_result


class TranscriptContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = InstalledModel("example/model", "abc123", "/models/example", "2026-08-09T00:00:00+00:00")
        self.media = SourceMedia(Path("/private/source.wav"), Path("batch/source.wav"))
        self.runtime = FakeSpeechRuntime(Transcript("Hello world", [Segment("Hello", 0.0, 1.25)], "en"))

    def test_workflow_returns_canonical_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.wav"
            source.write_bytes(b"canonical source bytes")
            media = SourceMedia(source, Path("batch/source.wav"))
            result = run_transcription_workflow([media], self.model, TranscriptionOptions(), runtime=self.runtime)[0]
        payload = result.to_dict()

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["source"]["relative_path"], "batch/source.wav")
        self.assertNotIn("/private", json.dumps(payload))
        self.assertEqual(payload["transcripts"][0]["segments"][0]["start_ms"], 0)
        self.assertEqual(payload["provenance"]["runtime"]["name"], "test")
        self.assertEqual(payload["source"]["sha256"], hashlib.sha256(b"canonical source bytes").hexdigest())
        self.assertIn("precision", payload["provenance"])
        self.assertRegex(payload["result_digest"], r"^[0-9a-f]{64}$")

    def test_json_loader_validates_claimed_result_digest(self) -> None:
        result = run_transcription_workflow([self.media], self.model, TranscriptionOptions(), runtime=self.runtime)[0]
        payload = json.loads(result.to_json())
        payload["transcripts"][0]["text"] = "tampered"

        with self.assertRaises(ArtifactValidationError):
            TranscriptResult.from_json(json.dumps(payload))

    def test_json_loader_accepts_legacy_result_without_digest(self) -> None:
        result = run_transcription_workflow([self.media], self.model, TranscriptionOptions(), runtime=self.runtime)[0]
        payload = result.to_dict()
        payload.pop("result_digest")

        loaded = TranscriptResult.from_json(json.dumps(payload))

        self.assertIsNone(loaded.result_digest)
        self.assertEqual(loaded.transcripts[0].text, "Hello world")

    def test_source_digest_depends_on_bytes_not_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.wav"
            second = Path(temporary) / "renamed.wav"
            first.write_bytes(b"same bytes")
            second.write_bytes(b"same bytes")
            results = run_transcription_workflow(
                [SourceMedia(first, Path("first.wav")), SourceMedia(second, Path("renamed.wav"))],
                self.model,
                TranscriptionOptions(),
                runtime=self.runtime,
            )

        self.assertEqual(results[0].source.sha256, results[1].source.sha256)

    def test_operation_records_every_inference_affecting_option(self) -> None:
        options = TranscriptionOptions(
            language="es", timestamps=True, translate=True, chunk_length=42, stride_length=7, device="mps"
        )
        result = run_transcription_workflow([self.media], self.model, options, runtime=self.runtime)[0]
        payload = result.to_dict()

        self.assertEqual(
            payload["operation"]["parameters"],
            {
                "language": "es",
                "timestamps": True,
                "translate": True,
                "chunk_length": 42,
                "stride_length": 7,
                "requested_device": "mps",
                "effective_device": "cpu",
                "precision": "test",
            },
        )
        self.assertEqual(payload["operation"]["source_hash_policy"], "sha256_source_bytes")
        self.assertIn("environment", payload["provenance"]["runtime"])

    def test_json_txt_srt_vtt_are_deterministic_projections(self) -> None:
        result = run_transcription_workflow([self.media], self.model, TranscriptionOptions(), runtime=self.runtime)[0]

        self.assertEqual(render_output("txt", result, self.media, self.model, "cpu"), "Hello world\n")
        self.assertIn('"transcripts"', render_output("json", result, self.media, self.model, "cpu"))
        self.assertEqual(
            render_output("srt", result, self.media, self.model, "cpu"),
            "1\n00:00:00,000 --> 00:00:01,250\nHello\n",
        )
        self.assertEqual(
            render_output("vtt", result, self.media, self.model, "cpu"),
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.250\nHello\n",
        )

    def test_batch_result_contains_ordered_records_and_counts(self) -> None:
        results = run_transcription_workflow([self.media, SourceMedia(Path("/x/second.wav"), Path("second.wav"))], self.model, TranscriptionOptions(), runtime=self.runtime)
        batch = make_batch_result(results).to_dict()

        self.assertEqual([item["source"]["relative_path"] for item in batch["files"]], ["batch/source.wav", "second.wav"])
        self.assertEqual(batch["counts"]["completed"], 2)
        self.assertEqual(batch["counts"]["processed"], 2)
        self.assertEqual(batch["counts"]["reused_verified"], 0)

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

    def test_render_failure_leaves_no_final_projection_and_records_failed_before_commit(self) -> None:
        result = run_transcription_workflow([self.media], self.model, TranscriptionOptions(), runtime=self.runtime)[0]
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "source"

            def fail_txt_render(format_name, *args):
                if format_name == "txt":
                    raise MunError("cannot render")
                return render_output(format_name, *args)

            with patch("mun.core.render_output", side_effect=fail_txt_render):
                with self.assertRaises(MunError):
                    write_result_outputs(base, ["txt", "json"], result, translated=False, overwrite=False)

            receipt = json.loads(Path(f"{base}.receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["state"], "failed_before_commit")
            self.assertEqual(receipt["committed_paths"], [])
            self.assertEqual(receipt["uncommitted_paths"], [str(Path(f"{base}.json")), str(Path(f"{base}.txt"))])
            self.assertFalse(Path(f"{base}.txt").exists())
            self.assertFalse(Path(f"{base}.json").exists())
            self.assertEqual(list(Path(temporary).glob(".mun-stage-*")), [])

    def test_commit_failure_reports_exact_partial_commit_paths(self) -> None:
        result = run_transcription_workflow([self.media], self.model, TranscriptionOptions(), runtime=self.runtime)[0]
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "source"
            real_replace = Path.replace
            destinations = [Path(f"{base}.json"), Path(f"{base}.txt")]

            def fail_second_commit(path: Path, target: Path):
                if target == destinations[1]:
                    raise OSError("disk failure")
                return real_replace(path, target)

            with patch.object(Path, "replace", autospec=True, side_effect=fail_second_commit):
                with self.assertRaises(ExportCommitError) as caught:
                    write_result_outputs(base, ["txt", "json"], result, translated=False, overwrite=False)

            self.assertEqual(caught.exception.committed_paths, [destinations[0]])
            self.assertEqual(caught.exception.uncommitted_paths, [destinations[1]])
            receipt = json.loads(Path(f"{base}.receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["state"], "partial_commit")
            self.assertEqual(receipt["committed_paths"], [str(destinations[0])])
            self.assertEqual(receipt["uncommitted_paths"], [str(destinations[1])])
            self.assertTrue(destinations[0].exists())
            self.assertFalse(destinations[1].exists())
            self.assertEqual(list(Path(temporary).glob(".mun-stage-*")), [])

    def test_precommit_cancellation_writes_cancelled_receipt_without_final_paths(self) -> None:
        result = run_transcription_workflow([self.media], self.model, TranscriptionOptions(), runtime=self.runtime)[0]
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "source"
            with patch("mun.core.render_output", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    write_result_outputs(base, ["txt"], result, translated=False, overwrite=False)

            receipt = json.loads(Path(f"{base}.receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["state"], "cancelled")
            self.assertEqual(receipt["committed_paths"], [])
            self.assertFalse(Path(f"{base}.txt").exists())
            self.assertEqual(list(Path(temporary).glob(".mun-stage-*")), [])

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
        self.assertIn('"operation"', payload)
        self.assertIn('"parameters"', payload)
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

    def test_unrelated_txt_blocks_without_fabricating_transcript_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "file.wav"
            source.write_bytes(b"audio")
            (Path(temporary) / "file.txt").write_text("unrelated", encoding="utf-8")
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
            self.assertEqual(summaries[0].reuse_status, "conflict")
            self.assertEqual(summaries[0].transcripts, [])
            self.assertEqual(failures[0]["error"], "Existing outputs cannot be verified for reuse")

    def test_matching_canonical_json_is_reused_without_loading_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "file.wav"
            source.write_bytes(b"audio")
            media = SourceMedia(source, Path(source.name))
            canonical = run_transcription_workflow(
                [media], self.model, TranscriptionOptions(), runtime=self.runtime
            )[0]
            (Path(temporary) / "file.json").write_text(canonical.to_json(), encoding="utf-8")
            with patch("mun.core.load_pipeline") as load_pipeline:
                summaries, failures = run_batch(
                    [media],
                    self.model,
                    Path(temporary),
                    ["json"],
                    TranscriptionOptions(),
                    False,
                    lambda _: None,
                    jobs=4,
                )
            self.assertEqual(load_pipeline.call_count, 0)
            self.assertEqual(summaries[0].status, "completed")
            self.assertEqual(summaries[0].reuse_status, "reused_verified")
            self.assertEqual(summaries[0].transcripts[0].text, "Hello world")
            self.assertEqual(failures, [])

    def test_changed_source_or_parameters_reject_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "file.wav"
            source.write_bytes(b"original")
            media = SourceMedia(source, Path(source.name))
            canonical = run_transcription_workflow(
                [media], self.model, TranscriptionOptions(), runtime=self.runtime
            )[0]
            (Path(temporary) / "file.json").write_text(canonical.to_json(), encoding="utf-8")

            source.write_bytes(b"changed")
            source_results, source_failures = run_batch(
                [media], self.model, Path(temporary), ["json"], TranscriptionOptions(), False,
                lambda _: None, jobs=4,
            )
            source.write_bytes(b"original")
            option_results, option_failures = run_batch(
                [media], self.model, Path(temporary), ["json"], TranscriptionOptions(language="es"), False,
                lambda _: None, jobs=4,
            )

        self.assertEqual(source_results[0].reuse_status, "conflict")
        self.assertEqual(option_results[0].reuse_status, "conflict")
        self.assertEqual(len(source_failures), 1)
        self.assertEqual(len(option_failures), 1)

    def test_partial_projection_set_is_distinct_from_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "file.wav"
            source.write_bytes(b"audio")
            media = SourceMedia(source, Path(source.name))
            canonical = run_transcription_workflow(
                [media], self.model, TranscriptionOptions(), runtime=self.runtime
            )[0]
            (Path(temporary) / "file.json").write_text(canonical.to_json(), encoding="utf-8")

            summaries, failures = run_batch(
                [media], self.model, Path(temporary), ["json", "txt"], TranscriptionOptions(), False,
                lambda _: None, jobs=4,
            )

        self.assertEqual(summaries[0].reuse_status, "incomplete_output_set")
        self.assertEqual(len(failures), 1)

    def test_overwrite_replaces_only_requested_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "file.wav"
            source.write_bytes(b"audio")
            txt = Path(temporary) / "file.txt"
            json_path = Path(temporary) / "file.json"
            txt.write_text("old", encoding="utf-8")
            json_path.write_text("keep", encoding="utf-8")

            summaries, failures = run_batch(
                [SourceMedia(source, Path(source.name))], self.model, Path(temporary), ["txt"],
                TranscriptionOptions(), True, lambda _: None, runtime=self.runtime,
            )

            self.assertEqual(txt.read_text(encoding="utf-8"), "Hello world\n")
            self.assertEqual(json_path.read_text(encoding="utf-8"), "keep")
        self.assertEqual(summaries[0].reuse_status, "overwrite_required")
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
