from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mun.cli import _existing_ancestor, build_parser, command_transcribe, interactive_wizard, main
from mun.core import SourceMedia
from mun.models import InstalledModel
from mun.transcript import SourceRecord, TranscriptResult, make_provenance


class CliTests(unittest.TestCase):
    def test_review_commands_parse_narrow_apply_and_render_operations(self) -> None:
        apply_args = build_parser().parse_args(
            ["review", "apply", "machine.json", "corrections.json", "-o", "corrected.json"]
        )
        render_args = build_parser().parse_args(
            ["review", "render", "machine.json", "--corrections", "corrections.json", "--view", "corrected", "--format", "srt"]
        )

        self.assertEqual(apply_args.review_command, "apply")
        self.assertEqual(apply_args.output, Path("corrected.json"))
        self.assertEqual(render_args.review_command, "render")
        self.assertEqual(render_args.view, "corrected")
        self.assertEqual(render_args.format, "srt")

    def test_remote_code_download_parses_explicit_acknowledgement(self) -> None:
        args = build_parser().parse_args([
            "models", "download", "owner/model", "--trust-remote-code", "--acknowledge-remote-code"
        ])

        self.assertTrue(args.trust_remote_code)
        self.assertTrue(args.acknowledge_remote_code)

    def test_review_apply_and_render_do_not_rewrite_machine_json(self) -> None:
        from tests.test_review import correction_payload, machine_result

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            machine = root / "machine.json"
            corrections = root / "corrections.json"
            corrected = root / "corrected.json"
            result = machine_result()
            machine.write_text(result.to_json(), encoding="utf-8")
            original_bytes = machine.read_bytes()
            corrections.write_text(json.dumps(correction_payload(result)), encoding="utf-8")

            self.assertEqual(main(["review", "apply", str(machine), str(corrections), "-o", str(corrected)]), 0)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = main(
                    ["review", "render", str(machine), "--corrections", str(corrections), "--view", "corrected", "--format", "txt"]
                )

            self.assertEqual(status, 0)
            self.assertEqual(output.getvalue(), "Hello world. Next line.\n")
            self.assertEqual(machine.read_bytes(), original_bytes)
            self.assertEqual(json.loads(corrected.read_text(encoding="utf-8"))["review_state"], "reviewed")

    def test_guided_workflow_uses_an_installed_model_without_downloading_again(self) -> None:
        model = InstalledModel(
            "owner/model",
            "abc123",
            "/models/owner--model",
            "2026-08-11T00:00:00+00:00",
            status="installed",
        )
        captured: list[argparse.Namespace] = []

        def fake_transcribe(args: argparse.Namespace) -> int:
            captured.append(args)
            return 0

        with patch("mun.cli.load_config", return_value={}), \
            patch("mun.cli.models_root", return_value=Path("/models")), \
            patch("mun.cli.installed_models", return_value=[model]), \
            patch("mun.cli.download_model") as download, \
            patch("builtins.input", side_effect=["", "voice.wav", ""]), \
            patch("mun.cli.command_transcribe", side_effect=fake_transcribe):
            status = interactive_wizard()

        self.assertEqual(status, 0)
        download.assert_not_called()
        self.assertEqual(captured[0].model, "/models/owner--model")
        self.assertFalse(captured[0].benchmark)

    def test_guided_workflow_uses_a_model_immediately_after_download(self) -> None:
        model = InstalledModel(
            "owner/model",
            "abc123",
            "/models/owner--model",
            "2026-08-11T00:00:00+00:00",
            status="installed",
        )
        captured: list[argparse.Namespace] = []

        def fake_transcribe(args: argparse.Namespace) -> int:
            captured.append(args)
            return 0

        with patch("mun.cli.load_config", return_value={}), \
            patch("mun.cli.models_root", return_value=Path("/models")), \
            patch("mun.cli.installed_models", side_effect=[[], [model]]), \
            patch("mun.cli.load_catalog", return_value={"models": [{
                "id": "owner/model",
                "revision": "abc123",
                "weights_bytes": 1024,
                "license": "apache-2.0",
            }]}), \
            patch("mun.cli._confirm", return_value=True), \
            patch("mun.cli.download_model", return_value=model) as download, \
            patch("builtins.input", side_effect=["", "voice.wav", ""]), \
            patch("mun.cli.command_transcribe", side_effect=fake_transcribe):
            status = interactive_wizard()

        self.assertEqual(status, 0)
        download.assert_called_once_with("owner/model", Path("/models"), "abc123", False, False)
        self.assertEqual(captured[0].model, "/models/owner--model")

    def test_help_describes_the_product_and_common_workflow(self) -> None:
        help_text = build_parser().format_help()

        self.assertIn("Transcribe audio and video on your computer", help_text)
        self.assertIn("mun transcribe recordings/", help_text)

    def test_transcribe_defaults_parse(self) -> None:
        args = build_parser().parse_args(["transcribe", "voice.wav"])
        self.assertEqual(args.inputs, ["voice.wav"])
        self.assertIsNone(args.format)
        self.assertEqual(args.jobs, 1)

    def test_transcribe_accepts_jobs(self) -> None:
        args = build_parser().parse_args(["transcribe", "voice.wav", "--jobs", "4"])
        self.assertEqual(args.jobs, 4)

    def test_transcribe_accepts_benchmark(self) -> None:
        args = build_parser().parse_args(["transcribe", "voice.wav", "--benchmark"])
        self.assertTrue(args.benchmark)

    def test_no_arguments_without_tty_returns_usage_error(self) -> None:
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            status = main([])
        self.assertEqual(status, 2)
        self.assertIn("usage: mun", error.getvalue())

    def test_existing_ancestor_handles_fresh_model_directory(self) -> None:
        self.assertEqual(_existing_ancestor(Path("/missing/child")), Path("/"))

    def test_command_transcribe_reads_streamed_input_list_and_runs_batch_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_list = Path(temporary) / "inputs.txt"
            source_list.write_text("one.wav\n# ignore me\n\ntwo.wav\n", encoding="utf-8")
            captured: list[str] = []
            batch_requests: list[dict[str, object]] = []
            model = InstalledModel("owner/model", "abc123", f"{temporary}/model", "2026-08-11T00:00:00+00:00")

            def fake_discover_media(paths: list[str], include_hidden: bool = False):
                captured.extend(paths)
                return [
                    SourceMedia(Path("one.wav"), Path("one.wav")),
                    SourceMedia(Path("two.wav"), Path("two.wav")),
                ]

            result = TranscriptResult(
                schema_version=1,
                status="completed",
                source=SourceRecord("one.wav", "one.wav"),
                transcripts=[],
                speakers=[],
                diagnostics=[],
                provenance=make_provenance(
                    mun_version="0.1.0",
                    model_id="owner/model",
                    revision="abc123",
                    runtime_name="transformers",
                    runtime_version="0",
                    requested_device="cpu",
                    effective_device="cpu",
                    precision="float32",
                ),
            )

            def fake_run_batch(media, discovered_model, output_dir, formats, options, overwrite, progress, runtime=None, jobs=1):
                batch_requests.append(
                    {
                        "media_count": len(media),
                        "jobs": jobs,
                        "model_id": discovered_model.id,
                        "formats": formats,
                        "output_dir": str(output_dir),
                    }
                )
                return [result], []

            args = build_parser().parse_args(
                [
                    "transcribe",
                    "--input-list",
                    str(source_list),
                    "--model-dir",
                    temporary,
                    "--output-dir",
                    str(Path(temporary) / "out"),
                    "--jobs",
                    "1",
                ]
            )

            with patch("mun.cli.load_config", return_value={}), \
                patch("mun.cli.discover_media", side_effect=fake_discover_media), \
                patch("mun.cli.models_root", return_value=Path(temporary)), \
                patch("mun.cli.find_installed", return_value=model), \
                patch("mun.cli.load_pipeline", return_value=(SimpleNamespace(info=SimpleNamespace(effective_device="cpu"),), "cpu", "transformers")), \
                patch("mun.cli.run_batch", side_effect=fake_run_batch):
                status = command_transcribe(args)

            self.assertEqual(status, 0)
            self.assertEqual(captured, ["one.wav", "two.wav"])
            self.assertEqual(batch_requests[0]["media_count"], 2)
            self.assertEqual(batch_requests[0]["jobs"], 1)
            self.assertEqual(batch_requests[0]["formats"], ["txt"])

    def test_verified_reuse_succeeds_and_is_counted_separately(self) -> None:
        result = TranscriptResult(
            schema_version=1,
            status="completed",
            source=SourceRecord("one.wav", "one.wav"),
            transcripts=[],
            speakers=[],
            diagnostics=[],
            provenance=make_provenance(
                mun_version="0.1.0",
                model_id="owner/model",
                revision="abc123",
                runtime_name="transformers",
                runtime_version="0",
                requested_device="cpu",
                effective_device="cpu",
                precision="float32",
            ),
            reuse_status="reused_verified",
        )
        model = InstalledModel("owner/model", "abc123", "/models/model", "2026-08-11T00:00:00+00:00")
        args = build_parser().parse_args(["transcribe", "one.wav", "--summary-json", "--benchmark"])
        output = io.StringIO()
        error = io.StringIO()

        with patch("mun.cli.load_config", return_value={}), \
            patch("mun.cli.discover_media", return_value=[SourceMedia(Path("one.wav"), Path("one.wav"))]), \
            patch("mun.cli.models_root", return_value=Path("/models")), \
            patch("mun.cli.find_installed", return_value=model), \
            patch("mun.cli.load_pipeline", return_value=(SimpleNamespace(info=SimpleNamespace(effective_device="cpu")), "cpu", "transformers")), \
            patch("mun.cli.run_batch", return_value=([result], [])), \
            contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            status = command_transcribe(args)

        payload = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(payload["counts"]["processed"], 0)
        self.assertEqual(payload["counts"]["reused_verified"], 1)
        self.assertIn("processed=0 reused_verified=1", error.getvalue())

    def test_keyboard_interrupt_at_transcription_workflow_boundary_is_cancelled_and_nonzero(self) -> None:
        model = InstalledModel("owner/model", "abc123", "/models/model", "2026-08-11T00:00:00+00:00")
        args = build_parser().parse_args(["transcribe", "one.wav"])

        with patch("mun.cli.load_config", return_value={}), \
            patch("mun.cli.discover_media", return_value=[SourceMedia(Path("one.wav"), Path("one.wav"))]), \
            patch("mun.cli.models_root", return_value=Path("/models")), \
            patch("mun.cli.find_installed", return_value=model), \
            patch("mun.cli.load_pipeline", return_value=(SimpleNamespace(info=SimpleNamespace(effective_device="cpu")), "cpu", "transformers")), \
            patch("mun.cli.run_batch", side_effect=KeyboardInterrupt), \
            contextlib.redirect_stderr(io.StringIO()):
            status = command_transcribe(args)

        self.assertEqual(status, 1)


if __name__ == "__main__":
    unittest.main()
