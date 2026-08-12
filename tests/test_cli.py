from __future__ import annotations

import argparse
import contextlib
import io
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


if __name__ == "__main__":
    unittest.main()
