from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path

from mun.cli import _existing_ancestor, build_parser, main


class CliTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
