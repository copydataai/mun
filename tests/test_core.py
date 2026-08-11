from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from mun.core import (
    Segment,
    SourceMedia,
    Transcript,
    TranscriptionOptions,
    discover_media,
    output_base,
    render_output,
    transcribe_media,
)
from mun.models import InstalledModel


class DiscoveryTests(unittest.TestCase):
    def test_recursive_discovery_deduplicates_and_skips_hidden_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "recordings"
            root.mkdir()
            first = root / "one.wav"
            first.write_bytes(b"audio")
            nested = root / "nested"
            nested.mkdir()
            second = nested / "two.mp3"
            second.write_bytes(b"audio")
            hidden = root / ".hidden.wav"
            hidden.write_bytes(b"audio")
            (root / "notes.txt").write_text("not media")

            media = discover_media([str(root), str(first)], probe=lambda _: True)

            self.assertEqual([item.source for item in media], [second.resolve(), first.resolve()])
            self.assertEqual(media[0].relative, Path("recordings/nested/two.mp3"))

    def test_explicit_unknown_extension_is_probed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "voice.custom"
            source.write_bytes(b"audio")
            self.assertEqual(len(discover_media([str(source)], probe=lambda _: True)), 1)

    def test_colliding_relative_names_get_deterministic_suffix(self) -> None:
        used: set[Path] = set()
        first = SourceMedia(Path("/a/voice.wav"), Path("voice.wav"))
        second = SourceMedia(Path("/b/voice.wav"), Path("voice.wav"))
        self.assertEqual(output_base(Path("out"), first, used), Path("out/voice"))
        self.assertRegex(output_base(Path("out"), second, used).name, r"voice-[0-9a-f]{8}")


class OutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.media = SourceMedia(Path("/private/source.wav"), Path("batch/source.wav"))
        self.model = InstalledModel(
            id="example/model",
            revision="a" * 40,
            path="/models/example",
            installed_at="2026-08-09T00:00:00+00:00",
        )
        self.transcript = Transcript(
            "Hello world",
            [Segment("Hello", 0.0, 1.25), Segment("world", 1.25, 2.5)],
            "en",
        )

    def test_json_uses_relative_source_and_schema_version(self) -> None:
        payload = json.loads(render_output("json", self.transcript, self.media, self.model, "cpu"))
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["source"], "batch/source.wav")
        self.assertNotIn("/private", json.dumps(payload))

    def test_srt_has_expected_timestamps(self) -> None:
        output = render_output("srt", self.transcript, self.media, self.model, "cpu")
        self.assertIn("00:00:00,000 --> 00:00:01,250", output)
        self.assertIn("2\n00:00:01,250 --> 00:00:02,500", output)

    def test_vtt_has_header(self) -> None:
        self.assertTrue(render_output("vtt", self.transcript, self.media, self.model, "cpu").startswith("WEBVTT\n"))


class ProbeTests(unittest.TestCase):
    def test_is_media_uses_cached_ffprobe_results(self) -> None:
        from mun import core

        core._FFPROBE_CACHE.clear()
        core._BINARY_PATH_CACHE.clear()

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "audio.wav"
            source.write_bytes(b"audio")

            fake = type("Result", (), {"returncode": 0, "stdout": '{"streams": [{"codec_type": "audio", "codec_name": "pcm_s16le", "channels": 1, "sample_rate": "16000"}]}'})()

            with patch("mun.core.subprocess.run", return_value=fake) as run_probe, patch("shutil.which", return_value="/usr/bin/ffprobe"):
                self.assertTrue(core.is_media(source))
                self.assertTrue(core._can_use_source_audio_directly(source))
                self.assertTrue(core.is_media(source))
                self.assertTrue(core._can_use_source_audio_directly(source))

            self.assertEqual(run_probe.call_count, 1)

    def test_transcribe_media_skips_ffmpeg_when_source_is_ready(self) -> None:
        ready_path = Path("/tmp/ready.wav")
        transcript = Transcript("Hello", [Segment("Hello", 0.0, 1.0)], "en")

        with patch("mun.core._can_use_source_audio_directly", return_value=True), patch(
            "mun.core._convert_media"
        ) as convert_media, patch("mun.core._run_pipeline", return_value=transcript) as pipeline:
            result, translated = transcribe_media(object(), ready_path, "whisper", TranscriptionOptions())

        self.assertEqual(result, transcript)
        self.assertIsNone(translated)
        convert_media.assert_not_called()
        self.assertEqual(len(pipeline.mock_calls), 1)

    def test_transcribe_media_converts_when_source_is_not_ready(self) -> None:
        source = Path("/tmp/raw.audio")
        prepared = Path("/tmp/prepared.wav")
        transcript = Transcript("Hello", [Segment("Hello", 0.0, 1.0)], "en")

        with patch("mun.core._can_use_source_audio_directly", return_value=False), patch(
            "mun.core._audio_input_path", return_value=(prepared, True)
        ) as audio_input, patch("mun.core._run_pipeline", return_value=transcript) as pipeline:
            result, _ = transcribe_media(object(), source, "whisper", TranscriptionOptions())

        self.assertEqual(result, transcript)
        audio_input.assert_called_once()
        called_source = audio_input.call_args.args[0]
        self.assertEqual(called_source, source)
        self.assertEqual(pipeline.call_args.args[1], prepared)


if __name__ == "__main__":
    unittest.main()
