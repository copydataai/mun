from __future__ import annotations

import json
import hashlib
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
    def test_runtime_provenance_distinguishes_direct_and_converted_audio(self) -> None:
        from mun.runtime import TransformersRuntime

        runtime = TransformersRuntime.__new__(TransformersRuntime)
        runtime.info = type(
            "Info",
            (),
            {
                "name": "transformers",
                "version": "1",
                "requested_device": "cpu",
                "effective_device": "cpu",
                "precision": "float32",
                "model_type": "whisper",
            },
        )()
        runtime._run_pipeline = lambda path, options, task: Transcript("Hello", [], "en")

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.wav"
            source.write_bytes(b"source wav")
            direct_probe = type(
                "Probe",
                (),
                {"media_format": "wav", "codec": "pcm_s16le", "sample_rate_hz": 16000, "channels": 1},
            )()
            with patch("mun.runtime._can_use_source_audio_directly", return_value=True), patch(
                "mun.runtime._probe_media_details", return_value=direct_probe
            ):
                runtime.transcribe(source, TranscriptionOptions())
            direct = runtime.prepared_media

            prepared_bytes = b"prepared wav bytes"

            def prepare(_source, destination):
                destination.write_bytes(prepared_bytes)
                return destination, True

            with patch("mun.runtime._can_use_source_audio_directly", return_value=False), patch(
                "mun.runtime._audio_input_path", side_effect=prepare
            ), patch("mun.runtime.ffmpeg_version", return_value="ffmpeg 8.0"):
                runtime.transcribe(source, TranscriptionOptions())
            converted = runtime.prepared_media

        self.assertFalse(direct.used)
        self.assertEqual(direct.sha256, hashlib.sha256(b"source wav").hexdigest())
        self.assertTrue(converted.used)
        self.assertEqual(converted.sha256, hashlib.sha256(prepared_bytes).hexdigest())
        self.assertEqual(converted.media_format, "wav")
        self.assertEqual(converted.codec, "pcm_s16le")
        self.assertEqual(converted.sample_rate_hz, 16000)
        self.assertEqual(converted.channels, 1)
        self.assertEqual(converted.converter.name, "ffmpeg")
        self.assertEqual(converted.converter.version, "ffmpeg 8.0")

    def test_runtime_provenance_records_probed_direct_media_format(self) -> None:
        from mun import core
        from mun.runtime import TransformersRuntime

        core._FFPROBE_CACHE.clear()
        core._BINARY_PATH_CACHE.clear()
        runtime = TransformersRuntime.__new__(TransformersRuntime)
        runtime.info = type(
            "Info",
            (),
            {
                "name": "transformers",
                "version": "1",
                "requested_device": "cpu",
                "effective_device": "cpu",
                "precision": "float32",
                "model_type": "whisper",
            },
        )()
        runtime._run_pipeline = lambda path, options, task: Transcript("Hello", [], "en")

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.aiff"
            source.write_bytes(b"direct aiff bytes")
            fake = type(
                "Result",
                (),
                {
                    "returncode": 0,
                    "stdout": json.dumps(
                        {
                            "streams": [
                                {"codec_type": "audio", "codec_name": "pcm_s16be", "channels": 1, "sample_rate": "16000"}
                            ],
                            "format": {"format_name": "aiff"},
                        }
                    ),
                },
            )()
            with patch("mun.core.subprocess.run", return_value=fake) as run_probe, patch(
                "shutil.which", return_value="/usr/bin/ffprobe"
            ):
                runtime.transcribe(source, TranscriptionOptions())

        direct = runtime.prepared_media
        self.assertFalse(direct.used)
        self.assertEqual(direct.media_format, "aiff")
        self.assertEqual(direct.codec, "pcm_s16be")
        self.assertEqual(direct.sample_rate_hz, 16000)
        self.assertEqual(direct.channels, 1)
        self.assertIsNone(direct.converter)
        self.assertEqual(run_probe.call_count, 1)

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

    def test_is_media_false_when_ffprobe_reports_failure(self) -> None:
        from mun import core

        core._FFPROBE_CACHE.clear()
        core._BINARY_PATH_CACHE.clear()

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "audio.wav"
            source.write_bytes(b"audio")
            fake = type("Result", (), {"returncode": 1, "stdout": ""})()

            with patch("mun.core.subprocess.run", return_value=fake), patch("shutil.which", return_value="/usr/bin/ffprobe"):
                self.assertFalse(core.is_media(source))
                self.assertFalse(core._can_use_source_audio_directly(source))

    def test_is_media_false_when_ffprobe_output_is_malformed(self) -> None:
        from mun import core

        core._FFPROBE_CACHE.clear()
        core._BINARY_PATH_CACHE.clear()

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "audio.wav"
            source.write_bytes(b"audio")
            fake = type("Result", (), {"returncode": 0, "stdout": "this is not json"})()

            with patch("mun.core.subprocess.run", return_value=fake), patch("shutil.which", return_value="/usr/bin/ffprobe"):
                self.assertFalse(core.is_media(source))
                self.assertFalse(core._can_use_source_audio_directly(source))


if __name__ == "__main__":
    unittest.main()
