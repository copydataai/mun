from __future__ import annotations

import json
import contextlib
import io
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

from mun.cli import main
from mun.core import Segment, Transcript
from mun.models import InstalledModel
from mun.quality import QualityError, run_quality_qualification
from mun.runtime import FakeSpeechRuntime


class FakePublicRuntime:
    physical_execution = False

    def transcribe(self, source: Path, exact_tuple: dict) -> dict:
        return {"text": source.read_text().strip(), "segments": [], "capabilities": {"transcription": "passed"}}


class QualityRunnerTests(unittest.TestCase):
    def manifest(self, root: Path) -> dict:
        source = root / "fixture.txt"
        source.write_text("hello world", encoding="utf-8")
        return {
            "schema_version": 1,
            "generated_at": "2026-08-12T00:00:00Z",
            "expiry_days": 30,
            "tuple": {
                "model": {"repository": "example/model", "revision": "abc", "artifact_sha256": "a" * 64},
                "runtime": {"name": "test", "version": "0"},
                "device": {"requested": "cpu", "effective": "cpu", "precision": "test"},
            },
            "fixtures": [{
                "id": "english-meeting", "source": "fixture.txt", "source_sha256": sha256(source.read_bytes()).hexdigest(),
                "reference_text": "hello world", "language": "en", "domain": "meeting",
                "license": "CC0", "consent": "independently_supplied", "allowed_metrics": ["wer", "cer"],
            }],
            "policy": {"mandatory_strata": [{"language": "en", "domain": "meeting", "max_wer": 0.0, "max_cer": 0.0}]},
        }

    def test_cli_selects_real_public_adapter_without_downloading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.manifest(root)
            manifest_path = root / "manifest.json"
            output = root / "record.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            model = InstalledModel("example/model", "abc", "/models/example", "2026-08-12T00:00:00Z")
            runtime = FakeSpeechRuntime(Transcript("hello world", [Segment("hello world", 0.0, 1.0)], "en"))
            runtime.model_artifact_sha256 = "a" * 64
            runtime.test_installed_model = model

            with patch("mun.cli.find_installed", return_value=model), \
                 patch("mun.cli.load_pipeline", return_value=(runtime, None)), \
                 patch("mun.cli.download_model") as download, \
                 contextlib.redirect_stdout(io.StringIO()):
                status = main([
                    "qualify-run", str(manifest_path), "-o", str(output), "--real-runtime", "--device", "cpu",
                ])

            self.assertEqual(status, 0)
            self.assertEqual(download.call_count, 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["tuple"], manifest["tuple"])

    def test_cli_blocks_expired_tuple_drift_and_missing_mandatory_strata(self) -> None:
        for mutation, reason in (
            (lambda manifest: manifest.update({"generated_at": "2020-01-01T00:00:00Z", "expiry_days": 1}), "expired_evidence"),
            (lambda manifest: manifest["tuple"]["runtime"].update({"version": "drifted"}), "tuple_drift"),
            (lambda manifest: manifest["policy"].update({"mandatory_strata": [{"language": "fr", "domain": "meeting", "max_wer": 0.0}]}), "mandatory_stratum_failed"),
        ):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest = self.manifest(root)
                mutation(manifest)
                manifest_path = root / "manifest.json"
                output = root / "record.json"
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with contextlib.redirect_stdout(io.StringIO()):
                    status = main([
                        "qualify-run", str(manifest_path), "-o", str(output), "--deterministic-fake-runtime",
                    ])

                record = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(status, 0)
                self.assertEqual(record["status"], "failed")
                self.assertEqual(record["status_reason"], reason)

    def test_executes_fixtures_reports_strata_and_never_self_asserts_hardware(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "fixture.txt"
            source.write_text("hello world")
            manifest = {
                "schema_version": 1,
                "generated_at": "2026-08-12T00:00:00Z",
                "expiry_days": 30,
                "tuple": {"model": "fake@abc", "runtime": "fake-1", "device": "synthetic"},
                "fixtures": [{
                    "id": "english-meeting", "source": "fixture.txt", "source_sha256": sha256(source.read_bytes()).hexdigest(),
                    "reference_text": "hello world", "language": "en", "domain": "meeting",
                    "license": "CC0", "consent": "independently_supplied", "allowed_metrics": ["wer", "cer"],
                }],
                "policy": {"mandatory_strata": [{"language": "en", "domain": "meeting", "max_wer": 0.0, "max_cer": 0.0}]},
            }
            record = run_quality_qualification(manifest, base_dir=root, runtime=FakePublicRuntime())
            self.assertEqual(record["status"], "eligible")
            self.assertEqual(record["status_reason"], "requires_physical_qualification")
            self.assertEqual(record["fixtures"][0]["wer"], 0.0)
            self.assertEqual(record["strata"][0]["outcome"], "passed")
            self.assertNotIn("quality_score", record)

    def test_missing_reference_digest_or_failed_mandatory_stratum_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "f.txt").write_text("actual")
            manifest = {
                "schema_version": 1, "generated_at": "2026-08-12T00:00:00Z", "expiry_days": 1,
                "tuple": {"model": "m", "runtime": "r", "device": "d"},
                "fixtures": [{"id": "f", "source": "f.txt", "source_sha256": sha256(b"actual").hexdigest(), "reference_text": "different", "language": "en", "domain": "meeting", "license": "x", "consent": "supplied", "allowed_metrics": ["wer"]}],
                "policy": {"mandatory_strata": [{"language": "en", "domain": "meeting", "max_wer": 0.0}]},
            }
            record = run_quality_qualification(manifest, base_dir=root, runtime=FakePublicRuntime())
            self.assertEqual(record["status"], "failed")
            broken = json.loads(json.dumps(manifest))
            del broken["fixtures"][0]["reference_text"]
            with self.assertRaisesRegex(QualityError, "reference"):
                run_quality_qualification(broken, base_dir=root, runtime=FakePublicRuntime())


if __name__ == "__main__":
    unittest.main()
