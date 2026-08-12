from __future__ import annotations

import json
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

from mun.quality import QualityError, run_quality_qualification


class FakePublicRuntime:
    physical_execution = False

    def transcribe(self, source: Path, exact_tuple: dict) -> dict:
        return {"text": source.read_text().strip(), "segments": [], "capabilities": {"transcription": "passed"}}


class QualityRunnerTests(unittest.TestCase):
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
