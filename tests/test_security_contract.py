from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mun.core import Segment, SourceMedia, Transcript, TranscriptionOptions, run_transcription_workflow
from mun.errors import MunError
from mun.models import InstalledModel, _write_metadata, remove_model_with_receipt, remove_transient_with_receipt
from mun.runtime import FakeSpeechRuntime


class SecurityContractTests(unittest.TestCase):
    def assert_private_home_absent(self, value: object) -> None:
        serialized = json.dumps(value)
        self.assertNotIn(str(Path.home()), serialized)
        self.assertNotIn(Path.home().name, serialized)

    def test_canonical_results_expose_typed_taint_and_agent_ineligibility(self) -> None:
        runtime = FakeSpeechRuntime(Transcript("Model words", [Segment("Model words", 0.0, 1.0)], "en"))
        media = SourceMedia(Path("source.wav"), Path("source.wav"))

        for remote_code, expected_model_trust in ((False, "verified_artifact"), (True, "unsafe_remote_code")):
            model = InstalledModel("owner/model", "abc", "/models/model", "2026-08-12T00:00:00Z", trust_remote_code=remote_code)
            result = run_transcription_workflow([media], model, TranscriptionOptions(), runtime=runtime)[0].to_dict()

            self.assertEqual(
                result["trust"],
                {
                    "media": "untrusted_bytes",
                    "model": expected_model_trust,
                    "content": "untrusted_model_output",
                    "review": None,
                },
            )
            self.assertEqual(result["agent_eligibility"]["status"], "ineligible")
            self.assertNotIn("trusted", result["agent_eligibility"]["reason"].lower())

    def test_model_deletion_receipt_states_exact_scope_and_exclusions(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.home()) as temporary:
            root = Path(temporary) / "models"
            target = root / "owner--model--abc"
            target.mkdir(parents=True)
            (target / "weights").write_bytes(b"1234")
            model = InstalledModel("owner/model", "abc", str(target), "2026-08-12T00:00:00Z")
            _write_metadata(target, model)

            receipt = remove_model_with_receipt(root, "owner/model")

        payload = receipt.to_dict()
        self.assertEqual(payload["attempted_paths"], ["owner--model--abc"])
        self.assertEqual(payload["removed_model"]["path"], "owner--model--abc")
        self.assertEqual(receipt.removed_model.path, "owner--model--abc")
        self.assert_private_home_absent(payload)
        self.assertEqual(receipt.result, "deleted")
        self.assertGreater(receipt.estimated_bytes, 4)
        self.assertIn("backups", receipt.exclusions)
        self.assertIn("APFS snapshots", receipt.exclusions)
        self.assertIn("swap", receipt.exclusions)
        self.assertIn("filesystem remnants", receipt.exclusions)
        self.assertIn("exports", receipt.exclusions)
        self.assertIn("third-party caches", receipt.exclusions)
        self.assertFalse(target.exists())

    def test_transient_deletion_receipts_redact_deleted_and_missing_paths(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.home()) as temporary:
            root = Path(temporary) / "models"
            root.mkdir()
            target = root / ".owner--model.download"
            target.mkdir()
            (target / "partial").write_bytes(b"1234")

            deleted = remove_transient_with_receipt(root, target).to_dict()
            missing = remove_transient_with_receipt(root, target).to_dict()

        self.assertEqual(deleted["attempted_paths"], [".owner--model.download"])
        self.assertEqual(deleted["result"], "deleted")
        self.assertEqual(missing["attempted_paths"], [".owner--model.download"])
        self.assertEqual(missing["result"], "not_found")
        self.assert_private_home_absent(deleted)
        self.assert_private_home_absent(missing)

    def test_model_deletion_error_redacts_an_absolute_target(self) -> None:
        root = Path.home() / "private-owner" / "models"
        target = root / "missing-model"

        with self.assertRaises(MunError) as raised:
            remove_model_with_receipt(root, str(target))

        self.assertIn("~/private-owner/models/missing-model", str(raised.exception))
        self.assert_private_home_absent(str(raised.exception))

    def test_transient_deletion_refuses_paths_outside_managed_root(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.home()) as temporary:
            root = Path(temporary) / "models"
            root.mkdir()
            outside = Path(temporary) / "secret.txt"
            outside.write_text("secret", encoding="utf-8")

            with self.assertRaises(MunError) as raised:
                remove_transient_with_receipt(root, outside)

            self.assertTrue(outside.exists())
            self.assertIn("managed", str(raised.exception))
            self.assert_private_home_absent(str(raised.exception))


if __name__ == "__main__":
    unittest.main()
