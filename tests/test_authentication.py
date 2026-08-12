from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat

from mun.authentication import AuthenticationError, sign_artifact, verify_artifact


class AuthenticationTests(unittest.TestCase):
    def test_ephemeral_ed25519_signature_binds_artifact_receipt_role_and_fingerprint(self) -> None:
        private = Ed25519PrivateKey.generate()
        key_bytes = private.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
        artifact = {"schema_version": 1, "artifact_kind": "mun-source-grounded-transcript-acceptance", "status": "accepted"}
        receipt = {"schema_version": 1, "artifact_sha256": "a" * 64}
        envelope = sign_artifact(artifact, receipt, key_bytes, "transcript-producer")
        outcome = verify_artifact(artifact, receipt, envelope)
        self.assertTrue(outcome["valid"])
        self.assertEqual(outcome["producer_role"], "transcript-producer")
        self.assertFalse(outcome["claims"]["accuracy"])
        self.assertNotIn("PRIVATE", json.dumps(envelope))

        altered = {**artifact, "status": "changed"}
        with self.assertRaisesRegex(AuthenticationError, "artifact"):
            verify_artifact(altered, receipt, envelope)
        with self.assertRaisesRegex(AuthenticationError, "receipt"):
            verify_artifact(artifact, {**receipt, "artifact_sha256": "b" * 64}, envelope)
        with self.assertRaisesRegex(AuthenticationError, "algorithm"):
            verify_artifact(artifact, receipt, {**envelope, "algorithm": "unknown"})
        with self.assertRaisesRegex(AuthenticationError, "fingerprint"):
            verify_artifact(artifact, receipt, {**envelope, "public_key_fingerprint": "0" * 64})

    def test_verify_cli_rejects_malformed_envelopes_with_one_json_record_and_no_traceback(self) -> None:
        private = Ed25519PrivateKey.generate()
        key_bytes = private.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
        artifact = {"schema_version": 1, "artifact_kind": "acceptance", "status": "accepted"}
        receipt = {"schema_version": 1, "artifact_sha256": "a" * 64}
        valid = sign_artifact(artifact, receipt, key_bytes, "transcript-producer")
        malformed = {
            "missing-field": {key: value for key, value in valid.items() if key != "signature"},
            "unknown-algorithm": {**valid, "algorithm": "RSA"},
            "bad-base64": {**valid, "signature": "%%%"},
            "bad-signature": {**valid, "signature": "AA=="},
            "fingerprint-mismatch": {**valid, "public_key_fingerprint": "0" * 64},
            "bad-canonicalization": {**valid, "canonicalization": "json"},
            "extra-field": {**valid, "signature_v2": valid["signature"]},
            "wrong-claims-shape": {**valid, "claims": []},
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_path = root / "artifact.json"
            receipt_path = root / "receipt.json"
            envelope_path = root / "envelope.json"
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

            cases = list(malformed.items()) + [
                ("envelope-array", []),
                ("envelope-invalid-json", "{"),
            ]
            for name, envelope in cases:
                with self.subTest(name=name):
                    if isinstance(envelope, str):
                        envelope_path.write_text(envelope, encoding="utf-8")
                    else:
                        envelope_path.write_text(json.dumps(envelope), encoding="utf-8")
                    completed = subprocess.run(
                        [sys.executable, "-m", "mun", "verify", str(artifact_path), str(receipt_path), str(envelope_path)],
                        text=True, capture_output=True, check=False,
                    )
                    records = completed.stdout.splitlines()
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertEqual(len(records), 1, completed)
                    payload = json.loads(records[0])
                    self.assertEqual(payload["valid"], False)
                    self.assertIsInstance(payload["error"]["code"], str)
                    self.assertNotIn("Traceback", completed.stdout + completed.stderr)
                    self.assertNotIn(str(root), completed.stdout + completed.stderr)

    def test_verify_cli_rejects_altered_artifact_and_receipt_as_json(self) -> None:
        private = Ed25519PrivateKey.generate()
        key_bytes = private.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
        artifact = {"schema_version": 1, "status": "accepted"}
        receipt = {"schema_version": 1, "artifact_sha256": "a" * 64}
        envelope = sign_artifact(artifact, receipt, key_bytes, "producer")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            envelope_path = root / "envelope.json"
            envelope_path.write_text(json.dumps(envelope), encoding="utf-8")
            for name, changed_artifact, changed_receipt in (
                ("artifact", {**artifact, "status": "changed"}, receipt),
                ("receipt", artifact, {**receipt, "artifact_sha256": "b" * 64}),
            ):
                with self.subTest(name=name):
                    artifact_path = root / "artifact.json"
                    receipt_path = root / "receipt.json"
                    artifact_path.write_text(json.dumps(changed_artifact), encoding="utf-8")
                    receipt_path.write_text(json.dumps(changed_receipt), encoding="utf-8")
                    completed = subprocess.run(
                        [sys.executable, "-m", "mun", "verify", str(artifact_path), str(receipt_path), str(envelope_path)],
                        text=True, capture_output=True, check=False,
                    )
                    payload = json.loads(completed.stdout)
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertEqual(payload["valid"], False)
                    self.assertIn(name, payload["error"]["code"])
                    self.assertNotIn("Traceback", completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
