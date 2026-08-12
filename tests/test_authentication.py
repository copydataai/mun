from __future__ import annotations

import json
import unittest

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


if __name__ == "__main__":
    unittest.main()
