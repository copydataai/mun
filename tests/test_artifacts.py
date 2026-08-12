from __future__ import annotations

import copy
import json
import unittest

from mun.artifacts import ArtifactValidationError, canonical_json_bytes, machine_result_digest, validate_machine_result


class ArtifactIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = {
            "schema_version": 1,
            "status": "completed",
            "source": {"name": "source.wav", "relative_path": "source.wav", "sha256": "source-digest"},
            "transcripts": [
                {
                    "kind": "original",
                    "language": {"tag": "en", "source": "detected", "confidence": None},
                    "text": "Hello\nworld",
                    "segments": [{"id": "segment_1", "start_ms": 0, "end_ms": 1000, "text": "Hello\nworld"}],
                }
            ],
            "speakers": [],
            "diagnostics": [],
            "provenance": {
                "mun_version": "0.1.0",
                "created_at": "2026-08-12T18:00:00Z",
                "model": {"repository": "example/model", "revision": "abc123", "artifact_sha256": "manifest-digest"},
                "runtime": {"name": "transformers", "version": "5.14.0", "environment": {"machine": "arm64"}},
                "requested_device": "auto",
                "effective_device": "cpu",
                "precision": "float32",
            },
            "operation": {
                "parameters": {"language": None, "timestamps": True, "translate": False, "chunk_length": 30, "stride_length": 5},
                "prepared_media": {"used": True, "sha256": "prepared-digest"},
                "source_hash_policy": "sha256_source_bytes",
            },
            "overlap_ms": 0,
        }

    def test_canonical_bytes_ignore_receipt_fields_order_and_platform_newlines(self) -> None:
        first = copy.deepcopy(self.payload)
        first["result_digest"] = "old-claim"
        second = json.loads(json.dumps(first).replace("\\n", "\\r\\n"))
        second["provenance"]["created_at"] = "2026-08-12T19:00:00Z"
        second = dict(reversed(list(second.items())))

        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
        self.assertNotIn(b"created_at", canonical_json_bytes(first))
        self.assertNotIn(b"result_digest", canonical_json_bytes(first))

    def test_equivalent_results_created_at_different_times_have_same_digest(self) -> None:
        later = copy.deepcopy(self.payload)
        later["provenance"]["created_at"] = "2026-08-13T18:00:00Z"

        self.assertEqual(machine_result_digest(self.payload), machine_result_digest(later))

    def test_every_derivation_input_changes_the_digest(self) -> None:
        mutations = {
            "source": lambda value: value["source"].update(sha256="different-source"),
            "prepared input": lambda value: value["operation"]["prepared_media"].update(sha256="different-prepared"),
            "model artifact": lambda value: value["provenance"]["model"].update(artifact_sha256="different-model"),
            "runtime": lambda value: value["provenance"]["runtime"].update(version="different-runtime"),
            "parameters": lambda value: value["operation"]["parameters"].update(chunk_length=31),
            "transcript": lambda value: value["transcripts"][0].update(text="Different transcript"),
            "diagnostics": lambda value: value["diagnostics"].append({"severity": "warning", "code": "changed", "message": "Changed", "stage": "test", "recoverable": True}),
        }
        expected = machine_result_digest(self.payload)
        for label, mutate in mutations.items():
            with self.subTest(label):
                changed = copy.deepcopy(self.payload)
                mutate(changed)
                self.assertNotEqual(expected, machine_result_digest(changed))

    def test_claimed_digest_mismatch_is_a_typed_validation_failure(self) -> None:
        claimed = {**self.payload, "result_digest": "0" * 64}

        with self.assertRaises(ArtifactValidationError) as raised:
            validate_machine_result(claimed)

        self.assertEqual(raised.exception.code, "result_digest_mismatch")
        self.assertEqual(str(raised.exception), "Machine result digest does not match its contents")

    def test_legacy_result_without_digest_remains_valid(self) -> None:
        self.assertEqual(validate_machine_result(self.payload), self.payload)


if __name__ == "__main__":
    unittest.main()
