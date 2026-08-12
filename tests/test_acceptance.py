from __future__ import annotations

import json
import unittest
from hashlib import sha256

from mun.acceptance import AcceptanceError, create_acceptance
from tests.test_review import machine_result


class AcceptanceTests(unittest.TestCase):
    def test_acceptance_binds_each_segment_and_fails_closed(self) -> None:
        result = machine_result()
        result = type(result)(**{**result.__dict__, "source": type(result.source)(**{**result.source.__dict__, "sha256": sha256(b"media").hexdigest()})})
        overlay = {
            "schema_version": 1,
            "machine_result_digest": result.result_digest,
            "source_sha256": result.source.sha256,
            "decisions": [
                {"transcript_kind": "original", "segment_id": "segment_1", "disposition": "accepted"},
                {"transcript_kind": "original", "segment_id": "segment_2", "disposition": "exception", "reason": "inaudible"},
            ],
        }
        with self.assertRaisesRegex(AcceptanceError, "unresolved exception"):
            create_acceptance(result, overlay, {"policy_id": "strict", "waivers": []})
        artifact, receipt = create_acceptance(
            result,
            overlay,
            {"policy_id": "strict", "waivers": [{"segment_id": "segment_2", "reason": "bounded inaudible interval"}]},
        )
        self.assertEqual(artifact["status"], "accepted_with_waiver")
        self.assertEqual(artifact["segments"][0]["source_sha256"], result.source.sha256)
        self.assertEqual(artifact["segments"][0]["interval"], {"start_ms": 0, "end_ms": 1250})
        self.assertEqual(artifact["trust"]["content"], "untrusted_model_output")
        self.assertEqual(artifact["agent_eligibility"]["status"], "ineligible")
        self.assertFalse(artifact["claims"]["truth"])
        self.assertEqual(receipt["artifact_sha256"], sha256(json.dumps(artifact, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest())

    def test_rejects_stale_duplicate_absent_and_overlapping_segments(self) -> None:
        result = machine_result()
        result = type(result)(**{**result.__dict__, "source": type(result.source)(**{**result.source.__dict__, "sha256": sha256(b"media").hexdigest()})})
        base = {"schema_version": 1, "machine_result_digest": result.result_digest, "source_sha256": result.source.sha256}
        for decisions, message in [
            ([{"transcript_kind": "original", "segment_id": "missing", "disposition": "accepted"}], "absent"),
            ([{"transcript_kind": "original", "segment_id": "segment_1", "disposition": "accepted"}] * 2, "duplicate"),
        ]:
            with self.assertRaisesRegex(AcceptanceError, message):
                create_acceptance(result, {**base, "decisions": decisions}, {"policy_id": "strict", "waivers": []})
        stale = {**base, "machine_result_digest": "0" * 64, "decisions": []}
        with self.assertRaisesRegex(AcceptanceError, "stale"):
            create_acceptance(result, stale, {"policy_id": "strict", "waivers": []})


if __name__ == "__main__":
    unittest.main()
