from __future__ import annotations

import json
import contextlib
import io
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

from mun.acceptance import AcceptanceError, create_acceptance
from mun.cli import main
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
            {"policy_id": "strict", "waivers": [{"transcript_kind": "original", "segment_id": "segment_2", "reason": "bounded inaudible interval"}]},
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

    def test_waivers_require_one_exact_existing_variant_segment_pair(self) -> None:
        result = machine_result()
        result = type(result)(**{
            **result.__dict__,
            "source": type(result.source)(**{**result.source.__dict__, "sha256": sha256(b"media").hexdigest()}),
            "result_digest": None,
        })
        overlay = {
            "schema_version": 1,
            "machine_result_digest": result.result_digest,
            "source_sha256": result.source.sha256,
            "decisions": [
                {"transcript_kind": "original", "segment_id": "segment_1", "disposition": "accepted"},
                {"transcript_kind": "original", "segment_id": "segment_2", "disposition": "exception", "reason": "inaudible"},
            ],
        }
        invalid = [
            [{"segment_id": "segment_2", "reason": "ambiguous"}],
            [{"transcript_kind": "english_translation", "segment_id": "segment_2", "reason": "cross variant"}],
            [{"transcript_kind": "original", "segment_id": "missing", "reason": "absent"}],
            [{"transcript_kind": "original", "segment_id": "segment_2", "reason": "one"}] * 2,
            [{"transcript_kind": "original", "segment_id": "segment_2", "reason": "   "}],
            "not-a-list",
        ]

        for waivers in invalid:
            with self.subTest(waivers=waivers), self.assertRaises(AcceptanceError):
                create_acceptance(result, overlay, {"policy_id": "strict", "waivers": waivers})

    def test_review_accept_cli_fails_closed_for_ambiguous_waiver(self) -> None:
        result = machine_result()
        result = type(result)(**{
            **result.__dict__,
            "source": type(result.source)(**{**result.source.__dict__, "sha256": sha256(b"media").hexdigest()}),
            "result_digest": None,
        })
        overlay = {
            "schema_version": 1,
            "machine_result_digest": result.result_digest,
            "source_sha256": result.source.sha256,
            "decisions": [
                {"transcript_kind": "original", "segment_id": "segment_1", "disposition": "accepted"},
                {"transcript_kind": "original", "segment_id": "segment_2", "disposition": "exception", "reason": "inaudible"},
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            machine = root / "machine.json"
            overlay_path = root / "overlay.json"
            policy = root / "policy.json"
            output = root / "accepted.json"
            machine.write_text(result.to_json(), encoding="utf-8")
            overlay_path.write_text(json.dumps(overlay), encoding="utf-8")
            policy.write_text(json.dumps({
                "policy_id": "strict",
                "waivers": [{"segment_id": "segment_2", "reason": "ambiguous"}],
            }), encoding="utf-8")

            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                status = main([
                    "review", "accept", str(machine), str(overlay_path), "--policy", str(policy), "-o", str(output),
                ])

            self.assertEqual(status, 1)
            self.assertFalse(output.exists())
            self.assertIn("waiver", errors.getvalue().lower())


if __name__ == "__main__":
    unittest.main()
