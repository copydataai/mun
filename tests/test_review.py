from __future__ import annotations

import hashlib
import json
import unittest

from mun.artifacts import canonical_json_bytes
from mun.review import CorrectionError, CorrectionSet, apply_corrections, render_reviewed
from mun.transcript import (
    Language,
    SourceRecord,
    TranscriptResult,
    TranscriptSegment,
    TranscriptVariant,
    make_provenance,
)


def machine_result() -> TranscriptResult:
    return TranscriptResult(
        schema_version=1,
        status="completed",
        source=SourceRecord("meeting.wav", "meeting.wav"),
        transcripts=[
            TranscriptVariant(
                kind="original",
                language=Language("en", "forced"),
                text="Helo world. Next line.",
                segments=[
                    TranscriptSegment("segment_1", 0, 1250, "Helo world."),
                    TranscriptSegment("segment_2", 1250, 2400, "Next line."),
                ],
            )
        ],
        speakers=[],
        diagnostics=[],
        provenance=make_provenance(
            mun_version="0.1.0",
            model_id="owner/model",
            revision="abc123",
            runtime_name="test",
            runtime_version="1",
            requested_device="cpu",
            effective_device="cpu",
            precision="float32",
        ),
    )


def correction_payload(result: TranscriptResult, **changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "correction_set_id": "review-2026-08-12-a",
        "created_at": "2026-08-12T19:00:00Z",
        "parent_result_digest": result.result_digest,
        "review_state": "reviewed",
        "corrections": [
            {
                "transcript_kind": "original",
                "segment_id": "segment_1",
                "original_text_digest": hashlib.sha256(b"Helo world.").hexdigest(),
                "replacement": "Hello world.",
                "note": "Corrected a recognition typo.",
            }
        ],
    }
    payload.update(changes)
    return payload


class ReviewTests(unittest.TestCase):
    def test_application_preserves_machine_result_and_records_distinct_identity(self) -> None:
        result = machine_result()
        before_dict = result.to_dict()
        before_json = result.to_json()
        overlay = CorrectionSet.from_json(json.dumps(correction_payload(result)))

        corrected = apply_corrections(result, overlay)

        self.assertEqual(result.to_dict(), before_dict)
        self.assertEqual(result.to_json(), before_json)
        self.assertEqual(corrected.transcript.transcripts[0].segments[0].text, "Hello world.")
        self.assertEqual(corrected.transcript.transcripts[0].text, "Hello world. Next line.")
        self.assertEqual(corrected.parent_result_digest, result.result_digest)
        self.assertEqual(corrected.review_state, "reviewed")
        self.assertNotEqual(corrected.export_digest, result.result_digest)
        self.assertEqual(corrected.export_digest, hashlib.sha256(canonical_json_bytes(corrected.identity_dict())).hexdigest())

        later = correction_payload(result, created_at="2026-08-12T20:00:00Z")
        self.assertNotEqual(overlay.digest, CorrectionSet.from_json(json.dumps(later)).digest)

    def test_parent_and_stale_segment_mismatches_fail_closed(self) -> None:
        result = machine_result()
        wrong_parent = correction_payload(result, parent_result_digest="0" * 64)
        with self.assertRaisesRegex(CorrectionError, "parent machine result"):
            apply_corrections(result, CorrectionSet.from_json(json.dumps(wrong_parent)))

        stale = correction_payload(result)
        stale["corrections"][0]["original_text_digest"] = hashlib.sha256(b"older text").hexdigest()  # type: ignore[index]
        with self.assertRaisesRegex(CorrectionError, "segment text"):
            apply_corrections(result, CorrectionSet.from_json(json.dumps(stale)))

    def test_corrected_txt_srt_and_vtt_replace_text_without_changing_timings(self) -> None:
        result = machine_result()
        corrected = apply_corrections(result, CorrectionSet.from_json(json.dumps(correction_payload(result))))

        self.assertEqual(render_reviewed("txt", result, corrected, "corrected"), "Hello world. Next line.\n")
        self.assertEqual(
            render_reviewed("srt", result, corrected, "corrected"),
            "1\n00:00:00,000 --> 00:00:01,250\nHello world.\n\n"
            "2\n00:00:01,250 --> 00:00:02,400\nNext line.\n",
        )
        self.assertIn("00:00:00.000 --> 00:00:01.250\nHello world.", render_reviewed("vtt", result, corrected, "corrected"))
        self.assertEqual(render_reviewed("txt", result, corrected, "machine"), "Helo world. Next line.\n")

    def test_corrected_json_has_explicit_state_and_treats_content_as_data(self) -> None:
        result = machine_result()
        payload = correction_payload(result)
        payload["corrections"][0]["replacement"] = "<script>alert('data')</script>"  # type: ignore[index]
        overlay = CorrectionSet.from_json(json.dumps(payload))
        corrected = apply_corrections(result, overlay)

        exported = json.loads(render_reviewed("json", result, corrected, "corrected"))

        self.assertEqual(exported["view"], "corrected")
        self.assertEqual(exported["review_state"], "reviewed")
        self.assertEqual(exported["parent_result_digest"], result.result_digest)
        self.assertEqual(exported["correction_set_id"], "review-2026-08-12-a")
        self.assertRegex(exported["export_digest"], r"^[0-9a-f]{64}$")
        self.assertEqual(exported["transcript"]["transcripts"][0]["segments"][0]["text"], "<script>alert('data')</script>")
        self.assertNotIn("authentic", json.dumps(exported).lower())
        self.assertNotIn("truth", json.dumps(exported).lower())

        machine_export = json.loads(render_reviewed("json", result, corrected, "machine"))
        self.assertEqual(machine_export["view"], "machine")
        self.assertEqual(machine_export["review_state"], "unreviewed")
        self.assertEqual(machine_export["export_digest"], result.result_digest)
        self.assertNotEqual(machine_export["export_digest"], exported["export_digest"])

    def test_format_rejects_duplicate_targets_invalid_state_and_oversized_note(self) -> None:
        result = machine_result()
        duplicate = correction_payload(result)
        duplicate["corrections"].append(dict(duplicate["corrections"][0]))  # type: ignore[union-attr,index]
        with self.assertRaises(CorrectionError):
            CorrectionSet.from_json(json.dumps(duplicate))
        with self.assertRaises(CorrectionError):
            CorrectionSet.from_json(json.dumps(correction_payload(result, review_state="approved")))
        oversized = correction_payload(result)
        oversized["corrections"][0]["note"] = "x" * 501  # type: ignore[index]
        with self.assertRaises(CorrectionError):
            CorrectionSet.from_json(json.dumps(oversized))


if __name__ == "__main__":
    unittest.main()
