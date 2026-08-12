from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from mun.cli import main
from mun.qualification import (
    QualificationError,
    create_qualification_record,
    missing_tested_claims,
    qualification_is_expired,
)


FIXTURES = Path(__file__).parent / "fixtures" / "qualification"


class QualificationTests(unittest.TestCase):
    def manifest(self) -> dict:
        return json.loads((FIXTURES / "run-manifest.json").read_text(encoding="utf-8"))

    def test_manifest_records_exact_tuple_hashes_measurements_and_capabilities(self) -> None:
        record = create_qualification_record(self.manifest(), base_dir=FIXTURES)

        self.assertEqual(record["status"], "eligible")
        self.assertTrue(record["unsigned"])
        self.assertEqual(record["tuple"]["device"]["effective"], "mps")
        self.assertEqual(record["tuple"]["device"]["precision"], "float16")
        self.assertEqual(
            record["fixtures"][0]["sha256"],
            "6d2de7ac128b6c305cdebcf59ab84fcfa8f072e014d210a0277c667af6936f70",
        )
        self.assertEqual(record["timing"]["warm"]["median_seconds"], 5.0)
        self.assertAlmostEqual(record["timing"]["warm"]["median_rtf"], 1 / 12)
        self.assertEqual(record["peak_memory"]["measurement"], "allocator-visible")
        self.assertEqual([row["name"] for row in record["capabilities"]], ["timestamps", "transcription"])
        self.assertNotIn("accuracy", record)
        self.assertNotIn("quality", record)

    def test_metadata_only_evidence_can_never_be_tested(self) -> None:
        manifest = self.manifest()
        manifest["physical_execution"] = False

        record = create_qualification_record(manifest, base_dir=FIXTURES)

        self.assertEqual(record["status"], "eligible")
        self.assertEqual(record["status_reason"], "observed_execution_unavailable")

    def test_caller_asserted_physical_execution_cannot_produce_tested(self) -> None:
        manifest = self.manifest()
        manifest["physical_execution"] = True

        record = create_qualification_record(manifest, base_dir=FIXTURES)

        self.assertEqual(record["status"], "eligible")
        self.assertEqual(record["status_reason"], "observed_execution_unavailable")

    def test_physical_failure_and_matrix_exclusion_have_distinct_statuses(self) -> None:
        failed = self.manifest()
        failed["capabilities"][0] = {
            "name": "transcription",
            "outcome": "failed",
            "diagnostic": "out of memory",
        }
        unsupported = self.manifest()
        unsupported["run_outcome"] = "unsupported"

        failed_record = create_qualification_record(failed, base_dir=FIXTURES)
        unsupported_record = create_qualification_record(unsupported, base_dir=FIXTURES)

        self.assertEqual(failed_record["status"], "failed")
        self.assertEqual(unsupported_record["status"], "unsupported")

    def test_material_tuple_change_or_elapsed_expiry_invalidates_record(self) -> None:
        record = create_qualification_record(self.manifest(), base_dir=FIXTURES)
        changed = deepcopy(record["tuple"])
        changed["runtime"]["version"] = "5.15.0"

        self.assertTrue(
            qualification_is_expired(
                record,
                current_tuple=changed,
                now=datetime(2026, 8, 13, tzinfo=UTC),
            )
        )
        self.assertTrue(qualification_is_expired(record, now=datetime(2026, 11, 11, tzinfo=UTC)))
        self.assertFalse(qualification_is_expired(record, now=datetime(2026, 8, 13, tzinfo=UTC)))

    def test_missing_or_expired_rows_block_an_advertised_tested_claim(self) -> None:
        record = create_qualification_record(self.manifest(), base_dir=FIXTURES)
        claims = [
            {"tuple_digest": record["tuple_digest"], "capability": "transcription"},
            {"tuple_digest": record["tuple_digest"], "capability": "english-translation"},
        ]

        missing = missing_tested_claims([record], claims, now=datetime(2026, 8, 13, tzinfo=UTC))

        self.assertEqual(missing, claims)
        self.assertEqual(
            missing_tested_claims([record], claims[:1], now=datetime(2026, 11, 11, tzinfo=UTC)),
            claims[:1],
        )

    def test_missing_advertised_capability_row_cannot_create_tested_record(self) -> None:
        manifest = self.manifest()
        manifest["capabilities"] = manifest["capabilities"][:1]

        with self.assertRaisesRegex(QualificationError, "missing capability outcomes: timestamps"):
            create_qualification_record(manifest, base_dir=FIXTURES)

    def test_cli_writes_deterministic_unsigned_local_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "qualification.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main([
                    "qualify", str(FIXTURES / "run-manifest.json"), "-o", str(output_path)
                ])

            record = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(record["status"], "eligible")
        self.assertTrue(record["unsigned"])
        self.assertIn("unsigned local qualification record", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
