from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mun.journal import OperationJournal, resume_journal


class JournalTests(unittest.TestCase):
    def test_fault_boundaries_classify_fail_closed_and_resume_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "batch.journal.json"
            binding = {
                "source_sha256": "a" * 64,
                "tuple": {"model": "m", "runtime": "r", "artifact": "b" * 64},
                "options": {"timestamps": True},
                "projections": ["json", "txt"],
                "destination": {"root": "transcripts", "overwrite": False},
            }
            journal = OperationJournal.create(path, [binding])
            self.assertEqual(journal.classify()[0]["classification"], "safely-resumable")
            journal.transition("a" * 64, "inference_completed", evidence={"result_digest": "c" * 64})
            self.assertEqual(journal.classify()[0]["classification"], "safely-resumable")
            journal.transition("a" * 64, "render_staged", evidence={"staged": ["x.json", "x.txt"]})
            self.assertEqual(journal.classify()[0]["classification"], "must-recompute")
            journal.transition("a" * 64, "partial_commit", evidence={"committed": ["x.json"], "uncommitted": ["x.txt"]})
            self.assertEqual(journal.classify()[0]["classification"], "conflict")

            first = resume_journal(path)
            second = resume_journal(path)
            self.assertEqual(first, second)
            self.assertEqual(first["sources"][0]["evidence"]["committed"], ["x.json"])

    def test_binding_change_is_indeterminate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "batch.journal.json"
            journal = OperationJournal.create(path, [{"source_sha256": "a" * 64, "tuple": {}, "options": {}, "projections": [], "destination": {}}])
            payload = json.loads(path.read_text())
            payload["sources"][0]["binding_digest"] = "0" * 64
            path.write_text(json.dumps(payload))
            self.assertEqual(OperationJournal.load(path).classify()[0]["classification"], "indeterminate")


if __name__ == "__main__":
    unittest.main()
