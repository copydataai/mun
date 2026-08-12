from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from mun.containment import ContainmentError, run_contained


class ContainmentTests(unittest.TestCase):
    def test_hang_and_output_flood_fail_with_machine_readable_receipts(self) -> None:
        with self.assertRaises(ContainmentError) as hung:
            run_contained([sys.executable, "-c", "import time; time.sleep(2)"], timeout_seconds=0.1)
        self.assertIn("wall_clock_timeout", hung.exception.receipt["failed"])

        with self.assertRaises(ContainmentError) as flooded:
            run_contained([sys.executable, "-c", "print('x' * 100000)"], max_output_bytes=1024)
        self.assertIn("bounded_output", flooded.exception.receipt["failed"])

    def test_symlink_output_escape_is_rejected_and_receipt_is_honest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "managed"
            root.mkdir()
            outside = Path(temporary) / "outside"
            outside.write_text("private")
            link = root / "result.wav"
            link.symlink_to(outside)
            with self.assertRaises(ContainmentError) as raised:
                run_contained([sys.executable, "-c", "pass"], managed_root=root, outputs=[link])
            self.assertIn("managed_output_paths", raised.exception.receipt["failed"])
            self.assertIn("kernel_isolation", raised.exception.receipt["unsupported"])
            self.assertNotIn(str(Path.home()), str(raised.exception.receipt))

    def test_sanitized_environment_closed_stdin_and_output_shape(self) -> None:
        result = run_contained(
            [sys.executable, "-c", "import os,sys; print(os.getenv('SECRET')); print(sys.stdin.read())"],
            expected_stdout="text",
        )
        self.assertEqual(result.stdout.strip(), "None")
        self.assertIn("sanitized_environment", result.receipt["enforced"])
        self.assertIn("closed_stdin", result.receipt["enforced"])


if __name__ == "__main__":
    unittest.main()
