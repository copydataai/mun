from __future__ import annotations

import sys
import tempfile
import time
import unittest
import os
from pathlib import Path

from mun.containment import ContainmentError, run_contained


class ContainmentTests(unittest.TestCase):
    def test_sustained_combined_flood_is_stopped_at_the_streaming_budget(self) -> None:
        script = (
            "import os\n"
            "block=b'x'*4096\n"
            "while True:\n"
            " os.write(1, block); os.write(2, block)\n"
        )
        started = time.monotonic()
        with self.assertRaises(ContainmentError) as flooded:
            run_contained([sys.executable, "-c", script], max_output_bytes=8192, timeout_seconds=5)

        self.assertLess(time.monotonic() - started, 2)
        self.assertIn("bounded_output", flooded.exception.receipt["failed"])

    @unittest.skipUnless(os.name == "posix", "process-group descendant check requires POSIX")
    def test_timeout_kills_forked_descendant_and_closes_pipes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pid_path = Path(temporary) / "child.pid"
            script = (
                "import os,time,sys\n"
                "pid=os.fork()\n"
                "if pid==0:\n"
                f" open({str(pid_path)!r},'w').write(str(os.getpid()))\n"
                " while True: os.write(1,b'x'); time.sleep(.01)\n"
                "while True: time.sleep(1)\n"
            )
            with self.assertRaises(ContainmentError):
                run_contained([sys.executable, "-c", script], timeout_seconds=0.2)
            child_pid = int(pid_path.read_text(encoding="utf-8"))

            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                try:
                    os.kill(child_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.02)
            else:
                self.fail("forked helper descendant survived containment cleanup")

    def test_cancellation_terminates_helper_and_reports_cleanup(self) -> None:
        started = time.monotonic()
        with self.assertRaises(ContainmentError) as cancelled:
            run_contained(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                timeout_seconds=20,
                cancelled=lambda: time.monotonic() - started > 0.1,
            )

        self.assertIn("cancelled", cancelled.exception.receipt["failed"])
        self.assertIn("process_group_terminated", cancelled.exception.receipt["observed"])

    def test_malformed_json_probe_fails_closed_and_process_is_reaped(self) -> None:
        with self.assertRaises(ContainmentError) as malformed:
            run_contained(
                [sys.executable, "-c", "print('{not-json')"],
                expected_stdout="json",
            )
        self.assertIn("stdout_shape", malformed.exception.receipt["failed"])
        self.assertIn("process_reaped", malformed.exception.receipt["observed"])

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
