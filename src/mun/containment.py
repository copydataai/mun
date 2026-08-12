from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from .errors import MunError


@dataclass(frozen=True)
class ContainedResult:
    returncode: int
    stdout: str
    stderr: str
    receipt: dict[str, object]


class ContainmentError(MunError):
    def __init__(self, message: str, receipt: dict[str, object]) -> None:
        super().__init__(message)
        self.receipt = receipt


def _limits() -> None:
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if hasattr(resource, "RLIMIT_AS") and sys.platform != "darwin":
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))


def run_contained(
    argv: Sequence[str],
    *,
    timeout_seconds: float = 60,
    max_output_bytes: int = 1024 * 1024,
    managed_root: Path | None = None,
    outputs: Sequence[Path] = (),
    max_file_bytes: int = 2 * 1024**3,
    expected_stdout: Literal["text", "json"] = "text",
) -> ContainedResult:
    if not argv or not all(isinstance(item, str) and "\x00" not in item for item in argv):
        raise ValueError("argv must be a non-empty sequence of strings")
    receipt: dict[str, object] = {
        "schema_version": 1,
        "receipt_kind": "mun-execution-containment",
        "enforced": ["argv_only", "sanitized_environment", "closed_stdin", "bounded_output", "wall_clock_timeout", "process_group_cancellation"],
        "observed": [],
        "unsupported": ["kernel_isolation", "decoder_memory_safety", "swap_exclusion", "filesystem_remanence", "remote_code_isolation"],
        "failed": [],
    }
    preexec = None
    if os.name == "posix":
        preexec = _limits
        receipt["enforced"].append("posix_resource_limits")  # type: ignore[union-attr]
    else:
        receipt["unsupported"].append("posix_resource_limits")  # type: ignore[union-attr]
    environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
    try:
        process = subprocess.Popen(
            list(argv), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, env=environment, text=False, preexec_fn=preexec,
        )
        try:
            stdout_bytes, stderr_bytes = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.communicate()
            receipt["failed"].append("wall_clock_timeout")  # type: ignore[union-attr]
            raise ContainmentError("Contained helper exceeded its wall-clock timeout", receipt) from exc
    except (OSError, subprocess.SubprocessError) as exc:
        receipt["failed"].append("process_start")  # type: ignore[union-attr]
        raise ContainmentError("Contained helper could not be started", receipt) from exc
    if len(stdout_bytes) + len(stderr_bytes) > max_output_bytes:
        receipt["failed"].append("bounded_output")  # type: ignore[union-attr]
        raise ContainmentError("Contained helper exceeded its output limit", receipt)
    stdout = stdout_bytes.decode("utf-8", "replace")
    stderr = stderr_bytes.decode("utf-8", "replace")
    if expected_stdout == "json":
        try:
            json.loads(stdout)
        except json.JSONDecodeError as exc:
            receipt["failed"].append("stdout_shape")  # type: ignore[union-attr]
            raise ContainmentError("Contained helper emitted malformed JSON", receipt) from exc
    if managed_root is not None:
        root = managed_root.resolve()
        for output in outputs:
            try:
                if output.is_symlink() or not output.resolve().is_relative_to(root):
                    raise ValueError
                if output.exists() and (not output.is_file() or output.stat().st_size > max_file_bytes):
                    raise ValueError
            except (OSError, ValueError) as exc:
                receipt["failed"].append("managed_output_paths")  # type: ignore[union-attr]
                raise ContainmentError("Contained helper produced an unsafe output path or shape", receipt) from exc
        receipt["enforced"].append("managed_output_paths")  # type: ignore[union-attr]
    receipt["observed"] = ["process_exited", "stdout_shape_valid", "output_limits_observed"]
    return ContainedResult(process.returncode, stdout, stderr, receipt)
