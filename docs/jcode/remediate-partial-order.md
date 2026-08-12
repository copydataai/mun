# J-Code remediation — remove partial-commit order dependence

You are the managed J-Code remediator in `/Users/josesanchez/Developer/public/mun-jcode-challenges-2026-08-12`. Do not launch nested agents.

The prior commit `62c62dc` is not accepted. Its isolated repetitions passed, but a fresh full suite failed deterministically twice:

```text
ERROR test_partial_commit_resume_preserves_verified_projection_writes_missing_remainder_and_repeats
src/mun/journal.py:218 write_result_outputs
src/mun/core.py:1140 MunError: Output already exists or cannot be verified: .../out/source.json
JournalError: Resumed projections conflict with the journal-bound result

ERROR test_public_batch_recovers_from_every_durable_boundary_and_repeated_resume (boundary='partial_commit')
same source.json verification failure
```

Diagnose shared/global/order-dependent state, nondeterministic serialization, time/receipt fields, runtime identity leakage, or projection digest mismatch. The fix must make verification derive expected bytes from the journal-bound canonical result and stable operation data—not a newly generated volatile field or process-global state. Preserve fail-closed conflict behavior and never blanket-overwrite.

Add a regression that runs the relevant recovery cases after the entire preceding test sequence or otherwise explicitly perturbs ordering/time/global state and proves stable verified reuse. Run the exact authoritative command at least three fresh times:

`uv run python -m unittest discover -s tests -v`

All three must pass. Then run `uv build`, CLI smoke, and `git diff --check`.

Create exactly one commit: `fix: make partial resume verification deterministic`

Leave a clean worktree and report exact results.