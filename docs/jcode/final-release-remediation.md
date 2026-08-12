# Final J-Code remediation — Mun release blockers

You are the managed J-Code remediator in `/Users/josesanchez/Developer/public/mun-jcode-challenges-2026-08-12`. Do not launch nested agents. Resolve exactly these independently reproduced blockers test-first, no unrelated cleanup.

Create two commits in order:

1. `fix: isolate batch journals by exact binding`
   - A completed `mun-batch.journal.json` from one source currently breaks a later independent batch with another source in the same output directory.
   - Before reusing a journal, compare the complete canonical binding set (sources, exact tuple, options, projections, destinations). Resume only an exact matching incomplete operation.
   - For a distinct new operation, create a distinct durable journal identity or archive/rotate the completed prior journal without losing evidence. Never silently attach new sources to stale bindings and never mask journal errors as generic transcription failures.
   - Preserve idempotent resume and conflicts. Add public `run_batch`/CLI tests for sequential distinct batches, exact resume, stale incomplete mismatch, and completed evidence retention.

2. `fix: reject malformed authentication envelopes`
   - `mun verify` must reject missing/malformed fields, unknown algorithms, bad base64/signatures, fingerprint mismatch, bad canonicalization, altered receipt/artifact, and extra ambiguous envelope forms with deterministic machine-readable output and no traceback.
   - Move all indexing/parsing into controlled validation and bound diagnostics without secret/private path disclosure.
   - Add CLI process tests asserting one parseable JSON record, nonzero exit, and no traceback for malformed cases.

After each commit run focused tests and `git diff --check`. Final: exact full unittest discovery three fresh times, `uv build`, Python 3.11 clean-wheel and 3.12 if available, CLI smokes, clean worktree. Report exact evidence.