# J-Code remediation packet — Mun specification findings

You are the managed J-Code remediator in `/Users/josesanchez/Developer/public/mun-jcode-challenges-2026-08-12`. Do not launch nested agents. Read the original packet and all current code/tests. Preserve existing behavior and commits. Resolve every finding below through production public seams, test-first. No unrelated cleanup.

Create exactly these four commits in order:

1. `fix: integrate resumable batch journals`
   - Integrate versioned journal creation and atomic transition updates into the real batch/transcription/export workflow before inference and at pre-inference, post-inference/pre-render, staged-render, and partial-commit boundaries.
   - `mun resume` must use a real runner, verify exact bindings, perform safe remaining work, persist transitions, and be idempotent. Classification-only behavior is insufficient.
   - Add public workflow fault-injection tests for process death at every required boundary and repeated resume.

2. `fix: bound hostile helper output while streaming`
   - Replace unbounded `communicate()` capture with incremental concurrent draining and a strict combined byte budget; terminate the full process group immediately on overflow/timeout/cancellation.
   - Ensure descendants are killed and cleanup is deterministic. Add sustained-flood, forking-child, cancellation, malformed probe, symlink, and cleanup tests through the public containment seam.

3. `fix: enforce bounded acceptance waivers`
   - Require each waiver to target exactly one existing `(transcript_kind, segment_id)`.
   - Reject absent, duplicate, ambiguous, cross-variant, or malformed waivers. Preserve deterministic canonical output and fail-closed acceptance.
   - Add CLI/module regression tests.

4. `fix: execute and expire quality evidence`
   - Add an explicit opt-in adapter over the real public transcription workflow for `qualify-run`, bind observed exact runtime/model/artifact tuple, and keep deterministic fake runtime for normal tests.
   - Compare expiry to current time; expired evidence and tuple drift must block eligible/tested/release status.
   - Add public CLI tests for real-adapter selection without downloading, expiry, tuple drift, and mandatory strata.

After each commit run focused tests and `git diff --check`. Final gates: full unittest discovery, `uv build`, CLI smokes, clean Python 3.11 wheel smoke and Python 3.12 if available. Leave a clean worktree. Report exact SHAs/results/skips and do not claim physical quality, truth, consent, universal determinism, or secure erasure.