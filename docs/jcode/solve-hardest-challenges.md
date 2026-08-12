# J-Code implementation packet — solve Mun's hardest challenges

You are the managed J-Code implementer inside `/Users/josesanchez/Developer/public/mun-jcode-challenges-2026-08-12` on branch `feat/solve-hardest-challenges`. Do not launch J-Code, Codex, or any nested agent. Work autonomously. Inspect `CONTEXT.md`, `README.md`, current implementation, tests, and accepted design docs before editing.

## Objective

Resolve every software-addressable gap behind Mun's hardest challenges. Do not falsely claim that source code can guarantee transcript truth, speaker consent, human identity, universal secure erasure, model determinism, or physical-hardware behavior. For those external boundaries, build executable qualification/evidence interfaces and fail-closed release gates rather than prose-only promises.

## Required packets and exact micro-commits

Complete these in order. For each packet: add a failing test through a public seam, implement the smallest deep module, run focused checks, then create exactly one focused commit with the specified message. No unrelated cleanup.

### M1 — restore authoritative test portability
Commit: `test: make review fixtures importable`

- Reproduce the current clean-checkout failure where `uv run python -m unittest discover -s tests -v` cannot import `tests.test_review`.
- Fix test/package invocation portability without weakening assertions or relying on a developer's stale environment.
- Prove both the documented unittest command and an installed-wheel smoke path can import the package correctly.

### M2 — source-grounded acceptance artifact
Commit: `feat: add source-grounded transcript acceptance`

Create a narrow public CLI/module interface for turning immutable machine output plus correction overlays into a source-grounded acceptance artifact.

Required behavior:
- Every reviewed segment remains linked to immutable source digest and bounded media interval.
- Track explicit segment dispositions: accepted, corrected, exception, and unreviewed.
- Reject stale overlays, overlapping/invalid intervals, changed machine identity, duplicate decisions, and claims about absent segments.
- Final acceptance fails closed while unreviewed segments or unresolved exceptions remain unless an explicit bounded waiver is recorded with a reason.
- Acceptance never removes transcript taint or agent-ineligibility and never claims truth, consent, custody, or reviewer identity.
- Generate deterministic, canonical JSON plus an adjacent receipt that binds source, machine-result, overlay, acceptance policy, and emitted projections.
- Add CLI help and README workflow examples. Prefer extending existing `mun review` interfaces rather than a parallel framework.

### M3 — producer authentication and chain verification
Commit: `feat: authenticate accepted transcript artifacts`

Add optional producer authentication that can be independently verified without Mun storing private keys.

Required behavior:
- Sign canonical artifact bytes via a narrow adapter to a well-defined local signing mechanism already reasonably available on supported platforms, or a small justified dependency.
- Verification must bind the signature, public identity material/fingerprint, canonical artifact digest, schema version, and producer-declared role.
- Never confuse authentication with transcript accuracy, reviewer identity, consent, or authorization.
- Reject altered content, altered receipts, unknown algorithms, mismatched fingerprints, malformed signatures, and ambiguous canonicalization.
- Private key material must never be copied into Mun configuration, logs, receipts, tests, or repository fixtures.
- Add deterministic tests using ephemeral keys and a CLI verify command with machine-readable output.

### M4 — crash-safe resumable batch execution
Commit: `feat: resume interrupted transcription batches`

Deepen the existing receipt/interruption machinery into a durable resume interface.

Required behavior:
- Persist a versioned operation journal before inference and update it atomically at externally observable transitions.
- Bind resume to source digest, exact model/runtime/artifact tuple, options, requested projections, and destination semantics.
- On restart classify each source as verified-complete, safely resumable, must-recompute, conflict, or indeterminate.
- Never silently reuse partial or unverifiable outputs; preserve exact evidence for committed and uncommitted paths.
- Handle process death at pre-inference, post-inference/pre-render, staged-render, and partial-commit boundaries using deterministic fault-injection tests.
- Provide `mun resume <journal>` or an equally narrow public CLI seam and idempotent repeated resume behavior.

### M5 — hostile-media execution containment
Commit: `feat: contain untrusted media processing`

Strengthen FFmpeg/ffprobe and model-helper execution as far as supported-platform source code can honestly enforce.

Required behavior:
- Centralize subprocess policy in a deep module with argv-only execution, sanitized environment, closed stdin, bounded stdout/stderr, wall-clock timeout, output/file-size limits where enforceable, process-group cancellation, and deterministic cleanup.
- Apply OS resource limits on supported POSIX systems and report unsupported containment predicates explicitly.
- Prevent symlink/path escape from managed temporary roots and reject output type/shape mismatches.
- Produce a machine-readable containment receipt recording enforced, observed, unsupported, and failed predicates.
- Do not call this a sandbox unless it is one. Document residual kernel, decoder, swap, filesystem, and optional remote-code risks.
- Add adversarial fake-helper tests for hangs, output floods, child processes, symlinks, malformed probe output, and cancellation.

### M6 — executable quality and hardware qualification
Commit: `feat: run transcript quality qualification`

Turn quality/reproducibility qualification from metadata conversion into an executable evidence runner.

Required behavior:
- Versioned fixture-manifest interface with source digest, independently supplied reference text/segments, language/domain labels, license/consent metadata, and allowed metrics.
- Execute the real public transcription workflow for an exact tuple; capture timing, memory when measurable, capability outcomes, normalized error metrics, failures, and artifact digests.
- Do not invent one universal quality score. Report per-fixture and stratified measurements and require explicit acceptance thresholds in policy.
- Missing fixtures, missing references, expired evidence, unqualified tuple changes, or failed mandatory strata must block `tested`/release status.
- Network/model-heavy tests must be explicit opt-in; normal tests use a deterministic fake public runtime and cannot self-assert physical execution.
- Provide machine-readable CLI output and documented commands.

## Cross-cutting acceptance

- Preserve the current CLI's documented behavior unless strengthened fail-closed behavior is required.
- Tests target public CLI/module seams; avoid implementation-coupled mocks when a fake adapter suffices.
- No credentials, transcript content, absolute home paths, or raw media in diagnostics/receipts unless explicitly user-owned output.
- Update docs honestly using Implemented / Requires physical qualification / Not enforceable classifications.
- Run after each packet: focused tests and `git diff --check`.
- Final gates:
  - `uv run python -m unittest discover -s tests -v`
  - `uv build`
  - clean-wheel smoke in temporary Python 3.11 and 3.12 environments if available
  - CLI help and all new machine-readable command smokes
  - `git diff --check`
- Leave the worktree clean with the six exact commits above following the pre-existing prompt commit.
- Final response must list commit SHAs, exact test/build results, skips, and remaining non-software qualification boundaries. Do not claim completion for a check you did not run.
