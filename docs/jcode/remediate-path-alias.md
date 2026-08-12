# J-Code remediation — canonicalize resume artifact paths

You are the managed J-Code remediator in `/Users/josesanchez/Developer/public/mun-jcode-challenges-2026-08-12`. Do not launch nested agents.

The orchestrator reproduced the exact root cause. In `_partial_commit`:

- committed file digest equals the journal artifact digest;
- reconstructed canonical JSON bytes equal the committed file exactly;
- journal artifact path is `/var/folders/.../out/source.json`;
- resumed destination is `/private/var/folders/.../out/source.json` because macOS resolves `/var` through `/private/var`;
- `resume_artifacts` uses raw path strings, so verified digest lookup fails solely on the alias.

Fix path identity consistently at the persistence/verification seam. Use a deterministic canonical path representation for journal artifacts, committed/uncommitted paths, destinations, and lookups while preserving user-facing relative-path behavior and not following unsafe symlinks outside the already validated destination root. Maintain backward compatibility with existing journals containing macOS alias spellings: normalize safely on load/lookup rather than making them unrecoverable. Do not weaken digest verification.

Add regression tests for `/var` vs `/private/var` aliasing and a generic symlink/alias mismatch where safe, plus conflicting bytes. Run:

- the focused test directly;
- exact `uv run python -m unittest discover -s tests -v` three fresh times from a separate clean shell;
- `uv build`, CLI smoke, `git diff --check`.

Create exactly one commit: `fix: canonicalize partial resume paths`

Leave a clean worktree and report exact results.