# J-Code remediation — Mun partial-commit replay

You are the managed J-Code remediator in `/Users/josesanchez/Developer/public/mun-jcode-challenges-2026-08-12`. Do not launch nested agents. A fresh independent full suite failed:

- `test_public_batch_recovers_from_every_durable_boundary_and_repeated_resume`, boundary `partial_commit`
- `resume_batch_journal` called `write_result_outputs`
- `MunError: Output already exists: .../out/source.json`

Fix the real correctness bug test-first. Resume from a partial commit must inspect and verify already committed projections against the journal-bound machine result/digests, preserve correct files, reject conflicting or unverifiable files, write only missing projections safely, and remain idempotent on repeated resume. Never use blanket overwrite that could replace unrelated/conflicting output. Add deterministic regression coverage for verified existing projection, conflicting existing projection, missing remainder, and repeated resume.

Create exactly one commit: `fix: resume verified partial commits`

Run focused test repeatedly enough to expose order/state leakage, then full unittest discovery, `uv build`, CLI smoke, and `git diff --check`. Leave a clean worktree and report exact results.