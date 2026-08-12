# Bounded transcript replay

Accepted for Mun 1.0 on 2026-08-12.

`mun replay <result.json>` verifies a completed canonical transcript result
against the recorded source, model, runtime, device, precision, and operation
parameters. Replay is bounded evidence. It is not a promise that model inference
is deterministic.

## Outcomes

- `exact_match`: canonical identity bytes and every available export projection
  match. This is the deterministic fake-runtime qualification outcome.
- `projection_match`: TXT, SRT, and VTT projections match, but canonical record
  metadata differs.
- `within_tolerance`: an opt-in tolerance data file admits normalized text and
  timestamp differences for an otherwise exact replay tuple.
- `semantic_drift`: replay completed but exceeded the permitted comparison.
- `environment_mismatch`: the model repository, immutable revision, artifact
  digest, runtime name/version, device, precision, or recorded host environment
  changed. Tolerances cannot override this outcome.
- `artifact_unavailable`: the source or installed pinned model cannot be used.
- `unsupported_replay`: the result or tolerance schema cannot be replayed.

Source existence and SHA-256 are checked before inference. The complete
model/runtime tuple and stable runtime environment are also checked before
inference. A model or runtime change can therefore never qualify as an exact,
projection, or tolerated match.

The JSON response includes an exact unified diff of replayed transcript text for
semantic comparisons. Exit status is zero for the three matching outcomes, two
for unsupported replay, and one for every other non-match.

## Fixture tiers

The deterministic tier uses `FakeSpeechRuntime` and asserts exact canonical
identity plus exact available export bytes. No tolerance is supplied or reported,
and this tier runs in the normal offline unit suite.

The live-model qualification tier is opt-in. It requires a locally installed,
pinned model and fixture source. Its normalization and timing limits are loaded
from `fixtures/replay/live-tolerances.json`. The file currently requires NFKC,
case-folded, whitespace-collapsed text with zero word error and permits at most
50 ms timestamp movement. These are explicit qualification bounds, never a claim
of deterministic inference.

Normal tests do not download models. Set `MUN_REPLAY_LIVE_MODEL` only when the
pinned qualification model and its fixture are already installed. Without it,
the live gate reports an exact unittest skip.

## Usage

```console
mun replay transcripts/source.json
mun replay transcripts/source.json --source recordings/source.wav
mun replay transcripts/source.json --tolerances fixtures/replay/live-tolerances.json
```
