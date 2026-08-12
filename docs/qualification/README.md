# Exact-tuple qualification records

`mun qualify` converts a completed local-run manifest into one versioned,
unsigned JSON record. It does not execute a model, sign evidence, approve a
support claim, or publish anything.

```sh
mun qualify path/to/run-manifest.json -o qualification.json
```

Fixture paths in the manifest are relative to the manifest. Mun reads the exact
bytes and records their SHA-256 and length. The generated record also binds:

- Mun version and revision;
- runtime name, version, and runtime-pack manifest digest;
- model repository, immutable revision, artifact digest, and model-manifest digest;
- OS, kernel, architecture, and Python version;
- physical device, backend, driver/runtime, accelerator target, requested and
  effective device, precision, and physical memory;
- cold load/end-to-end timing, every warm sample, its median and RTF;
- peak process and accelerator memory when measurable; and
- one outcome for every advertised capability.

See `tests/fixtures/qualification/run-manifest.json` for the input shape. The
text fixtures beside it are synthetic Apache-2.0 project test data. They test
the record harness only and are not speech-quality fixtures or physical
qualification evidence.

## Status and claim rules

- `eligible`: metadata or compatibility evidence only. Setting
  `physical_execution` to false can never produce `tested`.
- `tested`: the exact physical tuple ran, the run passed, all advertised
  capability rows are present and passed, and cold/warm measurements exist.
- `failed`: physical execution or an advertised capability failed.
- `unsupported`: the tuple is outside the supported matrix.

Missing, failed, non-physical, or expired rows block an advertised `tested`
claim. `missing_tested_claims` is the release-gating seam for checking exact
tuple-digest and capability pairs. Publication tooling must use it before
displaying reviewed records as tested.

Records expire at `expires_at`, and immediately when any material tuple field
or tuple-bound manifest/artifact digest changes. Generate a new record after a
Mun, runtime, model artifact, OS, driver, device, precision, or other tuple
change. Patch carry-forward still requires the smoke procedure defined by the
hardware policy.

## Trust boundary

The output is deliberately marked `unsigned` and `local_record`. It is local
evidence supplied by the operator, not an attestation. Review, signature,
acceptance of regressions, release gating, and publication remain external.
Qualification establishes runtime compatibility and observed execution only.
It does not establish transcript accuracy, comparative model quality, model
safety, or suitability for a language, domain, or person.
