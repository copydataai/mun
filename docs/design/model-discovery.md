# Model discovery and resource guidance

Status: accepted for Mun 1.0 on 2026-08-09.

## Decision

Mun uses a task-first model setup for non-technical users. Model setup asks three
plain-language questions before presenting one recommendation:

1. Task: transcribe, translate to English, or identify speakers.
2. Spoken language: auto-detect, a selected language, or multilingual.
3. Priority: balanced, best accuracy, lowest memory, or fastest.

The recommendation names the model, download size, estimated memory, compatible
device, capabilities, license or gating requirement, and evidence status. The
user can accept it or open the complete model library to compare alternatives.

The Electron application and interactive CLI use the same recommendation inputs
and machine-readable result. Neither client implements its own ranking policy.

## Evidence language

- **Eligible**: metadata and Mun's compatibility rules indicate the exact model,
  runtime, device, precision, and capability tuple may work. It is not a tested
  claim.
- **Tested**: Mun has qualifying physical evidence for that exact tuple.
- **Installed**: the pinned artifacts are present and verified locally. This does
  not imply every device or capability is tested.
- **Gated**: acquisition requires the user to accept upstream terms and
  authenticate with Hugging Face.
- **Not measured**: Mun has no qualifying speed or memory measurement. It never
  substitutes popularity or an estimate as measured performance.

Resource estimates and fit scores must be labeled as estimates. Hugging Face
metadata is discovery input only; a successful pinned runtime load is the
compatibility check.

## Advanced library

The complete library is secondary to task-first setup. It supports search,
device filtering, comparison, installation, validation, update, and removal.
Each model exposes its immutable revision, runtime, precision, disk and memory
requirements, available performance evidence, capabilities, license, gating,
and device-specific compatibility.

Removal previews the exact artifacts and reclaimed disk space. Shared artifacts
are retained while another installed model or runtime references them.

## Failure behavior

Discovery or recommendation failure does not start a download. Installation is
transactional: a failed download or validation leaves no model marked installed
and preserves any previous usable revision. Gated models explain the required
upstream action without accepting terms on the user's behalf.

Before an installation is atomically finalized, Mun writes a versioned artifact
manifest containing every downloaded file's normalized relative path, byte
length, and SHA-256, plus the source repository, immutable revision,
installation timestamp, `trust_remote_code` choice, and an aggregate manifest
digest. The model metadata file and manifest file are the only allowed
untracked files. Any other added file, including a later cache file, produces
`unexpected_file`; cache data present during installation is recorded and
verified like every other artifact.

Mun verifies this manifest before the first Transformers load. Verification
reports `verified`, `missing`, `modified`, `unexpected_file`, or
`unsafe_remote_code`. An older installation without a manifest reports the
typed `manifest_missing` result with reinstall guidance rather than crashing.
Missing and changed required files fail closed. The aggregate digest is retained
for later provenance records.

Artifact hashes establish only that local bytes match the recorded installation.
They cannot prove model safety, parser safety, absence of malicious behavior, or
semantic quality. An installation with `trust_remote_code` is always described
as unsafe, never tested or safe, even when all bytes verify.

## Prototype verdict

The accepted concept was **C — Task-first setup** from
`prototype/model-discovery` at commit `9d275b1`. The prototype remains
throwaway evidence and is not merged into the product branch.
