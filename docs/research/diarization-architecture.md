# Diarization architecture and trust boundary

Decision for Mun 1.0, researched 2026-08-09.

## Decision

Mun 1.0 offers speaker diarization as an explicit, optional local stage powered
by `pyannote.audio` 4.x and the pinned
[`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
pipeline. The research baseline is `pyannote.audio` 4.0.7 and model revision
`3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`; the release manifest must replace
these with the exact versions and hashes that pass Mun's hardware suite.

This is the only supported diarization pipeline in 1.0. Mun does not present
arbitrary Hub diarization repositories as compatible, does not use a hosted
pyannote service, and does not make diarization part of an ASR runtime. The
pipeline is a separate stage that enriches the canonical transcript result.

`community-1` is the narrowest choice that meets the contract: it runs locally,
returns ordinary overlap-preserving diarization and a separate exclusive
diarization intended for transcript reconciliation, supports offline loading,
and is maintained for `pyannote.audio` 4.x. The Python package is MIT licensed;
the model is CC BY 4.0 and gated by a user agreement.

## Dependency and process boundary

- Plain transcription never imports `pyannote`, checks for its model, asks for
  authentication, or downloads diarization assets.
- CLI installations expose diarization through an optional dependency group.
  The packaged desktop installs it in a versioned diarization runtime pack, not
  in the base ASR engine.
- The CLI owns orchestration and the transcript schema. A worker process owns
  the `pyannote.audio` import, pipeline load, and inference. This contains
  dependency conflicts and accelerator memory, but is **not** a security
  sandbox: it runs with the user's filesystem and process permissions.
- Mun decodes each source once to timeline-preserving, mono 16 kHz PCM using its
  bundled converter and passes waveform plus sample rate to the worker. It does
  not trim silence or time-stretch the audio. ASR and diarization therefore use
  the same zero-based source timeline and do not depend on pyannote's decoder.
- The worker returns only diarization intervals, labels, progress, structured
  diagnostics, runtime/model provenance, and resource observations. It never
  formats user outputs.

The optional pack is justified despite reusing PyTorch: pyannote 4.x currently
requires a newer and much larger dependency set than Mun's base contract. A
separate pack keeps plain transcription installable and prevents one optional
feature from determining the base engine's release cadence.

## Timestamp and speaker integration

Mun consumes both pipeline outputs for different purposes:

1. `speaker_diarization` is authoritative for detected turns, overlap, and
   duration. It may contain more than one active speaker.
2. `exclusive_speaker_diarization` is used only to attach a single speaker to
   an ASR word or segment. It must not replace the ordinary turns or erase
   overlap statistics.

Pipeline seconds are converted to the nearest non-negative integer millisecond,
with half milliseconds rounded upward, then clamped to the decoded media
duration. Empty intervals are discarded. Upstream labels are normalized to
`speaker_1`, `speaker_2`, and so on by earliest ordinary-turn start, breaking a
tie by the upstream label. IDs remain meaningful only within one result.

For the original transcript variant:

- A timed word receives an additive `speaker_id` when one exclusive speaker has
  positive overlap. The greatest overlap wins; a tie uses normalized speaker
  ID. No overlap leaves the field absent.
- A segment receives the speaker with greatest exclusive overlap across its
  interval, using the same tie and no-overlap rules. Mun does not split, merge,
  retime, or rewrite ASR segments. Word labels preserve mid-segment speaker
  changes when word timestamps exist.
- Translation variants do not receive inferred speaker alignment. They retain
  speaker fields only when the producing runtime supplied aligned timestamps.

`assigned_speech_ms` is the sum of original ASR segment durations assigned to a
speaker. `speaking_ms` is the union of that speaker's ordinary diarization
turns. `overlap_ms` is the union of time during which two or more distinct
speakers are active in the ordinary diarization. All three are computed after
millisecond normalization. Detected speakers with no assigned ASR segment are
still retained with `assigned_speech_ms: 0`.

The result provenance gains an additive diarization stage containing the
pipeline repository and immutable revision, artifact hashes, pyannote/PyTorch
versions, requested and effective device, precision, speaker-count constraints,
and whether exclusive turns were available. Mun stores no voice embeddings and
makes no claim about a person's identity.

## Acquisition, authentication, and offline operation

The model remains outside application installers. Acquisition follows this
sequence:

1. Explain that diarization is optional, name the model and CC BY 4.0 license,
   and link to the publisher's gated agreement. Mun cannot accept the agreement
   for the user.
2. Authenticate through the official Hugging Face Hub client. CLI users may use
   `hf auth login`; the Electron flow invokes the client's browser/device login.
   Mun never accepts a token as a command-line argument.
3. Resolve the requested model to a commit and download that immutable snapshot
   into Mun's model directory. Record repository, commit, expected filenames,
   file hashes, size, license, and agreement URL before marking it installed.
4. Load only from the installed local path. Inference workers set
   `HF_HUB_OFFLINE=1` and `PYANNOTE_METRICS_ENABLED=0` before importing the
   runtime. They make no Hub version check and send no usage telemetry.

Hugging Face's client owns any persisted credential in its normal token store;
Mun does not copy it into Mun configuration, model manifests, logs, IPC,
transcripts, crash reports, or provenance. An `HF_TOKEN` supplied by the user is
honored by the Hub client but is never echoed. Logout and token deletion remain
Hugging Face operations. The token is needed for acquisition, not inference.

If the snapshot is already complete, a disconnected machine can diarize without
a token. If it is absent, offline mode fails immediately with an actionable
`diarization_model_not_installed` diagnostic rather than attempting the network.
Gated-repository failures distinguish missing authentication, unaccepted terms,
authorization failure, and network failure without including response bodies or
credentials.

## Trust boundary

The user explicitly trusts locally installed model code and artifacts, but Mun
keeps that trust narrow and visible:

- 1.0 allowlists the official `community-1` repository and expected artifact
  layout. Every install is pinned and hashed. A changed or additional file makes
  a new unverified artifact tuple.
- `trust_remote_code` is always false. Python modules, native libraries, shell
  hooks, and repository-defined runtime plugins are rejected. Supporting a new
  pipeline requires a reviewed Mun/runtime release, not a per-run flag.
- Model weight deserialization is still trusted input; pinning and hashing prove
  identity, not safety. The UI states that the optional model executes inside a
  local worker with the user's permissions.
- Audio and transcripts never cross the process boundary except between Mun's
  local CLI and local worker. Network access is permitted only to the Hub client
  during explicit search, authentication, and download.
- `speaker-diarization-precision-2`, `community-1-cloud`, and every other hosted
  endpoint are excluded because they send media or derived content off-device.

CC BY attribution for the model, the MIT notice for `pyannote.audio`, all
transitive notices, and the pinned artifact provenance appear in the application
notices/SBOM. Mun does not bundle or mirror the gated weights; each user obtains
them from the original repository after accepting its current terms.

## Hardware claims

The diarization stage follows Mun's exact-tuple hardware policy:

| Platform | Mun 1.0 diarization claim |
| --- | --- |
| Apple Silicon macOS | Local CPU is the baseline. MPS is unverified and `auto` routes this stage to CPU. |
| Linux x86-64 CPU | Local CPU is the baseline and must pass the release fixtures. |
| NVIDIA CUDA | Eligible because upstream documents `pipeline.to(torch.device("cuda"))`; advertised as tested only for a passing physical tuple. |
| AMD ROCm | Unverified. PyTorch's HIP compatibility does not prove this pipeline; `auto` routes this stage to CPU. |

ASR may still use MPS or ROCm while diarization uses CPU. Per-stage requested and
effective devices prevent the result from implying otherwise. An explicit MPS
or ROCm request is rejected as unsupported in 1.0. Under `auto`, a failing
accelerated job follows the hardware policy's single pre-commit CPU retry and
retains the accelerator diagnostic; an explicit device never falls back.

## Failure behavior

Diarization is opt-in and recoverable. A missing optional pack, missing/gated
model, load failure, unsupported device, out-of-memory error, invalid intervals,
or inference failure does not discard a successful transcription. The file is
`partial`, its `speakers` array is empty, segment/word speaker fields are absent,
and a stable diagnostic explains the failed stage. The batch continues and
exits nonzero under the accepted transcript contract.

No-speaker and one-speaker outputs are successful when structurally valid.
Silence produces no speakers. A user-supplied exact, minimum, or maximum speaker
count is passed through and recorded in provenance; invalid or contradictory
bounds fail validation before inference.

Outputs and their temporary files are committed only after schema validation.
Cancellation terminates the worker, removes temporary diarization artifacts,
and preserves already committed results from other files.

## Rejected alternatives

- **WhisperX** already combines faster-whisper, language-specific forced
  alignment, and pyannote diarization. Adopting it would create a second ASR
  orchestration path, restrict diarization to Whisper, and silently add alignment
  model acquisition. Mun instead uses pyannote directly with timestamps supplied
  by whichever admitted ASR runtime produced them.
- **NVIDIA NeMo diarization** is capable and Apache 2.0, but its cascaded path
  adds manifests, several model components, Hydra configuration, and a much
  broader NVIDIA-oriented runtime. It does not close a 1.0 requirement that the
  smaller pyannote stage leaves open.
- **pyannote Precision-2/cloud** reports better accuracy but requires a pyannoteAI
  API key and hosted processing. It violates Mun's local-only boundary.
- **Legacy `speaker-diarization-3.1`** remains useful for older pyannote versions,
  but Community-1 is the maintained 4.x pipeline and supplies exclusive turns
  without a second reconciliation implementation.
- **A generic diarization plugin API** is deferred. One supported pipeline does
  not justify an extension interface, and arbitrary Hub repositories cannot
  inherit this pipeline's compatibility or trust claims.

## Acceptance criteria

This decision is implemented when all of the following are true:

- plain-transcription installation and execution pass with no pyannote package,
  model, authentication, or network access;
- an explicit install flow handles gated terms/authentication, pins and hashes
  the official snapshot, records its license, and never logs or stores a token
  outside the Hugging Face client;
- a fresh offline process loads only the installed path with Hub offline mode
  and telemetry disabled, then produces regular and exclusive turns;
- deterministic tests cover label normalization, millisecond rounding, overlap,
  speaker durations, word/segment assignment, ties, silence, and missing timing;
- failure tests prove that diarization errors create a partial transcript while
  preserving successful ASR output and continuing the batch;
- provenance and `doctor` report the exact runtime/model/device tuple and never
  call MPS or ROCm diarization tested without physical evidence;
- packaged macOS and Linux clean-host tests exercise CPU diarization with the
  bundled converter, while the qualifying CUDA row exercises its runtime pack;
  and
- notices/SBOM include pyannote and model licenses/attribution, and no gated model
  weights are present in an installer or source distribution.

## Open evidence risks

- No current upstream support statement establishes Community-1 inference on
  MPS or ROCm. Those claims stay absent until Mun's physical matrix passes and
  upstream dependency versions are pinned.
- The model repository uses binary PyTorch weights rather than a documented
  safetensors-only path. Immutable hashes reduce supply-chain drift but do not
  make deserialization untrusted-safe.
- Gated terms and artifact contents can change independently of Mun. Every
  release must re-check the agreement, license metadata, repository revision,
  expected files, and attribution text.
- Diarization quality varies by domain, channel layout, language, noise, and
  speaker count. Functional hardware fixtures cannot support a universal
  accuracy claim; accuracy must be reported only for named benchmark fixtures.
- A worker subprocess is a dependency/failure boundary, not an OS sandbox.
  Strong isolation would require platform-specific sandboxing and is outside
  the Mun 1.0 claim.

## Primary sources

- The [Community-1 model card](https://huggingface.co/pyannote/speaker-diarization-community-1)
  documents the gate, CC BY 4.0 license, local execution, ordinary and exclusive
  outputs, speaker-count controls, CUDA transfer, and offline loading.
- The [`pyannote.audio` repository](https://github.com/pyannote/pyannote-audio)
  documents installation, telemetry fields and controls, current local/cloud
  paths, and is distributed under its [MIT license](https://github.com/pyannote/pyannote-audio/blob/main/LICENSE).
- The [`pyannote.audio` 4.0 release](https://github.com/pyannote/pyannote-audio/releases/tag/4.0.0)
  introduced Community-1 integration and exclusive diarization in the 4.x line.
- Hugging Face documents browser/device login and persisted credentials in its
  [authentication API](https://huggingface.co/docs/huggingface_hub/package_reference/authentication),
  immutable repository downloads in [`snapshot_download`](https://huggingface.co/docs/huggingface_hub/package_reference/file_download#huggingface_hub.snapshot_download),
  and disconnected operation with
  [`HF_HUB_OFFLINE`](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hf-hub-offline).
- The [WhisperX repository](https://github.com/m-bain/whisperX) documents its
  faster-whisper, forced-alignment, and pyannote coupling.
- NVIDIA's [cascaded diarization configuration](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/configs.html)
  documents NeMo's manifest and multi-component configuration surface.
- PyTorch documents the [MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html)
  and that [ROCm uses `torch.cuda` interfaces](https://docs.pytorch.org/docs/stable/notes/hip.html);
  neither document establishes pyannote pipeline compatibility.
