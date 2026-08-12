# Mun

**Private transcription for audio and video, on your own computer.**

Mun turns one file—or a whole folder—into plain transcripts, JSON, or subtitles with a speech model you choose. Your media is processed locally and your transcripts stay yours. The network is used only to find and download models from Hugging Face.

```sh
mun transcribe interview.m4a
```

Mun is intentionally small: a command-line tool, a local model, and files you can keep or move anywhere.

> **Alpha:** plain transcription, timestamped subtitles, and Whisper translation work today. Speaker diarization is planned, but not implemented.

## Why Mun

- **Private by design.** Audio and video are never sent to a transcription service. Mun has no telemetry.
- **Made for real folders.** Pass files and directories together; Mun discovers readable media recursively, preserves its structure, and continues past transcription failures.
- **Safe and inspectable.** Models are pinned to immutable revisions, existing transcripts are not overwritten by default, and JSON results record model and runtime provenance.

Canonical JSON also records typed trust. Source media is `untrusted_bytes`, verified model snapshots are `verified_artifact`, remote-code snapshots are `unsafe_remote_code`, and machine text is `untrusted_model_output`. Every transcript result is `ineligible` for autonomous agent use. Human correction changes the content label to `untrusted_content` and adds review metadata. It does not remove the taint.

## Quick start

### 1. Install Mun from a checkout

Mun requires:

- macOS 13+ on Apple Silicon, or a current x86-64 Linux distribution
- Python 3.11 or 3.12
- [FFmpeg](https://ffmpeg.org/) (`ffmpeg` and `ffprobe` on `PATH`)
- Enough disk and memory for the model you select

macOS:

```sh
brew install ffmpeg uv
uv tool install --python 3.11 .
```

Ubuntu or Debian:

```sh
sudo apt install ffmpeg pipx
pipx install .
```

Then verify the installation:

```sh
mun doctor
```

CUDA and ROCm require the PyTorch build appropriate for your system. Use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/); Mun detects the backend but does not change its own environment.

### 2. Transcribe

Run `mun` with no arguments for the guided workflow. On first use, Mun shows a pinned multilingual model, its download size, and its license before asking permission to download it.

Or use the scriptable workflow:

```sh
mun models download openai/whisper-small
mun transcribe interview.m4a
```

Models requiring repository Python code need an immutable snapshot plus an explicit, revision-specific acknowledgement:

```sh
mun models download owner/model --trust-remote-code --acknowledge-remote-code
```

For automation, `remote_code_acknowledgement` may be configured only as the exact `owner/model@immutable-revision` value. These installations remain unsafe and untested, and their transcript results remain tainted and agent-ineligible.

Transcripts are written to `./transcripts/`. Existing outputs are reused only when the complete requested projection set includes canonical JSON whose digest, source identity, operation parameters, model identity, and derived projections all validate. Unrelated files, invalid artifacts, and incomplete output sets block that source with a recoverable nonzero result. Pass `--overwrite` to replace only the output paths explicitly requested by the current command.

## Common workflows

### Transcribe a folder

```sh
mun transcribe recordings/
```

Directories are scanned recursively. Hidden directories and symlinked directories are skipped by default.

### Mix files and folders

```sh
mun transcribe meetings/ interview.mp4 voice-note.m4a
```

### Create text and structured JSON

```sh
mun transcribe recordings/ --format txt --format json
```

### Create subtitles

```sh
mun transcribe recordings/ --format srt --format vtt
```

SRT and VTT automatically enable timestamps.

### Translate speech into English

Language selection and translation require a compatible multilingual Whisper-family model:

```sh
mun transcribe spanish.m4a --language Spanish --translate
```

Mun preserves both variants, such as `spanish.original.txt` and `spanish.en.txt`. Without `--language`, compatible Whisper models choose the spoken language automatically. Mun does not claim detected-language metadata because Transformers does not expose it consistently.

### Process large CPU batches in parallel

```sh
mun transcribe recordings/ --jobs 4
mun transcribe recordings/ --jobs 4 --benchmark
```

`--jobs` uses parallel workers for CPU inference. Accelerator runs remain single-process to avoid duplicating model memory. `--benchmark` reports elapsed time and file throughput to stderr.

### Use Mun in a script

One source media file to stdout:

```sh
mun transcribe audio.wav --stdout --format json > transcript.json
```

A machine-readable batch result:

```sh
mun transcribe recordings/ --summary-json > result.json
```

Progress and warnings go to stderr, so redirected stdout remains valid.

### Apply human corrections without changing the machine result

Create a correction-set JSON that names the exact parent `result_digest` and
targets segments by transcript kind, segment ID, and SHA-256 digest of the exact
original segment text. Then validate and export a separate corrected record:

```sh
mun review apply transcript.json corrections.json -o transcript.corrected.json
mun review render transcript.json --view machine --format txt
mun review render transcript.json --corrections corrections.json --view corrected --format srt
```

`review apply` refuses to overwrite an existing output. Parent, target, or
original-text mismatches fail closed. The canonical machine JSON is never
rewritten. Corrected TXT, SRT, and VTT retain the machine segment timings, while
corrected JSON records `reviewed` or `unreviewed`, the parent and correction-set
identities, and a distinct export digest.

### Understand deletion receipts

Model removal prints a receipt with the exact managed path attempted, the app-visible result, and estimated bytes removed. Transient download cleanup uses the same scoped receipt contract. Receipts do not claim universal erasure. They explicitly exclude backups, APFS snapshots, swap, filesystem remnants, transcript exports, and third-party caches. Mun refuses deletion requests outside its managed model directory.

### Verify a transcript with bounded replay

```sh
mun replay transcripts/interview.json
```

Replay checks the source digest and exact model/runtime tuple before inference,
then reports a typed JSON outcome. Live-model tolerances are opt-in data files,
not deterministic claims. See [bounded transcript replay](docs/design/reproducibility.md).

## Outputs

| Format | Purpose |
|---|---|
| `txt` | Plain transcript; the default |
| `json` | Versioned transcript result with segments, diagnostics, and provenance |
| `srt` | SubRip subtitles with timestamps |
| `vtt` | WebVTT subtitles with timestamps |

For each source, Mun renders every requested projection into a private staging directory on the destination filesystem, hashes and validates the staged files, checks the complete destination set, and then commits files in deterministic path order. Existing destinations are refused unless `--overwrite` is explicit. Staging is removed after normal failure or cancellation.

Each attempt persists `<name>.receipt.json` with one of `completed`, `cancelled`, `failed_before_commit`, or `partial_commit`. A partial commit records the exact committed and uncommitted paths. Earlier completed sources remain in place if a later source fails or is cancelled. This is not multi-file filesystem atomicity: a failure during the ordered commit can leave the exact partial set described by the receipt. Success exits 0; failed, partial, and cancelled workflows exit 1; command-line usage errors exit 2.

Batch summary counts distinguish newly processed files from `reused_verified`, `conflict`, `incomplete_output_set`, and `overwrite_required` plans. A verified reuse can succeed without loading the speech model. Conflicts and incomplete sets are recoverable by moving the existing files or rerunning with `--overwrite`.

Artifact validation establishes internal consistency only. A matching digest proves that the JSON has not changed relative to its own recorded digest, not that the transcript was honestly produced, that the recorded provenance is true, or that a trusted producer signed it. Mun does not currently provide producer signatures.

Correction content and notes are untrusted data. A human review state records a
workflow state only. It does not claim truth, authenticity, honesty, accuracy,
consent, custody, or reviewer identity.

FFmpeg converts source media into temporary mono audio when needed. Temporary audio is removed after success, failure, or cancellation. Media already suitable for inference can bypass conversion.

## Models

Mun stores complete, pinned model snapshots in its own model directory. It does not delete or mutate Hugging Face's global cache.

```sh
mun models search whisper
mun models download OWNER/MODEL
mun models list
mun models info OWNER/MODEL
mun models remove OWNER/MODEL
```

Search is discovery, not qualification. It reports repository metadata such as model ID, library, gating, and popularity, but does not label a model `eligible` or `tested` without the exact revision, runtime, device, precision, and requested capability tuple required to support that claim.

`gated` is visible in search results when downloading requires upstream terms and Hugging Face authentication. `installed` appears in `mun models list` only after pinned artifacts are present locally.

Mun does not present model-card metrics as comparable accuracy scores. Datasets, languages, preprocessing, and metrics differ too much for that claim.

### Remote model code

Remote repository code is disabled by default. A model that requires it needs `--trust-remote-code`; Mun displays a warning and pins the model to an immutable commit. This executes third-party Python locally, so inspect and trust the repository before opting in.

Gated models use the standard Hugging Face login:

```sh
hf auth login
```

Mun never stores your Hugging Face token in its configuration.

## Offline mode

Once a model is installed, transcription can run with Hugging Face network access disabled:

```sh
mun transcribe audio.wav --offline
mun models search --offline
mun config set offline true
```

Offline search uses Mun's built-in catalog. Model downloads are rejected.

## Configuration

CLI flags override the user-level TOML configuration:

```sh
mun config show
mun config set model openai/whisper-small
mun config set output_dir transcripts
mun config set device auto
mun config reset
```

Supported keys are `model`, `model_dir`, `output_dir`, `device`, and `offline`.

## Troubleshooting

```sh
mun doctor
mun doctor --json
```

`mun doctor` checks Python, FFmpeg, PyTorch, the detected device, disk space, and installed models. Home-directory paths are redacted.

If MPS encounters an unsupported PyTorch operation, retry on CPU:

```sh
mun transcribe audio.wav --device cpu
```

## Scope

Mun does not provide cloud inference, a web server, an interactive transcript editor, summarization, speaker recognition, training, fine-tuning, or model conversion. Its review commands only validate immutable correction overlays and render machine or corrected views. Speaker diarization is the next optional capability only if its gated model and dependencies can remain separate from plain transcription.

A graphical interface may eventually wrap the same workflow. Mun has no GUI dependency or implementation scaffold today.

## Development

```sh
uv run python -m unittest discover -s tests -v
uv build
```

Normal tests do not download models. Live transcription remains an explicit manual smoke test because even small models add network time and hundreds of megabytes.

## License

Mun is licensed under Apache-2.0. Downloaded models keep their own licenses; review model details before use.
