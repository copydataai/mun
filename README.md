# Mun

Mun is a local-first command-line tool that turns batches of audio and video into text with speech models from Hugging Face. Media and transcripts remain on your computer; the network is used only to search for and download models.

> **Alpha:** the plain-transcription path is the first supported milestone. Timestamped subtitles and Whisper translation are included. Speaker diarization is planned but not yet implemented.

## Requirements

- macOS 13+ on Apple Silicon, or a current x86-64 Linux distribution
- Python 3.11 or 3.12
- [FFmpeg](https://ffmpeg.org/) (`ffmpeg` and `ffprobe` must be on `PATH`)
- Enough disk and memory for your selected model

On macOS:

```sh
brew install ffmpeg uv
```

On Ubuntu or Debian:

```sh
sudo apt install ffmpeg pipx
```

## Install

From a checkout:

```sh
uv tool install .
mun doctor
```

For development:

```sh
uv sync
uv run mun --help
```

CUDA and ROCm need the PyTorch build appropriate for your system. Follow the [official PyTorch installation selector](https://pytorch.org/get-started/locally/); Mun detects the backend but does not modify its own environment.

## First run

Run `mun` in a terminal for the guided workflow. On first use it recommends a pinned multilingual model, shows its download size and license, and asks before downloading anything.

The same workflow is scriptable:

```sh
mun models search whisper
mun models download openai/whisper-small
mun transcribe interview.m4a
```

Transcripts go to `./transcripts/` by default. Existing output is skipped unless `--overwrite` is supplied.

## Batch transcription

Pass any mixture of files and directories. Directories are scanned recursively; hidden directories and symlinked directories are skipped.

```sh
mun transcribe recordings/ meeting.mp4 voice-note.m4a
mun transcribe --input-list files.txt --format txt --format json
mun transcribe recordings/ --format srt --format vtt
```

FFmpeg converts each source to private temporary mono audio. Temporary audio is removed after success, failure, or cancellation. One speech model is loaded and inference runs sequentially so concurrent files do not duplicate model memory.

### Language and English translation

Language selection and translation require a compatible multilingual Whisper-family model:

```sh
mun transcribe spanish.m4a --language Spanish --translate
```

Translation preserves both outputs, for example `spanish.original.txt` and `spanish.en.txt`. Without `--language`, compatible Whisper models choose the spoken language automatically; this is not exposed as a reliable detected-language metadata field because Transformers does not standardize that output.

### Output formats

- `txt` — default plain transcript
- `json` — versioned structured transcript and segments
- `srt` — SubRip subtitles; enables timestamps
- `vtt` — WebVTT subtitles; enables timestamps

Machine-readable single-file output:

```sh
mun transcribe audio.wav --stdout --format json > transcript.json
```

Machine-readable batch summary:

```sh
mun transcribe recordings/ --summary-json > result.json
```

Progress and warnings are written to stderr, so redirected stdout stays valid.

## Models

Mun stores complete pinned snapshots in its own model directory. It does not delete or mutate Hugging Face's global cache.

```sh
mun models search --limit 20
mun models download OWNER/MODEL
mun models list
mun models info OWNER/MODEL
mun models remove OWNER/MODEL
```

Search results have honest compatibility labels:

- `tested` — a pinned revision verified on named hardware in Mun's reviewed catalog
- `metadata-compatible` — Hugging Face metadata says ASR + Transformers; loading is the real validation
- `unsupported` — wrong pipeline or runtime

Model-card metrics are not treated as comparable accuracy scores. Catalogued models remain `metadata-compatible` until a live device test is recorded; other untested models are marked `unverified` because datasets, preprocessing, languages, and metrics differ.

### Remote model code

Remote repository code is disabled by default. A model that needs it requires `--trust-remote-code`, displays a warning, and is pinned to an immutable commit. This executes third-party Python locally; inspect and trust the repository before opting in.

Gated models use the standard Hugging Face login:

```sh
hf auth login
```

Mun never stores your token in its configuration.

## Offline mode

After downloading a model:

```sh
mun transcribe audio.wav --offline
mun models search --offline
mun config set offline true
```

Offline search shows only Mun's built-in tested catalog. Downloads are rejected.

## Configuration

CLI flags override the user-level TOML file.

```sh
mun config show
mun config set model openai/whisper-small
mun config set output_dir transcripts
mun config set device auto
mun config reset
```

Supported keys are `model`, `model_dir`, `output_dir`, `device`, and `offline`.

## Troubleshooting

Run:

```sh
mun doctor
mun doctor --json
```

It checks Python, FFmpeg, PyTorch, the detected device, disk space, and installed models while redacting the home-directory prefix. Mun collects no telemetry.

If MPS hits an unsupported PyTorch operation, retry on CPU:

```sh
mun transcribe audio.wav --device cpu
```

If a batch contains a bad file, Mun continues with the remaining files, reports failures, and exits nonzero. Shared failures such as a model that cannot load stop before processing. Completed output is preserved; incomplete output is never renamed into place.

## Development

```sh
uv run python -m unittest discover -s tests -v
uv build
```

Normal tests do not download models. A live transcription smoke test is intentionally manual because even small models add network time and hundreds of megabytes.

## Scope

Mun does not provide cloud inference, a web server, transcript editing, summarization, speaker recognition, training, fine-tuning, or speech-model conversion. Speaker diarization is the next optional capability once its gated model and dependency flow can be kept separate from plain transcription.

A GUI may eventually wrap the same core workflow. There is no GUI dependency or implementation scaffold today.

## License

Mun is licensed under Apache-2.0. Downloaded models retain their own licenses; review the model details before use.
