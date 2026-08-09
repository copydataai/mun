# Model runtime and compatibility contract

Decision for Mun 1.0, researched 2026-08-09.

## Decision

Mun 1.0 supports three local runtime adapters behind one transcript contract:

1. **Transformers/PyTorch** is the required coverage runtime. It admits Hugging
   Face repositories that can be loaded by `AutoModelForCTC`,
   `AutoModelForTDT`, or `AutoModelForSpeechSeq2Seq` with their matching Auto
   processor. This covers Whisper, CTC families such as Wav2Vec2/HuBERT/WavLM,
   and other Transformers speech-sequence-to-sequence models.
2. **faster-whisper/CTranslate2** is an optional optimized Whisper runtime. It
   admits only CTranslate2-formatted Whisper repositories, or a Mun-derived
   artifact converted from a pinned Transformers-compatible Whisper revision.
   It is supported on CPU and NVIDIA GPU; its published wheels do not establish
   Apple-GPU or AMD-GPU support.
3. **whisper.cpp** is an optional native Whisper runtime. It admits only
   whisper.cpp GGML artifacts that Mun recognizes, including Mun-derived
   artifacts with recorded provenance. It supplies the native path for Apple
   Metal and may cover CPU, CUDA, Vulkan, and ROCm only where Mun ships and tests
   the corresponding binary.

The adapters are admitted because they add distinct value: broad Hugging Face
model coverage, a mature Python Whisper optimizer, and a portable native
Whisper implementation with first-class Apple support. They share no model
loader abstraction beyond the small operations Mun actually needs: inspect,
load, transcribe, and report capabilities.

Optimum ONNX is not a 1.0 runtime. It officially covers CTC and Whisper/speech-
to-text models, but duplicates the admitted families and adds another export,
provider, quantization, and test matrix without closing a required hardware or
capability gap. MLX, Core ML conversion outside whisper.cpp, OpenAI Whisper,
and arbitrary repository-specific runtimes are also outside the 1.0 contract.
Add a runtime only when it unlocks a tested model family or materially improves
a supported hardware target.

## Compatibility is not a repository label

`pipeline_tag=automatic-speech-recognition` and `library_name` are search hints.
They cannot prove that a repository contains complete files, loads under Mun's
pinned dependency versions, runs on a particular device, or provides a given
timestamp/language feature. Mun must never describe every ASR-tagged repository
as supported.

Compatibility is keyed by this tuple:

```
(source repository, immutable source revision, runtime, runtime version,
 artifact/quantization, OS/architecture, device, capability)
```

A converted model is a distinct artifact, not another revision of the source.
Its record includes the source repository and commit, converter name/version,
converter options, artifact hashes, and license. Mun never converts or chooses
a different runtime silently.

## User-visible states

| State | Exact claim Mun may make |
| --- | --- |
| `candidate` | Hub metadata suggests an admitted adapter may load it; not yet validated. |
| `eligible` | Static inspection found the files/config and model family required by one named adapter. |
| `installed` | A complete artifact is stored locally at an immutable revision and hashes/metadata were recorded. |
| `load-verified` | The named runtime loaded the artifact offline and completed Mun's smoke input on the recorded device. |
| `tested` | A pinned tuple passed the maintained golden fixtures for every advertised capability on named hardware. |
| `failed` | A recorded adapter/device attempt failed, with its error and versions; other tuples remain usable. |
| `unsupported` | Static inspection proves that no admitted adapter can load the artifact. |

Search returns `candidate`, `eligible`, or `unsupported`; it must not return
`tested` from model popularity or a family name. Download ends at `installed`.
First load can promote only the attempted runtime/device tuple to
`load-verified`. Only reviewed compatibility-matrix evidence grants `tested`.
Unknown and failed are not synonyms for unsupported.

Consequently, the current single global model `status` should become an
installation-integrity status plus per-runtime/device validation records. A
CUDA out-of-memory error, unsupported MPS operation, or missing optional
runtime must not mark the downloaded snapshot invalid. Only corruption,
incomplete download, or hash failure invalidates the installation itself.

## Adapter eligibility

### Transformers/PyTorch

Static eligibility requires all of the following:

- ASR pipeline metadata, or an explicit user-selected local repository;
- a configuration recognized by one of Transformers' ASR AutoModel classes;
- a matching processor/tokenizer/feature extractor that loads without network
  access after installation;
- weights accepted by Mun's pinned Transformers version; and
- no remote code for a supported claim.

`trust_remote_code` remains an explicit unsafe escape hatch. Such a model can
be installed and run at the user's request, pinned to a commit, but stays
`candidate/unsafe` and cannot enter Mun's tested catalog until its code is
vendored or the model works through an upstream loader.

### faster-whisper/CTranslate2

Static eligibility requires a recognized CTranslate2 Whisper artifact plus its
tokenizer and preprocessor data. A standard Transformers Whisper repository is
eligible for **conversion**, not directly for this adapter. Mun keeps the
source snapshot and derived artifact separately. faster-whisper exposes
language detection, segment and word timestamps, transcription, translation,
and multiple compute types, but Mun advertises each only after its smoke test.

### whisper.cpp

Static eligibility requires a recognized whisper.cpp model artifact and a
Mun-managed, versioned `whisper-cli` binary for the target platform. A standard
Transformers/OpenAI Whisper repository is eligible for conversion, not direct
loading. Backend claims are build-specific: a CPU binary does not imply Metal,
CUDA, Vulkan, or ROCm support. Mun records the binary hash and build features
alongside validation evidence.

## Capability contract

Capabilities are probed and stored, never inferred solely from a repository
name:

- `transcribe`: required for every load-verified model.
- `segment_timestamps` and `word_timestamps`: separate capabilities.
  Transformers documents timestamps for pure CTC and Whisper, not other
  speech-sequence-to-sequence models.
- `language_detection`, `language_selection`, and `translate_to_english`:
  separate capabilities, normally Whisper-only.
- `long_audio`: requires a fixture longer than the model window; a short-file
  success does not establish it.
- `precision` and `quantization`: artifact/runtime/device properties, not model
  marketing labels.

Diarization is a separate local processing stage and never an ASR runtime
capability. It may enrich the shared transcript schema without changing which
adapter produced the words.

The Electron application uses the CLI's versioned JSON protocol and does not
load a model runtime itself. Runtime progress and errors go to stderr or
structured events; the final transcript/result is valid JSON on stdout. This
keeps one compatibility implementation for both interfaces.

## Runtime selection

An explicit `--runtime` always wins. Otherwise Mun chooses, in order: a tested
installed tuple for the selected device; a load-verified installed tuple; then
Transformers as the coverage fallback. It may recommend conversion or another
adapter, but must ask before downloading or deriving artifacts. Selection and
the reason are included in JSON output.

## Minimum release evidence

Every catalog entry records model and artifact revisions, runtime/dependency
versions, OS and architecture, device, compute type, peak memory, real-time
factor, fixtures, capabilities, and test date. Tests compare normalized text
and timestamp tolerances, not byte-identical output across runtimes. CPU and
Apple-Silicon lanes are automated; signed reproducible reports may cover CUDA
and ROCm until dedicated runners exist.

## Primary sources

- [Transformers ASR pipeline registration](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/__init__.py) names the accepted PyTorch AutoModel classes.
- [Transformers ASR pipeline implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/automatic_speech_recognition.py) distinguishes CTC, Whisper, and other seq2seq behavior and timestamp limits.
- [Transformers ASR guide](https://huggingface.co/docs/transformers/main/tasks/asr) demonstrates processor sampling requirements and pipeline inference.
- [faster-whisper documentation](https://github.com/SYSTRAN/faster-whisper/blob/master/README.md) documents Hub-hosted CTranslate2 models, conversion from Transformers Whisper, language detection, compute types, and timestamps.
- [CTranslate2 installation](https://opennmt.net/CTranslate2/installation.html) and [hardware support](https://opennmt.net/CTranslate2/hardware_support.html) define published wheel and GPU limits.
- [whisper.cpp documentation](https://github.com/ggml-org/whisper.cpp) documents its custom artifacts, conversion, supported build backends, Metal path, and CLI input contract.
- [Optimum ONNX Runtime model support](https://huggingface.co/docs/optimum-onnx/en/onnxruntime/package_reference/modeling) documents its supported CTC and speech-sequence-to-sequence families.
- [PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html) and [ROCm semantics](https://docs.pytorch.org/docs/main/notes/hip.html) define the device APIs used by the coverage runtime.
