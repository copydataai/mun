# Hardware validation and performance policy

Status: accepted for Mun 1.0 on 2026-08-09.

## Decision

Mun support is an evidence claim about an exact tuple, not a claim about every device that exposes the same backend. A validation record identifies:

- Mun, runtime, model-artifact revision, FFmpeg, OS, kernel, architecture, and Python versions;
- device name, backend, driver/runtime versions, accelerator target (CUDA compute capability or ROCm `gfx` target), physical RAM, and VRAM or MPS recommended working-set size;
- requested and effective device and precision; and
- capability, fixture, result, timing, memory, fallback, and failure data.

The hardware-evidence states used by this policy are:

- `eligible`: the tuple satisfies the runtime pack's requirements and the current vendor compatibility matrix, but Mun has not run it;
- `tested`: the exact tuple passed this policy on physical hardware;
- `failed`: the exact tuple was run and failed, with a diagnostic;
- `unsupported`: it is outside Mun's published OS, architecture, runtime-pack, or vendor matrix.

Virtual machines, emulators, metadata, compilation, and a successful device probe cannot produce `tested`. A result expires for release gating when Mun, the model artifact, runtime pack, OS major, driver major, or effective precision changes. Patch-version results may be carried forward only after the smoke suite passes again.

## Mun 1.0 matrix

Every release must retain ordinary unit/build coverage on macOS and Linux. Hardware qualification is a separate, explicit job and must contain these physical rows:

| Claim | Required release row | Required devices | Initial precision policy |
| --- | --- | --- | --- |
| Apple Silicon | M1 with 8 GiB unified memory on macOS 14's latest patch; one current M-series Mac with at least 16 GiB on Mun's newest supported macOS | `mps`, plus `cpu` fallback | qualified adapter precision; Transformers baseline is FP16 on MPS and FP32 on CPU |
| Linux CPU | x86-64, 4 cores, 8 GiB RAM on Ubuntu 22.04's latest patch; repeat smoke tests on Ubuntu 24.04 | `cpu` | FP32 for Transformers; INT8 only for a separately qualified runtime artifact |
| NVIDIA | Tesla T4 (SM 7.5, 16 GiB) on Ubuntu 22.04's latest patch | `cuda`, plus `cpu` diagnostic comparison | FP16; BF16, INT8, and other quantization require their own tuple |
| AMD | Radeon RX 7900 XTX (`gfx1100`, 24 GiB) on an AMD-supported Ubuntu 22.04 point release | PyTorch's `cuda` API backed by HIP, plus `cpu` diagnostic comparison | FP16; BF16, INT8, and other quantization require their own tuple |

The exact OS/runtime pair must appear in the vendor matrix at release time. NVIDIA's driver must meet the runtime pack's CUDA compatibility requirement. AMD qualification is limited to GPU/OS/PyTorch combinations in AMD's current matrix. Legacy Vega is therefore not a Mun 1.0 support claim unless a chosen runtime's current vendor matrix includes the exact target and it passes physically.

The T4 and RX 7900 XTX are initial reproducible anchors, not assertions that neighboring products work. A new GPU family, lower-memory SKU, architecture, precision, runtime, or model artifact starts as `eligible` and gets a separate record. Release is blocked if any platform advertised as `tested` lacks an unexpired passing row; an unavailable lab machine means the claim is removed or remains `eligible`, not waived.

## Workloads and procedure

The repository keeps redistributable, content-addressed fixtures and their expected language/text metadata:

1. a 15-second English smoke clip;
2. a 60-second multilingual/cross-language set covering transcription, language detection, timestamps, and English translation where the model supports them; and
3. a 10-minute mixed speech/silence performance file, built once and identified by hash.

Each runtime uses the same pinned semantic model and fixture revisions. A derived/quantized artifact is identified separately and compared with the pinned FP32 CPU Transformers result. Audio decoding is included in end-to-end measurements; an additional inference-only measurement may be reported but cannot replace it.

For each tuple:

1. Start offline with an empty output directory and verify the effective backend and precision.
2. In a fresh process, measure model-load time and the smoke clip. This is the cold result.
3. Warm up once, then run the 10-minute fixture three times in one process. Synchronize the accelerator immediately before and after each timed interval. Report all samples and their median; do not discard slow samples.
4. Run every claimed capability once, then run a recursive batch containing valid audio, video, silence, a corrupt file, and colliding basenames. Verify atomic outputs, continued processing, nonzero batch status, and cleanup.
5. Repeat the smoke clip with the machine offline. CPU comparisons use the same model, fixture, decoder, and Mun revision.

Wall time uses a monotonic high-resolution clock. Real-time factor (RTF) is end-to-end wall seconds divided by decoded audio seconds; lower is better. Report cold start separately, median steady-state RTF, and batch throughput. `real-time` may be displayed only at RTF <= 1.00; `fast` only at RTF <= 0.50. Slower results may still be functionally supported but must show the observed RTF rather than a speed adjective.

## Resource reporting

Every sample records absolute peak process RSS and available system memory at start. CUDA and ROCm additionally record PyTorch peak allocated and peak reserved device bytes. MPS records sampled current/driver-allocated bytes and `recommended_max_memory`; because Apple Silicon uses unified memory, MPS and process values are shown separately and never added. PyTorch allocator numbers are labelled as allocator-visible, not total machine or device use.

Publish model guidance from observations, not parameter-count estimates:

- discrete accelerators: recommended VRAM is the observed peak reserved bytes plus 25%, rounded up to GiB; recommended host RAM is peak RSS plus 25%;
- Apple unified memory: recommended RAM is the greater of peak RSS and peak MPS driver allocation, plus 25% and 2 GiB for the OS, rounded up to GiB;
- CPU: recommended RAM is peak RSS plus 25% and 2 GiB for the OS, rounded up to GiB.

The report includes disk bytes for the installed artifact. Energy, temperature, and vendor-tool memory are optional diagnostics and never replace the portable fields above.

## Pass, regression, and fallback rules

A tuple passes when all of the following hold:

- load, smoke, offline, batch, and every advertised capability complete without fallback, crash, NaN/Inf, invalid timestamps, incomplete final output, or temporary-file residue;
- each accelerated/quantized transcript is no worse than the FP32 CPU reference by more than 1.0 absolute WER percentage point on the fixed fixtures;
- peak accelerator allocation is at most 80% of physical VRAM or MPS recommended working-set size, and peak RSS is at most 80% of physical RAM (values are assessed independently on unified memory);
- all three measured runs complete; and
- against the previous passing record for the same tuple, median RTF regresses by no more than 15% and measured peak memory by no more than 10%, unless the release notes accept and explain the regression.

The first passing record establishes, rather than compares against, the performance baseline. Absolute RTF does not decide functional support; it decides only the displayed speed tier.

`--device auto` may retry a whole file once on CPU after an accelerator load, unsupported-operation, or out-of-memory failure, provided no final output was committed. The result and summary must say that CPU was the effective device and preserve the accelerator diagnostic. An explicit device never falls back. Mun does not enable PyTorch's operation-level MPS fallback by default because it obscures the effective backend and makes evidence incomparable. A fallback result proves only the CPU tuple, never the failed accelerator tuple.

## Evidence format and acceptance criteria

The validator emits one versioned JSON record per tuple and a short Markdown table generated from those records. JSON is the source of truth and contains raw samples, medians, hashes, environment fields, diagnostics, and a `tested`, `failed`, or `eligible` outcome. Secrets, usernames, home paths, and unrelated device inventory are excluded.

This decision is implemented when:

- a deterministic validation command writes the versioned record described above;
- fixtures are pinned by hash and their redistribution terms are documented;
- release automation refuses a `tested` claim without an unexpired passing record;
- model search/details show observed RAM/VRAM, RTF, effective precision, and the exact tested tuple, or say `not measured`;
- `doctor` distinguishes unavailable, vendor-ineligible, eligible/unverified, and tested tuples; and
- release notes link the records and disclose missing physical rows, accepted regressions, and fallbacks.

## Tradeoffs

Exact-tuple claims are narrower than saying “CUDA” or “ROCm supported,” but they remain truthful across fast-moving driver and framework matrices. Fixed fixtures favor repeatability over broad accuracy evaluation; they detect hardware/runtime regressions and do not rank models. The 25% memory headroom and 15% timing threshold are operational guardrails, not universal hardware requirements, so raw observations remain visible.

## Primary references

- PyTorch documents MPS availability and device use in [MPS backend notes](https://docs.pytorch.org/docs/stable/notes/mps.html), explicit CPU fallback in [MPS environment variables](https://docs.pytorch.org/docs/stable/mps_environment_variables.html), and allocator/working-set measurements in the [`torch.mps` API](https://docs.pytorch.org/docs/stable/mps.html).
- Apple documents that Apple Silicon uses a [unified memory model](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-intel-and-amd-gpus), which is why CPU and MPS allocation figures are not summed.
- PyTorch's [CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html) require synchronization for accurate timing and distinguish allocated from reserved memory. PyTorch also documents that its memory profiler cannot see every external allocation in [Understanding CUDA Memory Usage](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html).
- PyTorch documents that [HIP reuses the `torch.cuda` interfaces](https://docs.pytorch.org/docs/stable/notes/hip.html), while AMD publishes the authoritative [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) and [PyTorch compatibility](https://rocm.docs.amd.com/en/docs-7.2.2/compatibility/ml-compatibility/pytorch-compatibility.html).
- NVIDIA publishes the driver/runtime rules in [CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
- Python documents the monotonic high-resolution [`time.perf_counter`](https://docs.python.org/3/library/time.html#time.perf_counter) and process resource fields from [`resource.getrusage`](https://docs.python.org/3/library/resource.html#resource.getrusage); the validator must normalize platform-specific RSS units before writing bytes.
