# Desktop distribution and update strategy

Research current as of 2026-08-09. This decision assumes Mun 1.0 targets Apple
Silicon macOS and x86-64 Linux, with an Electron GUI that treats the Mun CLI as
its local API.

## Decision

Ship the GUI with `electron-builder`, not Electron Forge. Electron recommends
Forge generally, but Mun specifically needs AppImage packaging and Linux
updates; `electron-builder` supports DMG, ZIP, AppImage, update metadata,
differential AppImage updates, and GitHub Releases in one release pipeline.

The Electron app must invoke a bundled, version-matched Mun engine. It must not
discover `mun`, Python, or FFmpeg from `PATH` and must not install anything
globally. The backend is a PyInstaller **one-folder** executable placed under
Electron's `extraResources`, alongside a pinned FFmpeg executable. PyInstaller
includes the Python interpreter and dependencies, and its documentation
recommends proving one-folder mode before one-file mode. One-folder also avoids
extracting a large PyTorch application to a temporary directory on every run.

The machine interface is a versioned CLI mode, for example
`mun api --protocol 1`: newline-delimited JSON requests on stdin, progress and
results on stdout, diagnostics on stderr, and a startup handshake containing
engine and protocol versions. Electron's main process locates it from
`process.resourcesPath` and starts it with an argument array and no shell. The
sandboxed renderer receives only narrow, validated preload APIs; it never gains
Node or process-spawning access.

## Release artifacts

| Target | Required artifact | Included engine |
| --- | --- | --- |
| Apple Silicon macOS | signed and notarized DMG plus ZIP update artifact | arm64 Python, PyTorch with CPU/MPS, FFmpeg |
| x86-64 Linux | AppImage | x86-64 Python, CPU PyTorch, FFmpeg |
| CLI users | PyPI wheel/sdist; Homebrew formula on macOS | installed independently, without Electron |

Build each native artifact on its target OS. PyInstaller output is OS- and
architecture-specific, and AppImage must be built on Linux (or in its Linux
builder image). Pin Electron, Node dependencies, Python, PyTorch, Transformers,
FFmpeg source revision/configuration, and the build container/toolchain.

Use the electron-builder AppImage static runtime toolset only after the exact
pinned version passes the supported Linux matrix. Its documentation currently
labels the new static runtime beta; until it proves reliable, test and document
the legacy FUSE2 behavior and extraction fallback rather than claiming
universal Linux support.

### GPU runtimes

Do not put CUDA and ROCm in the base AppImage and do not run `pip` inside the
installed application. They are large, mutually exclusive stacks with strict
driver, GPU, kernel, operating-system, ROCm, and PyTorch compatibility.

Publish CPU, CUDA, and ROCm engines as separate, immutable runtime packs. A pack
contains the whole one-folder Mun engine, its matching PyTorch build, and
FFmpeg; its manifest records protocol version, platform/architecture, PyTorch
and accelerator version, supported driver/OS range, SHA-256, size, and the
hardware matrix it passed. The base app contains the CPU engine. The runtime
manager may recommend and download one compatible GPU pack after detection and
explicit confirmation, then unpack to a versioned application-data directory,
run `mun doctor --json` plus a tiny inference smoke test, and atomically switch
the selected-runtime pointer. Keep the previously working pack until the new
one passes. On failure, select the previous pack automatically.

This keeps inference local while permitting network downloads of executable
runtimes and Hugging Face models. Runtime downloads need the same release
provenance and checksums as app artifacts. Models remain in Mun's existing
user-data model directory, outside the app and runtime packs, so application
updates neither duplicate nor remove them. Model files and runtime packs are
separate lifecycle domains.

PyTorch's installer explicitly selects CPU, CUDA, or ROCm builds, and AMD's
matrix restricts ROCm support to named Linux, kernel, GPU, ROCm, and PyTorch
combinations. A generic “AMD GPU” or “NVIDIA GPU” promise is therefore not a
valid release claim; only tested manifest combinations are supported.

### FFmpeg

Bundle `ffmpeg` and `ffprobe`; do not depend on `PATH` for desktop use. Build a
pinned LGPL-only configuration without `--enable-gpl` or `--enable-nonfree`.
Publish the corresponding FFmpeg source archive, configure command, changes,
copyright notices, and relinking information next to every release as required
by FFmpeg's distribution checklist. Treat codec availability as the exact
capability of that build, not “every media format.”

## Signing and notarization

For macOS, sign every nested Mach-O binary and library in the Python engine and
FFmpeg before signing the outer Electron app with a Developer ID Application
certificate, hardened runtime, secure timestamp, and least-privilege
entitlements. Do not use `com.apple.security.get-task-allow`. Have
electron-builder submit with Apple's `notarytool` integration and staple the
ticket. The release job must fail unless these independent checks pass:

1. `codesign --verify --deep --strict --verbose=2 Mun.app`
2. `spctl --assess --type execute --verbose=4 Mun.app`
3. `xcrun stapler validate Mun.app` and the DMG
4. a Gatekeeper launch and transcription smoke test on a clean macOS host

Use App Store Connect API-key credentials in CI secrets, never repository or
artifact contents. The Mac App Store is not a 1.0 target: direct Developer ID
distribution avoids adding App Sandbox constraints before the local engine and
model-download workflows have been validated there.

Linux has no equivalent platform signing gate. Publish SHA-256/SHA-512 sums,
an SBOM, and release provenance for the AppImage and runtime packs, and verify
the updater metadata checksum before installation.

## Updates and rollback

Publish desktop releases and electron-builder metadata to GitHub Releases via
`electron-updater`. Use two feeds: stable (`latest`) and opt-in beta. Check on
startup and periodically, show release notes and download size, download in the
background, and ask before restarting. Never update while a transcription or
model/runtime mutation is active. macOS requires both DMG and ZIP targets for
Squirrel.Mac update metadata; Linux uses the AppImage target and embedded
blockmap for differential downloads.

Roll out desktop updates in stages. The electron-builder documentation is
explicit that withdrawing a bad staged release requires a **higher** version;
there is no dependable automatic rollback of an already replaced Electron app.
Therefore:

- stop the feed and publish a fixed higher version for an emergency rollback;
- keep at least the two previous signed/notarized installers directly
  downloadable for manual recovery;
- preserve config, models, transcripts, and runtime packs across app updates;
- migrate persisted state only with backward-compatible or explicitly
  reversible migrations; and
- keep runtime-pack activation separately atomic, health-checked, and
  automatically reversible as described above.

Do not couple Electron, runtime-pack, and model updates into one transaction.
The app manifest declares the supported engine protocol range; an incompatible
pack is never activated and the bundled engine remains the fallback.

## Separate CLI installation

Keep the existing Python package as the canonical headless product. Publish the
same versioned core and API contract to PyPI and install it in an isolated
environment with `uv tool install mun`. Publish a Homebrew formula for macOS
that depends on FFmpeg and installs the Python CLI in its own environment.
Document hardware-specific PyTorch installation from the official selector;
the CLI must not silently replace a user's accelerator stack.

Desktop and standalone CLI installations may coexist and share the model store,
but they do not share executables or Python environments. The GUI always uses
its compatible bundled/runtime-pack engine. An advanced preference may point
to an external `mun` only after a protocol handshake, and is not required for
1.0.

## Release gates

- Reproducible native builds produce DMG, ZIP, AppImage, CLI distributions,
  checksums, SBOMs, notices, and FFmpeg corresponding source.
- A packaged-app test proves the Electron main process can start the bundled
  engine from a path containing spaces and non-ASCII characters, cancel it,
  survive malformed JSON, and preserve valid stdout framing.
- Clean-host tests cover install, first launch, CPU/MPS transcription, model
  persistence across update, interrupted update, and previous-installer
  recovery.
- CUDA and ROCm packs ship only for combinations in the tested compatibility
  matrix; detection errors fall back to CPU.
- macOS signing, notarization, stapling, Gatekeeper, and update installation all
  pass on the final artifacts; Linux AppImage launch and self-update pass on
  every claimed distribution.
- A release may not advertise rollback as automatic for the Electron shell.
  Automatic rollback is limited to runtime-pack activation.

## Rejected alternatives

- **Electron Forge:** Electron's general recommendation, but it does not meet
  Mun's AppImage and Linux updater requirement as directly as electron-builder.
- **System Python/FFmpeg for the GUI:** smaller artifact, but unsuitable for
  non-technical users and creates uncontrolled version drift.
- **PyInstaller one-file engine:** slower startup and temporary extraction of a
  very large native dependency tree without a distribution benefit inside an
  already packaged Electron app.
- **One AppImage containing CPU, CUDA, and ROCm:** largest download for every
  user and incompatible runtime assumptions.
- **Installing GPU wheels into the application at runtime:** mutates the shipped
  environment, complicates recovery, and makes support state non-reproducible.
- **Mac App Store first:** adds sandbox/signing constraints without improving
  the agreed direct-download destination.

## Primary sources

- Electron: [application packaging](https://www.electronjs.org/docs/latest/tutorial/application-distribution/),
  [Forge overview](https://www.electronjs.org/docs/latest/tutorial/forge-overview),
  [code signing](https://www.electronjs.org/docs/latest/tutorial/code-signing),
  [security checklist](https://www.electronjs.org/docs/latest/tutorial/security),
  and [`process.resourcesPath`](https://www.electronjs.org/docs/latest/api/process)
- electron-builder: [application contents and `extraResources`](https://www.electron.build/docs/contents/),
  [AppImage](https://www.electron.build/appimage/),
  [macOS configuration](https://www.electron.build/mac/),
  [notarization](https://www.electron.build/docs/notarization/), and
  [updates and staged rollout](https://www.electron.build/docs/features/auto-update/)
- Apple: [notarizing macOS software before distribution](https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution)
- PyInstaller: [bundling behavior and platform specificity](https://pyinstaller.org/en/stable/operating-mode.html)
- PyTorch: [platform-specific installation selector](https://pytorch.org/get-started/locally/)
  and [MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html)
- AMD: [ROCm compatibility matrix](https://rocm.docs.amd.com/en/develop/compatibility/compatibility-matrix.html)
- FFmpeg: [license and binary-distribution checklist](https://ffmpeg.org/legal.html)
- Hugging Face: [cache location controls](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)
- uv: [isolated tool installation](https://docs.astral.sh/uv/concepts/tools/)
