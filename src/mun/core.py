from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version as package_version
import subprocess
import tempfile
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable

from .errors import MunError
from .models import InstalledModel, VerificationResult, verify_installed_model
from . import __version__
from .transcript import (
    SCHEMA_VERSION,
    Diagnostic,
    ExportArtifact,
    ExportReceipt,
    ExportReceiptState,
    Language,
    OperationParameters,
    OperationRecord,
    PreparedMediaRecord,
    ReuseStatus,
    SourceRecord,
    TranscriptResult,
    TranscriptSegment,
    TranscriptVariant,
    make_batch_result,
    make_provenance,
    make_trust,
    render_json,
    render_srt,
    render_txt,
    render_vtt,
)

MEDIA_EXTENSIONS = {
    ".aac", ".aiff", ".alac", ".avi", ".flac", ".m4a", ".mkv", ".mov", ".mp3",
    ".mp4", ".mpeg", ".mpg", ".oga", ".ogg", ".opus", ".wav", ".webm", ".wma",
}
@dataclass(frozen=True)
class MediaProbe:
    is_audio: bool
    is_whisper_ready: bool
    media_format: str | None
    codec: str | None
    sample_rate_hz: int | None
    channels: int | None


_FFPROBE_CACHE: dict[Path, MediaProbe] = {}
_BINARY_PATH_CACHE: dict[str, str | None] = {}
SOURCE_HASH_POLICY = "sha256_source_bytes"
BATCH_INTERRUPTION_FILE = "mun-batch-interruption.json"


class ExportCommitError(MunError):
    def __init__(self, committed_paths: list[Path], uncommitted_paths: list[Path]) -> None:
        super().__init__("Export commit failed after some projections were committed")
        self.committed_paths = committed_paths
        self.uncommitted_paths = uncommitted_paths


def _cached_binary_path(name: str, missing_error: str) -> str:
    if name not in _BINARY_PATH_CACHE:
        _BINARY_PATH_CACHE[name] = shutil.which(name)
    path = _BINARY_PATH_CACHE[name]
    if path is None:
        raise MunError(missing_error)
    return path


@dataclass(frozen=True)
class SourceMedia:
    source: Path
    relative: Path


@dataclass(frozen=True)
class Segment:
    text: str
    start: float | None
    end: float | None
    speaker: str | None = None


@dataclass(frozen=True)
class Transcript:
    text: str
    segments: list[Segment]
    language: str | None = None


@dataclass(frozen=True)
class TranscriptionOptions:
    language: str | None = None
    timestamps: bool = False
    translate: bool = False
    chunk_length: int = 30
    stride_length: int = 5
    device: str = "auto"


_WORKER_RUNTIME: Any | None = None
_WORKER_MODEL: InstalledModel | None = None
_WORKER_OPTIONS: TranscriptionOptions | None = None


def _init_batch_worker(model: InstalledModel, options: TranscriptionOptions) -> None:
    from .runtime import create_transformers_runtime

    global _WORKER_RUNTIME, _WORKER_MODEL, _WORKER_OPTIONS
    _WORKER_RUNTIME = create_transformers_runtime(model, options.device)
    _WORKER_MODEL = model
    _WORKER_OPTIONS = options


def _run_batch_worker(payload: tuple[int, SourceMedia]) -> tuple[int, TranscriptResult]:
    index, media = payload
    if _WORKER_RUNTIME is None or _WORKER_MODEL is None or _WORKER_OPTIONS is None:
        raise MunError("Batch worker is not initialized")
    return index, transcribe_source(_WORKER_RUNTIME, media, _WORKER_MODEL, _WORKER_OPTIONS)


def _media_size_or_zero(media: SourceMedia) -> int:
    try:
        return media.source.stat().st_size
    except OSError:
        return 0


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _source_record(media: SourceMedia, sha256: str | None = None) -> SourceRecord:
    return SourceRecord(media.source.name, str(media.relative), sha256=sha256 if sha256 is not None else _sha256_file(media.source))


def _cancelled_result(
    media: SourceMedia,
    model: InstalledModel,
    options: TranscriptionOptions,
    runtime: Any,
    reuse_status: ReuseStatus,
) -> TranscriptResult:
    source_sha256 = _sha256_file(media.source)
    return TranscriptResult(
        schema_version=SCHEMA_VERSION,
        status="cancelled",
        source=_source_record(media, source_sha256),
        transcripts=[],
        speakers=[],
        diagnostics=[Diagnostic("warning", "batch_cancelled", "Batch transcription was cancelled", "transcription", True)],
        provenance=_provenance(runtime, model, options),
        operation=_operation(runtime, options, source_sha256),
        reuse_status=reuse_status,
        trust=make_trust(model.trust_remote_code),
    )


def _persist_batch_interruption(
    output_dir: Path,
    results: list[TranscriptResult],
    queued_unstarted: list[SourceMedia],
    unfinished_unknown: list[SourceMedia] | None = None,
) -> None:
    payload = {
        **make_batch_result(results).to_dict(),
        "status": "cancelled",
        "queued_unstarted_sources": [
            {"name": item.source.name, "relative_path": str(item.relative)} for item in queued_unstarted
        ],
        "unfinished_unknown_sources": [
            {"name": item.source.name, "relative_path": str(item.relative)}
            for item in unfinished_unknown or []
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / BATCH_INTERRUPTION_FILE
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _operation(runtime: Any, options: TranscriptionOptions, source_sha256: str | None) -> OperationRecord:
    prepared = getattr(runtime, "prepared_media", None)
    if prepared is None:
        prepared = PreparedMediaRecord(
            used=False,
            sha256=source_sha256,
            media_format=None,
            codec=None,
            sample_rate_hz=None,
            channels=None,
            converter=None,
        )
    return OperationRecord(
        parameters=OperationParameters(
            language=options.language,
            timestamps=options.timestamps,
            translate=options.translate,
            chunk_length=options.chunk_length,
            stride_length=options.stride_length,
            requested_device=options.device,
            effective_device=runtime.info.effective_device,
            precision=runtime.info.precision,
        ),
        prepared_media=prepared,
        source_hash_policy=SOURCE_HASH_POLICY,
    )


def _matches_reuse_identity(
    result: TranscriptResult,
    media: SourceMedia,
    model: InstalledModel,
    options: TranscriptionOptions,
    source_sha256: str | None,
    artifact_sha256: str,
    runtime: Any,
) -> bool:
    operation = result.operation
    if result.result_digest is None or operation is None:
        return False
    parameters = operation.parameters
    provenance = result.provenance
    environment = provenance.runtime.environment
    info = runtime.info
    runtime_artifact_sha256 = getattr(runtime, "model_artifact_sha256", None)
    return (
        result.status == "completed"
        and result.source.name == media.source.name
        and result.source.relative_path == str(media.relative)
        and result.source.sha256 == source_sha256
        and provenance.model.repository == model.id
        and provenance.model.revision == model.revision
        and provenance.model.artifact_sha256 == artifact_sha256
        and runtime_artifact_sha256 == artifact_sha256
        and result.trust.model == ("unsafe_remote_code" if model.trust_remote_code else "verified_artifact")
        and provenance.runtime.name == info.name
        and provenance.runtime.version == info.version
        and environment is not None
        and (
            environment.python_version,
            environment.python_implementation,
            environment.operating_system,
            environment.machine,
        ) == (platform.python_version(), platform.python_implementation(), sys.platform, platform.machine())
        and provenance.requested_device == options.device
        and provenance.effective_device == info.effective_device
        and provenance.precision == info.precision
        and operation.source_hash_policy == SOURCE_HASH_POLICY
        and parameters.language == options.language
        and parameters.timestamps == options.timestamps
        and parameters.translate == options.translate
        and parameters.chunk_length == options.chunk_length
        and parameters.stride_length == options.stride_length
        and parameters.requested_device == options.device
        and parameters.effective_device == info.effective_device
        and parameters.precision == info.precision
    )


def _load_reusable_result(
    json_path: Path,
    expected_paths: list[Path],
    media: SourceMedia,
    model: InstalledModel,
    options: TranscriptionOptions,
    artifact_sha256: str,
    runtime: Any,
) -> TranscriptResult | None:
    try:
        result = TranscriptResult.from_json(json_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, KeyError):
        return None
    if not _matches_reuse_identity(
        result, media, model, options, _sha256_file(media.source), artifact_sha256, runtime
    ):
        return None
    for path in expected_paths:
        if path == json_path:
            continue
        format_name = path.suffix.lstrip(".")
        kind = "english_translation" if f".en.{format_name}" in path.name else (
            "original" if f".original.{format_name}" in path.name else None
        )
        projection = result
        if kind is not None:
            from .transcript import select_variant

            try:
                projection = replace(result, transcripts=[select_variant(result, kind)])
            except ValueError:
                return None
        try:
            expected_content = render_output(format_name, projection, media, model, result.provenance.effective_device)
            if path.read_text(encoding="utf-8") != expected_content:
                return None
        except (OSError, MunError):
            return None
    return replace(result, reuse_status="reused_verified")


def _blocked_result(
    media: SourceMedia,
    model: InstalledModel,
    options: TranscriptionOptions,
    reuse_status: ReuseStatus,
    code: str,
    message: str,
) -> TranscriptResult:
    runtime = _batch_provenance_runtime(options.device, options.device)
    source_sha256 = _sha256_file(media.source)
    return TranscriptResult(
        schema_version=SCHEMA_VERSION,
        status="partial",
        source=_source_record(media, source_sha256),
        transcripts=[],
        speakers=[],
        diagnostics=[Diagnostic("warning", code, message, "output", True)],
        provenance=_provenance(runtime, model, options),
        operation=_operation(runtime, options, source_sha256),
        reuse_status=reuse_status,
        trust=make_trust(model.trust_remote_code),
    )


def _probe_media(path: Path) -> tuple[bool, bool]:
    """Return (is_audio, is_ready_for_whisper).

    The second result is true when the input can be passed directly to the speech
    pipeline without a temporary conversion to 16 kHz mono PCM.
    """
    probe = _probe_media_details(path)
    return probe.is_audio, probe.is_whisper_ready


def _probe_media_details(path: Path) -> MediaProbe:
    key = path.resolve()
    cached = _FFPROBE_CACHE.get(key)
    if cached is not None:
        return cached

    ffprobe = _cached_binary_path("ffprobe", "FFprobe is not installed. Install FFmpeg and try again.")
    result = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,channels,sample_rate:format=format_name",
            "-of", "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        probe = MediaProbe(False, False, None, None, None, None)
        _FFPROBE_CACHE[key] = probe
        return probe

    is_audio = False
    is_whisper_ready = False
    media_format = None
    codec_name = None
    sample_rate = None
    channels = None
    try:
        payload = json.loads(result.stdout)
        streams = payload.get("streams", [])
        if isinstance(streams, list) and streams:
            stream = streams[0]
            is_audio = stream.get("codec_type") == "audio" or "codec_type" not in stream
            channels = int(stream.get("channels", 0))
            sample_rate = int(stream.get("sample_rate", 0))
            codec_name = str(stream.get("codec_name", "")) or None
            is_whisper_ready = is_audio and channels == 1 and sample_rate == 16_000 and str(codec_name).startswith("pcm")
            format_record = payload.get("format", {})
            if isinstance(format_record, dict):
                media_format = str(format_record.get("format_name", "")) or None
    except (TypeError, ValueError, json.JSONDecodeError):
        is_audio = False
        is_whisper_ready = False
        media_format = None
        codec_name = None
        sample_rate = None
        channels = None

    probe = MediaProbe(is_audio, is_whisper_ready, media_format, codec_name, sample_rate, channels)
    _FFPROBE_CACHE[key] = probe
    return probe


def _can_use_source_audio_directly(path: Path) -> bool:
    return _probe_media(path)[1]


def _infer_effective_device(requested_device: str) -> str:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise MunError(f"Could not load model for device detection: {exc}") from exc

    return detect_device(requested_device, torch)


def _batch_provenance_runtime(
    requested_device: str,
    effective_device: str,
    verified_model_artifact_sha256: str | None = None,
) -> Any:
    from .runtime import RuntimeInfo

    try:
        transformers_version = package_version("transformers")
    except PackageNotFoundError:
        transformers_version = None
    precision = "float16" if effective_device.startswith("cuda") else "float32"
    info = RuntimeInfo(
        name="transformers",
        version=transformers_version,
        requested_device=requested_device,
        effective_device=effective_device,
        precision=precision,
        model_type="transformers",
    )
    return SimpleNamespace(
        info=info,
        model_artifact_sha256=verified_model_artifact_sha256,
    )


def discover_media(
    raw_paths: Iterable[str],
    include_hidden: bool = False,
    probe: Callable[[Path], bool] | None = None,
) -> list[SourceMedia]:
    probe = probe or is_media
    discovered: list[SourceMedia] = []
    seen: set[Path] = set()
    for raw_path in raw_paths:
        selected = Path(raw_path).expanduser()
        if not selected.exists():
            raise MunError(f"Input does not exist: {selected}")
        if selected.is_dir():
            root = selected.resolve()
            candidates = [candidate for candidate in root.rglob("*") if candidate.is_file() and not candidate.is_symlink()]
            candidates.sort()
            for candidate in candidates:
                relative_inside = candidate.relative_to(root)
                if not include_hidden and any(part.startswith(".") for part in relative_inside.parts):
                    continue
                if candidate.suffix.lower() not in MEDIA_EXTENSIONS:
                    continue
                _add_media(discovered, seen, candidate, Path(root.name) / relative_inside, probe)
        elif selected.is_file():
            resolved = selected.resolve()
            _add_media(discovered, seen, resolved, Path(resolved.name), probe)
        else:
            raise MunError(f"Input is not a regular file or directory: {selected}")
    if not discovered:
        raise MunError("No readable audio or video files were found")
    return discovered


def is_media(path: Path) -> bool:
    return _probe_media(path)[0]


def _audio_input_path(source: Path, converted: Path) -> tuple[Path, bool]:
    """Return the path to pass to ASR and whether a temporary conversion was used."""
    if _can_use_source_audio_directly(source):
        return source, False
    _convert_media(source, converted)
    return converted, True


def load_pipeline(model: InstalledModel, requested_device: str = "auto") -> tuple[Any, str, str]:
    from .runtime import create_transformers_runtime

    runtime = create_transformers_runtime(model, requested_device)
    return runtime, runtime.info.effective_device, runtime.info.model_type


def detect_device(requested: str, torch_module: Any) -> str:
    from .runtime import detect_device as runtime_detect_device

    return runtime_detect_device(requested, torch_module)


def transcribe_media(
    speech_pipeline: Any,
    source: Path,
    model_type: str,
    options: TranscriptionOptions,
) -> tuple[Transcript, Transcript | None]:
    if hasattr(speech_pipeline, "transcribe"):
        return speech_pipeline.transcribe(source, options)
    needs_timestamps = options.timestamps
    if needs_timestamps and model_type not in {"whisper", "wav2vec2", "hubert", "wavlm"}:
        raise MunError(f"The selected {model_type} model cannot provide timestamps")
    if (options.language or options.translate) and model_type != "whisper":
        raise MunError("Language selection and English translation require a Whisper-family model")
    if _can_use_source_audio_directly(source):
        original = _run_pipeline(speech_pipeline, source, model_type, options, task="transcribe")
        translated = None
        if options.translate:
            translated = _run_pipeline(speech_pipeline, source, model_type, options, task="translate")
        return original, translated

    with tempfile.TemporaryDirectory(prefix="mun-") as temporary_directory:
        prepared_audio, _ = _audio_input_path(source, Path(temporary_directory) / "audio.wav")
        original = _run_pipeline(speech_pipeline, prepared_audio, model_type, options, task="transcribe")
        translated = None
        if options.translate:
            translated = _run_pipeline(speech_pipeline, prepared_audio, model_type, options, task="translate")
        return original, translated


def run_batch(
    media: list[SourceMedia],
    model: InstalledModel,
    output_dir: Path,
    formats: list[str],
    options: TranscriptionOptions,
    overwrite: bool,
    progress: Callable[[str], None],
    runtime: Any | None = None,
    jobs: int = 1,
) -> tuple[list[TranscriptResult], list[dict[str, str]]]:
    summaries: list[TranscriptResult] = []
    failures: list[dict[str, str]] = []
    used_bases: set[Path] = set()

    if jobs < 1:
        raise MunError("--jobs must be a positive integer")

    jobs = min(jobs, os.cpu_count() or 1)

    queued: list[tuple[int, SourceMedia, Path]] = []
    queued_status: dict[int, ReuseStatus] = {}
    queued_results: dict[int, TranscriptResult] = {}
    model_verification: VerificationResult | None = None
    reuse_runtime = runtime

    for index, item in enumerate(media, start=1):
        base = output_base(output_dir, item, used_bases)
        expected = output_paths(base, formats, options.translate)
        existing = [path for path in expected if path.exists()]
        if overwrite:
            queued.append((index, item, base))
            queued_status[index] = "overwrite_required" if existing else "queued"
            continue
        if not existing:
            queued.append((index, item, base))
            queued_status[index] = "queued"
            continue
        if len(existing) != len(expected):
            message = "Existing outputs form an incomplete projection set"
            queued_results[index] = _blocked_result(
                item, model, options, "incomplete_output_set", "incomplete_output_set", message
            )
            failures.append({"source": str(item.relative), "error": message})
            progress(f"[{index}/{len(media)}] blocked {item.source} (incomplete output set)")
            continue
        json_path = next((path for path in expected if path.suffix == ".json"), None)
        reused = None
        if json_path is not None:
            if model_verification is None:
                model_verification = verify_installed_model(model)
            if (
                model_verification.status in {"verified", "unsafe_remote_code"}
                and model_verification.artifact_digest is not None
            ):
                if reuse_runtime is None:
                    try:
                        reuse_effective_device = _infer_effective_device(options.device)
                        reuse_runtime = _batch_provenance_runtime(
                            options.device,
                            reuse_effective_device,
                            verified_model_artifact_sha256=model_verification.artifact_digest,
                        )
                    except MunError:
                        reuse_runtime = None
            if (
                model_verification.status in {"verified", "unsafe_remote_code"}
                and model_verification.artifact_digest is not None
                and reuse_runtime is not None
            ):
                reused = _load_reusable_result(
                    json_path,
                    expected,
                    item,
                    model,
                    options,
                    model_verification.artifact_digest,
                    reuse_runtime,
                )
        if reused is not None:
            queued_results[index] = reused
            progress(f"[{index}/{len(media)}] reused {item.source} (verified canonical JSON)")
            continue
        message = "Existing outputs cannot be verified for reuse"
        queued_results[index] = _blocked_result(item, model, options, "conflict", "output_conflict", message)
        failures.append({"source": str(item.relative), "error": message})
        progress(f"[{index}/{len(media)}] blocked {item.source} (output conflict)")

    if not queued:
        for index in sorted(queued_results):
            summaries.append(queued_results[index])
        return summaries, failures

    queued_count = len(queued)
    jobs = min(jobs, queued_count)

    if runtime is None and jobs > 1:
        effective_device = _infer_effective_device(options.device)
        if effective_device != "cpu":
            progress(f"[{jobs} workers requested but effective device is {effective_device}; falling back to single-threaded batch")
            jobs = 1
    else:
        effective_device = runtime.info.effective_device if runtime is not None else options.device

    if jobs > 1 and effective_device != "cpu":
        progress(f"[{jobs} workers requested but effective device is {effective_device}; falling back to single-threaded batch")
        jobs = 1

    if jobs == 1:
        runtime = runtime or load_pipeline(model, options.device)[0]
        effective_device = runtime.info.effective_device

    if runtime is None:
        fallback_runtime = _batch_provenance_runtime(options.device, effective_device)
    else:
        fallback_runtime = runtime

    if jobs == 1:
        for queued_position, (index, item, base) in enumerate(queued):
            progress(f"[{index}/{len(media)}] transcribing {item.source} ({effective_device}, {model.id})")
            try:
                result = transcribe_source(runtime, item, model, options)
                result = replace(result, reuse_status=queued_status[index])
                if result.status == "completed":
                    write_result_outputs(base, formats, result, options.translate, overwrite=overwrite)
                queued_results[index] = result
                progress(f"[{index}/{len(media)}] {result.status} {item.source}")
            except KeyboardInterrupt:
                queued_results[index] = _cancelled_result(
                    item, model, options, runtime, queued_status[index]
                )
                interruption_results = [queued_results[key] for key in sorted(queued_results)]
                _persist_batch_interruption(
                    output_dir,
                    interruption_results,
                    [entry[1] for entry in queued[queued_position + 1:]],
                )
                raise
            except ExportCommitError:
                queued_results[index] = TranscriptResult(
                    schema_version=SCHEMA_VERSION,
                    status="partial",
                    source=_source_record(item),
                    transcripts=[],
                    speakers=[],
                    diagnostics=[Diagnostic("error", "partial_commit", "Export commit was only partially completed", "export", False)],
                    provenance=_provenance(runtime, model, options),
                    operation=_operation(runtime, options, _sha256_file(item.source)),
                    reuse_status=queued_status[index],
                    trust=make_trust(model.trust_remote_code),
                )
                progress(f"[{index}/{len(media)}] partial {item.source} (export commit incomplete)")
            except Exception as exc:
                queued_results[index] = TranscriptResult(
                    schema_version=SCHEMA_VERSION,
                    status="failed",
                    source=_source_record(item),
                    transcripts=[],
                    speakers=[],
                    diagnostics=[Diagnostic("error", "transcription_failed", "Transcription failed", "transcription", False)],
                    provenance=_provenance(runtime, model, options),
                    operation=_operation(runtime, options, _sha256_file(item.source)),
                    reuse_status=queued_status[index],
                    trust=make_trust(model.trust_remote_code),
                )
                progress(f"[{index}/{len(media)}] failed {item.source}: {exc}")

        for index in sorted(queued_results):
            result = queued_results[index]
            summaries.append(result)
            if result.status == "failed":
                failures.append({"source": str(result.source.relative_path), "error": result.diagnostics[0].message if result.diagnostics else "failed"})
        return summaries, failures

    queued.sort(key=lambda entry: _media_size_or_zero(entry[1]), reverse=True)

    completed: dict[int, TranscriptResult] = {}
    base_by_index = {index: base for index, _, base in queued}
    media_by_index = {index: item for index, item, _ in queued}

    executor = ProcessPoolExecutor(max_workers=jobs, initializer=_init_batch_worker, initargs=(model, options))
    futures: dict[Any, int] = {}
    try:
        for index, item, _ in queued:
            futures[executor.submit(_run_batch_worker, (index, item))] = index
        for future in as_completed(futures):
            index = futures[future]
            item = media_by_index[index]
            progress(f"[{index}/{len(media)}] transcribing {item.source} ({effective_device}, {model.id})")
            try:
                _, result = future.result()
                result = replace(result, reuse_status=queued_status[index])
            except Exception:
                result = TranscriptResult(
                    schema_version=SCHEMA_VERSION,
                    status="failed",
                    source=_source_record(item),
                    transcripts=[],
                    speakers=[],
                    diagnostics=[Diagnostic("error", "transcription_failed", "Transcription failed", "transcription", False)],
                    provenance=_provenance(fallback_runtime, model, options),
                    operation=_operation(fallback_runtime, options, _sha256_file(item.source)),
                    reuse_status=queued_status[index],
                    trust=make_trust(model.trust_remote_code),
                )

            completed[index] = result
    except KeyboardInterrupt:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)

        for index in sorted(completed):
            result = completed[index]
            base = base_by_index[index]
            if result.status == "completed":
                try:
                    write_result_outputs(base, formats, result, options.translate, overwrite=overwrite)
                except ExportCommitError:
                    result = replace(
                        result,
                        status="partial",
                        transcripts=[],
                        diagnostics=[Diagnostic("error", "partial_commit", "Export commit was only partially completed", "export", False)],
                    )
            queued_results[index] = result

        unfinished = [(index, item) for index, item, _ in queued if index not in completed]
        for unfinished_index, unfinished_media in unfinished:
            queued_results[unfinished_index] = _cancelled_result(
                unfinished_media,
                model,
                options,
                fallback_runtime,
                queued_status[unfinished_index],
            )

        _persist_batch_interruption(
            output_dir,
            [queued_results[index] for index in sorted(queued_results)],
            [],
            [item for _, item in unfinished],
        )
        raise
    else:
        executor.shutdown(wait=True)

    ordered = sorted(completed)
    for index in ordered:
        result = completed[index]
        queued_results[index] = result

    for index in sorted(queued_results):
        result = queued_results[index]
        base = base_by_index.get(index)
        if result.status == "completed" and base is not None:
            try:
                write_result_outputs(base, formats, result, options.translate, overwrite=overwrite)
            except ExportCommitError:
                result = replace(
                    result,
                    status="partial",
                    transcripts=[],
                    diagnostics=[Diagnostic("error", "partial_commit", "Export commit was only partially completed", "export", False)],
                )
        summaries.append(result)
        if result.status in {"failed", "partial"}:
            failures.append({"source": str(result.source.relative_path), "error": result.diagnostics[0].message if result.diagnostics else "failed"})
        progress(f"[{index}/{len(media)}] {result.status} {result.source.name}")

    return summaries, failures


def transcribe_source(runtime: Any, media: SourceMedia, model: InstalledModel, options: TranscriptionOptions) -> TranscriptResult:
    source_sha256 = _sha256_file(media.source)
    source = _source_record(media, source_sha256)
    try:
        original, translated = runtime.transcribe(media.source, options)
        transcripts = [_variant("original", original, options.language, options.language is not None)]
        if translated:
            transcripts.append(_variant("english_translation", translated, "en", False))
        return TranscriptResult(
            schema_version=SCHEMA_VERSION,
            status="completed",
            source=source,
            transcripts=transcripts,
            speakers=[],
            diagnostics=[],
            provenance=_provenance(runtime, model, options),
            operation=_operation(runtime, options, source_sha256),
            trust=make_trust(model.trust_remote_code),
        )
    except Exception:
        return TranscriptResult(
            schema_version=SCHEMA_VERSION,
            status="failed",
            source=source,
            transcripts=[],
            speakers=[],
            diagnostics=[Diagnostic("error", "transcription_failed", "Transcription failed", "transcription", False)],
            provenance=_provenance(runtime, model, options),
            operation=_operation(runtime, options, source_sha256),
            trust=make_trust(model.trust_remote_code),
        )


def run_transcription_workflow(
    media: list[SourceMedia],
    model: InstalledModel,
    options: TranscriptionOptions,
    runtime: Any | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[TranscriptResult]:
    runtime = runtime or load_pipeline(model, options.device)[0]
    results: list[TranscriptResult] = []
    for index, item in enumerate(media, start=1):
        if progress:
            progress(f"[{index}/{len(media)}] transcribing {item.source} ({runtime.info.effective_device}, {model.id})")
        result = transcribe_source(runtime, item, model, options)
        results.append(result)
        if progress:
            progress(f"[{index}/{len(media)}] {result.status} {item.source}")
    return results


def output_base(output_dir: Path, media: SourceMedia, used: set[Path]) -> Path:
    candidate = output_dir / media.relative.with_suffix("")
    if candidate in used:
        suffix = hashlib.sha256(str(media.source).encode()).hexdigest()[:8]
        candidate = candidate.with_name(f"{candidate.name}-{suffix}")
    used.add(candidate)
    return candidate


def output_paths(base: Path, formats: list[str], translated: bool) -> list[Path]:
    paths: list[Path] = []
    for format_name in formats:
        if translated and format_name != "json":
            paths.extend((Path(f"{base}.original.{format_name}"), Path(f"{base}.en.{format_name}")))
        else:
            paths.append(Path(f"{base}.{format_name}"))
    return paths


def write_outputs(
    base: Path,
    formats: list[str],
    media: SourceMedia,
    model: InstalledModel,
    device: str,
    original: Transcript,
    translated: Transcript | None,
    overwrite: bool,
) -> list[Path]:
    versions = [("original" if translated else "", original)]
    if translated:
        versions.append(("en", translated))
    written: list[Path] = []
    for label, transcript in versions:
        labelled_base = Path(f"{base}.{label}") if label else base
        for format_name in formats:
            path = Path(f"{labelled_base}.{format_name}")
            if path.exists() and not overwrite:
                raise MunError(f"Output already exists: {path}")
            content = render_output(format_name, transcript, media, model, device)
            _atomic_write(path, content)
            written.append(path)
    return written


def write_result_outputs(
    base: Path,
    formats: list[str],
    result: TranscriptResult,
    translated: bool,
    overwrite: bool,
) -> list[Path]:
    projections: list[tuple[Path, str, TranscriptResult]] = []
    for format_name in formats:
        variants = [("", None)]
        if translated and format_name != "json":
            variants = [("original", "original"), ("en", "english_translation")]
        for label, kind in variants:
            if kind and not any(variant.kind == kind for variant in result.transcripts):
                continue
            labelled_base = Path(f"{base}.{label}") if label else base
            path = Path(f"{labelled_base}.{format_name}")
            if kind:
                from .transcript import select_variant
                single = TranscriptResult(**{**result.__dict__, "transcripts": [select_variant(result, kind)]})
            else:
                single = result
            projections.append((path, format_name, single))

    requested_destinations = [path for path, _, _ in projections]
    projections.sort(key=lambda projection: str(projection[0]))
    destinations = [path for path, _, _ in projections]
    receipt_path = Path(f"{base}.receipt.json")
    artifacts: list[ExportArtifact] = []
    committed: list[Path] = []

    def persist_receipt(state: ExportReceiptState) -> None:
        receipt = ExportReceipt(
            schema_version=1,
            state=state,  # type: ignore[arg-type]
            source=result.source.relative_path,
            result_digest=result.result_digest,
            artifacts=artifacts,
            committed_paths=[str(path) for path in committed],
            uncommitted_paths=[str(path) for path in destinations if path not in committed],
        )
        _atomic_write(receipt_path, receipt.to_json())

    existing = [path for path in destinations if path.exists()]
    if existing and not overwrite:
        persist_receipt("failed_before_commit")
        raise MunError(f"Output already exists: {existing[0]}")

    base.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".mun-stage-", dir=base.parent))
    os.chmod(staging, 0o700)
    staged: list[tuple[Path, Path]] = []
    try:
        for index, (destination, format_name, single) in enumerate(projections):
            content = render_output(
                format_name,
                single,
                SourceMedia(Path(result.source.name), Path(result.source.relative_path)),
                InstalledModel(result.provenance.model.repository, result.provenance.model.revision or "", "", ""),
                result.provenance.effective_device,
            )
            staged_path = staging / f"{index:04d}-{destination.name}"
            staged_path.write_text(content, encoding="utf-8")
            digest = _sha256_file(staged_path)
            if digest is None:
                raise MunError(f"Cannot validate staged export: {destination}")
            if format_name == "json":
                TranscriptResult.from_json(staged_path.read_text(encoding="utf-8"))
            artifacts.append(ExportArtifact(str(destination), digest, staged_path.stat().st_size))
            staged.append((staged_path, destination))

        for staged_path, destination in staged:
            try:
                staged_path.replace(destination)
            except BaseException as exc:
                persist_receipt("partial_commit" if committed else ("cancelled" if isinstance(exc, KeyboardInterrupt) else "failed_before_commit"))
                if committed:
                    raise ExportCommitError(committed.copy(), [path for path in destinations if path not in committed]) from exc
                raise
            committed.append(destination)
        persist_receipt("completed")
        return requested_destinations
    except KeyboardInterrupt:
        if not committed:
            persist_receipt("cancelled")
        raise
    except ExportCommitError:
        raise
    except BaseException:
        if not committed:
            persist_receipt("failed_before_commit")
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def render_output(
    format_name: str,
    transcript: Transcript | TranscriptResult,
    media: SourceMedia,
    model: InstalledModel,
    device: str,
) -> str:
    if isinstance(transcript, TranscriptResult):
        try:
            if format_name == "txt":
                return render_txt(transcript)
            if format_name == "json":
                return render_json(transcript)
            if format_name == "srt":
                return render_srt(transcript)
            if format_name == "vtt":
                return render_vtt(transcript)
        except ValueError as exc:
            raise MunError(str(exc)) from exc
    if format_name == "txt":
        return transcript.text.strip() + "\n"
    if format_name == "json":
        payload = {
            "schema_version": 1,
            "source": str(media.relative),
            "model": {"id": model.id, "revision": model.revision},
            "device": device,
            "language": transcript.language,
            "text": transcript.text,
            "segments": [asdict(segment) for segment in transcript.segments],
            "warnings": [],
            "errors": [],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if format_name in {"srt", "vtt"}:
        if not transcript.segments or any(segment.start is None or segment.end is None for segment in transcript.segments):
            raise MunError(f"{format_name.upper()} output requires timestamps")
        return _render_subtitles(transcript.segments, format_name)
    raise MunError(f"Unknown output format: {format_name}")


def _add_media(
    discovered: list[SourceMedia],
    seen: set[Path],
    candidate: Path,
    relative: Path,
    probe: Callable[[Path], bool],
) -> None:
    resolved = candidate.resolve()
    if resolved in seen or not probe(resolved):
        return
    seen.add(resolved)
    discovered.append(SourceMedia(resolved, relative))


def _convert_media(source: Path, destination: Path) -> None:
    ffmpeg = _cached_binary_path("ffmpeg", "FFmpeg is not installed. Install FFmpeg and try again.")
    result = subprocess.run(
        [
            ffmpeg, "-nostdin", "-v", "error", "-i", str(source), "-map", "0:a:0",
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-y", str(destination),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown FFmpeg error"
        raise MunError(f"Media conversion failed: {message}")


def _run_pipeline(
    speech_pipeline: Any,
    wav_path: Path,
    model_type: str,
    options: TranscriptionOptions,
    task: str,
) -> Transcript:
    kwargs: dict[str, Any] = {
        "chunk_length_s": options.chunk_length,
        "stride_length_s": options.stride_length,
    }
    if options.timestamps:
        kwargs["return_timestamps"] = True if model_type == "whisper" else "word"
    if model_type == "whisper":
        kwargs["task"] = task
        if options.language:
            kwargs["language"] = options.language
    result = speech_pipeline(str(wav_path), **kwargs)
    chunks = result.get("chunks", []) if isinstance(result, dict) else []
    segments = []
    for chunk in chunks:
        timestamp = chunk.get("timestamp") or (None, None)
        segments.append(Segment(chunk.get("text", "").strip(), timestamp[0], timestamp[1]))
    return Transcript(text=result["text"].strip(), segments=segments, language=options.language)


def _render_subtitles(segments: list[Segment], format_name: str) -> str:
    lines = ["WEBVTT", ""] if format_name == "vtt" else []
    for index, segment in enumerate(segments, start=1):
        if format_name == "srt":
            lines.append(str(index))
        lines.append(f"{_timestamp(segment.start or 0, format_name)} --> {_timestamp(segment.end or 0, format_name)}")
        lines.extend([segment.text, ""])
    return "\n".join(lines).rstrip() + "\n"


def _timestamp(seconds: float, format_name: str) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    separator = "," if format_name == "srt" else "."
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}{separator}{millis:03d}"


def _variant(kind: str, transcript: Transcript, language: str | None, forced: bool) -> TranscriptVariant:
    return TranscriptVariant(
        kind=kind,  # type: ignore[arg-type]
        language=Language(language or transcript.language, "forced" if forced else ("model" if kind == "english_translation" else "unknown")),
        text=transcript.text,
        segments=[
            TranscriptSegment(
                id=f"segment_{index}",
                start_ms=_seconds_to_ms(segment.start),
                end_ms=_seconds_to_ms(segment.end),
                text=segment.text,
                speaker_id=segment.speaker,
                words=[],
            )
            for index, segment in enumerate(transcript.segments, start=1)
        ],
    )


def _seconds_to_ms(seconds: float | None) -> int | None:
    return None if seconds is None else round(seconds * 1000)


def _provenance(runtime: Any, model: InstalledModel, options: TranscriptionOptions):
    info = runtime.info
    return make_provenance(
        mun_version=__version__,
        model_id=model.id,
        revision=model.revision,
        artifact_sha256=getattr(runtime, "model_artifact_sha256", None),
        runtime_name=info.name,
        runtime_version=info.version,
        requested_device=options.device,
        effective_device=info.effective_device,
        precision=info.precision,
    )


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
