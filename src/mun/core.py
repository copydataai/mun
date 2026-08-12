from __future__ import annotations

import hashlib
import json
import os
import shutil
from importlib.metadata import PackageNotFoundError, version as package_version
import subprocess
import tempfile
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .errors import MunError
from .models import InstalledModel
from . import __version__
from .transcript import (
    SCHEMA_VERSION,
    Diagnostic,
    Language,
    SourceRecord,
    TranscriptResult,
    TranscriptSegment,
    TranscriptVariant,
    make_provenance,
    render_json,
    render_srt,
    render_txt,
    render_vtt,
)

MEDIA_EXTENSIONS = {
    ".aac", ".aiff", ".alac", ".avi", ".flac", ".m4a", ".mkv", ".mov", ".mp3",
    ".mp4", ".mpeg", ".mpg", ".oga", ".ogg", ".opus", ".wav", ".webm", ".wma",
}
_FFPROBE_CACHE: dict[Path, tuple[bool, bool]] = {}
_BINARY_PATH_CACHE: dict[str, str | None] = {}


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


def _probe_media(path: Path) -> tuple[bool, bool]:
    """Return (is_audio, is_ready_for_whisper).

    The second result is true when the input can be passed directly to the speech
    pipeline without a temporary conversion to 16 kHz mono PCM.
    """
    key = path.resolve()
    cached = _FFPROBE_CACHE.get(key)
    if cached is not None:
        return cached

    ffprobe = _cached_binary_path("ffprobe", "FFprobe is not installed. Install FFmpeg and try again.")
    result = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,channels,sample_rate",
            "-of", "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        _FFPROBE_CACHE[key] = (False, False)
        return False, False

    is_audio = False
    is_whisper_ready = False
    try:
        payload = json.loads(result.stdout)
        streams = payload.get("streams", [])
        if isinstance(streams, list) and streams:
            stream = streams[0]
            is_audio = stream.get("codec_type") == "audio" or "codec_type" not in stream
            channels = int(stream.get("channels", 0))
            sample_rate = int(stream.get("sample_rate", 0))
            codec_name = stream.get("codec_name", "")
            is_whisper_ready = is_audio and channels == 1 and sample_rate == 16_000 and str(codec_name).startswith("pcm")
    except (TypeError, ValueError, json.JSONDecodeError):
        is_audio = False
        is_whisper_ready = False

    _FFPROBE_CACHE[key] = (is_audio, is_whisper_ready)
    return is_audio, is_whisper_ready


def _can_use_source_audio_directly(path: Path) -> bool:
    return _probe_media(path)[1]


def _infer_effective_device(requested_device: str) -> str:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise MunError(f"Could not load model for device detection: {exc}") from exc

    return detect_device(requested_device, torch)


def _batch_provenance_runtime(requested_device: str, effective_device: str) -> Any:
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
    return SimpleNamespace(info=info)


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
    skipped: list[tuple[int, SourceMedia, Path]] = []

    queued_results: dict[int, TranscriptResult] = {}

    for index, item in enumerate(media, start=1):
        base = output_base(output_dir, item, used_bases)
        expected = output_paths(base, formats, options.translate)
        if not overwrite and any(path.exists() for path in expected):
            skipped.append((index, item, base))
            continue
        queued.append((index, item, base))

    if not queued:
        fallback_runtime = runtime or _batch_provenance_runtime(options.device, options.device)
        for index, item, _ in skipped:
            progress(f"[{index}/{len(media)}] skipped {item.source} (output exists)")
            queued_results[index] = TranscriptResult(
                schema_version=SCHEMA_VERSION,
                status="partial",
                source=SourceRecord(item.source.name, str(item.relative)),
                transcripts=[],
                speakers=[],
                diagnostics=[Diagnostic("warning", "output_exists", "Output already exists", "output", True)],
                provenance=_provenance(fallback_runtime, model, options),
            )
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

    for index, item, _ in skipped:
        progress(f"[{index}/{len(media)}] skipped {item.source} (output exists)")
        queued_results[index] = TranscriptResult(
            schema_version=SCHEMA_VERSION,
            status="partial",
            source=SourceRecord(item.source.name, str(item.relative)),
            transcripts=[],
            speakers=[],
            diagnostics=[Diagnostic("warning", "output_exists", "Output already exists", "output", True)],
            provenance=_provenance(fallback_runtime, model, options),
        )

    if jobs == 1:
        for index, item, base in queued:
            progress(f"[{index}/{len(media)}] transcribing {item.source} ({effective_device}, {model.id})")
            try:
                result = transcribe_source(runtime, item, model, options)
                if result.status == "completed":
                    write_result_outputs(base, formats, result, options.translate, overwrite=overwrite)
                queued_results[index] = result
                progress(f"[{index}/{len(media)}] {result.status} {item.source}")
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                queued_results[index] = TranscriptResult(
                    schema_version=SCHEMA_VERSION,
                    status="failed",
                    source=SourceRecord(item.source.name, str(item.relative)),
                    transcripts=[],
                    speakers=[],
                    diagnostics=[Diagnostic("error", "transcription_failed", "Transcription failed", "transcription", False)],
                    provenance=_provenance(runtime, model, options),
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

    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_batch_worker, initargs=(model, options)) as executor:
        futures = {
            executor.submit(_run_batch_worker, (index, item)): index for index, item, _ in queued
        }
        for future in as_completed(futures):
            index = futures[future]
            item = media_by_index[index]
            progress(f"[{index}/{len(media)}] transcribing {item.source} ({effective_device}, {model.id})")
            try:
                _, result = future.result()
            except Exception:
                result = TranscriptResult(
                    schema_version=SCHEMA_VERSION,
                    status="failed",
                    source=SourceRecord(item.source.name, str(item.relative)),
                    transcripts=[],
                    speakers=[],
                    diagnostics=[Diagnostic("error", "transcription_failed", "Transcription failed", "transcription", False)],
                    provenance=_provenance(fallback_runtime, model, options),
                )

            completed[index] = result

    ordered = sorted(completed)
    for index in ordered:
        result = completed[index]
        queued_results[index] = result

    for index in sorted(queued_results):
        result = queued_results[index]
        base = base_by_index.get(index)
        if result.status == "completed" and base is not None:
            write_result_outputs(base, formats, result, options.translate, overwrite=overwrite)
        summaries.append(result)
        if result.status == "failed":
            failures.append({"source": str(result.source.relative_path), "error": result.diagnostics[0].message if result.diagnostics else "failed"})
        progress(f"[{index}/{len(media)}] {result.status} {result.source.name}")

    return summaries, failures


def transcribe_source(runtime: Any, media: SourceMedia, model: InstalledModel, options: TranscriptionOptions) -> TranscriptResult:
    try:
        original, translated = runtime.transcribe(media.source, options)
        transcripts = [_variant("original", original, options.language, options.language is not None)]
        if translated:
            transcripts.append(_variant("english_translation", translated, "en", False))
        return TranscriptResult(
            schema_version=SCHEMA_VERSION,
            status="completed",
            source=SourceRecord(media.source.name, str(media.relative)),
            transcripts=transcripts,
            speakers=[],
            diagnostics=[],
            provenance=_provenance(runtime, model, options),
        )
    except Exception:
        return TranscriptResult(
            schema_version=SCHEMA_VERSION,
            status="failed",
            source=SourceRecord(media.source.name, str(media.relative)),
            transcripts=[],
            speakers=[],
            diagnostics=[Diagnostic("error", "transcription_failed", "Transcription failed", "transcription", False)],
            provenance=_provenance(runtime, model, options),
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
    written: list[Path] = []
    for format_name in formats:
        projections = [("", None)]
        if translated and format_name != "json":
            projections = [("original", "original"), ("en", "english_translation")]
        for label, kind in projections:
            if kind and not any(variant.kind == kind for variant in result.transcripts):
                continue
            labelled_base = Path(f"{base}.{label}") if label else base
            path = Path(f"{labelled_base}.{format_name}")
            if path.exists() and not overwrite:
                raise MunError(f"Output already exists: {path}")
            if kind:
                from .transcript import select_variant
                single = TranscriptResult(**{**result.__dict__, "transcripts": [select_variant(result, kind)]})
            else:
                single = result
            _atomic_write(
                path,
                render_output(
                    format_name,
                    single,
                    SourceMedia(Path(result.source.name), Path(result.source.relative_path)),
                    InstalledModel(result.provenance.model.repository, result.provenance.model.revision or "", "", ""),
                    result.provenance.effective_device,
                ),
            )
            written.append(path)
    return written


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
