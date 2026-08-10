from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .errors import MunError
from .models import InstalledModel

MEDIA_EXTENSIONS = {
    ".aac", ".aiff", ".alac", ".avi", ".flac", ".m4a", ".mkv", ".mov", ".mp3",
    ".mp4", ".mpeg", ".mpg", ".oga", ".ogg", ".opus", ".wav", ".webm", ".wma",
}


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
            candidates = sorted(root.rglob("*"))
            for candidate in candidates:
                if candidate.is_dir() or candidate.is_symlink():
                    continue
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
    if not shutil.which("ffprobe"):
        raise MunError("FFprobe is not installed. Install FFmpeg and try again.")
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_type", "-of", "default=nw=1:nk=1", str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and "audio" in result.stdout.splitlines()


def load_pipeline(model: InstalledModel, requested_device: str = "auto") -> tuple[Any, str, str]:
    try:
        import torch
        from transformers import AutoConfig, pipeline

        device = detect_device(requested_device, torch)
        config = AutoConfig.from_pretrained(model.path, local_files_only=True)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        pipeline_device: str | int = device if device != "cpu" else -1
        speech_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model.path,
            config=config,
            device=pipeline_device,
            dtype=dtype,
            trust_remote_code=model.trust_remote_code,
        )
        return speech_pipeline, device, config.model_type
    except Exception as exc:
        raise MunError(f"Could not load {model.id}: {exc}") from exc


def detect_device(requested: str, torch_module: Any) -> str:
    if requested != "auto":
        if requested.startswith("cuda") and not torch_module.cuda.is_available():
            raise MunError("CUDA/ROCm was requested but is unavailable")
        if requested == "mps" and not torch_module.backends.mps.is_available():
            raise MunError("MPS was requested but is unavailable")
        if requested not in {"cpu", "mps"} and not requested.startswith("cuda"):
            raise MunError(f"Unknown device: {requested}")
        return requested
    if torch_module.cuda.is_available():
        return "cuda:0"
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def transcribe_media(
    speech_pipeline: Any,
    source: Path,
    model_type: str,
    options: TranscriptionOptions,
) -> tuple[Transcript, Transcript | None]:
    needs_timestamps = options.timestamps
    if needs_timestamps and model_type not in {"whisper", "wav2vec2", "hubert", "wavlm"}:
        raise MunError(f"The selected {model_type} model cannot provide timestamps")
    if (options.language or options.translate) and model_type != "whisper":
        raise MunError("Language selection and English translation require a Whisper-family model")
    with tempfile.TemporaryDirectory(prefix="mun-") as temporary_directory:
        wav_path = Path(temporary_directory) / "audio.wav"
        _convert_media(source, wav_path)
        original = _run_pipeline(speech_pipeline, wav_path, model_type, options, task="transcribe")
        translated = None
        if options.translate:
            translated = _run_pipeline(speech_pipeline, wav_path, model_type, options, task="translate")
        return original, translated


def run_batch(
    media: list[SourceMedia],
    model: InstalledModel,
    output_dir: Path,
    formats: list[str],
    options: TranscriptionOptions,
    overwrite: bool,
    progress: Callable[[str], None],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    speech_pipeline, device, model_type = load_pipeline(model, options.device)
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    used_bases: set[Path] = set()
    for index, item in enumerate(media, start=1):
        base = output_base(output_dir, item, used_bases)
        expected = output_paths(base, formats, options.translate)
        if not overwrite and any(path.exists() for path in expected):
            progress(f"[{index}/{len(media)}] skipped {item.source} (output exists)")
            summaries.append({"source": str(item.relative), "status": "skipped", "outputs": []})
            continue
        progress(f"[{index}/{len(media)}] transcribing {item.source} ({device}, {model.id})")
        try:
            original, translated = transcribe_media(speech_pipeline, item.source, model_type, options)
            written = write_outputs(
                base, formats, item, model, device, original, translated, overwrite=overwrite
            )
            summaries.append(
                {"source": str(item.relative), "status": "complete", "outputs": [str(path) for path in written]}
            )
            progress(f"[{index}/{len(media)}] complete {item.source}")
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            failures.append({"source": str(item.relative), "error": str(exc)})
            progress(f"[{index}/{len(media)}] failed {item.source}: {exc}")
    return summaries, failures


def output_base(output_dir: Path, media: SourceMedia, used: set[Path]) -> Path:
    candidate = output_dir / media.relative.with_suffix("")
    if candidate in used:
        suffix = hashlib.sha256(str(media.source).encode()).hexdigest()[:8]
        candidate = candidate.with_name(f"{candidate.name}-{suffix}")
    used.add(candidate)
    return candidate


def output_paths(base: Path, formats: list[str], translated: bool) -> list[Path]:
    suffixes = [".original", ".en"] if translated else [""]
    return [Path(f"{base}{suffix}.{format_name}") for suffix in suffixes for format_name in formats]


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


def render_output(
    format_name: str,
    transcript: Transcript,
    media: SourceMedia,
    model: InstalledModel,
    device: str,
) -> str:
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
    if not shutil.which("ffmpeg"):
        raise MunError("FFmpeg is not installed. Install FFmpeg and try again.")
    result = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(source), "-map", "0:a:0",
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
