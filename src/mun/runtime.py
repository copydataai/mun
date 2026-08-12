from __future__ import annotations

import hashlib
import subprocess
import tempfile
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

from .core import (
    Segment,
    Transcript,
    TranscriptionOptions,
    _audio_input_path,
    _cached_binary_path,
    _can_use_source_audio_directly,
    _probe_media_details,
)
from .errors import MunError
from .models import InstalledModel, verify_installed_model
from .transcript import ConverterIdentity, ModelTrust, PreparedMediaRecord


@dataclass(frozen=True)
class RuntimeInfo:
    name: str
    version: str | None
    requested_device: str
    effective_device: str
    precision: str | None
    model_type: str
    model_trust: ModelTrust = "verified_artifact"


class SpeechRuntime(Protocol):
    info: RuntimeInfo

    def transcribe(self, source: Path, options: TranscriptionOptions) -> tuple[Transcript, Transcript | None]: ...


class TransformersRuntime:
    def __init__(self, model: InstalledModel, requested_device: str = "auto") -> None:
        if model.trust_remote_code and not model.remote_code_acknowledged:
            raise MunError("Remote repository code has not been explicitly acknowledged for this pinned revision")
        verification = verify_installed_model(model)
        if verification.status not in {"verified", "unsafe_remote_code"}:
            detail = f" ({', '.join(verification.paths)})" if verification.paths else ""
            raise MunError(
                f"Model integrity verification failed: {verification.status}{detail}. {verification.guidance}".strip()
            )
        try:
            import torch
            from transformers import AutoConfig, pipeline

            device = detect_device(requested_device, torch)
            config = AutoConfig.from_pretrained(model.path, local_files_only=True)
            dtype = torch.float16 if device.startswith("cuda") else torch.float32
            pipeline_device: str | int = device if device != "cpu" else -1
            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=model.path,
                config=config,
                device=pipeline_device,
                dtype=dtype,
                trust_remote_code=model.trust_remote_code,
            )
            loaded_verification = verify_installed_model(model)
            if (
                loaded_verification.status not in {"verified", "unsafe_remote_code"}
                or loaded_verification.artifact_digest is None
                or loaded_verification.artifact_digest != verification.artifact_digest
            ):
                raise MunError("The installed model changed while the runtime was loading")
            self.model_artifact_sha256 = loaded_verification.artifact_digest
            self.info = RuntimeInfo(
                name="transformers",
                version=_package_version("transformers"),
                requested_device=requested_device,
                effective_device=device,
                precision="float16" if dtype is torch.float16 else "float32",
                model_type=config.model_type,
                model_trust="unsafe_remote_code" if model.trust_remote_code else "verified_artifact",
            )
        except Exception as exc:
            raise MunError(f"Could not load {model.id}: {exc}") from exc

    def transcribe(self, source: Path, options: TranscriptionOptions) -> tuple[Transcript, Transcript | None]:
        self.prepared_media = None
        model_type = self.info.model_type
        if options.timestamps and model_type not in {"whisper", "wav2vec2", "hubert", "wavlm"}:
            raise MunError(f"The selected {model_type} model cannot provide timestamps")
        if (options.language or options.translate) and model_type != "whisper":
            raise MunError("Language selection and English translation require a Whisper-family model")
        if _can_use_source_audio_directly(source):
            wav_path = source
            self.prepared_media = _prepared_media_record(wav_path, used=False)
            original = self._run_pipeline(wav_path, options, task="transcribe")
            translated = self._run_pipeline(wav_path, options, task="translate") if options.translate else None
            return original, translated

        with tempfile.TemporaryDirectory(prefix="mun-") as temporary_directory:
            wav_path, _ = _audio_input_path(source, Path(temporary_directory) / "audio.wav")
            self.prepared_media = _prepared_media_record(wav_path, used=True)
            original = self._run_pipeline(wav_path, options, task="transcribe")
            translated = self._run_pipeline(wav_path, options, task="translate") if options.translate else None
            return original, translated

    def _run_pipeline(self, wav_path: Path, options: TranscriptionOptions, task: str) -> Transcript:
        kwargs: dict[str, Any] = {"chunk_length_s": options.chunk_length, "stride_length_s": options.stride_length}
        if options.timestamps:
            kwargs["return_timestamps"] = True if self.info.model_type == "whisper" else "word"
        if self.info.model_type == "whisper":
            kwargs["task"] = task
            if options.language:
                kwargs["language"] = options.language
        result = self._pipeline(str(wav_path), **kwargs)
        chunks = result.get("chunks", []) if isinstance(result, dict) else []
        segments = []
        for chunk in chunks:
            timestamp = chunk.get("timestamp") or (None, None)
            segments.append(Segment(chunk.get("text", "").strip(), timestamp[0], timestamp[1]))
        language = options.language if task == "transcribe" else "en"
        return Transcript(text=result["text"].strip(), segments=segments, language=language)


class FakeSpeechRuntime:
    def __init__(self, original: Transcript, translated: Transcript | None = None) -> None:
        self._original = original
        self._translated = translated
        self.info = RuntimeInfo("test", "0", "test", "cpu", "test", "whisper")
        self.prepared_media: PreparedMediaRecord | None = None

    def transcribe(self, source: Path, options: TranscriptionOptions) -> tuple[Transcript, Transcript | None]:
        return self._original, self._translated if options.translate else None


def create_transformers_runtime(model: InstalledModel, requested_device: str = "auto") -> SpeechRuntime:
    return TransformersRuntime(model, requested_device)


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


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def ffmpeg_version() -> str | None:
    try:
        ffmpeg = _cached_binary_path("ffmpeg", "FFmpeg is not installed. Install FFmpeg and try again.")
        result = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True, check=False)
    except (MunError, OSError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout.splitlines()[0].strip()


def _prepared_media_record(path: Path, *, used: bool) -> PreparedMediaRecord:
    probe = None if used else _probe_media_details(path)
    return PreparedMediaRecord(
        used=used,
        sha256=_sha256_file(path),
        media_format="wav" if used else probe.media_format,
        codec="pcm_s16le" if used else probe.codec,
        sample_rate_hz=16_000 if used else probe.sample_rate_hz,
        channels=1 if used else probe.channels,
        converter=ConverterIdentity("ffmpeg", ffmpeg_version()) if used else None,
    )
