from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import sys
import contextlib
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import __version__
from .config import config_path, load_config, reset_config, set_config
from .core import (
    TranscriptionOptions,
    discover_media,
    load_pipeline,
    render_output,
    run_batch,
    run_transcription_workflow,
)
from .errors import MunError
from .transcript import make_batch_result
from .models import (
    download_model,
    find_installed,
    installed_models,
    load_catalog,
    model_details,
    models_root,
    remote_model_summary,
    remove_model,
    search_models,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mun",
        description="Transcribe audio and video on your computer",
        epilog="Get started: mun transcribe recordings/",
    )
    parser.add_argument("--version", action="version", version=f"mun {__version__}")
    subcommands = parser.add_subparsers(dest="command")

    transcribe = subcommands.add_parser("transcribe", help="transcribe audio and video files")
    transcribe.add_argument("inputs", nargs="*", help="files or directories")
    transcribe.add_argument("--input-list", type=Path, help="newline-separated paths")
    transcribe.add_argument("-m", "--model", help="installed model ID or managed model path")
    transcribe.add_argument("--model-dir", help="managed model directory")
    transcribe.add_argument("-o", "--output-dir", help="output directory")
    transcribe.add_argument("-f", "--format", action="append", choices=("txt", "json", "srt", "vtt"))
    transcribe.add_argument("--timestamps", action="store_true", help="include segment timestamps")
    transcribe.add_argument("--language", help="spoken language name or code (Whisper only)")
    transcribe.add_argument("--translate", action="store_true", help="also translate speech to English")
    transcribe.add_argument("--include-hidden", action="store_true", help="include media inside hidden directories")
    transcribe.add_argument("--overwrite", action="store_true", help="replace existing transcript files")
    transcribe.add_argument("--device", default=None, help="auto, cpu, mps, or cuda[:index]")
    transcribe.add_argument("--offline", action="store_true", help="disable Hugging Face network access")
    transcribe.add_argument("--stdout", action="store_true", help="write one TXT or JSON result to stdout")
    transcribe.add_argument("--summary-json", action="store_true", help="write batch summary JSON to stdout")
    transcribe.add_argument("--jobs", type=int, default=1, help="parallel batch workers (CPU runs only)")
    transcribe.add_argument("--benchmark", action="store_true", help="report elapsed batch time and file throughput")
    transcribe.add_argument("--chunk-length", type=int, default=30, help=argparse.SUPPRESS)
    transcribe.add_argument("--stride-length", type=int, default=5, help=argparse.SUPPRESS)

    model_commands = subcommands.add_parser("models", help="search and manage speech models")
    model_commands.add_argument("--model-dir", help="managed model directory")
    model_subcommands = model_commands.add_subparsers(dest="models_command", required=True)
    search = model_subcommands.add_parser("search", help="search compatible Hugging Face models")
    search.add_argument("query", nargs="?")
    search.add_argument("--limit", type=int, default=20)
    search.add_argument("--offline", action="store_true")
    search.add_argument("--json", action="store_true")
    download = model_subcommands.add_parser("download", help="download a pinned model snapshot")
    download.add_argument("model_id")
    download.add_argument("--revision")
    download.add_argument("--trust-remote-code", action="store_true")
    download.add_argument("--yes", action="store_true")
    download.add_argument("--offline", action="store_true")
    list_parser = model_subcommands.add_parser("list", help="list installed models")
    list_parser.add_argument("--json", action="store_true")
    info = model_subcommands.add_parser("info", help="show an installed model")
    info.add_argument("target")
    info.add_argument("--json", action="store_true")
    remove = model_subcommands.add_parser("remove", help="remove a managed model")
    remove.add_argument("target")
    remove.add_argument("--yes", action="store_true")

    config = subcommands.add_parser("config", help="show or change user defaults")
    config_subcommands = config.add_subparsers(dest="config_command", required=True)
    config_subcommands.add_parser("show")
    config_set = config_subcommands.add_parser("set")
    config_set.add_argument("key")
    config_set.add_argument("value")
    config_reset = config_subcommands.add_parser("reset")
    config_reset.add_argument("--yes", action="store_true")

    doctor = subcommands.add_parser("doctor", help="diagnose the local runtime")
    doctor.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    try:
        if not argv:
            if not sys.stdin.isatty():
                parser.print_help(sys.stderr)
                return 2
            return interactive_wizard()
        args = parser.parse_args(argv)
        if args.command == "transcribe":
            return command_transcribe(args)
        if args.command == "models":
            return command_models(args)
        if args.command == "config":
            return command_config(args)
        if args.command == "doctor":
            return command_doctor(args)
        parser.print_help()
        return 2
    except KeyboardInterrupt:
        print("\nCancelled; completed outputs were preserved.", file=sys.stderr)
        return 130
    except MunError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def interactive_wizard() -> int:
    print("Mun — local speech-to-text\n")
    config = load_config()
    root = models_root(config)
    models = [model for model in installed_models(root) if model.status in {"installed", "ready"}]
    if not models:
        default = load_catalog()["models"][0]
        print(f"No speech model is installed. Recommended: {default['id']}")
        print(f"Download: {_human_bytes(default['weights_bytes'])}; license: {default['license']}")
        if not _confirm("Download it now?"):
            print(f"Run 'mun models search' or 'mun models download {default['id']}' when ready.")
            return 0
        download_model(default["id"], root, default["revision"], False, False)
        models = [model for model in installed_models(root) if model.status in {"installed", "ready"}]
    print("Installed models:")
    for index, model in enumerate(models, start=1):
        print(f"  {index}. {model.id} ({model.revision[:12]})")
    choice = input("Model [1]: ").strip()
    try:
        model = models[int(choice or "1") - 1]
    except (ValueError, IndexError) as exc:
        raise MunError("Invalid model selection") from exc
    entered_paths = input("Drag or enter files/folders (quote paths with spaces): ").strip()
    if not entered_paths:
        raise MunError("At least one file or directory is required")
    output_dir = input("Output directory [transcripts]: ").strip() or "transcripts"
    args = argparse.Namespace(
        inputs=shlex.split(entered_paths), input_list=None, model=model.path, model_dir=str(root),
        output_dir=output_dir, format=["txt"], timestamps=False, language=None, translate=False,
        include_hidden=False, overwrite=False, device=None, offline=False, stdout=False,
        summary_json=False, jobs=1, benchmark=False, chunk_length=30, stride_length=5,
    )
    return command_transcribe(args)


def command_transcribe(args: argparse.Namespace) -> int:
    config = load_config()
    raw_paths = list(args.inputs)
    if args.input_list:
        try:
            with args.input_list.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if stripped and not stripped.lstrip().startswith("#"):
                        raw_paths.append(stripped)
        except OSError as exc:
            raise MunError(f"Cannot read input list {args.input_list}: {exc}") from exc
    if not raw_paths:
        raise MunError("At least one file or directory is required")
    formats = list(dict.fromkeys(args.format or ["txt"]))
    timestamps = args.timestamps or bool({"srt", "vtt"} & set(formats))
    if args.stdout and (len(formats) != 1 or formats[0] not in {"txt", "json"}):
        raise MunError("--stdout requires exactly one TXT or JSON format")
    media = discover_media(raw_paths, args.include_hidden)
    if args.stdout and len(media) != 1:
        raise MunError("--stdout requires exactly one source media file")
    root = models_root(config, args.model_dir)
    model = find_installed(root, args.model or config.get("model"))
    options = TranscriptionOptions(
        language=args.language,
        timestamps=timestamps,
        translate=args.translate,
        chunk_length=args.chunk_length,
        stride_length=args.stride_length,
        device=args.device or config.get("device", "auto"),
    )
    if args.jobs < 1:
        raise MunError("--jobs must be a positive integer")

    with _offline_environment(args.offline or config.get("offline", False)):
        runtime: Any | None = load_pipeline(model, options.device)[0] if args.jobs == 1 else None
        if args.stdout:
            runtime = runtime or load_pipeline(model, options.device)[0]
            results = run_transcription_workflow(media, model, options, runtime=runtime)
            result = results[0]
            if result.status != "completed":
                for diagnostic in result.diagnostics:
                    print(f"{diagnostic.severity}: {diagnostic.message}", file=sys.stderr)
                return 1
            sys.stdout.write(render_output(formats[0], result, media[0], model, runtime.info.effective_device))
            return 0
        output_dir = Path(args.output_dir or config.get("output_dir", "transcripts")).expanduser().resolve()
        def progress(message: str) -> None:
            print(message, file=sys.stderr)

        started_at = time.perf_counter() if args.benchmark else None
        summaries, failures = run_batch(
            media,
            model,
            output_dir,
            formats,
            options,
            args.overwrite,
            progress,
            runtime=runtime,
            jobs=args.jobs,
        )
        if args.benchmark and started_at is not None:
            elapsed = time.perf_counter() - started_at
            processed = sum(result.reuse_status in {"queued", "overwrite_required"} for result in summaries)
            reused = sum(result.reuse_status == "reused_verified" for result in summaries)
            throughput = processed / elapsed if elapsed else 0.0
            print(
                f"Benchmark: files={len(media)} processed={processed} reused_verified={reused} failed={len(failures)} "
                f"duration_s={elapsed:.2f} files_per_s={throughput:.2f} jobs={args.jobs}",
                file=sys.stderr,
            )
        if args.summary_json:
            json.dump(make_batch_result(summaries).to_dict(), sys.stdout, indent=2)
            sys.stdout.write("\n")
        incomplete = any(result.status != "completed" for result in summaries)
        if not args.summary_json and failures:
            print(f"Completed with {len(failures)} failure(s).", file=sys.stderr)
        return 1 if incomplete else 0


def command_models(args: argparse.Namespace) -> int:
    config = load_config()
    root = models_root(config, args.model_dir)
    if args.models_command == "search":
        records = search_models(args.query, args.limit, args.offline or config.get("offline", False))
        _print_records(records, args.json, ("id", "library", "gated", "downloads"))
        return 0
    if args.models_command == "list":
        records = [asdict(model) for model in installed_models(root)]
        _print_records(records, args.json, ("id", "revision", "status", "path"))
        return 0
    if args.models_command == "download":
        if args.offline or config.get("offline", False):
            raise MunError("Cannot download models in offline mode")
        summary = remote_model_summary(args.model_id, args.revision)
        _print_mapping(summary)
        if args.trust_remote_code:
            print("WARNING: this model may execute Python code from its repository.", file=sys.stderr)
        if not args.yes and not _confirm("Download this immutable model snapshot?"):
            print("Cancelled.")
            return 0
        model = download_model(args.model_id, root, args.revision, args.trust_remote_code, False)
        print(f"Installed {model.id}@{model.revision} in {_redact_home(model.path)}")
        return 0
    if args.models_command == "info":
        details = model_details(root, args.target)
        if args.json:
            print(json.dumps(details, indent=2))
        else:
            _print_mapping(details["installed"])
            if details["catalog"]:
                print("Catalog:")
                _print_mapping(details["catalog"])
        return 0
    if args.models_command == "remove":
        model = find_installed(root, args.target)
        if not args.yes and not _confirm(f"Remove {model.id}@{model.revision[:12]}?"):
            print("Cancelled.")
            return 0
        removed, reclaimed = remove_model(root, model.path)
        print(f"Removed {removed.id}; reclaimed {_human_bytes(reclaimed)}")
        return 0
    raise MunError("Unknown models command")


@contextlib.contextmanager
def _offline_environment(enabled: bool):
    if not enabled:
        yield
        return
    previous = {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def command_config(args: argparse.Namespace) -> int:
    if args.config_command == "show":
        print(f"Configuration: {_redact_home(str(config_path()))}")
        _print_mapping(load_config())
        return 0
    if args.config_command == "set":
        path = set_config(args.key, args.value)
        print(f"Updated {_redact_home(str(path))}")
        return 0
    if args.config_command == "reset":
        if not args.yes and not _confirm(f"Remove {_redact_home(str(config_path()))}?"):
            print("Cancelled.")
            return 0
        print("Configuration reset." if reset_config() else "No configuration file exists.")
        return 0
    raise MunError("Unknown config command")


def command_doctor(args: argparse.Namespace) -> int:
    config = load_config()
    root = models_root(config)
    report: dict[str, Any] = {
        "mun": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "ffprobe": shutil.which("ffprobe") is not None,
        "model_dir": _redact_home(str(root)),
        "model_dir_free": _human_bytes(shutil.disk_usage(_existing_ancestor(root)).free),
        "installed_models": len(installed_models(root)),
        "offline": bool(config.get("offline", False)),
    }
    try:
        import torch

        report.update(
            torch=torch.__version__,
            device=(
                "cuda/rocm" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            ),
        )
    except ImportError:
        report.update(torch="missing", device="unavailable")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_mapping(report)
    return 0 if report["ffmpeg"] and report["ffprobe"] and report["torch"] != "missing" else 1


def _print_records(records: list[dict[str, Any]], as_json: bool, columns: tuple[str, ...]) -> None:
    if as_json:
        print(json.dumps(records, indent=2))
        return
    if not records:
        print("No models found.")
        return
    widths = {column: max(len(column), *(len(str(row.get(column, ""))) for row in records)) for column in columns}
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in records:
        values = [str(row.get(column, "")) for column in columns]
        if "revision" in columns:
            values[columns.index("revision")] = values[columns.index("revision")][:12]
        print("  ".join(value.ljust(widths[column]) for value, column in zip(values, columns)))


def _print_mapping(mapping: dict[str, Any]) -> None:
    if not mapping:
        print("  (defaults)")
        return
    for key, value in mapping.items():
        if key == "path":
            value = _redact_home(str(value))
        print(f"  {key}: {value}")


def _confirm(prompt: str) -> bool:
    if not sys.stdin.isatty():
        raise MunError(f"Confirmation required: {prompt} Use --yes in non-interactive mode.")
    return input(f"{prompt} [y/N] ").strip().lower() in {"y", "yes"}


def _redact_home(value: str) -> str:
    home = str(Path.home())
    return value.replace(home, "~") if value.startswith(home) else value


def _existing_ancestor(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate == candidate.parent:
            raise MunError(f"No existing parent directory for {path}")
        candidate = candidate.parent
    return candidate


def _human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return str(value)
