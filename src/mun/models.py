from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

from .config import default_model_dir
from .errors import MunError

METADATA_FILE = "mun-model.json"
MANIFEST_FILE = "mun-artifact-manifest.json"
MANIFEST_VERSION = 1
VerificationStatus = Literal[
    "verified",
    "missing",
    "modified",
    "unexpected_file",
    "unsafe_remote_code",
    "manifest_missing",
]


@dataclass(frozen=True)
class InstalledModel:
    id: str
    revision: str
    path: str
    installed_at: str
    status: str = "installed"
    trust_remote_code: bool = False


@dataclass(frozen=True)
class VerificationResult:
    status: VerificationStatus
    artifact_digest: str | None = None
    paths: tuple[str, ...] = ()
    guidance: str = ""


def load_catalog() -> dict[str, Any]:
    return json.loads(files("mun").joinpath("catalog.json").read_text(encoding="utf-8"))


def models_root(config: dict[str, Any], override: str | None = None) -> Path:
    return Path(override or config.get("model_dir") or default_model_dir()).expanduser().resolve()


def installed_models(root: Path) -> list[InstalledModel]:
    if not root.exists():
        return []
    models: list[InstalledModel] = []
    for metadata_path in root.glob(f"*/{METADATA_FILE}"):
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            models.append(InstalledModel(**data))
        except (OSError, json.JSONDecodeError, TypeError):
            continue
    return sorted(models, key=lambda model: (model.id, model.installed_at))


def find_installed(root: Path, model_id: str | None) -> InstalledModel:
    ready = [model for model in installed_models(root) if model.status in {"installed", "ready"}]
    if model_id:
        candidate_path = Path(model_id).expanduser()
        if candidate_path.exists():
            metadata = _read_metadata(candidate_path.resolve())
            if metadata.status not in {"installed", "ready"}:
                raise MunError(f"Model installation is not ready: {candidate_path}")
            return metadata
        ready = [model for model in ready if model.id == model_id or model.path == model_id]
    if not ready:
        target = f" '{model_id}'" if model_id else ""
        raise MunError(f"No installed model{target}. Run: mun models download openai/whisper-small")
    return ready[-1]


def search_models(query: str | None, limit: int, offline: bool) -> list[dict[str, Any]]:
    catalog = load_catalog()["models"]
    if offline:
        needle = (query or "").lower()
        return [
            _catalog_search_record(model)
            for model in catalog
            if not needle or needle in model["id"].lower()
        ][:limit]
    try:
        from huggingface_hub import HfApi

        results = HfApi().list_models(
            pipeline_tag="automatic-speech-recognition",
            search=query,
            sort="downloads",
            limit=limit,
            expand=["pipeline_tag", "library_name", "downloads", "likes", "gated", "tags", "sha"],
        )
        return [_search_record(model) for model in results]
    except Exception as exc:
        raise MunError(f"Hugging Face search failed: {exc}") from exc


def remote_model_summary(model_id: str, revision: str | None = None) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
        size, _ = _download_plan(info.siblings or [])
        card_data = getattr(info, "card_data", None)
        return {
            "id": model_id,
            "revision": info.sha,
            "library": info.library_name or "unknown",
            "pipeline": info.pipeline_tag or "unknown",
            "gated": bool(info.gated),
            "license": getattr(card_data, "license", None) or "unknown",
            "download_size": _human_bytes(size) if size else "unknown",
        }
    except Exception as exc:
        raise MunError(f"Could not inspect {model_id}: {exc}") from exc


def download_model(
    model_id: str,
    root: Path,
    revision: str | None,
    trust_remote_code: bool,
    offline: bool,
) -> InstalledModel:
    if offline:
        raise MunError("Cannot download models in offline mode")
    try:
        from huggingface_hub import HfApi, snapshot_download

        info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
        if info.pipeline_tag != "automatic-speech-recognition":
            raise MunError(f"{model_id} is not tagged for automatic speech recognition")
        if info.library_name not in {None, "transformers"}:
            raise MunError(f"{model_id} uses {info.library_name}, not Transformers")
        sha = info.sha
        if not sha:
            raise MunError(f"Hugging Face did not return an immutable revision for {model_id}")
        existing = [model for model in installed_models(root) if model.id == model_id and model.revision == sha]
        if existing and existing[-1].status in {"installed", "ready"}:
            verification = verify_installed_model(existing[-1])
            if verification.status in {"verified", "unsafe_remote_code"}:
                return existing[-1]
        directory = root / f"{_slug(model_id)}--{sha[:12]}"
        temporary_directory = root / f".{directory.name}.download"
        if temporary_directory.exists():
            shutil.rmtree(temporary_directory)
        temporary_directory.mkdir(parents=True, exist_ok=False)
        model = InstalledModel(
            id=model_id,
            revision=sha,
            path=str(directory),
            installed_at=datetime.now(UTC).isoformat(),
            status="downloading",
            trust_remote_code=trust_remote_code,
        )
        needed_bytes, ignore_patterns = _download_plan(info.siblings or [])
        free_bytes = shutil.disk_usage(root).free
        if needed_bytes and free_bytes < needed_bytes * 1.1:
            shutil.rmtree(temporary_directory, ignore_errors=True)
            raise MunError(
                f"Insufficient disk space: need about {_human_bytes(needed_bytes)}, "
                f"have {_human_bytes(free_bytes)}"
            )
        snapshot_download(
            repo_id=model_id,
            revision=sha,
            local_dir=temporary_directory,
            ignore_patterns=ignore_patterns,
        )
        model = InstalledModel(**{**asdict(model), "status": "installed"})
        _write_manifest(temporary_directory, model)
        _write_metadata(temporary_directory, model)
        if directory.exists():
            shutil.rmtree(directory)
        temporary_directory.replace(directory)
        _write_metadata(directory, model)
        return model
    except MunError:
        if "temporary_directory" in locals():
            shutil.rmtree(temporary_directory, ignore_errors=True)
        raise
    except Exception as exc:
        if "temporary_directory" in locals():
            shutil.rmtree(temporary_directory, ignore_errors=True)
        raise MunError(f"Model download failed: {exc}") from exc


def remove_model(root: Path, target: str) -> tuple[InstalledModel, int]:
    matches = [
        model
        for model in installed_models(root)
        if model.id == target or model.path == str(Path(target).expanduser().resolve())
    ]
    if not matches:
        raise MunError(f"No managed model matches: {target}")
    if len(matches) > 1:
        revisions = ", ".join(model.revision[:12] for model in matches)
        raise MunError(f"Multiple revisions match {target}; use an installed path: {revisions}")
    model = matches[0]
    path = Path(model.path).resolve()
    root = root.resolve()
    if path.parent != root or not (path / METADATA_FILE).is_file():
        raise MunError("Refusing to remove a directory not managed by Mun")
    size = sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
    shutil.rmtree(path)
    return model, size


def model_details(root: Path, target: str) -> dict[str, Any]:
    model = find_installed(root, target)
    catalog = next((item for item in load_catalog()["models"] if item["id"] == model.id), None)
    return {"installed": asdict(model), "catalog": catalog}


def _read_metadata(path: Path) -> InstalledModel:
    metadata_path = path / METADATA_FILE
    try:
        return InstalledModel(**json.loads(metadata_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise MunError(f"Not a managed model directory: {path}") from exc


def _write_metadata(directory: Path, model: InstalledModel) -> None:
    path = directory / METADATA_FILE
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(asdict(model), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_installed_model(model: InstalledModel) -> VerificationResult:
    directory = Path(model.path)
    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.is_file():
        return VerificationResult(
            "manifest_missing",
            guidance=f"Reinstall {model.id} to create a verifiable artifact manifest.",
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest["files"]
        expected_digest = _manifest_digest({key: value for key, value in manifest.items() if key != "artifact_digest"})
        if (
            manifest.get("version") != MANIFEST_VERSION
            or manifest.get("source_repository") != model.id
            or manifest.get("source_revision") != model.revision
            or manifest.get("installed_at") != model.installed_at
            or manifest.get("trust_remote_code") != model.trust_remote_code
            or manifest.get("artifact_digest") != expected_digest
            or not isinstance(files, list)
        ):
            raise ValueError("manifest metadata or digest mismatch")
        expected: dict[str, dict[str, Any]] = {}
        for record in files:
            relative = _validated_relative_path(record["path"])
            if relative in expected or not isinstance(record["bytes"], int) or not isinstance(record["sha256"], str):
                raise ValueError("invalid file record")
            expected[relative] = record
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return VerificationResult("modified", guidance=f"Reinstall {model.id}; its artifact manifest is invalid.")

    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if (path.is_file() or path.is_symlink())
        and path.relative_to(directory).as_posix() not in {METADATA_FILE, MANIFEST_FILE}
    }
    unexpected = sorted(actual - expected.keys())
    if unexpected:
        return VerificationResult(
            "unexpected_file",
            artifact_digest=manifest["artifact_digest"],
            paths=tuple(unexpected),
            guidance="Remove untracked files or reinstall the model.",
        )
    missing = sorted(expected.keys() - actual)
    if missing:
        return VerificationResult(
            "missing",
            artifact_digest=manifest["artifact_digest"],
            paths=tuple(missing),
            guidance=f"Reinstall {model.id}; required model files are missing.",
        )
    modified = []
    for relative, record in expected.items():
        path = directory / relative
        if path.is_symlink() or path.stat().st_size != record["bytes"] or _file_sha256(path) != record["sha256"]:
            modified.append(relative)
    if modified:
        return VerificationResult(
            "modified",
            artifact_digest=manifest["artifact_digest"],
            paths=tuple(sorted(modified)),
            guidance=f"Reinstall {model.id}; installed model files were modified.",
        )
    status: VerificationStatus = "unsafe_remote_code" if model.trust_remote_code else "verified"
    guidance = "Remote repository code is enabled; this installation is not tested or safe." if model.trust_remote_code else ""
    return VerificationResult(status, manifest["artifact_digest"], guidance=guidance)


def _write_manifest(directory: Path, model: InstalledModel) -> dict[str, Any]:
    records = []
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        if relative in {METADATA_FILE, MANIFEST_FILE}:
            continue
        if path.is_symlink():
            raise MunError(f"Refusing to install symlinked model artifact: {path}")
        if not path.is_file():
            continue
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": _file_sha256(path)})
    payload: dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "source_repository": model.id,
        "source_revision": model.revision,
        "installed_at": model.installed_at,
        "trust_remote_code": model.trust_remote_code,
        "files": records,
    }
    manifest = {**payload, "artifact_digest": _manifest_digest(payload)}
    path = directory / MANIFEST_FILE
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return manifest


def _validated_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("invalid relative path")
    path = Path(value)
    if path.is_absolute() or path.as_posix() != value or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("invalid relative path")
    return value


def _manifest_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _catalog_search_record(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": model["id"],
        "revision": model.get("revision"),
        "library": model.get("library", "transformers"),
        "pipeline": model.get("pipeline", "automatic-speech-recognition"),
        "gated": bool(model.get("gated", False)),
        "license": model.get("license", "unknown"),
    }


def _search_record(model: Any) -> dict[str, Any]:
    library = getattr(model, "library_name", None)
    return {
        "id": model.id,
        "revision": getattr(model, "sha", None),
        "library": library or "unknown",
        "pipeline": getattr(model, "pipeline_tag", None) or "unknown",
        "downloads": getattr(model, "downloads", 0) or 0,
        "likes": getattr(model, "likes", 0) or 0,
        "gated": bool(getattr(model, "gated", False)),
    }


def _download_plan(siblings: list[Any]) -> tuple[int, list[str]]:
    filenames = [sibling.rfilename for sibling in siblings]
    has_safetensors = any(name.endswith(".safetensors") for name in filenames)
    ignored = ["*.h5", "*.msgpack", "*.onnx", "*.tflite", "*.ot"]
    if has_safetensors:
        ignored.append("*.bin")
    wanted = [
        sibling
        for sibling in siblings
        if not any(Path(sibling.rfilename).match(pattern) for pattern in ignored)
    ]
    return sum((getattr(sibling, "size", 0) or 0) for sibling in wanted), ignored


def _slug(model_id: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "--" for character in model_id)


def _human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{value} B"
