from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

from .config import default_model_dir
from .errors import MunError

METADATA_FILE = "mun-model.json"


@dataclass(frozen=True)
class InstalledModel:
    id: str
    revision: str
    path: str
    installed_at: str
    status: str = "ready"
    trust_remote_code: bool = False


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
    ready = [model for model in installed_models(root) if model.status == "ready"]
    if model_id:
        candidate_path = Path(model_id).expanduser()
        if candidate_path.exists():
            metadata = _read_metadata(candidate_path.resolve())
            if metadata.status != "ready":
                raise MunError(f"Model installation is not ready: {candidate_path}")
            return metadata
        ready = [model for model in ready if model.id == model_id or model.path == model_id]
    if not ready:
        target = f" '{model_id}'" if model_id else ""
        raise MunError(f"No installed model{target}. Run: mun models download openai/whisper-small")
    return ready[-1]


def search_models(query: str | None, limit: int, offline: bool) -> list[dict[str, Any]]:
    catalog = load_catalog()["models"]
    tested = {model["id"]: model for model in catalog}
    if offline:
        needle = (query or "").lower()
        return [
            {**model, "compatibility": _catalog_compatibility(model)}
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
        return [_search_record(model, tested) for model in results]
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
        if existing and existing[-1].status == "ready":
            return existing[-1]
        directory = root / f"{_slug(model_id)}--{sha[:12]}"
        directory.mkdir(parents=True, exist_ok=True)
        model = InstalledModel(
            id=model_id,
            revision=sha,
            path=str(directory),
            installed_at=datetime.now(UTC).isoformat(),
            status="invalid",
            trust_remote_code=trust_remote_code,
        )
        _write_metadata(directory, model)
        needed_bytes, ignore_patterns = _download_plan(info.siblings or [])
        free_bytes = shutil.disk_usage(root).free
        if needed_bytes and free_bytes < needed_bytes * 1.1:
            raise MunError(
                f"Insufficient disk space: need about {_human_bytes(needed_bytes)}, "
                f"have {_human_bytes(free_bytes)}"
            )
        snapshot_download(
            repo_id=model_id,
            revision=sha,
            local_dir=directory,
            ignore_patterns=ignore_patterns,
        )
        model = InstalledModel(**{**asdict(model), "status": "ready"})
        _write_metadata(directory, model)
        return model
    except MunError:
        raise
    except Exception as exc:
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


def _search_record(model: Any, tested: dict[str, Any]) -> dict[str, Any]:
    library = getattr(model, "library_name", None)
    pipeline_tag = getattr(model, "pipeline_tag", None)
    if model.id in tested and _catalog_compatibility(tested[model.id]) == "tested":
        compatibility = "tested"
    elif pipeline_tag == "automatic-speech-recognition" and library in {None, "transformers"}:
        compatibility = "metadata-compatible"
    else:
        compatibility = "unsupported"
    return {
        "id": model.id,
        "revision": getattr(model, "sha", None),
        "library": library or "unknown",
        "downloads": getattr(model, "downloads", 0) or 0,
        "likes": getattr(model, "likes", 0) or 0,
        "gated": getattr(model, "gated", False),
        "compatibility": compatibility,
        "quality": tested.get(model.id, {}).get("quality", "unverified"),
    }


def _catalog_compatibility(model: dict[str, Any]) -> str:
    return "tested" if model.get("tested", {}).get("device") not in {None, "unverified"} else "metadata-compatible"


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
