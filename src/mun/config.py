from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path
from typing import Any

from .errors import MunError

CONFIG_KEYS = {"model", "model_dir", "output_dir", "device", "offline"}


def user_config_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "mun"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "mun"


def default_model_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "mun" / "models"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "mun" / "models"


def config_path() -> Path:
    return user_config_dir() / "config.toml"


def load_config() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        return {}
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise MunError(f"Cannot read configuration {path}: {exc}") from exc
    unknown = set(data) - CONFIG_KEYS
    if unknown:
        raise MunError(f"Unknown configuration field in {path}: {sorted(unknown)[0]}")
    return data


def set_config(key: str, raw_value: str) -> Path:
    if key not in CONFIG_KEYS:
        raise MunError(f"Unknown configuration key: {key}")
    data = load_config()
    if key == "offline":
        lowered = raw_value.lower()
        if lowered not in {"true", "false"}:
            raise MunError("offline must be true or false")
        data[key] = lowered == "true"
    else:
        data[key] = raw_value
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{name} = {_toml_value(data[name])}\n" for name in sorted(data)]
    _atomic_write(path, "".join(lines))
    return path


def reset_config() -> bool:
    path = config_path()
    if not path.exists():
        return False
    path.unlink()
    return True


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
