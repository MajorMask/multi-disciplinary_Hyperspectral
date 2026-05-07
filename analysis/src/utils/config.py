from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def load_config(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".yml", ".yaml"}:
        if yaml is None:
            raise ImportError("PyYAML is required to read YAML configs. Install with `pip install pyyaml`.")
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    raise ValueError(f"Unsupported configuration file format: {suffix}")


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
