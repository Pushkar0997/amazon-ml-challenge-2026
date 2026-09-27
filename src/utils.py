from __future__ import annotations

import json
from pathlib import Path


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
