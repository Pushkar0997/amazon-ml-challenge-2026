from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np


def normalize_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    s = unicodedata.normalize("NFKC", str(value)).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_name(value) -> str:
    s = normalize_text(value)
    replacements = {
        "incorporated": "inc",
        "corporation": "corp",
        "limited": "ltd",
        "company": "co",
        "private": "pvt",
    }
    tokens = [replacements.get(t, t) for t in s.split()]
    while tokens and tokens[-1] in {"inc", "corp", "ltd", "co", "pvt", "llc", "plc"}:
        tokens.pop()
    return " ".join(tokens)


def normalize_address(value) -> str:
    s = normalize_text(value)
    replacements = {
        "street": "st",
        "road": "rd",
        "avenue": "ave",
        "boulevard": "blvd",
        "drive": "dr",
        "lane": "ln",
        "highway": "hwy",
        "parkway": "pkwy",
        "place": "pl",
        "suite": "ste",
        "apartment": "apt",
    }
    return " ".join(replacements.get(t, t) for t in s.split())


def extract_postal(address: str) -> str:
    if not address:
        return ""
    matches = re.findall(r"\b\d{5,6}\b", str(address))
    return matches[-1] if matches else ""


def extract_street_number(address: str) -> str:
    if not address:
        return ""
    m = re.search(r"\b(\d{1,6}[a-zA-Z]?)\b", str(address))
    return m.group(1).lower() if m else ""


def extract_city(address: str) -> str:
    if not address:
        return ""
    parts = [p.strip() for p in re.split(r"[,|]", str(address)) if p.strip()]
    if len(parts) >= 2:
        return normalize_text(parts[-2])
    return ""


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
