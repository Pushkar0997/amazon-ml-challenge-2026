"""Shared helpers for the EDA steps. Stdlib + numpy only (polars/duckdb/pandas not installed;
nothing is downloaded). Files are streamed line by line, split on TAB only (no quote handling)."""
import json
import os
import re
import sys
import time
import unicodedata

import numpy as np
import psutil

SEED = 20260925
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # student_resource/
DATA = os.path.join(ROOT, "dataset")
EDA = os.path.join(ROOT, "eda")
OUT = os.path.join(EDA, "out")
CACHE = os.path.join(EDA, "cache")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

FILES = {
    "train_source1": "train/train_source1.tsv",
    "train_source2": "train/train_source2.tsv",
    "train_source3": "train/train_source3.tsv",
    "train_ground_truth": "train/train_ground_truth.tsv",
    "test_source1": "test/test_source1.tsv",
    "test_source2": "test/test_source2.tsv",
    "test_source3": "test/test_source3.tsv",
}
SOURCE_FILES = [k for k in FILES if "source" in k]
COUNTRIES = ["US", "India", "France"]
CCODE = {c: i for i, c in enumerate(COUNTRIES)}  # anything else -> 9


def path(key):
    return os.path.join(DATA, FILES[key])


def iter_lines(key):
    """Yield (lineno, fields) for data rows (header skipped). Splits on '\\n' only."""
    with open(path(key), encoding="utf-8", newline="\n") as f:
        next(f)
        for i, line in enumerate(f, start=2):
            yield i, line.rstrip("\n").split("\t")


def iter_records(key):
    """Yield (entity_id, name, address, country) for well-formed source rows."""
    for _, p in iter_lines(key):
        if len(p) == 4:
            yield p[0], p[1], p[2], p[3]


def id_int(eid):
    return int(eid[3:])


def cc(country):
    return CCODE.get(country, 9)


class _NormTable(dict):
    """str.translate table: Latin combining diacritics (U+0300-036F) -> deleted;
    punctuation/symbol/separator/control -> space; everything else (incl. Indic marks) kept."""

    def __missing__(self, k):
        if 0x300 <= k <= 0x36F or unicodedata.category(chr(k)) == "Cf":  # accents, ZWJ/ZWNJ
            v = None
        elif unicodedata.category(chr(k))[0] in "PSZC":
            v = " "
        else:
            v = k
        self[k] = v
        return v


_TABLE = _NormTable()


def norm(s):
    """Baseline normalisation: lowercase, strip accents, collapse punctuation+whitespace."""
    s = unicodedata.normalize("NFKD", s.lower()).translate(_TABLE)
    return " ".join(s.split())


def tokens(s):
    return norm(s).split()


# Postcode heuristics (documented in the report).
_IN_PIN = re.compile(r"(?<!\d)([1-9]\d{2})\s?(\d{3})(?!\d)")
_US_ZIP = re.compile(r"(?:\b[A-Za-z]{2}\s+|^\s*|,\s*)(\d{5})(?:-\d{4})?\s*(?:,|$)")
_FR_CP = re.compile(
    r"(?:^|,)\s*(\d{5})(?!\s+(?:r\b|r\.|rue|av|bd|boulevard|chemin|place|all|impasse|route|quai|cours))"
    r"(?:\s+[^\d,][^,]*)?\s*(?=,|$)", re.IGNORECASE)


def postcode(addr, country):
    if country == "India":
        m = _IN_PIN.search(addr)
        return m.group(1) + m.group(2) if m else None
    if country == "US":
        m = _US_ZIP.search(addr)
        return m.group(1) if m else None
    if country == "France":
        m = _FR_CP.search(addr)
        return m.group(1) if m else None
    return None


def trunc(s, n=120):
    s = s.replace("|", "/")
    return s if len(s) <= n else s[: n - 1] + "…"


class Step:
    """Times a step, records Windows peak working set, writes markdown section."""

    def __init__(self, name):
        self.name = name
        self.t0 = time.time()
        self.lines = []
        np.random.seed(SEED)

    def md(self, *lines):
        for l in lines:
            self.lines.append(l)
            print(l)
        sys.stdout.flush()

    def table(self, header, rows):
        self.md("| " + " | ".join(map(str, header)) + " |",
                "|" + "|".join("---" for _ in header) + "|")
        for r in rows:
            self.md("| " + " | ".join(map(str, r)) + " |")
        self.md("")

    def rss_gb(self):
        return psutil.Process().memory_info().rss / 1e9

    def finish(self, note=""):
        mi = psutil.Process().memory_info()
        peak = getattr(mi, "peak_wset", mi.rss) / 1e9
        dt = time.time() - self.t0
        with open(os.path.join(OUT, self.name + ".md"), "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")
        rt_path = os.path.join(OUT, "runtimes.json")
        rt = json.load(open(rt_path)) if os.path.exists(rt_path) else {}
        rt[self.name] = {"seconds": round(dt, 1), "peak_gb": round(peak, 2), "note": note}
        json.dump(rt, open(rt_path, "w"), indent=1)
        print(f"[{self.name}] {dt:.0f}s, peak working set {peak:.2f} GB")


def fmt(x, d=1):
    return f"{x:,.{d}f}"


def pct(a, b, d=1):
    return "n/a" if b == 0 else f"{100.0 * a / b:.{d}f}%"
