"""Single source of text normalization for blocking and matching.

Pipeline (per record): normalize(name, address, country) -> dict of derived fields.
See the module docstrings below for the exact rules. No network calls, no external
data — only cache/state_aliases.json and cache/city_aliases.json, which this module
itself mines from the training ground truth (see `mine_alias_maps`).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from indic_transliteration import sanscript
except ImportError:  # pragma: no cover - dependency is required at runtime, not at import-time for tests of pure-python paths
    sanscript = None

ROOT = Path(__file__).resolve().parents[1]
# Overridable via env var (not just a CLI flag / function arg) because on
# Windows, multiprocessing.Pool workers are spawned as fresh interpreters that
# re-import this module from scratch — a plain global reassigned in the
# parent's main() after import time would not be visible to them, but an
# inherited environment variable is.
DATA_DIR = Path(os.environ.get("PREPROCESS_DATA_DIR", str(ROOT / "student_resource" / "dataset")))
CACHE_DIR = Path(os.environ.get("PREPROCESS_CACHE_DIR", str(ROOT / "cache")))

# ---------------------------------------------------------------------------
# Base character-level normalisation
# ---------------------------------------------------------------------------
# Same principle as student_resource/eda/common.py's _NormTable (already fixed
# there for the Devanagari-splitting bug): zero-width / format characters (Cf)
# are DELETED, not turned into spaces, so a ZWJ/ZWNJ inside a Devanagari
# conjunct doesn't split the word in two. Punctuation/symbol/separator/control
# characters become a single space. Latin combining diacritics (U+0300-036F)
# are deleted (accent stripping) but combining marks belonging to other
# scripts (Devanagari matras, viramas, etc., which live outside that range)
# are kept untouched.


class _NormTable(dict):
    def __missing__(self, k):
        if 0x300 <= k <= 0x36F or unicodedata.category(chr(k)) == "Cf":
            v = None
        elif unicodedata.category(chr(k))[0] in "PSZC":
            v = " "
        else:
            v = k
        self[k] = v
        return v


_TABLE = _NormTable()


def _base_norm(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s.lower()).translate(_TABLE)
    return " ".join(s.split())


def _ascii_fold(s: str) -> str:
    """Strip any remaining combining marks so transliterated text is pure ASCII."""
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


NULL_TOKENS = {"null", "na", "n a", "none", "nan", "-", ""}

# ---------------------------------------------------------------------------
# Script detection
# ---------------------------------------------------------------------------
_SCRIPT_BLOCKS = [
    ("devanagari", 0x0900, 0x097F),
    ("bengali", 0x0980, 0x09FF),
    ("gurmukhi", 0x0A00, 0x0A7F),
    ("gujarati", 0x0A80, 0x0AFF),
    ("oriya", 0x0B00, 0x0B7F),
    ("tamil", 0x0B80, 0x0BFF),
    ("telugu", 0x0C00, 0x0C7F),
    ("kannada", 0x0C80, 0x0CFF),
    ("malayalam", 0x0D00, 0x0D7F),
]


def detect_script(s: str) -> str:
    """Dominant script of the raw string: 'latin', one of the Indic block names,
    or 'other' when there are no alphabetic characters at all."""
    if not s:
        return "other"
    counts: dict[str, int] = {}
    latin_count = 0
    for ch in s:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        matched = False
        for name, lo, hi in _SCRIPT_BLOCKS:
            if lo <= cp <= hi:
                counts[name] = counts.get(name, 0) + 1
                matched = True
                break
        if not matched:
            latin_count += 1
    if counts:
        top_script, top_n = max(counts.items(), key=lambda kv: kv[1])
        if top_n >= latin_count:
            return top_script
    if latin_count:
        return "latin"
    return "other"


_SANSCRIPT_SCHEME = {}
if sanscript is not None:
    _SANSCRIPT_SCHEME = {
        "devanagari": sanscript.DEVANAGARI,
        "bengali": sanscript.BENGALI,
        "gurmukhi": sanscript.GURMUKHI,
        "gujarati": sanscript.GUJARATI,
        "oriya": sanscript.ORIYA,
        "tamil": sanscript.TAMIL,
        "telugu": sanscript.TELUGU,
        "kannada": sanscript.KANNADA,
        "malayalam": sanscript.MALAYALAM,
    }

# Small set of English-loanword folds applied after ASCII-folding a
# transliterated word. Indian scripts often spell borrowed English words
# phonetically (e.g. Telugu <phu-d> for "food"); this single substitution was
# enough to clear the golden test within the transliteration timebox — it is
# a heuristic, not a general phonetic model, and is intentionally not tuned
# further.
_LOANWORD_FOLDS = [("ph", "f")]


def transliterate_core(text: str, script: str) -> str:
    """Transliterate non-Latin text to a Latin, ASCII-folded form.

    Steps: script -> IAST via indic-transliteration, drop the trailing bare
    'a' inherent-vowel artefact IAST leaves on Devanagari words (Devanagari
    doesn't mark schwa-deletion; Telugu/Tamil/etc. already mark word-final
    virama so this is usually a no-op there), fold all diacritics to ASCII,
    apply the loanword folds above.
    """
    scheme = _SANSCRIPT_SCHEME.get(script)
    if scheme is None or sanscript is None or not text:
        return text
    try:
        iast = sanscript.transliterate(text, scheme, sanscript.IAST)
    except Exception:
        return text
    # Anusvara (ṃ) before a dental is conventionally read/written as 'n' in
    # informal English romanisation (Hindi "आनंद" -> "Anand", not "Anamd").
    iast = iast.replace("ṃ", "n")
    words = []
    for w in iast.split():
        if len(w) > 1 and w[-1] == "a":
            w = w[:-1]
        words.append(w)
    out = _ascii_fold(" ".join(words)).lower()
    for a, b in _LOANWORD_FOLDS:
        out = out.replace(a, b)
    return out


# ---------------------------------------------------------------------------
# Legal-form vocabulary
# ---------------------------------------------------------------------------
UNIVERSAL_LEGAL = {
    "inc": "inc", "incorporated": "inc",
    "corp": "corp", "corporation": "corp",
    "llc": "llc",
    "llp": "llp",
    "lp": "lp",
    "pllc": "pllc",
    "pc": "pc",
    "plc": "plc",
    "co": "co", "company": "co",
    "ltd": "ltd", "limited": "ltd",
}

INDIA_LEGAL = {
    "pvt": "pvt", "private": "pvt",
    "ltd": "ltd", "limited": "ltd",
    "llp": "llp",
    # Devanagari (Hindi)
    "प्राइवेट": "pvt", "प्रा": "pvt", "प्रायवेट": "pvt",
    "लिमिटेड": "ltd", "लि": "ltd",
    # Telugu
    "ప్రైవేట్": "pvt", "లిమిటెడ్": "ltd",
    # Kannada
    "ಪ್ರೈವೇಟ್": "pvt", "ಲಿಮಿಟೆಡ್": "ltd",
    # Tamil
    "பிரைவேட்": "pvt", "லிமிடெட்": "ltd",
}

FRANCE_LEGAL = {
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl",
    "sa": "sa", "sci": "sci", "snc": "snc", "ei": "ei",
}

def _norm_vocab_keys(vocab: dict) -> dict:
    # NFKD decomposes some Indic compatibility vowel signs (e.g. Telugu AI)
    # into two codepoints, same as _base_norm does on real input, so a
    # dictionary literal must go through the same normalisation to match.
    return {_base_norm(k): v for k, v in vocab.items()}


_COUNTRY_LEGAL_VOCAB = {
    "US": _norm_vocab_keys(UNIVERSAL_LEGAL),
    "India": _norm_vocab_keys({**UNIVERSAL_LEGAL, **INDIA_LEGAL}),
    "France": _norm_vocab_keys({**UNIVERSAL_LEGAL, **FRANCE_LEGAL}),
}


def _legal_vocab(country: str) -> dict:
    return _COUNTRY_LEGAL_VOCAB.get(country, _COUNTRY_LEGAL_VOCAB["US"])


def _strip_legal_form(tokens: list[str], country: str) -> tuple[str, list[str]]:
    vocab = _legal_vocab(country)
    matched = [t for t in tokens if t in vocab]
    if not matched:
        return "", tokens
    codes = {vocab[t] for t in matched}
    if {"pvt", "ltd"} <= codes:
        rest = sorted(codes - {"pvt", "ltd"})
        legal_form = "_".join(["pvt_ltd"] + rest)
    else:
        legal_form = "_".join(sorted(codes))
    matched_set = set(matched)
    remaining = [t for t in tokens if t not in matched_set]
    return legal_form, remaining


# ---------------------------------------------------------------------------
# Domain detection
# ---------------------------------------------------------------------------
_DOMAIN_RE = re.compile(r"\b[a-z0-9][a-z0-9-]*\.(com|net|org|io|biz|info|in|co)\b", re.IGNORECASE)
_DOMAIN_TOKENS = {"www", "com", "net", "org", "io", "biz", "info", "co", "in", "uk"}


def _is_domain(raw_name: str) -> bool:
    return bool(_DOMAIN_RE.search(raw_name or ""))


# ---------------------------------------------------------------------------
# Address abbreviation tables (abbreviation -> canonical full form)
# ---------------------------------------------------------------------------
GENERIC_ADDR_ABBR = {
    "st": "street", "str": "street",
    "rd": "road",
    "ave": "avenue", "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "hwy": "highway",
    "pkwy": "parkway",
    "pl": "place",
    "ste": "suite",
    "apt": "apartment",
    "no": "number",
}

INDIA_ADDR_ABBR = {
    "nr": "near", "opp": "opposite", "soc": "society",
    "flr": "floor", "no": "number", "h": "house",
}

FRANCE_ADDR_ABBR = {
    "r": "rue", "n": "number",
    "bd": "boulevard", "av": "avenue", "ave": "avenue",
    "all": "allee", "imp": "impasse", "rte": "route",
    "chem": "chemin", "fg": "faubourg", "pl": "place",
}

_COUNTRY_ADDR_ABBR = {
    "US": {**GENERIC_ADDR_ABBR},
    "India": {**GENERIC_ADDR_ABBR, **INDIA_ADDR_ABBR},
    "France": {**GENERIC_ADDR_ABBR, **FRANCE_ADDR_ABBR},
}


def _addr_abbr(country: str) -> dict:
    return _COUNTRY_ADDR_ABBR.get(country, GENERIC_ADDR_ABBR)


def _expand_addr_tokens(norm_text: str, country: str) -> str:
    abbr = _addr_abbr(country)
    return " ".join(abbr.get(t, t) for t in norm_text.split())


# ---------------------------------------------------------------------------
# Address numbers
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"\d+")


def extract_addr_numbers(raw_address: str) -> list[str]:
    nums, seen = [], set()
    for m in _NUM_RE.finditer(raw_address or ""):
        canon = str(int(m.group(0)))
        if canon not in seen:
            seen.add(canon)
            nums.append(canon)
    return nums


def extract_addr_components(raw_address: str, country: str) -> list[str]:
    out = []
    for part in (raw_address or "").split(","):
        p = _base_norm(part)
        if not p or p in NULL_TOKENS:
            continue
        out.append(_expand_addr_tokens(p, country))
    return out


# ---------------------------------------------------------------------------
# State / city alias maps (mined from train; see mine_alias_maps)
# ---------------------------------------------------------------------------
_STATE_SEED = {
    "France": {},
    "_generic": {},
}
_CITY_SEED = {
    "France": {},
    "_generic": {},
}

_alias_cache: dict[str, dict] = {}


def _load_alias_maps():
    if _alias_cache:
        return _alias_cache["state"], _alias_cache["city"]
    state_path = CACHE_DIR / "state_aliases.json"
    city_path = CACHE_DIR / "city_aliases.json"
    state_map = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else dict(_STATE_SEED)
    city_map = json.loads(city_path.read_text(encoding="utf-8")) if city_path.exists() else dict(_CITY_SEED)
    _alias_cache["state"] = state_map
    _alias_cache["city"] = city_map
    return state_map, city_map


def load_alias_maps():
    """Public accessor for the raw (unpruned, with counts) state/city alias
    maps — used by src/features.py to decide per-pair whether a city_canon
    value is trustworthy enough to compare (see CITY_ALIAS_MIN_COUNT there).
    Preprocessing itself (state_canon/city_canon in the parquet output) always
    uses these maps unpruned; any confidence filtering happens downstream at
    feature time, not by changing what gets written to the parquet."""
    return _load_alias_maps()


def _canon_lookup(alias_map: dict, country: str, value: str) -> str:
    if not value:
        return ""
    table = alias_map.get(country) or alias_map.get("_generic") or {}
    entry = table.get(value)
    if entry:
        return entry["canonical"] if isinstance(entry, dict) else entry
    return value


def state_canon(addr_components: list[str], country: str) -> str:
    if not addr_components:
        return ""
    state_map, _ = _load_alias_maps()
    return _canon_lookup(state_map, country, addr_components[-1])


def city_canon(addr_components: list[str], country: str) -> str:
    if len(addr_components) < 2:
        return ""
    _, city_map = _load_alias_maps()
    return _canon_lookup(city_map, country, addr_components[-2])


# ---------------------------------------------------------------------------
# Per-record normalize()
# ---------------------------------------------------------------------------
def normalize(name: str, address: str, country: str) -> dict:
    name = name or ""
    address = address or ""
    country = country or ""

    script = detect_script(name)
    name_norm = _base_norm(name)
    tokens = name_norm.split()

    legal_form, remaining = _strip_legal_form(tokens, country)

    is_domain = _is_domain(name)
    if is_domain:
        remaining = [t for t in remaining if t not in _DOMAIN_TOKENS]

    name_core = " ".join(remaining)
    name_sorted = " ".join(sorted(remaining))
    name_compact = "".join(remaining)

    if script == "latin" or script == "other":
        name_translit = name_core
    else:
        name_translit = transliterate_core(name_core, script)

    addr_norm = _expand_addr_tokens(_base_norm(address), country)
    addr_components = extract_addr_components(address, country)
    addr_numbers = extract_addr_numbers(address)
    state = state_canon(addr_components, country)
    city = city_canon(addr_components, country)

    return {
        "country": country,
        "script": script,
        "name_norm": name_norm,
        "name_core": name_core,
        "name_sorted": name_sorted,
        "name_compact": name_compact,
        "name_translit": name_translit,
        "legal_form": legal_form,
        "is_domain": is_domain,
        "addr_norm": addr_norm,
        "addr_components": addr_components,
        "addr_numbers": addr_numbers,
        "state_canon": state,
        "city_canon": city,
    }


# ---------------------------------------------------------------------------
# Alias mining (train-only, cached to cache/*.json)
# ---------------------------------------------------------------------------
def _iter_tsv(path: Path):
    with open(path, encoding="utf-8", newline="\n") as f:
        header = next(f).rstrip("\n").split("\t")
        for line in f:
            yield dict(zip(header, line.rstrip("\n").split("\t")))


def _is_alpha_component(s: str) -> bool:
    """Sanity guard against noisy address components (house numbers, phone-like
    digit runs, landmark fragments) leaking into the alias maps: a real
    state/city name is letters (plus combining marks, e.g. Indic matras/virama
    which are Unicode category Mn/Mc, not "alphabetic") and spaces only."""
    stripped = s.replace(" ", "")
    return bool(stripped) and all(
        ch.isalpha() or unicodedata.category(ch)[0] == "M" for ch in stripped
    )


def _looks_like_alias_pair(canonical_side: str, variant_side: str) -> bool:
    """True when the variant side looks like a script/abbreviation variant of
    the canonical (Latin, spelled-out) side rather than an unrelated string."""
    if not canonical_side or not variant_side or canonical_side == variant_side:
        return False
    if not _is_alpha_component(canonical_side) or not _is_alpha_component(variant_side):
        return False
    if detect_script(variant_side) != "latin":
        return True
    if len(variant_side) <= 3 and len(canonical_side) > 3:
        return True
    return False


def mine_alias_maps(
    max_pairs_per_country: int = 150_000,
    seed: int = 42,
    data_dir: Path | None = None,
    cache_dir: Path | None = None,
    sample_s1: int | None = None,
) -> tuple[dict, dict]:
    """Mine state/city alias maps from true S1<->S2/S3 pairs in train.

    Uses pairs where the S1 address is Latin-script/spelled-out and the
    matched address's corresponding component is either non-Latin or a short
    abbreviation, so we learn e.g. 'GJ' -> 'gujarat' or the Devanagari state
    name -> its Latin S1 form. Deterministic: the same fixed-order scan of the
    files is used every run, so results are identical across runs.

    `sample_s1` caps how many Source-1 rows are read at all (for a cheap local
    smoke run); `data_dir`/`cache_dir` override the module defaults so the
    same code path runs against a tiny local checkout or the full Kaggle copy.
    """
    data_dir = data_dir or DATA_DIR
    cache_dir = cache_dir or CACHE_DIR
    train_dir = data_dir / "train"
    s1_path = train_dir / "train_source1.tsv"
    gt_path = train_dir / "train_ground_truth.tsv"

    s1_info: dict[str, tuple[list[str], str]] = {}
    for row in _iter_tsv(s1_path):
        addr = extract_addr_components(row["business_address"], row["country"])
        s1_info[row["entity_id"]] = (addr, row["country"])
        if sample_s1 is not None and len(s1_info) >= sample_s1:
            break

    needed: dict[str, str] = {}
    per_country_count: dict[str, int] = {}
    for row in _iter_tsv(gt_path):
        s1_id = row["source1_entity_id"]
        info = s1_info.get(s1_id)
        if info is None:
            continue
        country = info[1]
        if per_country_count.get(country, 0) >= max_pairs_per_country:
            continue
        matched = [x for x in row["matched_entity_ids"].split(",") if x]
        if not matched:
            continue
        per_country_count[country] = per_country_count.get(country, 0) + 1
        for mid in matched:
            needed[mid] = s1_id

    state_counts: dict[str, dict[str, dict[str, int]]] = {}
    city_counts: dict[str, dict[str, dict[str, int]]] = {}

    def _record(counts, country, variant, canonical):
        bucket = counts.setdefault(country, {}).setdefault(variant, {})
        bucket[canonical] = bucket.get(canonical, 0) + 1

    for fname in ("train_source2.tsv", "train_source3.tsv"):
        for row in _iter_tsv(train_dir / fname):
            s1_id = needed.get(row["entity_id"])
            if s1_id is None:
                continue
            s1_addr, country = s1_info[s1_id]
            match_addr = extract_addr_components(row["business_address"], row["country"])
            if s1_addr and match_addr:
                s1_state, m_state = s1_addr[-1], match_addr[-1]
                if _looks_like_alias_pair(s1_state, m_state):
                    _record(state_counts, country, m_state, s1_state)
                elif _looks_like_alias_pair(m_state, s1_state):
                    _record(state_counts, country, s1_state, m_state)
            if len(s1_addr) >= 2 and len(match_addr) >= 2:
                s1_city, m_city = s1_addr[-2], match_addr[-2]
                if _looks_like_alias_pair(s1_city, m_city):
                    _record(city_counts, country, m_city, s1_city)
                elif _looks_like_alias_pair(m_city, s1_city):
                    _record(city_counts, country, s1_city, m_city)

    def _finalize(counts, min_count=3):
        out = {}
        for country, variants in counts.items():
            table = {}
            for variant, canon_counts in variants.items():
                best_canon = max(canon_counts.items(), key=lambda kv: kv[1])
                if best_canon[1] < min_count:
                    continue  # drop low-confidence singleton/noise observations
                table[variant] = {"canonical": best_canon[0], "count": best_canon[1]}
            out[country] = table
        return out

    state_map = _finalize(state_counts)
    city_map = _finalize(city_counts)

    for seed_map, target in ((_STATE_SEED, state_map), (_CITY_SEED, city_map)):
        for country, table in seed_map.items():
            target.setdefault(country, {})
            for k, v in table.items():
                target[country].setdefault(k, {"canonical": v, "count": 0})

    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "state_aliases.json").write_text(json.dumps(state_map, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    (cache_dir / "city_aliases.json").write_text(json.dumps(city_map, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    _alias_cache.clear()
    return state_map, city_map


# ---------------------------------------------------------------------------
# CLI: stream a split, normalize with multiprocessing, write per-source parquet
# ---------------------------------------------------------------------------
def _process_row(args):
    entity_id, name, address, country = args
    rec = normalize(name, address, country)
    rec["entity_id"] = entity_id
    rec["business_name"] = name
    rec["business_address"] = address
    return rec


_OUTPUT_COLUMNS = [
    "entity_id", "business_name", "business_address", "country", "script",
    "name_norm", "name_core", "name_sorted", "name_compact", "name_translit",
    "legal_form", "is_domain",
    "addr_norm", "addr_components", "addr_numbers", "state_canon", "city_canon",
]


def _read_rows(path: Path, sample_n: int | None = None):
    with open(path, encoding="utf-8", newline="\n") as f:
        header = next(f).rstrip("\n").split("\t")
        idx = {c: i for i, c in enumerate(header)}
        n = 0
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != len(header):
                continue
            yield (
                fields[idx["entity_id"]],
                fields[idx["business_name"]],
                fields[idx["business_address"]],
                fields[idx["country"]],
            )
            n += 1
            if sample_n is not None and n >= sample_n:
                return


def _records_to_table(records: list[dict]) -> pa.Table:
    df = pd.DataFrame.from_records(records)[_OUTPUT_COLUMNS]
    df["addr_components"] = df["addr_components"].apply(list)
    df["addr_numbers"] = df["addr_numbers"].apply(list)
    return pa.Table.from_pandas(df, preserve_index=False)


def preprocess_file(
    src_path: Path,
    out_path: Path,
    sample_n: int | None = None,
    n_workers: int | None = None,
    batch_size: int = 20_000,
) -> int:
    """Stream `src_path` -> `out_path` in batches via pyarrow.ParquetWriter,
    so memory holds at most one batch of records at a time instead of the
    whole (potentially multi-million-row) file. This is what made the earlier
    unsampled local run thrash on an 8GB machine — see AGENT_LOG.md."""
    n_workers = n_workers or os.cpu_count() or 1
    rows = _read_rows(src_path, sample_n=sample_n)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer: pq.ParquetWriter | None = None
    total = 0

    def flush(batch: list[dict]):
        nonlocal writer, total
        if not batch:
            return
        table = _records_to_table(batch)
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), table.schema)
        else:
            table = table.cast(writer.schema)
        writer.write_table(table)
        total += len(batch)

    try:
        batch: list[dict] = []
        if n_workers > 1:
            with mp.Pool(n_workers) as pool:
                for rec in pool.imap(_process_row, rows, chunksize=2000):
                    batch.append(rec)
                    if len(batch) >= batch_size:
                        flush(batch)
                        batch = []
        else:
            for r in rows:
                batch.append(_process_row(r))
                if len(batch) >= batch_size:
                    flush(batch)
                    batch = []
        flush(batch)
    finally:
        if writer is not None:
            writer.close()

    if total == 0:
        # No data rows at all: still write a correctly-typed empty file.
        empty = pd.DataFrame(columns=_OUTPUT_COLUMNS)
        pq.write_table(pa.Table.from_pandas(empty, preserve_index=False), str(out_path))

    return total


def main():
    global DATA_DIR, CACHE_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--sample-s1", type=int, default=None, help="Cap Source-1 rows to N for a quick smoke run; Source-2/3 are capped to N*5 (~the true S1:S2+S3 ratio) so blocking/matching on the sample stays realistic")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None, help="Override the dataset root (contains train/ and test/); defaults to student_resource/dataset")
    parser.add_argument("--cache-dir", type=str, default=None, help="Override the cache root (alias maps + preprocessed/ output); defaults to ./cache")
    parser.add_argument("--mine-aliases", action="store_true", help="(Re)mine <cache-dir>/state_aliases.json and city_aliases.json from train before preprocessing")
    args = parser.parse_args()

    if args.data_dir:
        os.environ["PREPROCESS_DATA_DIR"] = args.data_dir
        DATA_DIR = Path(args.data_dir)
    if args.cache_dir:
        os.environ["PREPROCESS_CACHE_DIR"] = args.cache_dir
        CACHE_DIR = Path(args.cache_dir)

    if args.mine_aliases or not (CACHE_DIR / "state_aliases.json").exists():
        print("Mining state/city alias maps from train...")
        t0 = time.time()
        mine_alias_maps(data_dir=DATA_DIR, cache_dir=CACHE_DIR, sample_s1=args.sample_s1)
        print(f"  done in {time.time() - t0:.1f}s")

    split_dir = DATA_DIR / args.split
    prefix = args.split
    out_dir = CACHE_DIR / "preprocessed"

    for i in (1, 2, 3):
        src = split_dir / f"{prefix}_source{i}.tsv"
        out = out_dir / f"{args.split}_source{i}.parquet"
        sample_n = args.sample_s1 if i == 1 else (args.sample_s1 * 5 if args.sample_s1 else None)
        t0 = time.time()
        n = preprocess_file(src, out, sample_n=sample_n, n_workers=args.workers)
        dt = time.time() - t0
        print(f"source{i}: {n:,} rows -> {out} in {dt:.1f}s ({n / dt if dt else 0:,.0f} rec/s)")


if __name__ == "__main__":
    main()
