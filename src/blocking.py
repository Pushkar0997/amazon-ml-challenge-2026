from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def _default_max_candidates() -> int:
    try:
        cfg = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
        return int(cfg.get("blocking", {}).get("max_candidates_per_source1", 30))
    except Exception:
        return 30


MAX_CANDIDATES_PER_SOURCE1 = _default_max_candidates()


def _keys_for_row(r) -> list[tuple]:
    """Blocking keys: country + addr_numbers (paired with state_canon, so a
    house number only collides within the same state) and country + name_core
    tokens. Replaces the old postal-code / name-prefix keys — postcodes are
    near-absent in this data (see EDA_REPORT.md) and name prefixes miss
    reordered/typo'd names that name_core tokens still catch."""
    country = str(r.get("country_norm", ""))
    if not country:
        return []

    name_core = str(r.get("name_core", ""))
    state = str(r.get("state_canon", ""))
    # Lists round-trip through parquet as numpy arrays, whose truthiness is
    # ambiguous for `x or []` — check for None explicitly instead.
    addr_numbers = r.get("addr_numbers")
    if addr_numbers is None:
        addr_numbers = []

    keys = []

    for num in addr_numbers:
        keys.append(("country", country, "addrnum_state", str(num), state))

    for token in sorted(set(name_core.split()))[:4]:
        if len(token) >= 3:
            keys.append(("country", country, "namecore_tok", token))

    return keys


def build_index(df: pd.DataFrame, allowed_keys: set[tuple] | None = None):
    index = defaultdict(set)

    for row_id, row in df.iterrows():
        for key in _keys_for_row(row):
            if allowed_keys is None or key in allowed_keys:
                index[key].add(row_id)

    return index


def _query_keys(s1: pd.DataFrame) -> set[tuple]:
    keys = set()

    for _, row in s1.iterrows():
        keys.update(_keys_for_row(row))

    return keys


def generate_candidates(
    s1: pd.DataFrame,
    target: pd.DataFrame,
    max_candidates: int = MAX_CANDIDATES_PER_SOURCE1,
):
    allowed_keys = _query_keys(s1)

    index = build_index(
        target,
        allowed_keys=allowed_keys,
    )

    rows = []

    for s1_index, row in s1.iterrows():

        candidates = set()

        for key in _keys_for_row(row):
            candidates.update(index.get(key, set()))

        if len(candidates) > max_candidates:
            candidates = set(
                sorted(candidates)[:max_candidates]
            )

        for target_index in candidates:
            rows.append(
                (
                    s1_index,
                    target_index,
                )
            )

    return pd.DataFrame(
        rows,
        columns=[
            "s1_index",
            "target_index",
        ],
    )


def generate_all_candidates(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    max_candidates: int = MAX_CANDIDATES_PER_SOURCE1,
):
    c2 = generate_candidates(
        s1,
        s2,
        max_candidates=max_candidates,
    )

    c2["target_source"] = "S2"

    c3 = generate_candidates(
        s1,
        s3,
        max_candidates=max_candidates,
    )

    c3["target_source"] = "S3"

    return pd.concat(
        [
            c2,
            c3,
        ],
        ignore_index=True,
    )


def recall_at_cap(candidates: pd.DataFrame, s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame, gt: pd.DataFrame) -> dict:
    """Fraction of true S1<->S2/S3 pairs (restricted to the S1 rows in `s1`)
    that survive into `candidates` (i.e. blocking's ceiling on recall before
    the matcher ever runs). Returns {"recall": float, "true_pairs": int, "found": int}."""
    target_map = {"S2": s2, "S3": s3}
    found_pairs = set()
    for c in candidates.itertuples(index=False):
        s1_id = s1.iloc[int(c.s1_index)]["entity_id"]
        target_id = target_map[c.target_source].iloc[int(c.target_index)]["entity_id"]
        found_pairs.add((s1_id, target_id))

    s1_ids = set(s1["entity_id"])
    total_true = 0
    found_true = 0
    for r in gt.itertuples(index=False):
        if r.source1_entity_id not in s1_ids:
            continue
        for target_id in str(r.matched_entity_ids).split(","):
            target_id = target_id.strip()
            if not target_id:
                continue
            total_true += 1
            if (r.source1_entity_id, target_id) in found_pairs:
                found_true += 1

    recall = found_true / total_true if total_true else 1.0
    return {"recall": recall, "true_pairs": total_true, "found": found_true}
