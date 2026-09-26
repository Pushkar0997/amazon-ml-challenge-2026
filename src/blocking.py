from __future__ import annotations

from collections import defaultdict

import pandas as pd


def _add(index, key, row_id):
    if key and key[-1]:
        index[key].add(row_id)


def _keys_for_row(r) -> list[tuple]:
    country = str(r.get("country_norm", ""))
    name = str(r.get("name_norm", ""))
    postal = str(r.get("postal", ""))
    city = str(r.get("city", ""))
    street = str(r.get("street_number", ""))

    keys = []

    if country and name:
        keys.append(("country", country, "np4", name[:4]))
        keys.append(("country", country, "np6", name[:6]))

    if country and postal:
        keys.append(("country", country, "postal", postal))

    if country and city:
        keys.append(("country", country, "city", city))

    if country and street:
        keys.append(("country", country, "street", street))

    if country and name:
        for token in sorted(set(name.split()))[:4]:
            if len(token) >= 4:
                keys.append(("country", country, "token", token))

    return keys


def build_index(df: pd.DataFrame, allowed_keys: set[tuple] | None = None):
    index = defaultdict(set)

    for row_id, row in df.iterrows():
        for key in _keys_for_row(row):
            if allowed_keys is None or key in allowed_keys:
                _add(index, key, row_id)

    return index


def _query_keys(s1: pd.DataFrame) -> set[tuple]:
    keys = set()

    for _, row in s1.iterrows():
        keys.update(_keys_for_row(row))

    return keys


def generate_candidates(
    s1: pd.DataFrame,
    target: pd.DataFrame,
    max_candidates: int = 500,
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
    max_candidates: int = 500,
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