from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz.fuzz import (
    ratio,
    partial_ratio,
    token_set_ratio,
    token_sort_ratio,
)


def _sim(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return ratio(a, b) / 100.0


def build_features(s1, targets, candidates):
    rows = []

    for c in candidates.itertuples(index=False):
        a = s1.iloc[int(c.s1_index)]
        b = targets[c.target_source].iloc[int(c.target_index)]

        n1, n2 = a["name_norm"], b["name_norm"]
        ad1, ad2 = a["address_norm"], b["address_norm"]

        name_ratio = _sim(n1, n2)
        addr_ratio = _sim(ad1, ad2)

        rows.append({
            "s1_index": c.s1_index,
            "target_index": c.target_index,
            "target_source": c.target_source,
            "name_exact": int(n1 == n2 and bool(n1)),
            "name_ratio": name_ratio,
            "name_partial": partial_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0,
            "name_token_set": token_set_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0,
            "name_token_sort": token_sort_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0,
            "address_exact": int(ad1 == ad2 and bool(ad1)),
            "address_ratio": addr_ratio,
            "address_partial": partial_ratio(ad1, ad2) / 100.0 if ad1 and ad2 else 0.0,
            "address_token_set": token_set_ratio(ad1, ad2) / 100.0 if ad1 and ad2 else 0.0,
            "postal_match": int(bool(a["postal"]) and a["postal"] == b["postal"]),
            "city_match": int(bool(a["city"]) and a["city"] == b["city"]),
            "street_number_match": int(bool(a["street_number"]) and a["street_number"] == b["street_number"]),
            "country_match": int(a["country_norm"] == b["country_norm"]),
            "name_len_diff": abs(len(n1) - len(n2)),
            "address_len_diff": abs(len(ad1) - len(ad2)),
            "combined_ratio": _sim(
                (n1 + " " + ad1).strip(),
                (n2 + " " + ad2).strip(),
            ),
        })

    return pd.DataFrame(rows)


FEATURE_COLUMNS = [
    "name_exact",
    "name_ratio",
    "name_partial",
    "name_token_set",
    "name_token_sort",
    "address_exact",
    "address_ratio",
    "address_partial",
    "address_token_set",
    "postal_match",
    "city_match",
    "street_number_match",
    "country_match",
    "name_len_diff",
    "address_len_diff",
    "combined_ratio",
]
