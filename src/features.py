from __future__ import annotations

import pandas as pd
from rapidfuzz.fuzz import (
    ratio,
    partial_ratio,
    token_set_ratio,
    token_sort_ratio,
)

from .preprocess import load_alias_maps

# A city_canon value is only trusted for feature purposes when the alias
# entry that produced it (if any) is itself high-confidence. This does NOT
# change what src/preprocess.py writes to the parquet (state_canon/city_canon
# there stay exactly as mined) — it only filters city_canon at scoring time.
# Identity fallbacks (no alias substitution happened at all, city_canon is
# just the raw normalised text) are still trusted: the low-confidence risk is
# specifically in a *wrong* alias substitution, not in a plain literal value.
CITY_ALIAS_MIN_COUNT = 20
CITY_ALIAS_MIN_KEY_LEN = 3


def _sim(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return ratio(a, b) / 100.0


def _as_list(x) -> list:
    # Lists round-trip through parquet as numpy arrays, whose truthiness is
    # ambiguous for `x or []` — check for None explicitly and always return a
    # plain list.
    return [] if x is None else list(x)


def _trusted_city(row) -> str:
    city = row.get("city_canon") or ""
    if not city:
        return ""
    comps = _as_list(row.get("addr_components"))
    if len(comps) < 2:
        return city
    variant = comps[-2]
    if len(variant) < CITY_ALIAS_MIN_KEY_LEN:
        return ""
    _, city_map = load_alias_maps()
    table = city_map.get(row.get("country", "")) or city_map.get("_generic") or {}
    entry = table.get(variant)
    if entry is not None and entry.get("count", 0) < CITY_ALIAS_MIN_COUNT:
        return ""
    return city


def build_features(s1, targets, candidates):
    rows = []

    for c in candidates.itertuples(index=False):
        a = s1.iloc[int(c.s1_index)]
        b = targets[c.target_source].iloc[int(c.target_index)]

        n1, n2 = a["name_core"], b["name_core"]
        ns1, ns2 = a["name_sorted"], b["name_sorted"]
        nc1, nc2 = a["name_compact"], b["name_compact"]
        ad1, ad2 = a["addr_norm"], b["addr_norm"]

        name_core_ratio = _sim(n1, n2)
        best_name_ratio = name_core_ratio
        if a["script"] != "latin" or b["script"] != "latin":
            best_name_ratio = max(best_name_ratio, _sim(a["name_translit"], b["name_translit"]))

        addr_components_a = set(_as_list(a.get("addr_components")))
        addr_components_b = set(_as_list(b.get("addr_components")))
        if addr_components_a or addr_components_b:
            addr_component_jaccard = len(addr_components_a & addr_components_b) / len(addr_components_a | addr_components_b)
        else:
            addr_component_jaccard = 0.0

        addr_numbers_a = set(_as_list(a.get("addr_numbers")))
        addr_numbers_b = set(_as_list(b.get("addr_numbers")))
        addr_number_overlap = len(addr_numbers_a & addr_numbers_b)

        state1, state2 = a.get("state_canon") or "", b.get("state_canon") or ""
        city1, city2 = _trusted_city(a), _trusted_city(b)
        legal1, legal2 = a.get("legal_form") or "", b.get("legal_form") or ""

        rows.append({
            "s1_index": c.s1_index,
            "target_index": c.target_index,
            "target_source": c.target_source,
            "name_core_exact": int(n1 == n2 and bool(n1)),
            "name_core_ratio": name_core_ratio,
            "name_core_partial": partial_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0,
            "name_core_token_set": token_set_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0,
            "name_core_token_sort": token_sort_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0,
            "name_sorted_ratio": _sim(ns1, ns2),
            "name_compact_exact": int(nc1 == nc2 and bool(nc1)),
            "name_compact_ratio": _sim(nc1, nc2),
            "name_best_ratio": best_name_ratio,
            "address_exact": int(ad1 == ad2 and bool(ad1)),
            "address_ratio": _sim(ad1, ad2),
            "address_partial": partial_ratio(ad1, ad2) / 100.0 if ad1 and ad2 else 0.0,
            "address_component_jaccard": addr_component_jaccard,
            "addr_number_overlap": addr_number_overlap,
            "state_match": int(bool(state1) and state1 == state2),
            "city_match": int(bool(city1) and city1 == city2),
            "legal_form_match": int(bool(legal1) and legal1 == legal2),
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
    "name_core_exact",
    "name_core_ratio",
    "name_core_partial",
    "name_core_token_set",
    "name_core_token_sort",
    "name_sorted_ratio",
    "name_compact_exact",
    "name_compact_ratio",
    "name_best_ratio",
    "address_exact",
    "address_ratio",
    "address_partial",
    "address_component_jaccard",
    "addr_number_overlap",
    "state_match",
    "city_match",
    "legal_form_match",
    "country_match",
    "name_len_diff",
    "address_len_diff",
    "combined_ratio",
]
