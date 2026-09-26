from __future__ import annotations

import pandas as pd


def build_matching_results(s1, s2, s3, scored_candidates, threshold):
    accepted = scored_candidates[scored_candidates["probability"] >= threshold].copy()

    # Conservative guard: very weak name + address candidates are rejected.
    accepted = accepted[
        ~(
            (accepted["name_ratio"] < 0.35)
            & (accepted["address_ratio"] < 0.20)
            & (accepted["postal_match"] == 0)
        )
    ]

    s2_ids = set(s2["entity_id"])
    s3_ids = set(s3["entity_id"])

    matches = {}

    for r in accepted.itertuples(index=False):
        s1_id = s1.iloc[int(r.s1_index)]["entity_id"]
        target_df = s2 if r.target_source == "S2" else s3
        target_id = target_df.iloc[int(r.target_index)]["entity_id"]

        if r.target_source == "S2" and target_id not in s2_ids:
            continue
        if r.target_source == "S3" and target_id not in s3_ids:
            continue

        matches.setdefault(s1_id, set()).add(target_id)

    rows = []
    for s1_id in s1["entity_id"]:
        ids = sorted(matches.get(s1_id, set()))
        rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(ids),
        })

    return pd.DataFrame(rows)


def build_candidate_output(s1, s2, s3, candidates):
    rows = []

    for r in candidates.itertuples(index=False):
        s1_id = s1.iloc[int(r.s1_index)]["entity_id"]
        target_df = s2 if r.target_source == "S2" else s3
        target_id = target_df.iloc[int(r.target_index)]["entity_id"]

        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_id": target_id,
        })

    return pd.DataFrame(rows).drop_duplicates()
