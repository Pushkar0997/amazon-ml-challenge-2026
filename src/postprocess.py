from __future__ import annotations

import pandas as pd


def _resolve_ids(s1, s2, s3, df):
    s1_ids = [s1.iloc[int(i)]["entity_id"] for i in df["s1_index"]]
    target_ids = [
        (s2.iloc[int(ti)]["entity_id"] if src == "S2" else s3.iloc[int(ti)]["entity_id"])
        for ti, src in zip(df["target_index"], df["target_source"])
    ]
    return s1_ids, target_ids


def build_matching_results(s1, s2, s3, scored_candidates, threshold):
    accepted = scored_candidates[scored_candidates["probability"] >= threshold].copy()

    # Conservative guard: very weak name + address candidates are rejected.
    accepted = accepted[
        ~(
            (accepted["name_core_ratio"] < 0.35)
            & (accepted["address_ratio"] < 0.20)
            & (accepted["state_match"] == 0)
        )
    ]

    if accepted.empty:
        rows = [{"source1_entity_id": s1_id, "matched_entity_ids": ""} for s1_id in s1["entity_id"]]
        return pd.DataFrame(rows)

    accepted["s1_id"], accepted["target_id"] = _resolve_ids(s1, s2, s3, accepted)

    # Many-to-one: the ground truth guarantees each S2/S3 record matches at
    # most one S1 entity, so a target claimed by several S1 candidates goes to
    # whichever one scored highest — everyone else's claim on it is dropped.
    # This directly trades recall for precision on contested targets, which is
    # what F0.5 (precision weighted 2x) rewards.
    group = accepted.groupby(["target_source", "target_id"])["probability"]
    accepted["best_score"] = group.transform("max")
    accepted["second_best_score"] = group.transform(lambda s: s.nlargest(2).min() if len(s) > 1 else 0.0)
    accepted["margin"] = accepted["best_score"] - accepted["second_best_score"]

    winners = accepted[accepted["probability"] == accepted["best_score"]]
    winners = winners.drop_duplicates(subset=["target_source", "target_id"])

    s2_ids = set(s2["entity_id"])
    s3_ids = set(s3["entity_id"])

    matches = {}
    for r in winners.itertuples(index=False):
        if r.target_source == "S2" and r.target_id not in s2_ids:
            continue
        if r.target_source == "S3" and r.target_id not in s3_ids:
            continue
        matches.setdefault(r.s1_id, set()).add(r.target_id)

    rows = []
    for s1_id in s1["entity_id"]:
        ids = sorted(matches.get(s1_id, set()))
        rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(ids),
        })

    return pd.DataFrame(rows)


def build_candidate_output(s1, s2, s3, candidates):
    # One row per S1 with a comma-joined id list — same shape as
    # matching_results.tsv (the spec requires this; a prior version emitted
    # one row per (s1, candidate) pair with a singular column, which the
    # organizer's validator would reject).
    per_s1: dict[str, set[str]] = {}
    if not candidates.empty:
        s1_ids, target_ids = _resolve_ids(s1, s2, s3, candidates)
        for a, b in zip(s1_ids, target_ids):
            per_s1.setdefault(a, set()).add(b)

    rows = []
    for s1_id in s1["entity_id"]:
        ids = sorted(per_s1.get(s1_id, set()))
        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(ids),
        })

    return pd.DataFrame(rows)
