from __future__ import annotations

import random

import pandas as pd


def ground_truth_map(gt):
    mapping = {}
    for r in gt.itertuples(index=False):
        ids = [x.strip() for x in str(r.matched_entity_ids).split(",") if x.strip()]
        mapping[r.source1_entity_id] = set(ids)
    return mapping


def add_pair_labels(features, s1, s2, s3, gt):
    gt_map = ground_truth_map(gt)
    labels = []

    for r in features.itertuples(index=False):
        s1_id = s1.iloc[int(r.s1_index)]["entity_id"]
        target_df = s2 if r.target_source == "S2" else s3
        target_id = target_df.iloc[int(r.target_index)]["entity_id"]
        labels.append(int(target_id in gt_map.get(s1_id, set())))

    out = features.copy()
    out["label"] = labels
    return out


def sample_training_rows(labeled, max_rows=500000, seed=42):
    pos = labeled[labeled["label"] == 1]
    neg = labeled[labeled["label"] == 0]

    if len(pos) + len(neg) <= max_rows:
        return labeled

    rng = random.Random(seed)

    pos_keep = min(len(pos), max(1, int(max_rows * 0.25)))
    neg_keep = max_rows - pos_keep

    pos_idx = rng.sample(list(pos.index), pos_keep)
    neg_idx = rng.sample(list(neg.index), min(len(neg), neg_keep))

    return labeled.loc[pos_idx + neg_idx].reset_index(drop=True)
