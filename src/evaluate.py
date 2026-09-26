from __future__ import annotations

import pandas as pd
from sklearn.metrics import fbeta_score


def score_macro_f05(pred_map, truth_map):
    scores = []

    for s1_id, truth in truth_map.items():
        pred = set(pred_map.get(s1_id, set()))

        tp = len(pred & truth)
        fp = len(pred - truth)
        fn = len(truth - pred)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else (1.0 if not pred and not truth else 0.0)

        denom = 0.25 * precision + recall
        f05 = 0.0 if denom == 0 else 1.25 * precision * recall / denom
        scores.append(f05)

    return sum(scores) / len(scores) if scores else 0.0


def load_truth(gt):
    result = {}
    for r in gt.itertuples(index=False):
        result[r.source1_entity_id] = {
            x.strip() for x in str(r.matched_entity_ids).split(",") if x.strip()
        }
    return result
