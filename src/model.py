from __future__ import annotations

from pathlib import Path

import joblib
from xgboost import XGBClassifier

from .evaluate import score_macro_f05
from .features import FEATURE_COLUMNS
from .postprocess import build_matching_results
from .utils import ensure_dir, save_json


def train_xgb(train_df, model_path):
    X = train_df[FEATURE_COLUMNS]
    y = train_df["label"]

    pos = int(y.sum())
    neg = int(len(y) - pos)
    scale_pos_weight = max(1.0, neg / max(pos, 1))

    model = XGBClassifier(
        n_estimators=450,
        max_depth=7,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )

    model.fit(X, y)
    ensure_dir(Path(model_path).parent)
    joblib.dump(model, model_path)
    return model


def predict_proba(model, features):
    return model.predict_proba(features[FEATURE_COLUMNS])[:, 1]


def choose_threshold(s1, s2, s3, valid_features, probabilities, truth_map, min_t=0.30, max_t=0.95, step=0.02):
    """Sweep thresholds and keep the one maximizing macro F0.5 — the actual
    competition metric (per-S1-entity F0.5, averaged) — rather than a flat
    pair-level F0.5 over candidate rows, which rewards a different trade-off
    (pair-level metrics ignore that singletons need an empty prediction to
    score 1.0, and that F0.5 is averaged per entity, not over all pairs
    pooled together). Runs the real many-to-one postprocessing
    (build_matching_results) at each candidate threshold so the threshold
    picked matches what actually produces matching_results.tsv."""
    scored = valid_features.copy()
    scored["probability"] = probabilities

    best = {"threshold": 0.70, "macro_f0_5": -1.0}
    t = min_t
    while t <= max_t + 1e-9:
        result = build_matching_results(s1, s2, s3, scored, t)
        pred_map = {
            row.source1_entity_id: {x for x in row.matched_entity_ids.split(",") if x}
            for row in result.itertuples(index=False)
        }
        score = score_macro_f05(pred_map, truth_map)
        if score > best["macro_f0_5"]:
            best = {"threshold": round(float(t), 4), "macro_f0_5": float(score)}
        t += step

    return best


def save_threshold(result, path):
    save_json(result, path)
