from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, fbeta_score
from xgboost import XGBClassifier

from .features import FEATURE_COLUMNS
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


def choose_threshold(valid_df, probabilities, min_t=0.30, max_t=0.95, step=0.02):
    y = valid_df["label"].to_numpy()
    best = {"threshold": 0.70, "f0_5": -1.0}

    t = min_t
    while t <= max_t + 1e-9:
        pred = (probabilities >= t).astype(int)
        score = fbeta_score(y, pred, beta=0.5, zero_division=0)
        if score > best["f0_5"]:
            best = {"threshold": round(float(t), 4), "f0_5": float(score)}
        t += step

    return best


def save_threshold(result, path):
    save_json(result, path)
