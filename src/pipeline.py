from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split

from .blocking import generate_all_candidates
from .data import (
    add_normalized_columns,
    load_split,
)
from .evaluate import (
    load_truth,
    score_macro_f05,
)
from .features import build_features
from .labels import (
    add_pair_labels,
    sample_training_rows,
)
from .model import (
    choose_threshold,
    predict_proba,
    save_threshold,
    train_xgb,
)
from .postprocess import (
    build_candidate_output,
    build_matching_results,
)
from .utils import (
    ensure_dir,
    save_json,
)


ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "dataset"
OUTPUT = ROOT / "output"
MODELS = ROOT / "models"


def _prepare(df):

    return add_normalized_columns(df)


def _candidate_features(
    s1,
    s2,
    s3,
    max_candidates=500,
):

    candidates = generate_all_candidates(
        s1,
        s2,
        s3,
        max_candidates=max_candidates,
    )

    target_map = {
        "S2": s2,
        "S3": s3,
    }

    features = build_features(
        s1,
        target_map,
        candidates,
    )

    return candidates, features


def _development_target_limit(
    max_s1,
):

    if max_s1 is None:
        return None

    return max(
        5000,
        min(
            20000,
            max_s1 * 20,
        ),
    )


def run_train(max_s1=None):

    ensure_dir(OUTPUT)
    ensure_dir(MODELS)

    target_limit = (
        _development_target_limit(
            max_s1
        )
    )

    print(
        "Loading training data..."
    )

    print(
        f"max_s1={max_s1}, "
        f"target_limit={target_limit}"
    )

    s1, s2, s3, gt = load_split(
        DATASET,
        "train",
        max_s1=max_s1,
        max_target_per_source=target_limit,
    )

    print(
        f"Loaded: "
        f"S1={len(s1):,}, "
        f"S2={len(s2):,}, "
        f"S3={len(s3):,}"
    )

    s1, s2, s3 = map(
        _prepare,
        (s1, s2, s3),
    )

    if len(s1) < 2:

        raise RuntimeError(
            "Need at least 2 Source-1 "
            "rows for train/validation split."
        )

    train_ids, valid_ids = (
        train_test_split(
            s1["entity_id"],
            test_size=0.15,
            random_state=42,
        )
    )

    train_ids = set(train_ids)
    valid_ids = set(valid_ids)

    train_s1 = (
        s1[
            s1["entity_id"]
            .isin(train_ids)
        ]
        .reset_index(drop=True)
    )

    valid_s1 = (
        s1[
            s1["entity_id"]
            .isin(valid_ids)
        ]
        .reset_index(drop=True)
    )

    train_gt = (
        gt[
            gt["source1_entity_id"]
            .isin(train_ids)
        ]
        .reset_index(drop=True)
    )

    valid_gt = (
        gt[
            gt["source1_entity_id"]
            .isin(valid_ids)
        ]
        .reset_index(drop=True)
    )

    print(
        "Generating training candidates..."
    )

    train_candidates, train_features = (
        _candidate_features(
            train_s1,
            s2,
            s3,
        )
    )

    print(
        f"Training candidates: "
        f"{len(train_candidates):,}"
    )

    train_features = add_pair_labels(
        train_features,
        train_s1,
        s2,
        s3,
        train_gt,
    )

    train_features = sample_training_rows(
        train_features,
        max_rows=500000,
        seed=42,
    )

    positives = int(
        train_features["label"].sum()
    )

    print(
        f"Training rows: "
        f"{len(train_features):,}"
    )

    print(
        f"Positive pairs: "
        f"{positives:,}"
    )

    if positives == 0:

        raise RuntimeError(
            "No positive training pairs "
            "were generated. Check blocking "
            "recall and ground-truth IDs."
        )

    print(
        "Training XGBoost..."
    )

    model = train_xgb(
        train_features,
        MODELS
        / "xgb_matcher.joblib",
    )

    print(
        "Generating validation candidates..."
    )

    valid_candidates, valid_features = (
        _candidate_features(
            valid_s1,
            s2,
            s3,
        )
    )

    print(
        f"Validation candidates: "
        f"{len(valid_candidates):,}"
    )

    valid_features = add_pair_labels(
        valid_features,
        valid_s1,
        s2,
        s3,
        valid_gt,
    )

    probabilities = predict_proba(
        model,
        valid_features,
    )

    threshold_result = choose_threshold(
        valid_features,
        probabilities,
        min_t=0.30,
        max_t=0.95,
        step=0.02,
    )

    save_threshold(
        threshold_result,
        OUTPUT / "threshold.json",
    )

    scored = valid_features.copy()

    scored["probability"] = (
        probabilities
    )

    pred_map = {}

    for r in scored[
        scored["probability"]
        >= threshold_result[
            "threshold"
        ]
    ].itertuples(index=False):

        s1_id = (
            valid_s1
            .iloc[int(r.s1_index)]
            ["entity_id"]
        )

        target_df = (
            s2
            if r.target_source == "S2"
            else s3
        )

        target_id = (
            target_df
            .iloc[int(r.target_index)]
            ["entity_id"]
        )

        pred_map.setdefault(
            s1_id,
            set(),
        ).add(target_id)

    truth_map = load_truth(
        valid_gt
    )

    macro_score = score_macro_f05(
        pred_map,
        truth_map,
    )

    metrics = {

        "validation_macro_f0_5":
            macro_score,

        "threshold":
            threshold_result[
                "threshold"
            ],

        "threshold_pair_f0_5":
            threshold_result[
                "f0_5"
            ],

        "training_candidate_rows":
            int(len(train_features)),

        "validation_candidate_rows":
            int(len(valid_features)),

        "training_positive_rows":
            positives,

        "validation_positive_rows":
            int(
                valid_features[
                    "label"
                ].sum()
            ),

        "dev_max_s1":
            max_s1,

        "dev_target_limit":
            target_limit,

        "train_source1_rows":
            int(len(train_s1)),

        "train_source2_rows":
            int(len(s2)),

        "train_source3_rows":
            int(len(s3)),
    }

    save_json(
        metrics,
        OUTPUT
        / "validation_metrics.json",
    )

    print(
        "Training complete:"
    )

    print(metrics)


def run_test(max_s1=None):

    ensure_dir(OUTPUT)
    ensure_dir(MODELS)

    model_path = (
        MODELS
        / "xgb_matcher.joblib"
    )

    threshold_path = (
        OUTPUT
        / "threshold.json"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            "Model not found. Run: "
            "python run_pipeline.py "
            "--mode train"
        )

    if threshold_path.exists():

        threshold = json.loads(
            threshold_path.read_text(
                encoding="utf-8"
            )
        )["threshold"]

    else:

        threshold = 0.70

    target_limit = (
        _development_target_limit(
            max_s1
        )
    )

    print(
        "Loading test data..."
    )

    print(
        f"max_s1={max_s1}, "
        f"target_limit={target_limit}"
    )

    s1, s2, s3 = load_split(
        DATASET,
        "test",
        max_s1=max_s1,
        max_target_per_source=target_limit,
    )

    print(
        f"Loaded: "
        f"S1={len(s1):,}, "
        f"S2={len(s2):,}, "
        f"S3={len(s3):,}"
    )

    s1, s2, s3 = map(
        _prepare,
        (s1, s2, s3),
    )

    print(
        "Generating test candidates..."
    )

    candidates, features = (
        _candidate_features(
            s1,
            s2,
            s3,
        )
    )

    print(
        f"Test candidates: "
        f"{len(candidates):,}"
    )

    import joblib

    model = joblib.load(
        model_path
    )

    probabilities = predict_proba(
        model,
        features,
    )

    scored = features.copy()

    scored["probability"] = (
        probabilities
    )

    result = build_matching_results(
        s1,
        s2,
        s3,
        scored,
        threshold,
    )

    candidate_output = (
        build_candidate_output(
            s1,
            s2,
            s3,
            candidates,
        )
    )

    result.to_csv(
        OUTPUT
        / "matching_results.tsv",
        sep="\t",
        index=False,
    )

    candidate_output.to_csv(
        OUTPUT
        / "candidate_pairs.tsv",
        sep="\t",
        index=False,
    )

    print(
        f"Wrote "
        f"{OUTPUT / 'matching_results.tsv'}"
    )

    print(
        f"Wrote "
        f"{OUTPUT / 'candidate_pairs.tsv'}"
    )

    print(
        f"Threshold: {threshold}"
    )

    print(
        f"Source-1 rows: "
        f"{len(s1):,}"
    )

    print(
        f"Candidate rows: "
        f"{len(candidate_output):,}"
    )


if __name__ == "__main__":

    run_train()
    run_test()