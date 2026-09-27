from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split

from .blocking import generate_all_candidates, recall_at_cap
from .data import (
    add_normalized_columns,
    load_split,
    read_tsv,
    sample_dataframe,
    validate_ground_truth,
    _matched_ids_for_source1,
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

DATASET = ROOT / "student_resource" / "dataset"
OUTPUT = ROOT / "output"
MODELS = ROOT / "models"


def _prepare(df):
    return add_normalized_columns(df)


def _load_prepared_split(data_dir, split, max_s1, target_limit, preprocessed_dir=None):
    """Load s1/s2/s3 (+ gt for train), already carrying src.preprocess's
    columns. With preprocessed_dir, reads the parquet a Kaggle preprocessing
    run already produced instead of recomputing normalize() here."""
    prefix = split

    if preprocessed_dir is None:
        if split == "train":
            s1, s2, s3, gt = load_split(data_dir, "train", max_s1=max_s1, max_target_per_source=target_limit)
        else:
            s1, s2, s3 = load_split(data_dir, "test", max_s1=max_s1, max_target_per_source=target_limit)
            gt = None
        s1, s2, s3 = map(_prepare, (s1, s2, s3))
        return (s1, s2, s3, gt) if split == "train" else (s1, s2, s3)

    preprocessed_dir = Path(preprocessed_dir)
    s1 = pd.read_parquet(preprocessed_dir / f"{prefix}_source1.parquet")
    s2 = pd.read_parquet(preprocessed_dir / f"{prefix}_source2.parquet")
    s3 = pd.read_parquet(preprocessed_dir / f"{prefix}_source3.parquet")

    if max_s1 is not None:
        s1 = s1.head(max_s1).reset_index(drop=True)

    if split == "train":
        gt = validate_ground_truth(
            read_tsv(Path(data_dir) / "train" / "train_ground_truth.tsv"),
            "train_ground_truth.tsv",
        )
        required_target_ids = _matched_ids_for_source1(gt, set(s1["entity_id"]))
    else:
        gt = None
        required_target_ids = set()

    if target_limit is not None:
        s2 = sample_dataframe(s2, target_limit, required_ids=required_target_ids)
        s3 = sample_dataframe(s3, target_limit, required_ids=required_target_ids)

    for df in (s1, s2, s3):
        df["country_norm"] = df["country"].map(lambda x: str(x).strip().lower())

    return (s1, s2, s3, gt) if split == "train" else (s1, s2, s3)


def _candidate_features(s1, s2, s3, max_candidates=None):
    kwargs = {} if max_candidates is None else {"max_candidates": max_candidates}
    candidates = generate_all_candidates(s1, s2, s3, **kwargs)

    target_map = {"S2": s2, "S3": s3}
    features = build_features(s1, target_map, candidates)

    return candidates, features


def _development_target_limit(max_s1):
    if max_s1 is None:
        return None
    return max(5000, min(20000, max_s1 * 20))


def run_train(max_s1=None, data_dir=None, preprocessed_dir=None):
    ensure_dir(OUTPUT)
    ensure_dir(MODELS)

    data_dir = data_dir or DATASET
    target_limit = _development_target_limit(max_s1)

    print("Loading training data...")
    print(f"max_s1={max_s1}, target_limit={target_limit}, preprocessed_dir={preprocessed_dir}")

    s1, s2, s3, gt = _load_prepared_split(data_dir, "train", max_s1, target_limit, preprocessed_dir)

    print(f"Loaded: S1={len(s1):,}, S2={len(s2):,}, S3={len(s3):,}")

    if len(s1) < 2:
        raise RuntimeError("Need at least 2 Source-1 rows for train/validation split.")

    train_ids, valid_ids = train_test_split(s1["entity_id"], test_size=0.15, random_state=42)
    train_ids, valid_ids = set(train_ids), set(valid_ids)

    train_s1 = s1[s1["entity_id"].isin(train_ids)].reset_index(drop=True)
    valid_s1 = s1[s1["entity_id"].isin(valid_ids)].reset_index(drop=True)

    train_gt = gt[gt["source1_entity_id"].isin(train_ids)].reset_index(drop=True)
    valid_gt = gt[gt["source1_entity_id"].isin(valid_ids)].reset_index(drop=True)

    print("Generating training candidates...")
    train_candidates, train_features = _candidate_features(train_s1, s2, s3)
    print(f"Training candidates: {len(train_candidates):,}")

    train_recall = recall_at_cap(train_candidates, train_s1, s2, s3, train_gt)
    print(f"Blocking recall@cap (train, cap={_blocking_cap()}): "
          f"{train_recall['recall']:.3f} ({train_recall['found']:,}/{train_recall['true_pairs']:,} true pairs)")

    train_features = add_pair_labels(train_features, train_s1, s2, s3, train_gt)
    train_features = sample_training_rows(train_features, max_rows=500000, seed=42)

    positives = int(train_features["label"].sum())
    print(f"Training rows: {len(train_features):,}")
    print(f"Positive pairs: {positives:,}")

    if positives == 0:
        raise RuntimeError("No positive training pairs were generated. Check blocking recall and ground-truth IDs.")

    print("Training XGBoost...")
    model = train_xgb(train_features, MODELS / "xgb_matcher.joblib")

    print("Generating validation candidates...")
    valid_candidates, valid_features = _candidate_features(valid_s1, s2, s3)
    print(f"Validation candidates: {len(valid_candidates):,}")

    valid_recall = recall_at_cap(valid_candidates, valid_s1, s2, s3, valid_gt)
    print(f"Blocking recall@cap (valid, cap={_blocking_cap()}): "
          f"{valid_recall['recall']:.3f} ({valid_recall['found']:,}/{valid_recall['true_pairs']:,} true pairs)")

    valid_features = add_pair_labels(valid_features, valid_s1, s2, s3, valid_gt)

    probabilities = predict_proba(model, valid_features)
    truth_map = load_truth(valid_gt)

    threshold_result = choose_threshold(
        valid_s1, s2, s3, valid_features, probabilities, truth_map,
        min_t=0.30, max_t=0.95, step=0.02,
    )
    save_threshold(threshold_result, OUTPUT / "threshold.json")

    scored = valid_features.copy()
    scored["probability"] = probabilities
    final_result = build_matching_results(valid_s1, s2, s3, scored, threshold_result["threshold"])
    pred_map = {
        row.source1_entity_id: {x for x in row.matched_entity_ids.split(",") if x}
        for row in final_result.itertuples(index=False)
    }
    macro_score = score_macro_f05(pred_map, truth_map)

    metrics = {
        "validation_macro_f0_5": macro_score,
        "threshold": threshold_result["threshold"],
        "threshold_macro_f0_5": threshold_result["macro_f0_5"],
        "training_candidate_rows": int(len(train_features)),
        "validation_candidate_rows": int(len(valid_features)),
        "training_positive_rows": positives,
        "validation_positive_rows": int(valid_features["label"].sum()),
        "blocking_recall_at_cap_train": train_recall,
        "blocking_recall_at_cap_valid": valid_recall,
        "blocking_max_candidates": _blocking_cap(),
        "dev_max_s1": max_s1,
        "dev_target_limit": target_limit,
        "train_source1_rows": int(len(train_s1)),
        "train_source2_rows": int(len(s2)),
        "train_source3_rows": int(len(s3)),
    }

    save_json(metrics, OUTPUT / "validation_metrics.json")

    print("Training complete:")
    print(metrics)


def _blocking_cap():
    from .blocking import MAX_CANDIDATES_PER_SOURCE1
    return MAX_CANDIDATES_PER_SOURCE1


def run_test(max_s1=None, data_dir=None, preprocessed_dir=None):
    ensure_dir(OUTPUT)
    ensure_dir(MODELS)

    data_dir = data_dir or DATASET
    model_path = MODELS / "xgb_matcher.joblib"
    threshold_path = OUTPUT / "threshold.json"

    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run: python run_pipeline.py --mode train")

    if threshold_path.exists():
        threshold = json.loads(threshold_path.read_text(encoding="utf-8"))["threshold"]
    else:
        threshold = 0.70

    target_limit = _development_target_limit(max_s1)

    print("Loading test data...")
    print(f"max_s1={max_s1}, target_limit={target_limit}, preprocessed_dir={preprocessed_dir}")

    s1, s2, s3 = _load_prepared_split(data_dir, "test", max_s1, target_limit, preprocessed_dir)

    print(f"Loaded: S1={len(s1):,}, S2={len(s2):,}, S3={len(s3):,}")

    print("Generating test candidates...")
    candidates, features = _candidate_features(s1, s2, s3)
    print(f"Test candidates: {len(candidates):,}")

    import joblib
    model = joblib.load(model_path)

    probabilities = predict_proba(model, features)

    scored = features.copy()
    scored["probability"] = probabilities

    result = build_matching_results(s1, s2, s3, scored, threshold)
    candidate_output = build_candidate_output(s1, s2, s3, candidates)

    result.to_csv(OUTPUT / "matching_results.tsv", sep="\t", index=False)
    candidate_output.to_csv(OUTPUT / "candidate_pairs.tsv", sep="\t", index=False)

    print(f"Wrote {OUTPUT / 'matching_results.tsv'}")
    print(f"Wrote {OUTPUT / 'candidate_pairs.tsv'}")
    print(f"Threshold: {threshold}")
    print(f"Source-1 rows: {len(s1):,}")
    print(f"Candidate rows: {len(candidate_output):,}")


if __name__ == "__main__":
    run_train()
    run_test()
