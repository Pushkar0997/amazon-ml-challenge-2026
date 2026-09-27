from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import run_train, run_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test", "all"], default="all")
    parser.add_argument("--max-s1", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset root (contains train/ and test/); defaults to student_resource/dataset")
    parser.add_argument("--preprocessed-dir", type=str, default=None, help="Directory of already-preprocessed parquet (from src.preprocess's CLI, e.g. a Kaggle input path) — skips recomputing normalize() if given")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else None
    preprocessed_dir = Path(args.preprocessed_dir) if args.preprocessed_dir else None

    if args.mode in ("train", "all"):
        run_train(max_s1=args.max_s1, data_dir=data_dir, preprocessed_dir=preprocessed_dir)

    if args.mode in ("test", "all"):
        run_test(max_s1=args.max_s1, data_dir=data_dir, preprocessed_dir=preprocessed_dir)


if __name__ == "__main__":
    main()
