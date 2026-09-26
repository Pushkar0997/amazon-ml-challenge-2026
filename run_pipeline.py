from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import run_train, run_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test", "all"], default="all")
    parser.add_argument("--max-s1", type=int, default=None)
    args = parser.parse_args()

    if args.mode in ("train", "all"):
        run_train(max_s1=args.max_s1)

    if args.mode in ("test", "all"):
        run_test(max_s1=args.max_s1)


if __name__ == "__main__":
    main()
