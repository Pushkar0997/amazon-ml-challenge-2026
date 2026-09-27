"""Evaluate src.preprocess against the eda/common.py baseline normalization.

Samples N train Source-1 entities (fixed seed) and, for each, compares its
true S1-S2/S1-S3 matches against a random same-country non-match, BEFORE
(eda/common.py `norm`) vs AFTER (src.preprocess `normalize`). Writes
<out-dir>/PREPROCESS_EVAL.md.

Two run modes, same code path:
  - local smoke test (tiny, fast, low-memory):
      python scripts/eval_preprocess.py --sample-s1 2000
  - full evaluation (Kaggle; this is the one the task spec's numbers refer to):
      python scripts/eval_preprocess.py --sample-s1 200000 --data-dir /kaggle/input/<slug>/dataset
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "student_resource" / "eda"))

import common as eda_common  # noqa: E402

from src.preprocess import (  # noqa: E402
    DATA_DIR as DEFAULT_DATA_DIR,
    _iter_tsv,
    detect_script,
    normalize,
)

SEED = 42


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def before_fields(name: str, address: str):
    tokens = set(eda_common.tokens(name))
    parts = [eda_common.norm(p) for p in (address or "").split(",")]
    components = {p for p in parts if p}
    import re
    numbers = set(re.findall(r"\d+", address or ""))
    name_norm = eda_common.norm(name)
    return tokens, components, numbers, name_norm


def after_fields(name: str, address: str, country: str):
    rec = normalize(name, address, country)
    return (
        set(rec["name_core"].split()),
        set(rec["addr_components"]),
        set(rec["addr_numbers"]),
        rec["name_core"],
        rec["name_translit"],
        rec["script"],
    )


class Accum:
    def __init__(self):
        self.n = 0
        self.name_jaccard_sum = 0.0
        self.name_jaccard_zero = 0
        self.name_core_eq = 0
        self.addr_jaccard_sum = 0.0
        self.addr_number_share = 0

    def add(self, name_j, name_eq, addr_j, num_share):
        self.n += 1
        self.name_jaccard_sum += name_j
        self.name_jaccard_zero += int(name_j == 0.0)
        self.name_core_eq += int(name_eq)
        self.addr_jaccard_sum += addr_j
        self.addr_number_share += int(num_share)

    def row(self, label):
        if self.n == 0:
            return f"| {label} | 0 | - | - | - | - | - |"
        return (
            f"| {label} | {self.n:,} | {self.name_jaccard_sum / self.n:.3f} | "
            f"{100 * self.name_jaccard_zero / self.n:.1f}% | "
            f"{100 * self.name_core_eq / self.n:.1f}% | "
            f"{self.addr_jaccard_sum / self.n:.3f} | "
            f"{100 * self.addr_number_share / self.n:.1f}% |"
        )


def _eval_pair(name1, addr1, country, trow, accums, acc, key):
    name2, addr2 = trow["business_name"], trow["business_address"]

    bt, bc, bn, bname_norm = before_fields(name1, addr1)
    bt2, bc2, bn2, bname_norm2 = before_fields(name2, addr2)
    before_name_j = jaccard(bt, bt2)
    before_addr_j = jaccard(bc, bc2)
    before_num_share = bool(bn & bn2)
    before_name_eq = bname_norm == bname_norm2 and bool(bname_norm)

    at, ac, an, aname_core, _, _ = after_fields(name1, addr1, country)
    at2, ac2, an2, aname_core2, _, _ = after_fields(name2, addr2, country)
    after_name_j = jaccard(at, at2)
    after_addr_j = jaccard(ac, ac2)
    after_num_share = bool(an & an2)
    after_name_eq = aname_core == aname_core2 and bool(aname_core)

    bucket = acc(key)
    bucket["before"].add(before_name_j, before_name_eq, before_addr_j, before_num_share)
    bucket["after"].add(after_name_j, after_name_eq, after_addr_j, after_num_share)


def _eval_translit_row(name1, addr1, country, trow, accum: Accum):
    name2, addr2 = trow["business_name"], trow["business_address"]
    _, _, _, _, translit1, _ = after_fields(name1, addr1, country)
    _, _, _, _, translit2, _ = after_fields(name2, addr2, country)
    t1, t2 = set(translit1.split()), set(translit2.split())
    accum.add(jaccard(t1, t2), translit1 == translit2 and bool(translit1), 0.0, False)


def write_report(out_path, accums, india_translit_pair, india_translit_nonpair, n_s1, dt, rec_per_sec, sample_s1, max_scan_rows):
    lines = []
    lines.append("# Preprocessing evaluation: BEFORE (eda/common.py) vs AFTER (src.preprocess)")
    lines.append("")
    lines.append(f"Sampled {n_s1:,} train Source-1 entities (seed={SEED}, requested sample-s1={sample_s1}, "
                  f"max-scan-rows={max_scan_rows or 'unlimited'}). For each, compared its true S1-S2/S1-S3 match "
                  "(when present) against one random same-country non-match.")
    lines.append("")
    lines.append("| country/pair/kind | n | mean name Jaccard | name Jaccard=0 % | name_core exact-eq % | mean addr-component Jaccard | share addr number % |")
    lines.append("|---|---|---|---|---|---|---|")

    for label in sorted(accums.keys()):
        lines.append(accums[label]["before"].row(f"{label}|BEFORE"))
        lines.append(accums[label]["after"].row(f"{label}|AFTER"))

    lines.append("")
    lines.append("## India pairs where the S2/S3 name is non-Latin (name_translit, AFTER only)")
    lines.append("")
    lines.append("| kind | n | mean translit-token Jaccard | Jaccard=0 % | translit exact-eq % |")
    lines.append("|---|---|---|---|---|")
    for label, accum in (("pair", india_translit_pair), ("non-pair", india_translit_nonpair)):
        if accum.n == 0:
            lines.append(f"| {label} | 0 | - | - | - |")
        else:
            lines.append(
                f"| {label} | {accum.n:,} | {accum.name_jaccard_sum / accum.n:.3f} | "
                f"{100 * accum.name_jaccard_zero / accum.n:.1f}% | "
                f"{100 * accum.name_core_eq / accum.n:.1f}% |"
            )

    lines.append("")
    lines.append("## Throughput")
    lines.append("")
    lines.append(f"Metric computation (normalize() called twice per comparison, BEFORE+AFTER): {rec_per_sec:,.0f} S1 entities/sec over {dt:.1f}s.")
    lines.append("")
    lines.append("See AGENT_LOG.md for the CLI's own end-to-end preprocess throughput and the full-run projection.")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-s1", type=int, default=200_000, help="Number of train Source-1 entities to sample (200000 = the full spec'd eval; use a small number, e.g. 2000, for a fast local smoke test)")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset root (contains train/); defaults to student_resource/dataset")
    parser.add_argument("--pool-cap", type=int, default=None, help="Max same-country S2/S3 records pooled for non-pair sampling; defaults to a value scaled to --sample-s1")
    parser.add_argument("--max-scan-rows", type=int, default=None, help="Cap how many rows of each S2/S3 file are streamed (bounds wall time for small --sample-s1 runs); defaults to unlimited when --sample-s1 >= 50000, else scaled down")
    parser.add_argument("--out-dir", type=str, default=None, help="Directory to write PREPROCESS_EVAL.md into; defaults to student_resource/eda/")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR
    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "student_resource" / "eda")
    out_path = out_dir / "PREPROCESS_EVAL.md"

    pool_cap = args.pool_cap or max(2_000, min(300_000, args.sample_s1 * 2))
    max_scan_rows = args.max_scan_rows
    if max_scan_rows is None and args.sample_s1 < 50_000:
        max_scan_rows = max(5_000, args.sample_s1 * 20)

    train_dir = data_dir / "train"

    print(f"Loading S1 (cap {args.sample_s1:,}) from {train_dir}...")
    # Source files are already effectively shuffled (entity ids are not
    # sequential within them), so taking the first N rows is a fair
    # fixed-seed-deterministic sample without the cost of loading + shuffling
    # the full multi-million-row file just to take a small subset.
    sample_s1_rows = []
    for row in _iter_tsv(train_dir / "train_source1.tsv"):
        sample_s1_rows.append(row)
        if len(sample_s1_rows) >= args.sample_s1:
            break
    sample_ids = {r["entity_id"]: r for r in sample_s1_rows}
    print(f"Sampled {len(sample_ids):,} S1 entities")

    print("Loading ground truth for the sample...")
    true_targets: dict[str, list[str]] = {}
    needed_ids: set[str] = set()
    for row in _iter_tsv(train_dir / "train_ground_truth.tsv"):
        s1_id = row["source1_entity_id"]
        if s1_id not in sample_ids:
            continue
        matched = [x for x in row["matched_entity_ids"].split(",") if x]
        true_targets[s1_id] = matched
        needed_ids.update(matched)

    print(f"Streaming S2/S3 (cap {max_scan_rows or 'unlimited'} rows/file) for true-match rows and country pools...")
    true_records: dict[str, dict] = {}
    pool_by_country_source: dict[tuple[str, str], list[dict]] = {}
    for src_name, src_file in (("S2", "train_source2.tsv"), ("S3", "train_source3.tsv")):
        n_scanned = 0
        for row in _iter_tsv(train_dir / src_file):
            eid = row["entity_id"]
            if eid in needed_ids:
                true_records[eid] = row
            key = (src_name, row["country"])
            bucket = pool_by_country_source.setdefault(key, [])
            if len(bucket) < pool_cap:
                bucket.append(row)
            n_scanned += 1
            if max_scan_rows is not None and n_scanned >= max_scan_rows:
                break

    print("Computing metrics...")
    t0 = time.time()
    n_processed = 0
    rng2 = random.Random(SEED + 1)

    accums = {}

    def acc(key):
        return accums.setdefault(key, {"before": Accum(), "after": Accum()})

    india_translit_pair = Accum()
    india_translit_nonpair = Accum()

    for s1_id, s1_row in sample_ids.items():
        country = s1_row["country"]
        name1, addr1 = s1_row["business_name"], s1_row["business_address"]
        matches = true_targets.get(s1_id, [])
        for source in ("S2", "S3"):
            true_match_id = next((m for m in matches if m.startswith(source)), None)
            pool = pool_by_country_source.get((source, country), [])
            if true_match_id and true_match_id in true_records:
                trow = true_records[true_match_id]
                _eval_pair(name1, addr1, country, trow, accums, acc, f"{country}|S1-{source}|pair")
                if country == "India" and detect_script(trow["business_name"]) != "latin":
                    _eval_translit_row(name1, addr1, country, trow, india_translit_pair)
            if pool:
                cand = None
                for _ in range(3):
                    cand = rng2.choice(pool)
                    if cand["entity_id"] != true_match_id:
                        break
                if cand is not None:
                    _eval_pair(name1, addr1, country, cand, accums, acc, f"{country}|S1-{source}|non-pair")
                    if country == "India" and detect_script(cand["business_name"]) != "latin":
                        _eval_translit_row(name1, addr1, country, cand, india_translit_nonpair)
        n_processed += 1

    dt = time.time() - t0
    rec_per_sec = n_processed / dt if dt else 0.0

    write_report(out_path, accums, india_translit_pair, india_translit_nonpair, n_processed, dt, rec_per_sec, args.sample_s1, max_scan_rows)


if __name__ == "__main__":
    main()
