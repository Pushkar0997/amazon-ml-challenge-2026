# Agent log

## 2026-09-27 — src/preprocess.py: normalization module + tests + Kaggle offload

### What was built
- `src/preprocess.py`: per-record `normalize(name, address, country) -> dict`
  producing `script, name_norm, name_core, name_sorted, name_compact,
  name_translit, legal_form, is_domain, addr_norm, addr_components,
  addr_numbers, state_canon, city_canon`. Script-aware (Latin accents
  stripped; Devanagari/Telugu/Kannada/Tamil/Gujarati/Bengali/etc. kept,
  zero-width chars deleted not spaced — same fix eda/common.py already had).
  Legal-suffix vocab for US, India (Latin + Devanagari/Telugu/Kannada/Tamil),
  France (SARL/SAS/SASU/EURL/SA/SCI/SNC/EI), generic fallback for any other
  country. Transliteration via indic-transliteration (MIT) -> IAST -> drop
  trailing bare-'a' inherent-vowel artifact -> ASCII-fold -> a couple of
  loanword folds (ph->f, anusvara m->n before dentals).
- `mine_alias_maps()`: mines `state_canon`/`city_canon` alias tables from true
  S1<->S2/S3 pairs in train (not hand-typed), saved to `cache/state_aliases.json`
  / `city_aliases.json` with observation counts, min-count-3 filter to drop
  noise. Both accept `data_dir`/`cache_dir`/`sample_s1` overrides.
- `tests/test_preprocess.py`: 13 golden-case tests from EDA_REPORT examples
  (SLV Marketing group, Prospect Brothers word-order, Vaughn Tailwind domain
  match, French `R.`->`rue`, French `N°`, leading-zero/hyphenated house
  numbers, Telugu Anand Food translit+legal_form, unseen-country Germany,
  Indic-script preservation, zero-width-char bug regression, alias-map
  lookups, empty-address edge case). **All 13 pass.**
- `scripts/eval_preprocess.py`: BEFORE (eda/common.py) vs AFTER (src.preprocess)
  comparison — name-token Jaccard, name-exact-eq %, addr-component Jaccard,
  addr-number sharing, on true pairs vs random same-country non-pairs, plus a
  name_translit-specific row for India non-Latin matches. Takes `--sample-s1`
  and `--data-dir` so the same script runs as a fast local smoke test or the
  full 200k-entity spec'd evaluation.
- `kaggle/run_preprocess.ipynb`: clones the repo (GitHub token from Kaggle
  Secrets, never hard-coded), installs requirements.txt, mines aliases, runs
  full train+test preprocessing, runs the full 200k eval, writes everything
  under `/kaggle/working/`, prints the eval report.
- Fixed `config.yaml` dataset paths (`dataset/train` -> `student_resource/dataset/train`,
  same for test) — they didn't match the actual data location.
- `requirements.txt`: pinned `indic-transliteration==2.3.82` (MIT), noted
  `rapidfuzz` is also MIT.

### Mid-course correction: moved heavy work off the dev machine
The dev machine is an i3 / 8GB RAM box. A first full local run of
`python -m src.preprocess --split train --workers 4` (no sample cap) took
2.2M rows for source1 fine (~3 min, matching the 50k-row benchmark's
~15-25k rec/s), but source2 (5M rows) then ran for **38+ minutes without
finishing** before it was killed — almost certainly memory pressure /
thrashing from accumulating millions of Python dict-of-lists records before
the final `pd.DataFrame.from_records()` call, not something the small-sample
benchmark caught. That run was killed; no corrupted output was left in
`cache/` (the only parquet files present are from the earlier deliberate
`--sample-s1` smoke runs, and `cache/preprocessed/` is now gitignored).

Per updated instructions, all heavy work (full preprocessing, full alias
mining at 150k pairs/country, the 200k-entity eval) is now designated
**Kaggle-only**, run via `kaggle/run_preprocess.ipynb`. Local work is capped
to: pytest golden tests, and a `--sample-s1 2000` smoke run of both the CLI
and the eval script.

### What ran locally (verified, within caps: <3 min wall, <1.5GB RAM)
- `pytest tests/test_preprocess.py -q` — 13 passed, ~2-4s wall.
- `python -m src.preprocess --split train --sample-s1 2000 --workers 2` — 8.5s
  wall: source1 2,000 rows (961 rec/s), source2 10,000 rows (4,526 rec/s),
  source3 10,000 rows (4,186 rec/s). Reused the already-mined
  `cache/state_aliases.json`/`city_aliases.json` (74 India state entries incl.
  16 native-script ones, 45 US state entries, mined earlier from a 150k-
  pairs-per-country scan before the machine-load rule was set — see below).
- `python scripts/eval_preprocess.py --sample-s1 2000 --out-dir <tmp>` — 5.7s
  wall. Produced sane BEFORE/AFTER separation (e.g. US S1-S2 non-pair name
  Jaccard=0 rate 91.3%->98.7%, India non-pair 72.7%->98.6%) but pair-side
  sample sizes were tiny (n=4-8 per country/source, since `--max-scan-rows`
  auto-caps at 40,000 rows/file for a sample this small) — not meaningful
  numbers, just a code-path check. The real numbers are a Kaggle-only
  deliverable.

Note: the alias maps currently committed in `cache/*.json` were mined
*before* the local-machine-load rule took effect, via a direct one-off
`mine_alias_maps()` call (~80-120s wall, not through the smoke-capped CLI
path). They're real and good (e.g. India 'మహారాష్ట్ర'/'महाराष्ट्र'-style
native-script entries at 20k+ occurrences, US 'AZ'->'arizona' at 7.9k), but
should be re-mined on Kaggle at full scale (`--mine-aliases`, no sample cap)
for the final submission so provenance is 100% consistent with "ran on
Kaggle," even though the current ones are already correct.

### What's pending on Kaggle (not yet run — needs a Kaggle dataset slug + GITHUB_TOKEN secret)
1. `kaggle/run_preprocess.ipynb`: fill in `DATASET_SLUG` and `GITHUB_REPO`,
   add the `GITHUB_TOKEN` secret, run top to bottom.
2. Full `--mine-aliases` (no cap) — re-mine and commit the resulting
   `cache/state_aliases.json`/`city_aliases.json` back from Kaggle output.
3. Full `python -m src.preprocess --split {train,test}` (no `--sample-s1`) —
   writes `cache/preprocessed/{split}_source{1,2,3}.parquet`. Projected under
   45 min on 4 cores based on the local 50k-row benchmark (~15-25k rec/s ->
   ~24.2M total records / 20k rec/s ≈ 20 min) — **this projection is not yet
   confirmed at full scale** given the source2 slowdown observed above; Kaggle
   likely has more RAM/cores than the dev box so it should be fine, but this
   is the first thing to check once the notebook runs.
4. `scripts/eval_preprocess.py --sample-s1 200000` — the real BEFORE/AFTER
   numbers against the exit criteria (India S1-S2 name-Jaccard=0 rate 28.4%
   -> under 12%; US/India non-pair metrics should not rise more than a few
   points). Writes `student_resource/eda/PREPROCESS_EVAL.md`.

### Not in scope for this task (flagged, not done)
`src/data.py:add_normalized_columns`, `src/blocking.py:_keys_for_row`,
`src/features.py:build_features`, and `src/pipeline.py:_prepare` all still
call the old `src/utils.py` normalizer (which strips non-ASCII entirely —
the root cause of the India name-Jaccard=0 problem). Switching them over to
`src.preprocess.normalize` is follow-up work, not part of this task's scope.

### Open items / next steps
- Run `kaggle/run_preprocess.ipynb` and pull back: full alias maps, full
  preprocessed parquet (or regenerate on whatever machine trains the model),
  and `student_resource/eda/PREPROCESS_EVAL.md` with real 200k-sample numbers
  against the exit criteria.
- Once the eval confirms the India Jaccard-0 target and clean non-pair
  metrics, wire `add_normalized_columns`/`_keys_for_row`/`build_features`
  over to `src.preprocess.normalize`.
- France has zero mined state/city alias entries (no train coverage for that
  country) — falls back to the identity (component text used as-is for
  `state_canon`/`city_canon`). Acceptable per spec ("generic fallback for any
  unseen country") but worth a manual seed list if France candidate quality
  turns out to need it.

## 2026-09-27 (later) — pipeline wiring, deadline tonight

Reminder from this point on: local machine stays capped to pytest + a
`--sample-s1 2000` smoke run, under 3 min / 1.5GB. All full-scale work
(training, full preprocessing, the 200k eval) is Kaggle-only.

### 1. Pruned noisy city aliases (superseded mid-task — moved to features.py)
First pass: `_load_alias_maps()` dropped any city alias with `count < 20` or
a key shorter than 3 characters at load time, so `city_canon` in the
preprocessed parquet would already reflect the filter. **Corrected
mid-task**: this changes preprocessing output for cities, which the follow-up
instruction explicitly said not to do. Reverted — `city_canon` in the parquet
is unpruned again, exactly as mined. The count/key-length filter now lives in
`src/features.py` (`_trusted_city`, `CITY_ALIAS_MIN_COUNT=20`,
`CITY_ALIAS_MIN_KEY_LEN=3`, see task 4 below): it looks up the raw variant
key (`addr_components[-2]`) against `preprocess.load_alias_maps()`'s
unpruned table at pair-scoring time, and only trusts `city_canon` for the
`city_match` feature when the backing alias entry (if any) clears the bar.
An identity fallback (no alias substitution — `city_canon` is just the raw
normalised text) is still trusted, since the risk here is specifically a
*wrong* low-confidence alias substitution, not a plain literal comparison.
Added `preprocess.load_alias_maps()` as the public accessor features.py uses.

### 2. Streamed parquet output via pyarrow.ParquetWriter
`preprocess_file()` used to accumulate every record into a Python list, then
call `pd.DataFrame.from_records()` + one `to_parquet()` at the end — this is
almost certainly why the earlier full local run hung 38+ minutes on
source2's 5M rows before being killed (see the first Kaggle-offload entry
above). Now writes in 20k-row batches through `pyarrow.ParquetWriter`, so
memory holds at most one batch at a time regardless of file size. Verified
locally at `--sample-s1 2000` scale only (8s wall, correct output); full-scale
behavior is a Kaggle-only concern per the local limits.

### 3. Wired src.preprocess.normalize into add_normalized_columns
`src/data.py:add_normalized_columns` now calls `src.preprocess.normalize`
per row and merges its columns onto the dataframe (keeping `country_norm`
for backward compat). Removed `src/utils.py`'s old ASCII-only
`normalize_name`/`normalize_address`/`extract_postal`/`extract_street_number`/
`extract_city` — that normalizer stripped all non-ASCII characters, which was
the root cause of the India name-Jaccard=0 problem the whole preprocessing
task exists to fix. Also added `data.sample_dataframe()`, a required-ids-aware
row-capper for already-loaded DataFrames, used by the `--preprocessed-dir`
path added in task 7 below.

### 4. features.py rewritten on the new columns
Replaced `name_norm`/`address_norm`/`postal`/`city`/`street_number` features
(columns that no longer exist) with: `name_core`/`name_sorted`/`name_compact`
rapidfuzz similarities; `name_best_ratio` = max(name_core ratio, name_translit
ratio) when either side's script is non-Latin (so an Indic-script name still
gets credit via its transliteration even when the in-script name_core text
doesn't literally match); `address_component_jaccard`; `addr_number_overlap`;
`state_match`/`city_match` (city filtered through the count/key-length check
from task 1); `legal_form_match`. `country_match` unchanged.

### 5. blocking.py: dropped postal keys, capped at 30 (config)
Postcodes are near-absent in this data (<1% coverage, EDA_REPORT.md) so the
old postal-code blocking key was dead weight. Replaced `_keys_for_row` with:
`(country, addr_number, state_canon)` keys (a house number only collides
with candidates in the same state) and `(country, name_core token)` keys
(name_core already has legal suffixes stripped, catching reordered/typo'd
names the old 4/6-char name-prefix key missed). `max_candidates_per_source1`
is now actually read from `config.yaml` (previously config.yaml wasn't
loaded by any code at all) and lowered 500 -> 30. Added
`blocking.recall_at_cap()`, wired into `pipeline.run_train`'s printout.

**Recall@cap on the smoke sample**: the `--sample-s1 2000 --preprocessed-dir`
run (see task 7) reported recall@cap ≈ 0.001 on both train and valid splits.
This is **not** a blocking regression — it's an artifact of how the tiny
smoke parquet was built (`src.preprocess --sample-s1 2000` takes the first
2,000 S1 rows and the first 10,000 S2/S3 rows *by file position*, with no
ground-truth-aware sampling), so the true matches for those particular 2,000
S1 entities are almost never among the first 10,000 S2/S3 rows. Confirmed
blocking itself works correctly with a small contrived example (2 S1 records
with known true S2/S3 matches, both scattered across the full column set):
`recall_at_cap` = 1.0, both true pairs found. The real recall@cap number is a
Kaggle-only deliverable (full data, no sampling artifact).

### 6. postprocess.py: many-to-one assignment + candidate_pairs.tsv format fix
`build_matching_results` now enforces the ground truth's many-to-one
constraint (no S2/S3 record matches more than one S1, confirmed in
EDA_REPORT.md): when several S1 candidates clear the threshold for the same
target, only the highest-scoring one keeps it (adds `best_score`/
`second_best_score`/`margin` columns — one groupby + `nlargest(2)` over the
already-threshold-accepted rows, so it's cheap). Also fixed a pre-existing
bug in `build_candidate_output`: it emitted one row per (s1, candidate) pair
under a singular `candidate_entity_id` column, but the spec requires one row
per S1 with a comma-joined `candidate_entity_ids` list (same shape as
`matching_results.tsv`) — the old format would have failed
`validate_submission.py` at full scale. Not one of the 8 listed tasks, but
directly blocks task 8's "run the validator" exit criterion, so fixed now.

### 7. Threshold tuning on macro F0.5; pipeline wiring
`model.choose_threshold` now sweeps thresholds by actually running
`build_matching_results` (many-to-one + guard) at each candidate value and
scoring with `evaluate.score_macro_f05` (per-S1-entity F0.5, averaged) —
replacing the old flat pair-level `fbeta_score` sweep, which doesn't match
the competition metric (macro-averaged per entity; a correctly-predicted
empty list for a singleton scores 1.0, which a pooled pair-level score can't
represent).

Also found and fixed, while getting the smoke run working:
- `pipeline.py`'s `DATASET` pointed at `ROOT/'dataset'`, which doesn't
  exist (should be `student_resource/dataset`) — this alone made
  `run_pipeline.py` completely unusable, unrelated to today's 8 tasks but a
  blocking bug for testing any of them end to end.
- Added `--data-dir`/`--preprocessed-dir` to `run_pipeline.py`/`pipeline.py`.
  `--preprocessed-dir` loads already-normalized parquet (e.g. from a Kaggle
  preprocessing run, or the local smoke parquet) instead of recomputing
  `add_normalized_columns`, using `data.sample_dataframe` for dev-mode
  capping.
- `features.py` crashed on parquet-loaded data: `addr_components`/
  `addr_numbers` come back as numpy arrays, and `array or []` raises
  "truth value of an array is ambiguous". Added `_as_list()`, which checks
  for `None` explicitly instead of relying on truthiness.

**Local smoke run** (within limits: <3 min, <1.5GB):
`python -m pytest tests/test_preprocess.py -q` → 13 passed.
`python run_pipeline.py --mode all --max-s1 2000 --preprocessed-dir cache/preprocessed`
(against locally-generated `--sample-s1 2000` parquet for train+test) → **1m50s
wall, no crash**, wrote `output/matching_results.tsv` and
`output/candidate_pairs.tsv` in the correct format. Did not attempt the raw
(non-preprocessed-dir) load path locally: `load_split`'s TSV sampling scans
the full multi-million-row S2/S3 files regardless of `--max-s1` (pre-existing
behavior, not touched today) and hung past 2 minutes with zero output on this
machine — exactly the kind of full-file-scan cost the earlier Kaggle-offload
decision was about. The `--preprocessed-dir` path exists precisely to avoid
this locally; the raw path is exercised for real on Kaggle via task 8's
notebook, which doesn't pass `--preprocessed-dir` for training.

Did not commit the smoke run's tiny-sample `output/*.tsv`,
`output/threshold.json`, `output/validation_metrics.json`, or
`models/xgb_matcher.joblib` — those are tracked deliverable files and the
smoke run's 2,000-row sample would have overwritten them with meaningless
numbers. Reverted them to the last real committed state with `git checkout`.

### 8. kaggle/run_pipeline.ipynb
Clones the repo (GitHub token from Kaggle Secrets, never hard-coded),
installs only whichever of numpy/pandas/scikit-learn/xgboost/rapidfuzz/
joblib/pyarrow/PyYAML/indic-transliteration/psutil Kaggle's image is missing
(checked via `importlib.import_module`, not a blind
`pip install -r requirements.txt`, to avoid fighting Kaggle's own pinned
stack), runs full preprocessing (alias mining + normalize train/test), trains
via `run_pipeline.py --mode train --preprocessed-dir ...`, runs inference via
`--mode test`, runs `student_resource/utils/validate_submission.py`, copies
outputs to `/kaggle/working/`, and prints the held-out validation macro F0.5
plus the chosen threshold and blocking recall@cap.

### What's pending on Kaggle (updated)
Everything in the original "pending on Kaggle" list above, plus:
9. `kaggle/run_pipeline.ipynb`: fill in `DATASET_SLUG`/`GITHUB_REPO`, run
   top to bottom. This is what produces the real recall@cap, the real
   validation macro F0.5, and a validator-clean `matching_results.tsv`/
   `candidate_pairs.tsv` at full scale — none of which are meaningful from
   the local 2,000-row smoke sample.
