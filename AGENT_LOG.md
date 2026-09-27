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

### 1. Pruned noisy city aliases at load time
`cache/city_aliases.json` was mined with only a min-count-3 filter (see
above), which let through a long tail of 2-3-letter fragment keys and
single-digit counts that are more noise than signal for city names
specifically (state aliases don't have this problem as badly — their
frequencies are much higher). `_load_alias_maps()` now drops any city alias
with `count < 20` or a key shorter than 3 characters at **load time**
(`_prune_city_aliases`), rather than re-mining the cached file — the raw
mined counts stay in `cache/city_aliases.json` for inspection, only the
in-memory table used by `city_canon()` is filtered. State aliases are
untouched (not asked for, and their counts are already much higher-confidence).
