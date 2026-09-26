# Amazon ML Challenge 2026 — Business Entity Resolution

Match noisy business records from three sources (Source 1 = clean reference,
Source 2/3 = noisy fragments) with no shared identifier. Scored on macro F0.5
(precision weighted 2x over recall); candidate-set size also affects final
ranking beyond the leaderboard score.

Full problem statement: `student_resource/README.md`.
Organizer rules and submission logistics: `docs/`.

## Status
See `AGENT_LOG.md` for current state and next steps.

## Key facts from EDA (`student_resource/eda/EDA_REPORT.md`)
- S1→S2/S3 mapping is strictly many-to-one: no S2/S3 record matches more than one S1.
- 5.6% of S1 entities are singletons (no match) in train.
- Postcodes are near-absent (<1% coverage) in US/India — not usable for blocking.
- India name matching is unreliable alone: ~28% of true S1–S2 pairs have zero
  name-token overlap (script mismatch, Devanagari/Telugu/Kannada vs Latin).
- Test has ~5.75 S2+S3 records per S1 vs ~4.68 in train — more distractors than
  train reflects; validation must be built to match this ratio.

## Repo layout
- `code/business_entity_resolution/src/` — the actual pipeline (blocking → features → model → inference)
- `student_resource/eda/` — investigation scripts and `EDA_REPORT.md`
- `output/` — regenerated `matching_results.tsv` + `candidate_pairs.tsv` (not tracked)
- `Documentation_template.md` — methodology write-up for final submission

## Running
See `code/business_entity_resolution/README.md` for exact reproduction steps.

## Rules that bind this repo
- No external data, APIs, or lookups (disqualification if violated).
- Final model must be MIT/Apache 2.0, ≤8B params.
- `output/candidate_pairs.tsv` must be a superset of `output/matching_results.tsv`.
- Validate before every leaderboard upload:
  `python3 student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test`