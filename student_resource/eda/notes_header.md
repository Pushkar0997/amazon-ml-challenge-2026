# EDA Report — Amazon ML Challenge 2026 (business entity resolution)

Read-only investigation, no modelling. Scripts: `student_resource/eda/stepN_*.py` (run from `student_resource/`,
e.g. `python eda/step4_noise.py`; steps 2–5 read caches written by steps 1–2), assembled by `eda/build_report.py`.
Seed 20260925 everywhere. Tooling note: polars/duckdb/pandas are **not installed** in this environment and nothing
was downloaded, so all files are streamed line by line with the Python stdlib (split on `\n`, then on `\t`, no quote
handling) plus numpy/scipy/sklearn. No file is ever loaded whole into memory. `n` = sample size of each number.
