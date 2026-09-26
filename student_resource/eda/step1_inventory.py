"""STEP 1 - inventory of the 7 data files. Streams each file; caches id/country arrays."""
import os
import re

import numpy as np

from common import CACHE, FILES, Step, cc, path, pct, trunc

S = Step("step1_inventory")
NULLISH = {"null", "<null>", "n/a", "na", "none", "nan", "-", "--", "nil", "#n/a"}
ID_RE = re.compile(r"^S[123]-\d+$")

S.md("## Step 1 — Inventory", "",
     "Parsing: raw bytes split on `\\n`, fields split on `\\t`, no quote handling. "
     "`empty` = zero-length/whitespace-only field; `null-like` = whole field is one of "
     "NULL/<NULL>/N/A/None/nan/-/… (case-insensitive). Every column is a string; "
     "`entity_id` checked against `^S[123]-\\d+$`.", "")

summary = []
col_rows = []
bad_examples = []
for key in FILES:
    p = path(key)
    size_mb = os.path.getsize(p) / 2**20
    n_nl = n_cr = 0
    last = b""
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):  # 16 MB chunks
            n_nl += block.count(b"\n"); n_cr += block.count(b"\r"); last = block
    no_final_nl = not last.endswith(b"\n")
    with open(p, encoding="utf-8", newline="\n") as f:
        header = f.readline().rstrip("\n").split("\t")
        ncol = len(header)
        empties = [0] * ncol
        nullish = [0] * ncol
        n_rows = n_bad = n_badid = n_prefix = 0
        ids, ccs = [], []
        countries = {}
        exp_prefix = "S" + key[-1] + "-" if "source" in key else "S1-"
        rows_with_quote = 0
        for ln, line in enumerate(f, start=2):
            n_rows += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) != ncol:
                n_bad += 1
                if len([e for e in bad_examples if e[0] == key]) < 2:
                    bad_examples.append((key, ln, len(parts), trunc(line.rstrip("\n"), 160)))
                continue
            if '"' in line:
                rows_with_quote += 1
            for j, v in enumerate(parts):
                sv = v.strip()
                if not sv:
                    empties[j] += 1
                elif sv.lower() in NULLISH:
                    nullish[j] += 1
            eid = parts[0]
            if not ID_RE.match(eid):
                n_badid += 1
                continue
            if not eid.startswith(exp_prefix):
                n_prefix += 1
            ids.append(int(eid[3:]))
            if ncol == 4:
                countries[parts[3]] = countries.get(parts[3], 0) + 1
                ccs.append(cc(parts[3]))
    ids = np.array(ids, dtype=np.int64)
    u, c = np.unique(ids, return_counts=True)
    n_dup = int((c > 1).sum())
    if ncol == 4:
        np.savez(os.path.join(CACHE, key + "_ids.npz"), ids=ids, cc=np.array(ccs, dtype=np.int8))
    summary.append((key, f"{size_mb:,.1f}", f"{n_nl:,}", f"{n_rows:,}", f"{n_rows - n_bad:,}",
                    f"{n_bad:,}", f"{n_badid:,}", f"{n_prefix:,}", f"{n_dup:,}",
                    f"{n_cr:,}", f"{rows_with_quote:,}",
                    ", ".join(f"{k}:{v:,}" for k, v in sorted(countries.items())) or "—"))
    col_rows.append((key, ", ".join(header), "all str",
                     "; ".join(f"{empties[j]:,} ({pct(empties[j], n_rows - n_bad, 2)})" for j in range(ncol)),
                     "; ".join(f"{nullish[j]:,}" for j in range(ncol))))
    print(key, "done", n_rows, "rows; no final newline" if no_final_nl else "")

S.md("### Files", "")
S.table(["file", "MB", "raw `\\n` count (incl. header)", "data rows", "well-formed rows",
         "malformed (≠ header field count)", "bad id format", "id prefix ≠ file", "duplicate ids",
         "`\\r` bytes", "rows containing `\"`", "country counts"], summary)
S.md("### Columns (per column in header order; % of well-formed rows)", "")
S.table(["file", "columns", "dtype", "empty", "null-like"], col_rows)
S.md("### Malformed row examples", "")
if bad_examples:
    S.table(["file", "line", "#fields", "raw line"], bad_examples)
else:
    S.md("None — every row in every file has exactly the header's field count.", "")
S.finish()
