"""STEP 3 - test-set shape vs train (row counts per source per country, ratios)."""
import os

import numpy as np

from common import CACHE, COUNTRIES, Step, pct

S = Step("step3_test_shape")
cnt = {}
for split in ("train", "test"):
    for k in (1, 2, 3):
        c = np.load(os.path.join(CACHE, f"{split}_source{k}_ids.npz"))["cc"]
        for i, name in enumerate(COUNTRIES):
            cnt[split, k, name] = int((c == i).sum())
        cnt[split, k, "all"] = len(c)
        assert (c == 9).sum() == 0, "unexpected country label"

S.md("## Step 3 — Test shape vs train", "", "All country labels in all files are within {US, India, France}; "
     "France appears only in test.", "")
rows = []
for split in ("train", "test"):
    for name in COUNTRIES + ["all"]:
        a, b, c = (cnt[split, k, name] for k in (1, 2, 3))
        if a == 0 and b == 0:
            continue
        rows.append((split, name, f"{a:,}", f"{b:,}", f"{c:,}", pct(a, cnt[split, 1, "all"]),
                     pct(b, cnt[split, 2, "all"]), pct(c, cnt[split, 3, "all"]),
                     f"{b / a:.3f}", f"{c / a:.3f}", f"{(b + c) / a:.3f}"))
S.table(["split", "country", "S1 rows", "S2 rows", "S3 rows", "S1 country share", "S2 country share",
         "S3 country share", "S2/S1", "S3/S1", "(S2+S3)/S1"], rows)
g = np.load(os.path.join(CACHE, "gt.npz"))
S.md(f"Reference (train GT): matched S2 per S1 = {g['n2'].mean():.3f}, matched S3 per S1 = {g['n3'].mean():.3f}; "
     f"the remaining S2/S3 rows are unmatched (step 2).", "")
S.finish()
