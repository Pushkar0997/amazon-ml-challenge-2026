"""STEP 0 - orient: list utils/ and top-level docs; extract README lines relevant to the brief."""
import os
import re

from common import ROOT, Step

S = Step("step0_orient")
S.md("## Step 0 — Orientation", "")
rows = []
for d in ["", "utils"]:
    for f in sorted(os.listdir(os.path.join(ROOT, d) if d else ROOT)):
        p = os.path.join(ROOT, d, f)
        if os.path.isfile(p):
            rows.append((os.path.join(d, f).replace("\\", "/"), f"{os.path.getsize(p):,} B"))
S.md("Files at `student_resource/` top level and in `utils/`:", "")
S.table(["file", "size"], rows)

readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
checks = [
    ("candidate_pairs.tsv mentioned", "candidate_pairs.tsv" in readme),
    ("'not scored on the leaderboard' (candidates)", "not scored on the leaderboard" in readme),
    ("'reduction ratio' mentioned", "reduction ratio" in readme),
    ("any statement that smaller candidate sets improve ranking",
     bool(re.search(r"smaller candidate|fewer candidates|candidate set size", readme, re.I))),
    ("model licence/size rule (MIT/Apache, ≤8B params)", "8 Billion" in readme),
    ("external data lookup prohibited", "STRICTLY NOT ALLOWED" in readme),
]
S.table(["README check", "result"], [(a, "yes" if b else "no") for a, b in checks])
S.finish()
