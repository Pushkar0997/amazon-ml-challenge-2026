"""Assemble eda/EDA_REPORT.md from eda/out/step*.md + eda/notes_*.md + runtimes.json."""
import json
import os

from common import EDA, OUT

STEPS = ["step0_orient", "step1_inventory", "step2_labels", "step3_test_shape", "step4_noise", "step5_blocking"]


def read(p):
    return open(p, encoding="utf-8").read().rstrip() + "\n\n"


parts = [read(os.path.join(EDA, "notes_header.md"))]
for s in STEPS:
    parts.append(read(os.path.join(OUT, s + ".md")))
    extra = os.path.join(EDA, f"notes_{s}.md")
    if os.path.exists(extra):
        parts.append(read(extra))
parts.append(read(os.path.join(EDA, "notes_surprises.md")))
rt = json.load(open(os.path.join(OUT, "runtimes.json")))
lines = ["## Runtime & peak RAM per step", "",
         "Machine: Windows 10, i3 (4 logical cores), 8 GB RAM, Python 3.13, numpy/scipy/sklearn; single process. "
         "Peak = Windows peak working set of the step's process (psutil `peak_wset`).", "",
         "| step | wall time | peak RAM | note |", "|---|---|---|---|"]
for s in STEPS:
    if s in rt:
        r = rt[s]
        lines.append(f"| {s} | {r['seconds'] / 60:.1f} min | {r['peak_gb']:.2f} GB | {r['note'] or '—'} |")
parts.append("\n".join(lines) + "\n")
text = "".join(parts)
open(os.path.join(EDA, "EDA_REPORT.md"), "w", encoding="utf-8").write(text)
print("lines:", text.count("\n"))
