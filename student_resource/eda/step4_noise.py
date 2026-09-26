"""STEP 4 - noise: examples, true-pair similarity stats, tokens, lengths, scripts."""
import heapq
import os
import random
import re
from array import array
from collections import Counter

import numpy as np

from common import (CACHE, COUNTRIES, SEED, SOURCE_FILES, Step, iter_records, norm, pct, postcode,
                    trunc)

S = Step("step4_noise")
rng = random.Random(SEED)
nrng = np.random.default_rng(SEED)

LATIN_ACC = re.compile("[À-ɏḀ-ỿ]")
NON_LATIN = re.compile("[^\u0000-ɏḀ-ỿ -⁯₠-⃏℀-⅏]")
NULL_TOK = re.compile(r"(?:^|,)\s*(?:null|<null>|n/a|none|nan)\s*(?=,|$)", re.IGNORECASE)
DOMAIN = re.compile(r"\.(?:com|in|net|org|co|fr|io|biz)\b", re.IGNORECASE)
TOKEN_SAMPLE = 4  # every 4th record contributes to token counts

# ---------------- load S1 train text + pair index ----------------
pairs = np.load(os.path.join(CACHE, "pairs.npz"))
p_s1, p_src, p_mid = pairs["s1"], pairs["src"], pairs["mid"]
s1_ids, s1_name, s1_addr, s1_cc = [], [], [], []
for eid, nm, ad, c in iter_records("train_source1"):
    s1_ids.append(int(eid[3:])); s1_name.append(nm); s1_addr.append(ad); s1_cc.append(c)
s1_ids = np.array(s1_ids, np.int64)
s1_order = np.argsort(s1_ids); s1_sorted = s1_ids[s1_order]
s1_pc = [postcode(a, c) for a, c in zip(s1_addr, s1_cc)]
s1_row = s1_order[np.searchsorted(s1_sorted, p_s1)]  # row index of each pair's S1
print("S1 loaded", S.rss_gb())

# example groups: 8 random matched S1 per country
gt = np.load(os.path.join(CACHE, "gt.npz"))
ex_groups = {}
for ci, c in enumerate(["US", "India"]):
    ids = gt["s1"][(gt["cc"] == COUNTRIES.index(c)) & (gt["n2"] + gt["n3"] > 0)]
    ex_groups[c] = [int(x) for x in nrng.choice(ids, 8, replace=False)]
want_s1 = {x for v in ex_groups.values() for x in v}
sel = np.isin(p_s1, list(want_s1))
want_mid = {(int(s), int(m)): int(a) for a, s, m in zip(p_s1[sel], p_src[sel], p_mid[sel])}
ex_text = {}

# ---------------- accumulators ----------------
lens = {}          # (split, src, country, field) -> array
flags = Counter()  # (split, src, country, flag)
recs = Counter()   # (split, src, country)
tok = {c: {"name": Counter(), "addr": Counter()} for c in COUNTRIES}
pair = Counter()   # (country, src, metric)
jac_bins = Counter()
hardest = {c: [] for c in ["US", "India"]}  # heap of (-jac, rand, text) keep 5 smallest

for fkey in SOURCE_FILES:
    split = fkey.split("_")[0]; src = int(fkey[-1])
    is_pair_src = split == "train" and src in (2, 3)
    if is_pair_src:
        m = p_src == src
        mids = p_mid[m]; o = np.argsort(mids); mids_sorted = mids[o]; rows_for = s1_row[m][o]
    buf = []

    def flush(buf):
        ids = np.fromiter((int(r[0][3:]) for r in buf), np.int64, len(buf))
        pos = np.searchsorted(mids_sorted, ids).clip(0, len(mids_sorted) - 1)
        hit = mids_sorted[pos] == ids
        for r, h, pp in zip(buf, hit, pos):
            if not h:
                continue
            _, nm, ad, c = r
            i = int(rows_for[pp])
            sn, sa = s1_name[i], s1_addr[i]
            pair[c, src, "n"] += 1
            pair[c, src, "eq_lower"] += nm.lower() == sn.lower()
            a, b = norm(nm), norm(sn)
            pair[c, src, "eq_norm"] += a == b
            A, B = set(a.split()), set(b.split())
            j = len(A & B) / len(A | B) if A | B else 1.0
            jb = 0 if j == 0 else (1 if j < .25 else 2 if j < .5 else 3 if j < .75 else 4 if j < 1 else 5)
            jac_bins[c, src, jb] += 1
            ja = set(norm(ad).split()); jb2 = set(norm(sa).split())
            aj = len(ja & jb2) / len(ja | jb2) if ja | jb2 else 1.0
            pair[c, src, "addr_jac_sum"] += aj
            pair[c, src, "name_jac_sum"] += j
            pc1, pc2 = s1_pc[i], postcode(ad, c)
            pair[c, src, "pc_s1"] += pc1 is not None
            pair[c, src, "pc_m"] += pc2 is not None
            if pc1 and pc2:
                pair[c, src, "pc_both"] += 1
                pair[c, src, "pc_agree"] += pc1 == pc2
            h_ = hardest[c]
            item = (-j, rng.random(), (f"S1-{s1_ids[i]}", sn, sa, r[0], nm, ad))
            if len(h_) < 5:
                heapq.heappush(h_, item)
            elif item > h_[0]:
                heapq.heapreplace(h_, item)

    for k, (eid, nm, ad, c) in enumerate(iter_records(fkey)):
        key = (split, src, c)
        recs[key] += 1
        L = lens.get(key)
        if L is None:
            L = lens[key] = (array("H"), array("H"))
        L[0].append(min(len(nm), 65535)); L[1].append(min(len(ad), 65535))
        if not nm.isascii():
            flags[key + ("name_nonascii",)] += 1
            flags[key + ("name_latin_acc",)] += LATIN_ACC.search(nm) is not None
            flags[key + ("name_nonlatin",)] += NON_LATIN.search(nm) is not None
        if not ad.isascii():
            flags[key + ("addr_nonascii",)] += 1
            flags[key + ("addr_latin_acc",)] += LATIN_ACC.search(ad) is not None
            flags[key + ("addr_nonlatin",)] += NON_LATIN.search(ad) is not None
        flags[key + ("pc",)] += postcode(ad, c) is not None
        flags[key + ("addr_null_tok",)] += NULL_TOK.search(ad) is not None
        flags[key + ("name_domain",)] += DOMAIN.search(nm) is not None
        flags[key + ("name_upper",)] += nm.isupper()
        if (split == "train" or c == "France") and k % TOKEN_SAMPLE == 0 and c in tok:
            tok[c]["name"].update(norm(nm).split()); tok[c]["addr"].update(norm(ad).split())
        if is_pair_src:
            if (src, int(eid[3:])) in want_mid:
                ex_text[(src, int(eid[3:]))] = (eid, nm, ad)
            buf.append((eid, nm, ad, c))
            if len(buf) >= 200000:
                flush(buf); buf = []
    if is_pair_src and buf:
        flush(buf)
    print(fkey, "done", S.rss_gb())

# ---------------- report ----------------
S.md("## Step 4 — Noise", "")
S.md("### Examples: random true-match groups (train; 8 per country, seed-fixed; addresses ≤120 chars)", "")
for c in ["US", "India"]:
    rows = []
    for g, s1 in enumerate(ex_groups[c], 1):
        i = int(s1_order[np.searchsorted(s1_sorted, s1)])
        rows.append((f"{c}-{g}", f"S1-{s1}", trunc(s1_name[i], 60), trunc(s1_addr[i])))
        for (src, mid), owner in sorted(want_mid.items()):
            if owner == s1:
                eid, nm, ad = ex_text[(src, mid)]
                rows.append(("", eid, trunc(nm, 60), trunc(ad)))
    S.table(["group", "id", "name", "address"], rows)

S.md("### Examples: 5 hardest true pairs per country (lowest normalised name-token Jaccard; ties broken at random)", "")
rows = []
for c in ["US", "India"]:
    for negj, _, (a, sn, sa, b, nm, ad) in sorted(hardest[c], key=lambda t: (-t[0], t[1])):
        rows.append((c, f"{-negj:.2f}", f"{a}: {trunc(sn, 50)} / {trunc(sa, 60)}", f"{b}: {trunc(nm, 50)} / {trunc(ad, 60)}"))
S.table(["country", "jaccard", "S1 name / address", "match name / address"], rows)

# France examples from test
fr = {}
for k in (1, 2, 3):
    ccv = np.load(os.path.join(CACHE, f"test_source{k}_ids.npz"))["cc"]
    fr[k] = np.flatnonzero(ccv == COUNTRIES.index("France"))
pick1 = set(nrng.choice(fr[1], 10, replace=False).tolist())
tot23 = len(fr[2]) + len(fr[3])
pick23 = nrng.choice(tot23, 10, replace=False)
pick = {1: pick1, 2: set(fr[2][pick23[pick23 < len(fr[2])]].tolist()),
        3: set(fr[3][pick23[pick23 >= len(fr[2])] - len(fr[2])].tolist())}
rows = []
for k in (1, 2, 3):
    for r, (eid, nm, ad, c) in enumerate(iter_records(f"test_source{k}")):
        if r in pick[k]:
            rows.append((eid, trunc(nm, 60), trunc(ad)))
S.md(f"### Examples: France (test, unlabeled) — 10 random S1 rows then 10 random S2/S3 rows", "")
S.table(["id", "name", "address"], rows)

S.md("### True-pair similarity (train, all true pairs; n = pairs)", "",
     "`eq lower` = names identical after `.lower()`; `eq norm` = identical after baseline normalisation; "
     "Jaccard on normalised whitespace tokens.", "")
rows = []
for c in ["US", "India"]:
    for src in (2, 3):
        n = pair[c, src, "n"]
        rows.append((c, f"S1–S{src}", f"{n:,}", pct(pair[c, src, "eq_lower"], n, 2), pct(pair[c, src, "eq_norm"], n, 2),
                     f"{pair[c, src, 'name_jac_sum'] / n:.3f}", f"{pair[c, src, 'addr_jac_sum'] / n:.3f}",
                     *[pct(jac_bins[c, src, b], n) for b in range(6)]))
S.table(["country", "pair", "n", "name eq lower", "name eq norm", "mean name Jaccard", "mean addr Jaccard",
         "name J=0", "(0,.25)", "[.25,.5)", "[.5,.75)", "[.75,1)", "J=1"], rows)

S.md("### Postcode/PIN (regex heuristics: India `[1-9]ddd ?ddd` anywhere not adjacent to digits; US 5-digit "
     "(+4) as its own comma component or after a 2-letter state; France 5-digit at start of a comma component, "
     "not followed by a street word)", "")
rows = []
for c in ["US", "India"]:
    for src in (2, 3):
        n = pair[c, src, "n"]
        rows.append((c, f"S1–S{src}", f"{n:,}", pct(pair[c, src, "pc_s1"], n, 2), pct(pair[c, src, "pc_m"], n, 2),
                     f"{pair[c, src, 'pc_both']:,}", pct(pair[c, src, "pc_agree"], pair[c, src, "pc_both"], 1)))
S.table(["country", "pair", "true pairs", "S1 side has pc", "S2/S3 side has pc", "both have pc",
         "agree | both"], rows)

S.md("### Per-record rates by split/source/country (n = records)", "",
     "`non-ASCII` any char > 0x7F; `accented Latin` U+00C0–024F/1E00–1EFF; `non-Latin script` any other letter "
     "block (Devanagari, Tamil, …); `null tok` address has a NULL/N/A/None component; `domain` name contains "
     ".com/.in/.net/…; `upper` name fully upper-case.", "")
rows = []
for key in sorted(recs):
    n = recs[key]; f = lambda k: pct(flags[key + (k,)], n, 2)
    rows.append((key[0], f"S{key[1]}", key[2], f"{n:,}", f(f"name_nonascii"), f("name_latin_acc"), f("name_nonlatin"),
                 f("addr_nonascii"), f("addr_latin_acc"), f("addr_nonlatin"), f("pc"), f("addr_null_tok"),
                 f("name_domain"), f("name_upper")))
S.table(["split", "src", "country", "n", "name non-ASCII", "name acc. Latin", "name non-Latin", "addr non-ASCII",
         "addr acc. Latin", "addr non-Latin", "addr has postcode", "addr null tok", "name domain", "name upper"], rows)

S.md("### Length distributions in characters (n = records; empty addresses included as 0)", "")
rows = []
for key in sorted(lens):
    for fi, fname in enumerate(["name", "address"]):
        v = np.frombuffer(lens[key][fi], dtype=np.uint16).astype(np.int32)
        q = np.percentile(v, [5, 25, 50, 75, 95])
        rows.append((key[0], f"S{key[1]}", key[2], fname, f"{len(v):,}", f"{v.mean():.1f}",
                     *[f"{x:.0f}" for x in q], f"{v.max()}"))
S.table(["split", "src", "country", "field", "n", "mean", "p5", "p25", "p50", "p75", "p95", "max"], rows)

S.md(f"### Top-30 normalised tokens per country (S1+S2+S3 pooled; every {TOKEN_SAMPLE}th record; US/India from "
     "train, France from test; share = % of sampled records' token occurrences)", "")
for c in COUNTRIES:
    for field in ("name", "addr"):
        C = tok[c][field]; tot = sum(C.values())
        S.md(f"- **{c} {field}** (tokens={tot:,}): " + ", ".join(f"{w} {100 * n / tot:.1f}" for w, n in C.most_common(30)))
S.md("")
S.finish()
