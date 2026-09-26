"""STEP 2 - label structure of train_ground_truth.tsv. Caches true pairs to eda/cache/pairs.npz."""
import os

import numpy as np

from common import CACHE, COUNTRIES, Step, iter_lines, pct

S = Step("step2_labels")


def load(key):
    z = np.load(os.path.join(CACHE, key + "_ids.npz"))
    o = np.argsort(z["ids"])
    return z["ids"][o], z["cc"][o]


s1_ids, s1_cc = load("train_source1")
src = {2: load("train_source2"), 3: load("train_source3")}
test_ids = {k: load(f"test_source{k}")[0] for k in (1, 2, 3)}

gt_s1, n2, n3 = [], [], []
p_s1, p_src, p_mid = [], [], []
intra_dup = bad_prefix = 0
for _, p in iter_lines("train_ground_truth"):
    s1 = int(p[0][3:])
    ids = [x for x in p[1].split(",") if x] if len(p) > 1 else []
    if len(ids) != len(set(ids)):
        intra_dup += 1
    a = b = 0
    for x in ids:
        pre = x[:3]
        if pre == "S2-":
            a += 1; p_src.append(2)
        elif pre == "S3-":
            b += 1; p_src.append(3)
        else:
            bad_prefix += 1; continue
        p_s1.append(s1); p_mid.append(int(x[3:]))
    gt_s1.append(s1); n2.append(a); n3.append(b)

gt_s1 = np.array(gt_s1, np.int64); n2 = np.array(n2); n3 = np.array(n3); n = n2 + n3
p_s1 = np.array(p_s1, np.int64); p_src = np.array(p_src, np.int8); p_mid = np.array(p_mid, np.int64)


def lookup(sorted_ids, sorted_cc, q):
    i = np.searchsorted(sorted_ids, q).clip(0, len(sorted_ids) - 1)
    found = sorted_ids[i] == q
    return found, np.where(found, sorted_cc[i], -1)


g_found, g_cc = lookup(s1_ids, s1_cc, gt_s1)
p_s1_found, p_s1cc = lookup(s1_ids, s1_cc, p_s1)
p_found = np.zeros(len(p_mid), bool); p_cc = np.full(len(p_mid), -1, np.int8)
for k in (2, 3):
    m = p_src == k
    f, c = lookup(*src[k], p_mid[m])
    p_found[m] = f; p_cc[m] = c
np.savez(os.path.join(CACHE, "pairs.npz"), s1=p_s1, src=p_src, mid=p_mid, s1cc=p_s1cc, mcc=p_cc)
np.savez(os.path.join(CACHE, "gt.npz"), s1=gt_s1, n2=n2, n3=n3, cc=g_cc)

N = len(gt_s1)
S.md("## Step 2 — Label structure (train)", "")
S.md(f"GT rows: {N:,}; unique S1 ids in GT: {len(np.unique(gt_s1)):,}; total true pairs: {len(p_mid):,} "
     f"(S2: {(p_src == 2).sum():,}, S3: {(p_src == 3).sum():,}). Rows with a repeated id inside the list: "
     f"{intra_dup:,}. Ids in lists without S2-/S3- prefix: {bad_prefix:,}.", "")

rows = []
for name, m in [("all", np.ones(N, bool))] + [(c, g_cc == i) for i, c in enumerate(COUNTRIES) if (g_cc == i).any()]:
    k = m.sum()
    rows.append((name, f"{k:,}", f"{(n[m] == 0).sum():,}", pct((n[m] == 0).sum(), k, 2),
                 f"{n[m].mean():.3f}", f"{n2[m].mean():.3f}", f"{n3[m].mean():.3f}",
                 pct(((n2[m] > 0) & (n3[m] == 0)).sum(), k), pct(((n3[m] > 0) & (n2[m] == 0)).sum(), k),
                 pct(((n2[m] > 0) & (n3[m] > 0)).sum(), k)))
S.md("### Singletons and mean match counts (country = S1 record's country)", "")
S.table(["country", "S1 entities", "singletons", "singleton %", "mean |matches|", "mean #S2", "mean #S3",
         "only-S2 %", "only-S3 %", "both %"], rows)


def dist(v):
    b = [(v == i).sum() for i in range(5)] + [(v >= 5).sum()]
    return [f"{x:,} ({pct(x, len(v))})" for x in b]


S.md("### Match-set size distribution (count (% of S1 entities in that country))", "")
rows = []
for name, m in [("all", np.ones(N, bool))] + [(c, g_cc == i) for i, c in enumerate(COUNTRIES) if (g_cc == i).any()]:
    for lab, v in [("S2+S3", n), ("S2 only count", n2), ("S3 only count", n3)]:
        rows.append([name, lab] + dist(v[m]))
S.table(["country", "counted", "0", "1", "2", "3", "4", "5+"], rows)
S.md(f"Max |matches| for one S1: {n.max()} (S2 max {n2.max()}, S3 max {n3.max()}). "
     f"Count with ≥10 matches: {(n >= 10).sum():,}.", "")

# many-to-one check
key = p_src.astype(np.int64) * 10**12 + p_mid
u, c = np.unique(key, return_counts=True)
multi = c > 1
S.md("### Is the S1 → S2/S3 mapping many-to-one?", "")
S.table(["", "S2", "S3"], [
    ["distinct ids in GT lists", f"{((u // 10**12) == 2).sum():,}", f"{((u // 10**12) == 3).sum():,}"],
    ["ids appearing in >1 S1 list", f"{(multi & ((u // 10**12) == 2)).sum():,}", f"{(multi & ((u // 10**12) == 3)).sum():,}"],
    ["max #S1 lists for one id", f"{c[(u // 10**12) == 2].max() if len(c) else 0}", f"{c[(u // 10**12) == 3].max() if len(c) else 0}"],
])

S.md("### S2/S3 records that match no S1 (by the record's own country)", "")
rows = []
for k in (2, 3):
    ids, ccs = src[k]
    matched = np.isin(ids, np.unique(p_mid[p_src == k]))
    for i, c in enumerate(COUNTRIES):
        m = ccs == i
        if m.any():
            rows.append((f"S{k}", c, f"{m.sum():,}", f"{(~matched[m]).sum():,}", pct((~matched[m]).sum(), m.sum(), 2)))
S.table(["source", "country", "records", "unmatched", "unmatched %"], rows)

S.md("### Id integrity", "")
S.table(["check", "count"], [
    ["GT S1 ids not in train_source1", f"{(~g_found).sum():,}"],
    ["train_source1 ids missing from GT", f"{(~np.isin(s1_ids, gt_s1)).sum():,}"],
    ["GT S2 ids not in train_source2", f"{(~p_found[p_src == 2]).sum():,}"],
    ["GT S3 ids not in train_source3", f"{(~p_found[p_src == 3]).sum():,}"],
    ["train S1 ids also in test S1", f"{np.isin(s1_ids, test_ids[1]).sum():,}"],
    ["train S2 ids also in test S2", f"{np.isin(src[2][0], test_ids[2]).sum():,}"],
    ["train S3 ids also in test S3", f"{np.isin(src[3][0], test_ids[3]).sum():,}"],
    ["S2 ids whose numeric part also exists as an S3 id (train)", f"{np.isin(src[2][0], src[3][0]).sum():,}"],
])

ok = p_found & p_s1_found
diff = ok & (p_cc != p_s1cc)
S.md(f"### Country disagreement within true pairs", "",
     f"Pairs with a differing country: {diff.sum():,} / {ok.sum():,} ({pct(diff.sum(), ok.sum(), 3)}).", "")
if diff.any():
    rows = []
    for a in range(3):
        for b in range(3):
            k = (diff & (p_s1cc == a) & (p_cc == b)).sum()
            if k:
                rows.append((COUNTRIES[a], COUNTRIES[b], f"{k:,}"))
    S.table(["S1 country", "match country", "pairs"], rows)
S.finish()
