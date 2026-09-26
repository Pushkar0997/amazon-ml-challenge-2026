"""STEP 5 - blocking probe on 3,000 matched + 1,000 singleton train S1 entities.

Pool = all train S2+S3 records of the S1's country; one country at a time, pool streamed from disk
(name+address TF-IDF over the full US pool would not fit in ~4 GB if materialised).
TF-IDF = char 3-grams via sklearn HashingVectorizer (2^23 buckets, raw tf) + smooth idf
ln((1+N)/(1+df))+1 fitted on the pool + l2 norm, i.e. TfidfVectorizer defaults up to hash collisions.
"""
import os
import sys
from array import array
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from common import CACHE, COUNTRIES, SEED, Step, iter_records, norm, pct, postcode

S = Step("step5_blocking")
rng = np.random.default_rng(SEED)
N_MATCHED, N_SINGLE = 3000, 1000
KS = [5, 10, 20, 50]
KMAX = 50
CHUNK = 20000
F = 2 ** 23
hv = HashingVectorizer(analyzer="char", ngram_range=(3, 3), lowercase=False, n_features=F,
                       alternate_sign=False, norm=None, dtype=np.float32)
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None  # optional pool cap for smoke tests

# ---- sample ----
gt = np.load(os.path.join(CACHE, "gt.npz"))
n = gt["n2"] + gt["n3"]
matched = rng.choice(gt["s1"][n > 0], N_MATCHED, replace=False)
single = rng.choice(gt["s1"][n == 0], N_SINGLE, replace=False)
sample = {int(x): True for x in matched} | {int(x): False for x in single}
pairs = np.load(os.path.join(CACHE, "pairs.npz"))
sel = np.isin(pairs["s1"], matched)
true_of = defaultdict(list)
for s1, src, mid in zip(pairs["s1"][sel], pairs["src"][sel], pairs["mid"][sel]):
    true_of[int(s1)].append((int(src), int(mid)))
queries = {}  # s1 -> (country, name, addr)
for eid, nm, ad, c in iter_records("train_source1"):
    s1 = int(eid[3:])
    if s1 in sample:
        queries[s1] = (c, nm, ad)


def pool_iter(c):
    k = 0
    for src in (2, 3):
        for eid, nm, ad, cc_ in iter_records(f"train_source{src}"):
            if cc_ == c:
                yield src, int(eid[3:]), nm, ad
                k += 1
                if LIMIT and k >= LIMIT:
                    return


def chunks(it):
    buf = []
    for r in it:
        buf.append(r)
        if len(buf) == CHUNK:
            yield buf; buf = []
    if buf:
        yield buf


def tfidf(texts, idf):
    X = hv.transform(texts)
    X.data *= idf[X.indices]
    return normalize(X, copy=False)


res = {}
for c in ["US", "India"]:
    qids = [s for s in queries if queries[s][0] == c]
    nq = len(qids)
    q_name = [norm(queries[s][1]) for s in qids]
    q_na = [f"{a} {norm(queries[s][2])}" for a, s in zip(q_name, qids)]
    q_pc = [postcode(queries[s][2], c) for s in qids]
    q_tok = [set(t.split()) for t in q_name]
    q_vocab = set().union(*q_tok)
    want = {}  # (src, mid) -> query index
    for qi, s in enumerate(qids):
        for key in true_of.get(s, []):
            want[key] = qi
    n_true = len(want)

    # ---- pass A: DF (tokens, trigrams), postcode counts, postings ----
    Np = 0
    tok_df = Counter()
    postings = defaultdict(lambda: array("I"))
    pc_count = Counter()
    pc_post = defaultdict(lambda: array("I"))
    qpcs = {p for p in q_pc if p}
    df_name = np.zeros(F, np.int64); df_na = np.zeros(F, np.int64)
    true_pidx = []  # (pool idx, query idx)
    for buf in chunks(pool_iter(c)):
        nn = [norm(r[2]) for r in buf]
        na = [f"{a} {norm(r[3])}" for a, r in zip(nn, buf)]
        for j, (r, t) in enumerate(zip(buf, nn)):
            pi = Np + j
            ts = set(t.split())
            tok_df.update(ts)
            for w in ts & q_vocab:
                postings[w].append(pi)
            p = postcode(r[3], c)
            if p:
                pc_count[p] += 1
                if p in qpcs:
                    pc_post[p].append(pi)
            qi = want.get((r[0], r[1]))
            if qi is not None:
                true_pidx.append((pi, qi))
        df_name += np.bincount(hv.transform(nn).indices, minlength=F)
        df_na += np.bincount(hv.transform(na).indices, minlength=F)
        Np += len(buf)
    print(c, "pass A done, pool", Np, "rss", round(S.rss_gb(), 2)); sys.stdout.flush()
    idf_name = (np.log((1 + Np) / (1 + df_name)) + 1).astype(np.float32)
    idf_na = (np.log((1 + Np) / (1 + df_na)) + 1).astype(np.float32)
    del df_name, df_na
    tp = np.array(true_pidx, np.int64).reshape(-1, 2)
    tp = tp[np.argsort(tp[:, 0])]
    truth = [set() for _ in range(nq)]
    for pi, qi in tp:
        truth[qi].add(int(pi))

    # ---- pass B: TF-IDF top-k ----
    reps = {"name": (tfidf(q_name, idf_name).T.tocsr(), idf_name),
            "name+addr": (tfidf(q_na, idf_na).T.tocsr(), idf_na)}
    topS = {r: np.full((0, nq), -1, np.float32) for r in reps}
    topI = {r: np.zeros((0, nq), np.int64) for r in reps}
    true_score = {r: np.full(len(tp), np.nan, np.float32) for r in reps}
    off = 0
    for buf in chunks(pool_iter(c)):
        nn = [norm(r[2]) for r in buf]
        texts = {"name": nn, "name+addr": [f"{a} {norm(r[3])}" for a, r in zip(nn, buf)]}
        lo, hi = np.searchsorted(tp[:, 0], [off, off + len(buf)])
        for r, (QT, idf) in reps.items():
            Sc = (tfidf(texts[r], idf) @ QT).toarray()  # chunk x nq
            if hi > lo:
                true_score[r][lo:hi] = Sc[tp[lo:hi, 0] - off, tp[lo:hi, 1]]
            k = min(KMAX, Sc.shape[0])
            part = np.argpartition(-Sc, k - 1, axis=0)[:k]
            cs = np.vstack([topS[r], np.take_along_axis(Sc, part, 0)])
            ci = np.vstack([topI[r], part + off])
            k2 = min(KMAX, cs.shape[0])
            keep = np.argpartition(-cs, k2 - 1, axis=0)[:k2]
            topS[r] = np.take_along_axis(cs, keep, 0); topI[r] = np.take_along_axis(ci, keep, 0)
        off += len(buf)
        if (off // CHUNK) % 25 == 0:
            print(c, "pass B", off, "rss", round(S.rss_gb(), 2)); sys.stdout.flush()
    for r in reps:
        o = np.argsort(-topS[r], axis=0)
        topS[r] = np.take_along_axis(topS[r], o, 0); topI[r] = np.take_along_axis(topI[r], o, 0)

    # ---- metrics ----
    R = {"nq": nq, "nq_matched": sum(1 for s in qids if sample[s]), "n_true": n_true, "pool": Np,
         "true_in_pool": len(tp)}
    # (a) postcode
    hit = cand = 0
    pc_sets = []
    for qi in range(nq):
        p = q_pc[qi]
        sset = set(pc_post[p]) if p else set()
        pc_sets.append(sset)
        cand += len(sset); hit += len(truth[qi] & sset)
    R["a"] = (hit, cand / nq)
    R["q_has_pc"] = sum(1 for p in q_pc if p)
    # (b) rare token
    thr = 0.01 * Np
    hit = cand = no_rare = 0
    for qi in range(nq):
        rare = [w for w in q_tok[qi] if 0 < tok_df[w] < thr]
        if not rare:
            no_rare += 1; continue
        u = np.unique(np.concatenate([np.frombuffer(postings[w], np.uint32) for w in rare]))
        cand += len(u)
        hit += len(truth[qi] & set(u.tolist()))
    R["b"] = (hit, cand / nq); R["b_no_rare"] = no_rare
    # (c), (d)
    for r, lab in [("name", "c"), ("name+addr", "d")]:
        for k in KS:
            hit = sum(len(truth[qi] & set(topI[r][:k, qi].tolist())) for qi in range(nq))
            R[lab, k] = (hit, min(k, Np))
    # (e) union a + c@10
    hit = cand = 0
    for qi in range(nq):
        u = pc_sets[qi] | set(topI["name"][:10, qi].tolist())
        cand += len(u); hit += len(truth[qi] & u)
    R["e"] = (hit, cand / nq)
    # score distributions
    is_m = np.array([sample[s] for s in qids])
    for r in reps:
        best_true = np.full(nq, np.nan, np.float32)
        for (pi, qi), sc in zip(tp, true_score[r]):
            best_true[qi] = np.fmax(best_true[qi], sc)
        R["top1_single", r] = topS[r][0, ~is_m]
        R["top1_matched", r] = topS[r][0, is_m]
        R["besttrue_matched", r] = best_true[is_m]
        R["besttrue_is_top1", r] = int(sum(1 for qi in np.flatnonzero(is_m) if topI[r][0, qi] in truth[qi]))
    res[c] = R
    print(c, "done", {k: v for k, v in R.items() if not isinstance(v, np.ndarray)}); sys.stdout.flush()
    del postings, tok_df, pc_post, reps

# ---- report ----
S.md("## Step 5 — Blocking probe (train)", "",
     f"Sample (seed {SEED}): {N_MATCHED:,} matched + {N_SINGLE:,} singleton S1 entities drawn from all train S1. "
     "Pool = all train S2+S3 of the S1's country. Normalisation: lowercase, NFKD + drop Latin combining accents, "
     "punctuation/symbols → space, whitespace collapsed. Pools processed one country at a time, streamed from disk "
     "(a materialised US name+address TF-IDF matrix would exceed the ~4 GB budget). TF-IDF: char 3-grams, "
     f"HashingVectorizer n_features=2^23, idf fitted on the pool (sklearn smooth-idf formula), l2-normalised. "
     "Mean candidates are averaged over all sampled S1 (matched + singletons).", "")
rows = []
for c in res:
    R = res[c]
    rows.append((c, f"{R['nq']:,}", f"{R['nq_matched']:,}", f"{R['nq'] - R['nq_matched']:,}", f"{R['n_true']:,}",
                 f"{R['true_in_pool']:,}", f"{R['pool']:,}", f"{R['q_has_pc']:,}", f"{R['b_no_rare']:,}"))
S.table(["country", "S1 sampled", "matched", "singletons", "true pairs", "true pairs in pool", "pool size",
         "S1 with postcode", "S1 with no rare name token"], rows)

strategies = [("a) same postcode", "a"), ("b) shares rare name token (df<1%)", "b")] + \
             [(f"c) name TF-IDF top-{k}", ("c", k)) for k in KS] + \
             [(f"d) name+addr TF-IDF top-{k}", ("d", k)) for k in KS] + [("e) a ∪ c@10", "e")]
rows = []
for lab, key in strategies:
    row = [lab]
    th = tc = tq = 0
    for c in res:
        h, m = res[c][key]
        row += [pct(h, res[c]["n_true"]), f"{m:,.1f}"]
        th += h; tc += m * res[c]["nq"]; tq += res[c]["nq"]
    row += [pct(th, sum(res[c]["n_true"] for c in res)), f"{tc / tq:,.1f}"]
    rows.append(row)
hdr = ["strategy"]
for c in res:
    hdr += [f"{c} pair recall (n={res[c]['n_true']:,})", f"{c} mean cands (n={res[c]['nq']:,})"]
hdr += ["all recall", "all mean cands"]
S.table(hdr, rows)


def q(v):
    v = v[~np.isnan(v)]
    return [f"{x:.3f}" for x in np.percentile(v, [5, 25, 50, 75, 95])] + \
        [pct((v >= .5).sum(), len(v)), pct((v >= .7).sum(), len(v)), pct((v >= .9).sum(), len(v))]


S.md("### TF-IDF cosine distributions", "",
     "`top-1 (singletons)` = best pool score for a singleton; `best true (matched)` = highest cosine among the entity's "
     "true matches; `top-1 (matched)` = best pool score for a matched entity, true or not.", "")
rows = []
for c in res:
    for r in ["name", "name+addr"]:
        for lab, key in [("top-1 (singletons)", "top1_single"), ("best true (matched)", "besttrue_matched"),
                         ("top-1 (matched)", "top1_matched")]:
            v = res[c][key, r]
            rows.append([c, r, lab, f"{(~np.isnan(v)).sum():,}"] + q(v))
S.table(["country", "text", "score", "n", "p5", "p25", "p50", "p75", "p95", "≥0.5", "≥0.7", "≥0.9"], rows)
rows = [(c, r, pct(res[c]["besttrue_is_top1", r], res[c]["nq_matched"])) for c in res for r in ["name", "name+addr"]]
S.table(["country", "text", "matched S1 whose top-1 pool record is a true match"], rows)
S.finish("one country at a time; pool streamed twice per country")
