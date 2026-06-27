"""Experiment 054 — k-reciprocal re-ranking (Zhong et al. 2017), the SOTA training-free re-ID reranker.

051 found CSLS (hubness) gives a small win. k-reciprocal re-ranking is stronger: it re-scores by the
Jaccard overlap of k-reciprocal neighbour sets (with local query expansion), exploiting the full
neighbourhood graph rather than just per-side means. If the true exemplar is a reciprocal neighbour of
the query, this surfaces it. Pure linear algebra on cached global+L256 (cheap-probe-first).

dev 10-seed CV paired vs cosine baseline and vs CSLS k=5 (051's win). Sealed test only if adopted.

    .venv/bin/python scripts/kreciprocal.py
"""

from __future__ import annotations

import collections
import datetime
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import explog  # noqa: E402
from eval_appearance import _git_sha  # noqa: E402
from eval_merged import load, block_split, BASE  # noqa: E402
from split_devtest import get_split  # noqa: E402

SEEDS = 10


def unit(Z):
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)


def ms(v):
    v = [x for x in v if x == x]
    return (round(st.mean(v), 1), round(st.pstdev(v), 1)) if v else (float("nan"), 0.0)


def re_ranking(qg, qq, gg, k1=20, k2=6, lam=0.3):
    """Standard k-reciprocal re-ranking. Inputs are distance blocks (1-cos). Returns query×gallery dist."""
    original_dist = np.concatenate([np.concatenate([qq, qg], 1),
                                    np.concatenate([qg.T, gg], 1)], 0).astype(np.float32)
    original_dist = original_dist / (np.max(original_dist, axis=0) + 1e-12)
    V = np.zeros_like(original_dist)
    initial_rank = np.argsort(original_dist, axis=1)
    qnum = qg.shape[0]; allnum = original_dist.shape[0]
    k1 = min(k1, allnum - 1)
    for i in range(allnum):
        fwd = initial_rank[i, :k1 + 1]
        bwd = initial_rank[fwd, :k1 + 1]
        recip = fwd[np.where(bwd == i)[0]]
        recip_exp = recip
        for j in recip:
            cand = initial_rank[j, :int(round(k1 / 2)) + 1]
            cand_bwd = initial_rank[cand, :int(round(k1 / 2)) + 1]
            cand_k = cand[np.where(cand_bwd == j)[0]]
            if len(np.intersect1d(cand_k, recip)) > 2.0 / 3 * len(cand_k):
                recip_exp = np.append(recip_exp, cand_k)
        recip_exp = np.unique(recip_exp)
        w = np.exp(-original_dist[i, recip_exp])
        V[i, recip_exp] = w / np.sum(w)
    if k2 > 1:
        Vqe = np.zeros_like(V)
        for i in range(allnum):
            Vqe[i] = np.mean(V[initial_rank[i, :k2]], axis=0)
        V = Vqe
    invIndex = [np.where(V[:, i] != 0)[0] for i in range(allnum)]
    jacc = np.zeros((qnum, allnum), np.float32)
    for i in range(qnum):
        tmp = np.zeros(allnum, np.float32)
        nz = np.where(V[i] != 0)[0]
        for j in nz:
            tmp[invIndex[j]] += np.minimum(V[i, j], V[invIndex[j], j])
        jacc[i] = 1 - tmp / (2 - tmp + 1e-12)
    final = jacc * (1 - lam) + original_dist[:qnum] * lam
    return final[:, qnum:]


def classmax_from_sim(S, Y, tr, cov, labs, li):
    cols = collections.defaultdict(list)
    for j, i in enumerate(tr):
        cols[li[Y[i]]].append(j)
    sc = np.full((len(cov), len(labs)), -9.0, np.float32)
    for c, ix in cols.items():
        sc[:, c] = S[:, ix].max(1)
    pred = sc.argmax(1)
    return 100 * np.mean([labs[pred[r]] == Y[cov[r]] for r in range(len(cov))])


def eval_cosine(Z, Y, tr, te):
    labset = set(Y[i] for i in tr); cov = [q for q in te if Y[q] in labset]
    if not cov:
        return float("nan")
    labs = sorted(labset); li = {l: j for j, l in enumerate(labs)}
    return classmax_from_sim(Z[cov] @ Z[tr].T, Y, tr, cov, labs, li)


def eval_csls(Z, Y, tr, te, k=5):
    labset = set(Y[i] for i in tr); cov = [q for q in te if Y[q] in labset]
    if not cov:
        return float("nan")
    labs = sorted(labset); li = {l: j for j, l in enumerate(labs)}
    sqg = Z[cov] @ Z[tr].T; gg = Z[tr] @ Z[tr].T; np.fill_diagonal(gg, -9)
    rg = np.sort(gg, 1)[:, -k:].mean(1); rq = np.sort(sqg, 1)[:, -k:].mean(1)
    return classmax_from_sim(2 * sqg - rq[:, None] - rg[None, :], Y, tr, cov, labs, li)


def eval_krecip(Z, Y, tr, te, k1=20, k2=6, lam=0.3):
    labset = set(Y[i] for i in tr); cov = [q for q in te if Y[q] in labset]
    if not cov:
        return float("nan")
    labs = sorted(labset); li = {l: j for j, l in enumerate(labs)}
    Zq, Zg = Z[cov], Z[tr]
    qg = 1 - Zq @ Zg.T; qq = 1 - Zq @ Zq.T; gg = 1 - Zg @ Zg.T
    final = re_ranking(qg, qq, gg, k1, k2, lam)
    return classmax_from_sim(-final, Y, tr, cov, labs, li)   # similarity = -distance


def main():
    rows = load(); Y = [r["label"] for r in rows]
    split = get_split()
    cnt = collections.Counter(Y)
    core = [i for i in range(len(rows)) if cnt[Y[i]] >= 2]
    dev = [i for i in core if split[rows[i]["image"]] == "dev"]
    test = [i for i in core if split[rows[i]["image"]] == "test"]
    block = [json.loads((BASE / "_blocks.json").read_text())[r["image"]] for r in rows]
    zg = unit(np.load(BASE / "_dino_cache.npy").astype(np.float32))
    zl = unit(np.load(BASE / "_local256_cache.npy").astype(np.float32))
    Z = unit(np.concatenate([zg, zl], 1))
    print(f"core {len(core)} | dev {len(dev)} / test {len(test)} | global+L256")

    splits = [block_split(dev, block, s) for s in range(SEEDS)]
    methods = {
        "cosine": lambda tr, te: eval_cosine(Z, Y, tr, te),
        "CSLS k=5": lambda tr, te: eval_csls(Z, Y, tr, te, 5),
        "k-recip k1=10": lambda tr, te: eval_krecip(Z, Y, tr, te, 10, 6, 0.3),
        "k-recip k1=20": lambda tr, te: eval_krecip(Z, Y, tr, te, 20, 6, 0.3),
        "k-recip k1=20 λ=0.5": lambda tr, te: eval_krecip(Z, Y, tr, te, 20, 6, 0.5),
    }
    res = {k: [fn(tr, te) for tr, te in splits] for k, fn in methods.items()}
    base = res["cosine"]

    def paired(a, b=base):
        d = [x - y for x, y in zip(a, b)]
        return round(st.mean(d), 2), int(sum(x > 0 for x in d))

    print("\n== dev-CV 10-seed (paired vs cosine) ==")
    table = {}
    for k in methods:
        dlt = paired(res[k]); table[k] = (ms(res[k]), dlt)
        print(f"  {k:20} top1 {ms(res[k])[0]}±{ms(res[k])[1]}  Δ vs cos {dlt[0]:+} ({dlt[1]}/10)")
    # also k-recip vs CSLS (is it better than the adopted CSLS?)
    krbest = max([k for k in methods if "recip" in k], key=lambda k: st.mean(res[k]))
    d_vs_csls = paired(res[krbest], res["CSLS k=5"])
    print(f"  best k-recip ({krbest}) vs CSLS k=5: Δ {d_vs_csls[0]:+} ({d_vs_csls[1]}/10)")

    cand = [k for k in methods if k != "cosine"]
    best = max(cand, key=lambda k: st.mean(res[k]))
    d_best = table[best][1]
    adopt = d_best[0] > 0 and d_best[1] >= 7
    sealed = ""
    if adopt:
        if "recip" in best:
            k1 = int(best.split("k1=")[1].split()[0]); lam = 0.5 if "0.5" in best else 0.3
            sb = eval_krecip(Z, Y, dev, test, k1, 6, lam)
        else:
            sb = eval_csls(Z, Y, dev, test, 5)
        s0 = eval_cosine(Z, Y, dev, test)
        sealed = f" | SEALED cos {round(s0,1)} → {best} {round(sb,1)}"
    print(f"\n  {'★ '+best+' ADOPT' if adopt else 'best '+best} (dev Δ{d_best[0]:+}, {d_best[1]}/10){sealed}")

    d = explog.EXP / "054-kreciprocal"; d.mkdir(parents=True, exist_ok=True)
    ks = list(methods)
    explog.bar(d / "fig1.png", [k.replace(" ", "\n") for k in ks], [table[k][0][0] for k in ks],
               "054 k-reciprocal vs CSLS vs cosine (dev-CV top1)", "%", ymax=40, errors=[table[k][0][1] for k in ks])

    rowmd = "\n".join(f"| {k} | {table[k][0][0]}±{table[k][0][1]} | {table[k][1][0]:+} | {table[k][1][1]}/10 |" for k in ks)
    verdict = (f"🟢 **{best}** 채택 (dev Δ{d_best[0]:+}, {d_best[1]}/10){sealed}."
               if adopt else
               f"🔴 best {best} (Δ{d_best[0]:+}, {d_best[1]}/10) — 채택기준 미달.")
    note = (f"k-reciprocal이 CSLS보다 {'나음' if d_vs_csls[0] > 0 else '못함/동급'} (Δ{d_vs_csls[0]:+}, {d_vs_csls[1]}/10) "
            f"→ {'더 강한 리랭커가 검색에 유효' if d_vs_csls[0] > 0.3 else 'CSLS와 사실상 동급, 리랭킹 한계'}.")
    report = f"""# 054 — k-reciprocal 재순위화 (Zhong 2017) vs CSLS

- 날짜: {datetime.date.today().isoformat()} · 커밋 `main @ {_git_sha()}` · `scripts/kreciprocal.py`
- clean 502 (dev {len(dev)}/test {len(test)} 봉인), global+L256, dev 10-seed paired. 봉인 test는 채택분만.
- 동기: 051의 CSLS(작은 승리)를 더 강한 reranker(Jaccard k-reciprocal)로 확장 가능한가.

## 결과 (paired Δ vs cosine)
| 방법 | dev-CV top1 | Δ | wins |
|---|---|---|---|
{rowmd}

- {note}

![methods](fig1.png)

## 판정
{verdict}
"""
    explog.write(d, report, {
        "title": "k-reciprocal 재순위화 vs CSLS", "date": datetime.date.today().isoformat(),
        "headline": f"best={best} dev Δ{d_best[0]:+}({d_best[1]}/10){sealed} | k-recip vs CSLS Δ{d_vs_csls[0]:+}",
        "adopt": bool(adopt), "best": best, "krecip_vs_csls": list(d_vs_csls),
        "devcv": {k: {"top1": table[k][0], "delta_vs_cos": table[k][1]} for k in methods}})
    print(f"\nwrote -> {d}  (1 figure)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
