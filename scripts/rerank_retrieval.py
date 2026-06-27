"""Experiment 051 — training-free retrieval re-ranking (re-ID theory) on global+L256.

The task is gallery retrieval (query pin → nearest class exemplar). The person-re-ID literature has
strong TRAINING-FREE rerankers that fix two well-known failure modes of raw cosine kNN:
  • hubness — in high-dim, some gallery vectors are everyone's neighbour and get wrongly retrieved.
    CSLS (Conneau 2018) subtracts each side's local mean similarity → 2cos − r_q − r_g.
  • non-reciprocity — the true match is often a *reciprocal* neighbour. k-reciprocal re-ranking
    (Zhong 2017) and average query expansion (AQE) use neighbourhood structure.
These are pure linear-algebra on the cached embeddings — exactly the cheap-probe-first regime.

Each modifies the instance-instance similarity; the readout stays class-max (our exemplar 1-NN).
dev 10-seed CV paired vs the global+L256 baseline (33.5); sealed test only if a method is adopted.

    .venv/bin/python scripts/rerank_retrieval.py
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


# ---- instance-similarity functions (query x gallery) ----
def sim_cosine(Zq, Zg, **kw):
    return Zq @ Zg.T


def sim_csls(Zq, Zg, k=8, **kw):
    sqg = Zq @ Zg.T
    gg = Zg @ Zg.T
    np.fill_diagonal(gg, -9.0)
    rg = np.sort(gg, axis=1)[:, -k:].mean(1)          # gallery hubness
    rq = np.sort(sqg, axis=1)[:, -k:].mean(1)         # query local mean
    return 2 * sqg - rq[:, None] - rg[None, :]


def sim_aqe(Zq, Zg, m=5, **kw):
    sqg = Zq @ Zg.T
    top = np.argsort(-sqg, axis=1)[:, :m]
    qexp = unit(Zq + Zg[top].mean(1))                 # augment query with top-m gallery mean
    return qexp @ Zg.T


def sim_dis(Zq, Zg, k=8, **kw):
    """Local-scaling / self-tuning: divide by per-query and per-gallery kNN scale."""
    sqg = Zq @ Zg.T
    gg = Zg @ Zg.T
    np.fill_diagonal(gg, -9.0)
    sg = np.sort(gg, axis=1)[:, -k:].mean(1) + 1e-6
    sq = np.sort(sqg, axis=1)[:, -k:].mean(1) + 1e-6
    return sqg / np.sqrt(sq[:, None] * sg[None, :])


METHODS = {
    "baseline cosine": (sim_cosine, {}),
    "CSLS k=5": (sim_csls, {"k": 5}), "CSLS k=8": (sim_csls, {"k": 8}), "CSLS k=15": (sim_csls, {"k": 15}),
    "AQE m=3": (sim_aqe, {"m": 3}), "AQE m=7": (sim_aqe, {"m": 7}),
    "local-scale k=8": (sim_dis, {"k": 8}),
}


def evalu(simfn, kw, Z, Y, tr, te):
    labset = set(Y[i] for i in tr)
    cov = [q for q in te if Y[q] in labset]
    if not cov:
        return float("nan")
    labs = sorted(labset); li = {l: j for j, l in enumerate(labs)}
    cols = collections.defaultdict(list)
    for j, i in enumerate(tr):
        cols[li[Y[i]]].append(j)
    Smat = simfn(Z[cov], Z[tr], **kw)
    sc = np.full((len(cov), len(labs)), -9.0, np.float32)
    for c, ix in cols.items():
        sc[:, c] = Smat[:, ix].max(1)
    pred = sc.argmax(1)
    return 100 * np.mean([labs[pred[r]] == Y[cov[r]] for r in range(len(cov))])


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
    res = {}
    for name, (fn, kw) in METHODS.items():
        res[name] = [evalu(fn, kw, Z, Y, tr, te) for tr, te in splits]
    base = res["baseline cosine"]

    def paired(a):
        d = [x - y for x, y in zip(a, base)]
        return round(st.mean(d), 2), int(sum(x > 0 for x in d))

    print("\n== dev-CV 10-seed (paired vs baseline cosine) ==")
    table = {}
    for name in METHODS:
        dlt = paired(res[name]); table[name] = (ms(res[name]), dlt)
        print(f"  {name:18} top1 {ms(res[name])[0]}±{ms(res[name])[1]}  Δ {dlt[0]:+} ({dlt[1]}/10)")

    cand = [n for n in METHODS if n != "baseline cosine"]
    best = max(cand, key=lambda n: st.mean(res[n]))
    d_best = table[best][1]
    adopt = d_best[0] > 0 and d_best[1] >= 7
    sealed = ""
    if adopt:
        fn, kw = METHODS[best]
        st_best = evalu(fn, kw, Z, Y, dev, test); st_base = evalu(sim_cosine, {}, Z, Y, dev, test)
        sealed = f" | SEALED: base {round(st_base,1)} → {best} {round(st_best,1)}"
        print(f"\n  ★ {best} adopted (dev Δ{d_best[0]:+}, {d_best[1]}/10){sealed}")
    else:
        print(f"\n  best non-baseline = {best} (dev Δ{d_best[0]:+}, {d_best[1]}/10) — no adoption (sealed test untouched)")

    d = explog.EXP / "051-rerank-retrieval"; d.mkdir(parents=True, exist_ok=True)
    ks = list(METHODS)
    explog.bar(d / "fig1_methods.png", [k.replace(" ", "\n") for k in ks], [table[k][0][0] for k in ks],
               "051 retrieval re-ranking (dev-CV top1, global+L256)", "%", ymax=40,
               errors=[table[k][0][1] for k in ks])
    explog.bar(d / "fig2_delta.png", [k.replace(" ", "\n") for k in ks if k != "baseline cosine"],
               [table[k][1][0] for k in ks if k != "baseline cosine"],
               "051 paired Δtop1 vs cosine baseline", "Δ top1 pp")

    rowmd = "\n".join(f"| {k} | {table[k][0][0]}±{table[k][0][1]} | {table[k][1][0]:+} | {table[k][1][1]}/10 |" for k in ks)
    verdict = (f"🟢 **{best}** 가 cosine을 이김 (dev Δ{d_best[0]:+}, {d_best[1]}/10){sealed} → 채택."
               if adopt else
               f"🔴 **재순위화 무효** — CSLS/AQE/local-scaling 어느 것도 cosine exemplar를 못 넘음 (best {best} Δ{d_best[0]:+}). "
               f"hubness/비대칭이 우리 검색의 병목이 아님 — exemplar 1-NN이 이미 이 축에서 최적.")
    report = f"""# 051 — 검색 재순위화 (CSLS·AQE·local-scaling, 학습-free)

- 날짜: {datetime.date.today().isoformat()} · 커밋 `main @ {_git_sha()}` · `scripts/rerank_retrieval.py`
- clean 502 (dev {len(dev)}/test {len(test)} 봉인), global+L256, dev 10-seed CV paired. 봉인 test는 채택분만.
- 동기: re-ID 문헌의 학습-free 부스터로 exemplar 1-NN의 hubness(CSLS)·비대칭(AQE)·국소밀도(local-scaling) 교정.

## 결과 (paired Δ vs cosine)
| 방법 | dev-CV top1 | Δ | wins |
|---|---|---|---|
{rowmd}

![methods](fig1_methods.png)
![delta](fig2_delta.png)

## 판정
{verdict}
"""
    explog.write(d, report, {
        "title": "검색 재순위화 (CSLS·AQE·local-scaling)", "date": datetime.date.today().isoformat(),
        "headline": f"best={best} dev Δ{d_best[0]:+}({d_best[1]}/10){' ADOPT'+sealed if adopt else ' — 무효(cosine 최적)'}",
        "adopt": bool(adopt), "best": best,
        "devcv": {k: {"top1": table[k][0], "delta": table[k][1]} for k in ks}})
    print(f"\nwrote -> {d}  (2 figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
