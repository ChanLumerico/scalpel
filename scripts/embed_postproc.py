"""Experiment 052 — training-free embedding post-processing on global+L256.

EDA (042) showed DINO space is dominated by REGION variance (region sep 0.10 ≫ tissue ≈0). If a few
principal directions carry that region/scene signal, removing or equalising them should expose the
finer within-region identity the cosine currently drowns. Classic training-free tricks:
  • all-but-the-top (Mu & Viswanath 2018) — subtract the top-d principal components (common signal).
  • PCA-whitening — decorrelate + equalise variance so no direction dominates the cosine.
  • per-dim standardisation (z-score).
  • image-context removal — subtract the mean embedding of the pins on the same photo (scene signal).
All transforms are FIT ON THE GALLERY (train fold) only and applied to the query → leak-safe. Cheap.

dev 10-seed CV paired vs raw global+L256 (33.5); sealed test only for an adopted transform.

    .venv/bin/python scripts/embed_postproc.py
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


# ---- transform factories: fit on gallery X (n,d) → return a function applied to any Z ----
def fit_identity(X):
    return lambda Z: Z


def fit_meancenter(X):
    mu = X.mean(0)
    return lambda Z: Z - mu


def fit_abtt(X, d=3):
    mu = X.mean(0); Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    U = Vt[:d]                                  # top-d principal directions
    return lambda Z: (Z - mu) - ((Z - mu) @ U.T) @ U


def fit_whiten(X, k=256, eps=1e-3):
    mu = X.mean(0); Xc = X - mu
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:k].T; s = S[:k] / np.sqrt(len(X))
    W = V @ np.diag(1.0 / (s + eps))            # PCA-whiten, keep top-k
    return lambda Z: (Z - mu) @ W


def fit_std(X):
    mu = X.mean(0); sd = X.std(0) + 1e-6
    return lambda Z: (Z - mu) / sd


FITTERS = {
    "baseline": fit_identity,
    "mean-center": fit_meancenter,
    "all-but-top d=1": lambda X: fit_abtt(X, 1), "all-but-top d=3": lambda X: fit_abtt(X, 3),
    "all-but-top d=5": lambda X: fit_abtt(X, 5), "all-but-top d=10": lambda X: fit_abtt(X, 10),
    "whiten k=128": lambda X: fit_whiten(X, 128), "whiten k=256": lambda X: fit_whiten(X, 256),
    "std z-score": fit_std,
}


def exemplar_top1(Zt, Y, tr, te):
    labset = set(Y[i] for i in tr)
    cov = [q for q in te if Y[q] in labset]
    if not cov:
        return float("nan")
    labs = sorted(labset); li = {l: j for j, l in enumerate(labs)}
    cols = collections.defaultdict(list)
    for j, i in enumerate(tr):
        cols[li[Y[i]]].append(j)
    sims = Zt[cov] @ Zt[tr].T
    sc = np.full((len(cov), len(labs)), -9.0, np.float32)
    for c, ix in cols.items():
        sc[:, c] = sims[:, ix].max(1)
    pred = sc.argmax(1)
    return 100 * np.mean([labs[pred[r]] == Y[cov[r]] for r in range(len(cov))])


def eval_fit(fitter, Z, Y, tr, te):
    T = fitter(Z[tr])
    Zt = unit(T(Z))
    return exemplar_top1(Zt, Y, tr, te)


def img_context_removed(Z, rows, core):
    """Subtract, from each pin embedding, the mean embedding of all pins on its image (scene signal)."""
    by = collections.defaultdict(list)
    for i in core:
        by[rows[i]["image"]].append(i)
    Zc = Z.copy()
    for img, idxs in by.items():
        m = Z[idxs].mean(0)
        for i in idxs:
            Zc[i] = Z[i] - m
    return unit(Zc)


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
    for name, fit in FITTERS.items():
        res[name] = [eval_fit(fit, Z, Y, tr, te) for tr, te in splits]
    # image-context removal (precomputed transform, not gallery-fit)
    Zic = img_context_removed(Z, rows, core)
    res["img-context-rm"] = [exemplar_top1(Zic, Y, tr, te) for tr, te in splits]

    base = res["baseline"]

    def paired(a):
        d = [x - y for x, y in zip(a, base)]
        return round(st.mean(d), 2), int(sum(x > 0 for x in d))

    print("\n== dev-CV 10-seed (paired vs baseline) ==")
    table = {}
    for name in res:
        dlt = paired(res[name]); table[name] = (ms(res[name]), dlt)
        print(f"  {name:18} top1 {ms(res[name])[0]}±{ms(res[name])[1]}  Δ {dlt[0]:+} ({dlt[1]}/10)")

    cand = [n for n in res if n != "baseline"]
    best = max(cand, key=lambda n: st.mean(res[n]))
    d_best = table[best][1]
    adopt = d_best[0] > 0 and d_best[1] >= 7
    sealed = ""
    if adopt:
        if best == "img-context-rm":
            stb = exemplar_top1(Zic, Y, dev, test)
        else:
            T = FITTERS[best](Z[dev]); stb = exemplar_top1(unit(T(Z)), Y, dev, test)
        st0 = exemplar_top1(Z, Y, dev, test)
        sealed = f" | SEALED base {round(st0,1)} → {best} {round(stb,1)}"
    print(f"\n  {'★ '+best+' ADOPT' if adopt else 'best '+best+' — no adoption'} (dev Δ{d_best[0]:+}, {d_best[1]}/10){sealed}")

    d = explog.EXP / "052-embed-postproc"; d.mkdir(parents=True, exist_ok=True)
    ks = list(res)
    explog.bar(d / "fig1.png", [k.replace(" ", "\n") for k in ks], [table[k][0][0] for k in ks],
               "052 embedding post-processing (dev-CV top1)", "%", ymax=40, errors=[table[k][0][1] for k in ks])
    explog.bar(d / "fig2_delta.png", [k.replace(" ", "\n") for k in ks if k != "baseline"],
               [table[k][1][0] for k in ks if k != "baseline"], "052 paired Δtop1 vs raw global+L256", "Δ pp")

    rowmd = "\n".join(f"| {k} | {table[k][0][0]}±{table[k][0][1]} | {table[k][1][0]:+} | {table[k][1][1]}/10 |" for k in ks)
    verdict = (f"🟢 **{best}** 채택 (dev Δ{d_best[0]:+}, {d_best[1]}/10){sealed}."
               if adopt else
               f"🔴 **후처리 무효** — all-but-top/whiten/std/img-context 어느 것도 raw cosine을 못 넘음 (best {best} Δ{d_best[0]:+}). "
               f"부위지배 분산을 선형 제거해도 미세정체성이 안 살아남 = 부위/정체성이 같은 부분공간에 얽혀 선형분리 불가.")
    report = f"""# 052 — 임베딩 후처리 (all-but-the-top·whitening·표준화·이미지맥락제거)

- 날짜: {datetime.date.today().isoformat()} · 커밋 `main @ {_git_sha()}` · `scripts/embed_postproc.py`
- clean 502 (dev {len(dev)}/test {len(test)} 봉인), global+L256, 변환은 갤러리(train fold)에서만 fit. dev 10-seed paired.
- 동기(042): DINO 공간은 부위 분산이 지배 → 그 상위 주성분을 제거/균등화하면 미세정체성이 드러날까.

## 결과 (paired Δ vs raw cosine)
| 변환 | dev-CV top1 | Δ | wins |
|---|---|---|---|
{rowmd}

![methods](fig1.png)
![delta](fig2_delta.png)

## 판정
{verdict}
"""
    explog.write(d, report, {
        "title": "임베딩 후처리 (all-but-top·whiten·std)", "date": datetime.date.today().isoformat(),
        "headline": f"best={best} dev Δ{d_best[0]:+}({d_best[1]}/10){' ADOPT'+sealed if adopt else ' — 무효'}",
        "adopt": bool(adopt), "best": best,
        "devcv": {k: {"top1": table[k][0], "delta": table[k][1]} for k in res}})
    print(f"\nwrote -> {d}  (2 figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
