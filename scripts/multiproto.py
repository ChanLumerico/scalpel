"""Experiment — multi-prototype & soft aggregation per class.

Our lesson: exemplar (max over ALL gallery items) ≫ mean prototype (mean washes
out detail). But max-over-all is sensitive to a single noisy gallery item, and
mean is too smooth. Between them: cluster each class's gallery into K sub-prototypes
(k-means) — captures multi-view / multi-instance modes without over-smoothing — and
a SOFT log-sum-exp aggregation (temperature interpolates mean↔max). Test whether
anything beats plain exemplar-max. 10-seed, paired vs exemplar.

    .venv/bin/python scripts/multiproto.py
"""

from __future__ import annotations

import collections
import datetime
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import explog  # noqa: E402
from eval_appearance import load_core, split_indices, embed, _git_sha  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402


def _kmeans(X, k, iters=15, seed=0):
    rng = np.random.default_rng(seed)
    if len(X) <= k:
        return X.copy()
    c = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        a = np.argmax(X @ c.T, axis=1)
        nc = []
        for j in range(k):
            m = X[a == j]
            nc.append(m.mean(0) if len(m) else c[j])
        nc = np.stack(nc)
        nc /= (np.linalg.norm(nc, axis=1, keepdims=True) + 1e-9)
        if np.allclose(nc, c):
            break
        c = nc
    return c


def topk_from_scores(sc, labels, Y, cv):
    o = np.argsort(-sc, axis=1)
    n1 = sum(int(labels[o[r, 0]] == Y[cv[r]]) for r in range(len(cv)))
    n5 = sum(int(Y[cv[r]] in [labels[o[r, t]] for t in range(5)]) for r in range(len(cv)))
    return 100 * n1 / len(cv), 100 * n5 / len(cv)


def main():
    base = Path("data/triples")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size
    core = load_core("data/triples/triples.jsonl", 2)
    Y = [r["label"] for r in core]
    print(f"core {len(core)}/{len(set(Y))} | device={device}")
    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    pool = GaussianPool(cfg.point).to(device)
    print("embed once..."); Z = embed(core, base, bb, pool, S, device).numpy().astype(np.float32)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)

    methods = ["mean", "kmeans-2", "kmeans-3", "exemplar", "lse-τ.1", "lse-τ.05"]
    acc = {m: ([], []) for m in methods}
    for seed in range(10):
        tr, te = split_indices(core, 0.3, seed)
        ytr = [Y[i] for i in tr]
        labels = sorted(set(ytr)); lidx = {l: j for j, l in enumerate(labels)}
        cols = collections.defaultdict(list)
        for j, l in enumerate(ytr):
            cols[lidx[l]].append(j)
        cv = [k for k in te if Y[k] in lidx]
        Zc = Z[cv]; Ztr = Z[tr]
        sims = Zc @ Ztr.T                                    # (Ncov, Ntr)
        C = len(labels)
        sc = {m: np.full((len(cv), C), -2.0, np.float32) for m in methods}
        for c, ix in cols.items():
            g = Ztr[ix]                                      # gallery of class c
            s = sims[:, ix]                                  # (Ncov, n_c)
            sc["exemplar"][:, c] = s.max(1)
            mean = g.mean(0); mean /= (np.linalg.norm(mean) + 1e-9)
            sc["mean"][:, c] = Zc @ mean
            for K, nm in [(2, "kmeans-2"), (3, "kmeans-3")]:
                cen = _kmeans(g, K, seed=seed)
                sc[nm][:, c] = (Zc @ cen.T).max(1)
            for tau, nm in [(0.1, "lse-τ.1"), (0.05, "lse-τ.05")]:
                sc[nm][:, c] = tau * np.log(np.exp(s / tau).sum(1) + 1e-30)
        for m in methods:
            a = topk_from_scores(sc[m], labels, Y, cv)
            acc[m][0].append(a[0]); acc[m][1].append(a[1])
        print(f"  seed {seed}: " + " ".join(f"{m} {acc[m][0][-1]:.0f}" for m in methods))

    ms = lambda v: (round(st.mean(v), 1), round(st.pstdev(v), 1))
    base1 = acc["exemplar"][0]
    rows = []
    for m in methods:
        t1, t5 = acc[m]
        dd = [a - b for a, b in zip(t1, base1)]
        rows.append((m, ms(t1), ms(t5), round(st.mean(dd), 1), sum(x > 0 for x in dd)))
    cand = [r for r in rows if r[0] != "exemplar"]
    best = max(cand, key=lambda r: r[3])
    verdict = ("exemplar-max보다 향상 (paired)" if (best[3] > 0 and best[4] >= 8)
               else "exemplar-max가 여전히 최선")
    print("\n== " + " | ".join(f"{m} {ms(acc[m][0])[0]}" for m in methods) +
          f" | best non-ex {best[0]} Δ{best[3]:+}({best[4]}/10) -> {verdict} ==")

    d = explog.next_dir("multiproto")
    explog.bar(d / "fig_mp.png", [r[0] for r in rows], [r[1][0] for r in rows],
               "Multi-prototype / soft aggregation: top1 (10-seed)", "%", ymax=100,
               errors=[r[1][1] for r in rows])
    tab = "\n".join(f"| {m} | {t1[0]}±{t1[1]}% | {t5[0]}% | {dd:+} ({w}/10) |" for m, t1, t5, dd, w in rows)
    report = f"""# 다중 프로토타입 / soft 집계 (multiproto)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/multiproto.py`

## 목적
교훈 = exemplar(전체 max) ≫ mean(뭉갬). 그 사이 — 클래스별 k-means K개 부분프로토타입(다중모드)과
soft log-sum-exp 집계(τ가 mean↔max 보간) — 가 plain max를 넘는지. 10-seed, paired vs exemplar.

## 결과 (paired vs exemplar)
| 집계 | top1 | top5 | Δtop1 |
|---|---|---|---|
{tab}

![mp](fig_mp.png)

## 판정
- 베스트(비-exemplar): **{best[0]}** Δtop1 {best[3]:+}%p ({best[4]}/10) → **{verdict}**

## 해석
- exemplar가 최선이면 → 갤러리가 작아(2-core) 부분프로토/soft가 줄 여지가 없음, max가 디테일 보존엔
  최적. soft가 약간 도우면 → 노이즈 완화 여지(추후 큰 갤러리에서 재평가).
"""
    explog.write(d, report, {
        "title": "다중 프로토타입/soft 집계", "date": datetime.date.today().isoformat(),
        "headline": f"best non-ex {best[0]} Δtop1 {best[3]:+}({best[4]}/10) → {verdict}",
        "methods": {m: {"top1": t1, "top5": t5, "dtop1": dd} for m, t1, t5, dd, w in rows}})
    print(f"wrote -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
