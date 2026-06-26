"""Experiment — feature-coherent point pooling (task-tailored, training-free).

GaussianPool weights patches by DISTANCE only, so a thin artery's embedding is
blended with the surrounding fat/adjacent tissue — bad for the look-alike
discrimination that dominates our errors. Feature-coherent pooling weights each
patch by BOTH spatial proximity AND feature similarity to the pin's own patch:

    w_i = softmax( -||p_i - q||^2 / 2σ^2  +  cos(token_i, seed) / τ )

so it pools the structure the pin sits on (e.g. the vessel along its length) and
suppresses dissimilar neighbours — a parameter-free region-grow from the pin. We
use a WIDE spatial σ so the feature term does the selecting. Sweep τ; compare
PAIRED to GaussianPool. Cached grids, exemplar 1-NN, 10 seeds.

    .venv/bin/python scripts/coherent_pool.py
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
import torch.nn.functional as F  # noqa: E402

import explog  # noqa: E402
from eval_appearance import load_core, split_indices, _git_sha  # noqa: E402
from learned_pooler import cache_grids, gather  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone  # noqa: E402


def seed_token(grid, q, S):
    g, _, d = grid.shape
    feat = grid.permute(2, 0, 1).unsqueeze(0)
    qn = (2.0 * (q / S) - 1.0).view(1, 1, 1, 2)
    return F.grid_sample(feat, qn, mode="bilinear", align_corners=False,
                         padding_mode="border").reshape(d)


def pool(grid, centers, q, S, mode, sigma, tau):
    g, _, d = grid.shape
    tok = grid.reshape(g * g, d)
    d2 = ((centers - q) ** 2).sum(-1)
    spatial = -d2 / (2.0 * sigma ** 2)
    if mode == "gauss":
        w = torch.softmax(spatial, 0)
    else:  # coherent
        seed = F.normalize(seed_token(grid, q, S), dim=0)
        feat = (F.normalize(tok, dim=1) @ seed) / tau
        w = torch.softmax(spatial + feat, 0)
    return F.normalize((w[:, None] * tok).sum(0), dim=0)


def pool_all(core, grids, wh, centers, S, device, mode, sigma, tau, chunk=128):
    Z = []
    qf = {}
    for i, r in enumerate(core):
        w, h = wh[r["image"]]
        qf[i] = (r["q"][0] * S / w, r["q"][1] * S / h)
    for s in range(0, len(core), chunk):
        ids = list(range(s, min(s + chunk, len(core))))
        g = gather(grids, core, ids, device)
        for n, i in enumerate(ids):
            q = torch.tensor(qf[i], device=device)
            Z.append(pool(g[n], centers, q, S, mode, sigma, tau).cpu())
    return torch.stack(Z).numpy().astype(np.float32)


def exrun(core, Y, Z):
    t1, t5 = [], []
    for seed in range(10):
        tr, te = split_indices(core, 0.3, seed)
        ytr = [Y[i] for i in tr]
        labels = sorted(set(ytr)); lidx = {l: j for j, l in enumerate(labels)}
        cols = collections.defaultdict(list)
        for j, l in enumerate(ytr):
            cols[lidx[l]].append(j)
        cv = [k for k in te if Y[k] in lidx]
        sims = Z[cv] @ Z[tr].T
        sc = np.full((len(cv), len(labels)), -2.0, np.float32)
        for c, ix in cols.items():
            sc[:, c] = sims[:, ix].max(1)
        o = np.argsort(-sc, axis=1)
        t1.append(100 * sum(int(labels[o[r, 0]] == Y[k]) for r, k in enumerate(cv)) / len(cv))
        t5.append(100 * sum(int(Y[k] in [labels[o[r, t]] for t in range(5)]) for r, k in enumerate(cv)) / len(cv))
    return t1, t5


def main():
    base = Path("data/triples")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size
    core = load_core("data/triples/triples.jsonl", 2)
    Y = [r["label"] for r in core]
    print(f"core {len(core)}/{len(set(Y))} | device={device}")
    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    centers = bb.patch_centers(device)
    print("caching grids..."); grids, wh = cache_grids(core, base, bb, S, device)

    configs = [("gauss σ40", "gauss", 40, None),
               ("coherent σ80 τ0.1", "coherent", 80, 0.1),
               ("coherent σ80 τ0.2", "coherent", 80, 0.2),
               ("coherent σ80 τ0.3", "coherent", 80, 0.3),
               ("coherent σ120 τ0.2", "coherent", 120, 0.2)]
    res = {}
    for name, mode, sig, tau in configs:
        Z = pool_all(core, grids, wh, centers, S, device, mode, sig, tau)
        res[name] = exrun(core, Y, Z)
        print(f"  {name:20s} top1 {st.mean(res[name][0]):.1f}±{st.pstdev(res[name][0]):.1f}  "
              f"top5 {st.mean(res[name][1]):.1f}")

    ms = lambda v: (round(st.mean(v), 1), round(st.pstdev(v), 1))
    base_t1 = res["gauss σ40"][0]
    rows = []
    for name, *_ in configs:
        t1, t5 = res[name]
        d = [a - b for a, b in zip(t1, base_t1)]
        rows.append((name, ms(t1), ms(t5), round(st.mean(d), 1), sum(x > 0 for x in d)))
    best = max(rows[1:], key=lambda r: r[3])
    verdict = "coherent 풀링이 도움 (paired)" if (best[3] > 0 and best[4] >= 8) else "효과 불명확/노이즈"
    print(f"\n== best coherent: {best[0]} Δtop1 {best[3]:+} ({best[4]}/10) -> {verdict} ==")

    d = explog.next_dir("coherent-pool")
    explog.bar(d / "fig_coherent.png", [r[0].replace(" ", "\n") for r in rows], [r[1][0] for r in rows],
               "Feature-coherent pooling: top1 (10-seed)", "%", ymax=100, errors=[r[1][1] for r in rows])
    tab = "\n".join(f"| {n} | {t1[0]}±{t1[1]}% | {t5[0]}±{t5[1]}% | {dd:+} ({w}/10) |" for n, t1, t5, dd, w in rows)
    report = f"""# Feature-coherent 풀링 (coherent-pool)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/coherent_pool.py`

## 목적
GaussianPool은 거리로만 풀링 → 핀 구조물에 주변 조직이 섞임. **거리 × 핀패치 특징유사도**로 풀링해
구조물만 모으면(param-free region-grow) 미세 look-alike 판별이 좋아지는지.

## 결과 (exemplar 1-NN, 10-seed, paired vs gauss σ40)
| 풀링 | top1 | top5 | Δtop1 |
|---|---|---|---|
{tab}

![coherent](fig_coherent.png)

## 판정
- 베스트: **{best[0]}** Δtop1 {best[3]:+}%p ({best[4]}/10) → **{verdict}**

## 해석
- 도움되면 → 정식 풀러 채택(학습 없이 미세판별↑). 무효면 → DINO 패치는 이미 구조-국소적이라 거리풀링으로
  충분(특징선택이 추가 정보를 안 줌).
"""
    explog.write(d, report, {
        "title": "Feature-coherent 풀링", "date": datetime.date.today().isoformat(),
        "headline": f"best={best[0]} Δtop1 {best[3]:+}({best[4]}/10) → {verdict}",
        "configs": {n: {"top1": t1, "top5": t5, "dtop1": dd} for n, t1, t5, dd, w in rows}})
    print(f"wrote -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
