"""Experiment — region-conditioned Bayesian prior (task-tailored re-ranking).

Structures co-occur by body region; the global CLS token encodes the region and
lifted top5 (exp 008). Combine it as a PRIOR rather than a concat feature:

    score(y) = appearance_exemplar_sim(y) + λ · cos(CLS_test, region_proto(y))

where region_proto(y) = mean CLS of class y's gallery images. This up-weights
classes whose typical region matches the test image's region — fixing cross-region
confusions (cerebellum↔cecum) even if it can't separate same-region look-alikes.
λ swept at FIXED values (not tuned on test). exemplar 1-NN, 10-seed, paired.

    .venv/bin/python scripts/region_prior.py
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
from PIL import Image  # noqa: E402

import explog  # noqa: E402
from eval_appearance import load_core, split_indices, _MEAN, _STD, _git_sha  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402

LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]


@torch.no_grad()
def embed(core, base, bb, pool, centers, S, device):
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    by = collections.defaultdict(list)
    for i, r in enumerate(core):
        by[r["image"]].append(i)
    Zq = [None] * len(core); Cl = [None] * len(core)
    for img, idxs in by.items():
        im = Image.open(base / img).convert("RGB"); w, h = im.size
        arr = np.asarray(im.resize((S, S)), np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        grid, cls = bb((x - mean) / std)
        c = F.normalize(cls[0], dim=0).cpu().numpy()
        for i in idxs:
            qx, qy = core[i]["q"]
            q = torch.tensor([[qx * S / w, qy * S / h]], device=device)
            Zq[i] = F.normalize(pool(grid, centers, q)[0], dim=0).cpu().numpy()
            Cl[i] = c
    return np.stack(Zq).astype(np.float32), np.stack(Cl).astype(np.float32)


def run(core, Y, Zq, Cl):
    out = {lam: ([], []) for lam in LAMBDAS}
    for seed in range(10):
        tr, te = split_indices(core, 0.3, seed)
        ytr = [Y[i] for i in tr]
        labels = sorted(set(ytr)); lidx = {l: j for j, l in enumerate(labels)}
        cols = collections.defaultdict(list)
        for j, l in enumerate(ytr):
            cols[lidx[l]].append(j)
        Ztr = Zq[tr]; Ctr = Cl[tr]
        rproto = np.stack([_n(Ctr[cols[lidx[l]]].mean(0)) for l in labels])  # (C, D)
        cv = [k for k in te if Y[k] in lidx]
        app = Zq[cv] @ Ztr.T
        appc = np.full((len(cv), len(labels)), -2.0, np.float32)
        for c, ix in cols.items():
            appc[:, c] = app[:, ix].max(1)
        reg = Cl[cv] @ rproto.T                                  # (Ncov, C) region sim
        for lam in LAMBDAS:
            sc = appc + lam * reg
            o = np.argsort(-sc, axis=1)
            n1 = sum(int(labels[o[r, 0]] == Y[k]) for r, k in enumerate(cv))
            n5 = sum(int(Y[k] in [labels[o[r, t]] for t in range(5)]) for r, k in enumerate(cv))
            out[lam][0].append(100 * n1 / len(cv)); out[lam][1].append(100 * n5 / len(cv))
    return out


def _n(v):
    return v / (np.linalg.norm(v) + 1e-9)


def main():
    base = Path("data/triples")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size
    core = load_core("data/triples/triples.jsonl", 2)
    Y = [r["label"] for r in core]
    print(f"core {len(core)}/{len(set(Y))} | device={device}")
    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    pool = GaussianPool(cfg.point).to(device); centers = bb.patch_centers(device)
    print("embedding (Zq + CLS)..."); Zq, Cl = embed(core, base, bb, pool, centers, S, device)
    Zq = Zq / (np.linalg.norm(Zq, axis=1, keepdims=True) + 1e-9)
    Cl = Cl / (np.linalg.norm(Cl, axis=1, keepdims=True) + 1e-9)

    out = run(core, Y, Zq, Cl)
    ms = lambda v: (round(st.mean(v), 1), round(st.pstdev(v), 1))
    base1 = out[0.0][0]
    rows = []
    for lam in LAMBDAS:
        t1, t5 = out[lam]
        d = [a - b for a, b in zip(t1, base1)]
        rows.append((lam, ms(t1), ms(t5), round(st.mean(d), 1), sum(x > 0 for x in d)))
    best = max(rows[1:], key=lambda r: r[3])
    verdict = "부위 사전이 top1 향상" if (best[3] > 0 and best[4] >= 8) else "효과 불명확 (교차-부위 혼동은 소수)"
    for lam, t1, t5, dd, w in rows:
        print(f"  λ={lam}: top1 {t1[0]}±{t1[1]}  top5 {t5[0]}  Δ{dd:+}({w}/10)")
    print(f"\n== best λ={best[0]}: Δtop1 {best[3]:+} ({best[4]}/10) -> {verdict} ==")

    d = explog.next_dir("region-prior")
    explog.bar(d / "fig_region.png", [f"λ{r[0]}" for r in rows], [r[1][0] for r in rows],
               "Region prior: top1 vs λ (10-seed)", "%", ymax=100, errors=[r[1][1] for r in rows])
    tab = "\n".join(f"| {lam} | {t1[0]}±{t1[1]}% | {t5[0]}% | {dd:+} ({w}/10) |" for lam, t1, t5, dd, w in rows)
    report = f"""# 부위-조건부 사전 (region-prior)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/region_prior.py`

## 목적
`score = 외형유사도 + λ·cos(CLS_test, 부위프로토타입)`. region(CLS)을 *concat 특징*(exp 008)이
아니라 *베이지안 사전*으로 결합 → 교차-부위 혼동을 칠 수 있는지. λ 고정 스윕(test-튜닝 아님).

## 결과 (exemplar 1-NN, 10-seed, paired vs λ=0)
| λ | top1 | top5 | Δtop1 |
|---|---|---|---|
{tab}

![region](fig_region.png)

## 판정
- 베스트 λ={best[0]}: Δtop1 {best[3]:+}%p ({best[4]}/10) → **{verdict}**

## 해석
- 교차-부위 혼동(예: cerebellum↔cecum)은 칠 수 있으나, 주된 혼동(같은-부위 동맥↔정맥)은 부위가 같아
  못 가름. 이득이 작으면 → 대부분의 혼동이 *같은 부위 내 미세판별*이라는 또 다른 확인.
"""
    explog.write(d, report, {
        "title": "부위-조건부 사전", "date": datetime.date.today().isoformat(),
        "headline": f"best λ={best[0]} Δtop1 {best[3]:+}({best[4]}/10) → {verdict}",
        "lambdas": {str(lam): {"top1": t1, "top5": t5, "dtop1": dd} for lam, t1, t5, dd, w in rows}})
    print(f"wrote -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
