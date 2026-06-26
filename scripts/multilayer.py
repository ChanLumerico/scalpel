"""Experiment — multi-layer DINO features (texture + semantics).

We only pool the LAST DINO layer. Early layers carry low-level texture (vessel-wall
striation, muscle fibre direction) that the semantic last layer may wash out — and
that texture is exactly what separates same-region look-alikes (artery vs vein).
Pool several layers at the pin, concatenate, exemplar 1-NN; PAIRED vs last-layer.

    .venv/bin/python scripts/multilayer.py
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
from scalpel.perception import DinoBackbone  # noqa: E402

LAYERSETS = {
    "last (L11)": [11],
    "L8+L11": [8, 11],
    "L5+L8+L11": [5, 8, 11],
    "L2+L5+L8+L11": [2, 5, 8, 11],
    "L2+L11 (tex+sem)": [2, 11],
}


def gpool(grid, centers, q, sigma=40.0):
    g, _, d = grid.shape
    tok = grid.reshape(g * g, d)
    w = torch.softmax(-((centers - q) ** 2).sum(-1) / (2 * sigma ** 2), 0)
    return F.normalize((w[:, None] * tok).sum(0), dim=0)


@torch.no_grad()
def embed(core, base, model, centers, S, device, layers):
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    g = int(S // 14)
    by = collections.defaultdict(list)
    for i, r in enumerate(core):
        by[r["image"]].append(i)
    Z = [None] * len(core)
    for n, (img, idxs) in enumerate(by.items(), 1):
        im = Image.open(base / img).convert("RGB"); w, h = im.size
        arr = np.asarray(im.resize((S, S)), np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        feats = model.get_intermediate_layers((x - mean) / std, n=layers, reshape=False, norm=True)
        grids = [f[0].reshape(g, g, -1) for f in feats]   # each (g,g,D)
        for i in idxs:
            qx, qy = core[i]["q"]
            q = torch.tensor([qx * S / w, qy * S / h], device=device)
            z = torch.cat([gpool(gr, centers, q) for gr in grids])
            Z[i] = F.normalize(z, dim=0).cpu().numpy()
        if n % 60 == 0:
            print(f"   {n}", flush=True)
    return np.stack(Z).astype(np.float32)


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
    model = bb._model

    res = {}
    for name, layers in LAYERSETS.items():
        Z = embed(core, base, model, centers, S, device, layers)
        res[name] = exrun(core, Y, Z)
        print(f"  {name:18s} top1 {st.mean(res[name][0]):.1f}±{st.pstdev(res[name][0]):.1f}  top5 {st.mean(res[name][1]):.1f}")

    ms = lambda v: (round(st.mean(v), 1), round(st.pstdev(v), 1))
    base1 = res["last (L11)"][0]
    rows = []
    for name in LAYERSETS:
        t1, t5 = res[name]
        d = [a - b for a, b in zip(t1, base1)]
        rows.append((name, ms(t1), ms(t5), round(st.mean(d), 1), sum(x > 0 for x in d)))
    best = max(rows[1:], key=lambda r: r[3])
    verdict = "다층 특징이 도움 (paired)" if (best[3] > 0 and best[4] >= 8) else "효과 불명확/노이즈"
    print(f"\n== best: {best[0]} Δtop1 {best[3]:+} ({best[4]}/10) -> {verdict} ==")

    d = explog.next_dir("multilayer")
    explog.bar(d / "fig_ml.png", [r[0].replace(" ", "\n") for r in rows], [r[1][0] for r in rows],
               "Multi-layer DINO features: top1 (10-seed)", "%", ymax=100, errors=[r[1][1] for r in rows])
    tab = "\n".join(f"| {n} | {t1[0]}±{t1[1]}% | {t5[0]}% | {dd:+} ({w}/10) |" for n, t1, t5, dd, w in rows)
    report = f"""# 다층 DINO 특징 (multilayer)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/multilayer.py`

## 목적
마지막 층(의미)만 쓰던 걸, **초기층(질감·결) + 후기층(의미)** 을 핀에서 풀링·concat. 동맥/정맥의
미세 질감 차이를 초기층이 잡는지. exemplar 1-NN, paired.

## 결과 (10-seed, paired vs last-layer)
| 층 조합 | top1 | top5 | Δtop1 |
|---|---|---|---|
{tab}

![ml](fig_ml.png)

## 판정
- 베스트: **{best[0]}** Δtop1 {best[3]:+}%p ({best[4]}/10) → **{verdict}**
"""
    explog.write(d, report, {
        "title": "다층 DINO 특징", "date": datetime.date.today().isoformat(),
        "headline": f"best={best[0]} Δtop1 {best[3]:+}({best[4]}/10) → {verdict}",
        "layersets": {n: {"top1": t1, "top5": t5, "dtop1": dd} for n, t1, t5, dd, w in rows}})
    print(f"wrote -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
