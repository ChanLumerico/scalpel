"""Experiment — M6' step 1: structured neighbour-region context (training-free).

The cheap context probe (exp 008) showed a GLOBAL region token (CLS) lifts top5
but not top1. Here we test a STRUCTURED relational signal: over-segment each photo
into regions (k-means on DINO patch tokens + position), find the pin's region and
its ADJACENT regions, and fuse their appearance into the point embedding. If
neighbour-region context lifts top1 where global context didn't, it justifies
building the full R-GCN relational expert; if not, structure also can't beat the
data limit at this scale.

Variants (each sub-vector L2-normalized, concatenated, re-normalized):
  base      = z_q (GaussianPool)
  +region   = [z_q ; pin-region mean]
  +neighbor = [z_q ; mean of adjacent regions]
  +both
Frozen, exemplar 1-NN, 10-seed paired. Foundation for M6'.

    .venv/bin/python scripts/relational_context.py
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
from scipy.cluster.vq import kmeans2  # noqa: E402

import explog  # noqa: E402
from eval_appearance import load_core, split_indices, _git_sha  # noqa: E402
from learned_pooler import cache_grids  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone  # noqa: E402

K = 12          # regions per image
BETA = 2.0      # spatial weight in clustering (region contiguity)


def segment(grid):
    """k-means over [unit token ; beta*pos] -> (g,g) region label map."""
    g, _, d = grid.shape
    tok = F.normalize(grid.reshape(g * g, d).float(), dim=1).numpy()
    ys, xs = np.divmod(np.arange(g * g), g)
    pos = np.stack([ys / g, xs / g], 1).astype(np.float32) * BETA
    feat = np.concatenate([tok, pos], 1)
    _, lab = kmeans2(feat, K, minit="++", seed=0, missing="warn")
    return lab.reshape(g, g), grid.reshape(g * g, d).float().numpy()


def region_means(lab, tokens):
    g = lab.shape[0]
    flat = lab.reshape(-1)
    means = {}
    for c in np.unique(flat):
        means[int(c)] = tokens[flat == c].mean(0)
    # adjacency from 4-neighbourhood on the grid
    adj = collections.defaultdict(set)
    for r in range(g):
        for col in range(g):
            a = lab[r, col]
            for dr, dc in ((1, 0), (0, 1)):
                rr, cc = r + dr, col + dc
                if rr < g and cc < g and lab[rr, cc] != a:
                    adj[int(a)].add(int(lab[rr, cc])); adj[int(lab[rr, cc])].add(int(a))
    return means, adj


def nrm(v):
    return v / (np.linalg.norm(v) + 1e-9)


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

    # per-image segmentation (cache by image)
    print("segmenting + region embeddings...")
    seg = {}
    for img, gr in grids.items():
        lab, tok = segment(gr)
        seg[img] = region_means(lab, tok) + (lab,)

    # per-triple embeddings for each variant
    p = cfg.backbone.patch_size
    g = cfg.backbone.grid_size
    Z = {v: np.zeros((len(core), 0)) for v in ["base", "region", "neighbor", "both"]}
    cols = {v: [] for v in Z}
    for i, r in enumerate(core):
        gr = grids[r["image"]].float()
        w, h = wh[r["image"]]
        qx, qy = r["q"][0] * S / w, r["q"][1] * S / h
        # GaussianPool z_q
        d2 = ((centers.cpu() - torch.tensor([qx, qy])) ** 2).sum(-1)
        wts = torch.softmax(-d2 / (2 * 40.0 ** 2), 0)
        zq = nrm((wts[:, None] * gr.reshape(g * g, -1)).sum(0).numpy())
        means, adj, lab = seg[r["image"]]
        pr = int(lab[min(g - 1, int(qy // p)), min(g - 1, int(qx // p))])
        zr = nrm(means[pr])
        nb = list(adj.get(pr, []))
        zn = nrm(np.mean([means[c] for c in nb], 0)) if nb else np.zeros_like(zr)
        cols["base"].append(zq)
        cols["region"].append(nrm(np.concatenate([zq, zr])))
        cols["neighbor"].append(nrm(np.concatenate([zq, zn])))
        cols["both"].append(nrm(np.concatenate([zq, zr, zn])))
    for v in Z:
        Z[v] = np.stack(cols[v]).astype(np.float32)

    def exrun(Zv):
        t1, t5 = [], []
        for seed in range(10):
            tr, te = split_indices(core, 0.3, seed)
            ytr = [Y[i] for i in tr]
            labels = sorted(set(ytr)); lidx = {l: j for j, l in enumerate(labels)}
            colmap = collections.defaultdict(list)
            for j, l in enumerate(ytr):
                colmap[lidx[l]].append(j)
            cv = [k for k in te if Y[k] in lidx]
            sims = Zv[cv] @ Zv[tr].T
            sc = np.full((len(cv), len(labels)), -2.0, np.float32)
            for c, ix in colmap.items():
                sc[:, c] = sims[:, ix].max(1)
            o = np.argsort(-sc, axis=1)
            t1.append(100 * sum(int(labels[o[r, 0]] == Y[k]) for r, k in enumerate(cv)) / len(cv))
            t5.append(100 * sum(int(Y[k] in [labels[o[r, t]] for t in range(5)]) for r, k in enumerate(cv)) / len(cv))
        return t1, t5

    res = {v: exrun(Z[v]) for v in Z}
    ms = lambda v: (round(st.mean(v), 1), round(st.pstdev(v), 1))
    base_t1 = res["base"][0]
    rows = []
    for v in ["base", "region", "neighbor", "both"]:
        t1, t5 = res[v]
        d = [a - b for a, b in zip(t1, base_t1)]
        rows.append((v, ms(t1), ms(t5), round(st.mean(d), 1), sum(x > 0 for x in d)))
    best = max(rows[1:], key=lambda r: r[3]) if len(rows) > 1 else rows[0]
    verdict = ("구조적 맥락이 top1 향상 → full R-GCN 진행" if best[3] > rows[0][1][1]
               else "구조적 맥락도 top1 무효 → 데이터/학습이 레버")
    print("\n== " + " | ".join(f"{v} {ms(res[v][0])[0]}(Δ{rows[k][3]:+})" for k, v in enumerate(['base','region','neighbor','both'])) + f" -> {verdict} ==")

    d = explog.next_dir("relational-context")
    explog.bar(d / "fig_rel.png", [r[0] for r in rows], [r[1][0] for r in rows],
               "Structured neighbour context: top1 (10-seed)", "%", ymax=100, errors=[r[1][1] for r in rows])
    tab = "\n".join(f"| {v} | {t1[0]}±{t1[1]}% | {t5[0]}±{t5[1]}% | {dd:+} ({w}/10) |"
                    for v, t1, t5, dd, w in rows)
    report = f"""# M6' 1단계: 구조화된 이웃-영역 맥락 (relational-context)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/relational_context.py`

## 목적
exp 008(전역 CLS)은 top5만 올렸다. **구조화된** 관계 신호 — 핀 영역의 *인접 영역* 외형 — 이 top1을
올리는지 검증(=M6' 토대). DINO 패치를 K={K} 영역으로 클러스터(공간가중 β={BETA}), 인접 영역을
임베딩에 결합, frozen exemplar 1-NN.

## 결과 (10-seed, paired vs base)
| 변형 | top1 | top5 | Δtop1 |
|---|---|---|---|
{tab}

![rel](fig_rel.png)

## 판정
- 베스트: **{best[0]}** Δtop1 {best[3]:+}%p ({best[4]}/10) → **{verdict}**

## 다음
- top1 향상 시 → **full R-GCN**(scene_graph + RelationalGNN, SupCon 학습)로 관계 추론 본격화.
- 무효 시 → 구조도 데이터 한계를 못 넘음(학습형 풀러·맥락과 일관) → 데이터가 결정적.
"""
    explog.write(d, report, {
        "title": "M6' 구조적 이웃 맥락", "date": datetime.date.today().isoformat(),
        "headline": f"best={best[0]} Δtop1 {best[3]:+}({best[4]}/10) → {verdict}",
        "variants": {v: {"top1": t1, "top5": t5, "dtop1": dd} for v, t1, t5, dd, w in rows}})
    print(f"wrote -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
