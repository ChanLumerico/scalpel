"""Experiment 053 — multi-layer DINO feature fusion at q (finer texture from shallower blocks).

The pooled embedding uses only the LAST block's patch tokens — semantic/region-level. Shallower
blocks carry finer texture and edge structure (vessel wall, fascicle pattern) that the last layer
abstracts away. Hypothesis: pooling several blocks at q and concatenating injects the fine cue that
distinguishes artery/vein and same-region look-alikes — a representation-level probe (unlike the
geometry tricks 051/052), the kind of thing that could touch the intrinsic limit rather than re-rank.

DINOv2 exposes intermediate layers in one forward. We pool σ40 at q for blocks {3,6,9,11}, build
concat variants, and compare (plain cosine) to global+L256 (33.5). Sealed test only if adopted.

    .venv/bin/python scripts/multilayer_fusion.py
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
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

import explog  # noqa: E402
from eval_appearance import _MEAN, _STD, _git_sha  # noqa: E402
from eval_merged import load, block_split, BASE  # noqa: E402
from split_devtest import get_split  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402

SEEDS = 10
LAYERS = [3, 6, 9, 11]


def unit(Z):
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)


def ms(v):
    v = [x for x in v if x == x]
    return (round(st.mean(v), 1), round(st.pstdev(v), 1)) if v else (float("nan"), 0.0)


@torch.no_grad()
def embed_layers(rows, core, bb, pool, centers, S, g, device):
    cache = BASE / "_mlayer_cache.npz"
    if cache.exists():
        d = np.load(cache)
        if d["idx"].shape[0] == len(rows):
            print("  cached multilayer")
            return {int(L): d[f"L{L}"] for L in LAYERS}
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    by = collections.defaultdict(list)
    for i in core:
        by[rows[i]["image"]].append(i)
    Z = {L: [None] * len(rows) for L in LAYERS}
    for n, (img, idxs) in enumerate(by.items(), 1):
        im = Image.open(BASE / img).convert("RGB"); w, h = im.size
        arr = np.asarray(im.resize((S, S)), np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        outs = bb._model.get_intermediate_layers((x - mean) / std, n=LAYERS, reshape=False, norm=True)
        grids = {L: outs[k].reshape(1, g, g, -1) for k, L in enumerate(LAYERS)}
        for i in idxs:
            qx, qy = rows[i]["q"]
            q = torch.tensor([[qx * S / w, qy * S / h]], device=device)
            for L in LAYERS:
                Z[L][i] = F.normalize(pool(grids[L], centers, q)[0], dim=0).cpu().numpy()
        if n % 200 == 0:
            print(f"   multilayer {n} imgs")
    out = {}
    for L in LAYERS:
        A = np.zeros((len(rows), len(Z[L][core[0]])), np.float32)
        for i in core:
            A[i] = Z[L][i]
        out[L] = A
    np.savez(cache, idx=np.arange(len(rows)), **{f"L{L}": out[L] for L in LAYERS})
    return out


def top1(Zt, Y, tr, te):
    labset = set(Y[i] for i in tr); cov = [q for q in te if Y[q] in labset]
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


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size; g = cfg.backbone.grid_size
    rows = load(); Y = [r["label"] for r in rows]
    split = get_split()
    cnt = collections.Counter(Y)
    core = [i for i in range(len(rows)) if cnt[Y[i]] >= 2]
    dev = [i for i in core if split[rows[i]["image"]] == "dev"]
    test = [i for i in core if split[rows[i]["image"]] == "test"]
    block = [json.loads((BASE / "_blocks.json").read_text())[r["image"]] for r in rows]
    zg = unit(np.load(BASE / "_dino_cache.npy").astype(np.float32))
    zl = unit(np.load(BASE / "_local256_cache.npy").astype(np.float32))
    print(f"core {len(core)} | dev {len(dev)} / test {len(test)}")

    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    pool = GaussianPool(cfg.point).to(device); centers = bb.patch_centers(device)
    print("embedding multilayer (blocks 3/6/9/11)...")
    ML = embed_layers(rows, core, bb, pool, centers, S, g, device)
    L = {k: unit(v) for k, v in ML.items()}

    Zbase = unit(np.concatenate([zg, zl], 1))   # global+L256 baseline
    variants = {
        "global+L256 (base)": Zbase,
        "L11 only": L[11],
        "L9+L11": unit(np.concatenate([L[9], L[11]], 1)),
        "L6+L9+L11": unit(np.concatenate([L[6], L[9], L[11]], 1)),
        "L3+L6+L9+L11": unit(np.concatenate([L[3], L[6], L[9], L[11]], 1)),
        "multilayer+L256": unit(np.concatenate([L[6], L[9], L[11], zl], 1)),
        "base+L6 (texture)": unit(np.concatenate([zg, zl, L[6]], 1)),
    }
    splits = [block_split(dev, block, s) for s in range(SEEDS)]
    res = {k: [top1(v, Y, tr, te) for tr, te in splits] for k, v in variants.items()}
    base = res["global+L256 (base)"]

    def paired(a):
        d = [x - y for x, y in zip(a, base)]
        return round(st.mean(d), 2), int(sum(x > 0 for x in d))

    print("\n== dev-CV 10-seed (paired vs global+L256) ==")
    table = {}
    for k in variants:
        dlt = paired(res[k]); table[k] = (ms(res[k]), dlt)
        print(f"  {k:22} top1 {ms(res[k])[0]}±{ms(res[k])[1]}  Δ {dlt[0]:+} ({dlt[1]}/10)")

    cand = [k for k in variants if k != "global+L256 (base)"]
    best = max(cand, key=lambda k: st.mean(res[k]))
    d_best = table[best][1]
    adopt = d_best[0] > 0 and d_best[1] >= 7
    sealed = ""
    if adopt:
        sb = top1(variants[best], Y, dev, test); s0 = top1(Zbase, Y, dev, test)
        sealed = f" | SEALED base {round(s0,1)} → {best} {round(sb,1)}"
    print(f"\n  {'★ '+best+' ADOPT' if adopt else 'best '+best+' — no adoption'} (dev Δ{d_best[0]:+}, {d_best[1]}/10){sealed}")

    d = explog.EXP / "053-multilayer-fusion"; d.mkdir(parents=True, exist_ok=True)
    ks = list(variants)
    explog.bar(d / "fig1.png", [k.replace(" ", "\n") for k in ks], [table[k][0][0] for k in ks],
               "053 multi-layer DINO fusion (dev-CV top1)", "%", ymax=40, errors=[table[k][0][1] for k in ks])
    explog.bar(d / "fig2_delta.png", [k.replace(" ", "\n") for k in ks if "base)" not in k],
               [table[k][1][0] for k in ks if "base)" not in k], "053 paired Δ vs global+L256", "Δ pp")

    rowmd = "\n".join(f"| {k} | {table[k][0][0]}±{table[k][0][1]} | {table[k][1][0]:+} | {table[k][1][1]}/10 |" for k in ks)
    verdict = (f"🟢 **{best}** 채택 (dev Δ{d_best[0]:+}, {d_best[1]}/10){sealed} — 얕은층 텍스처가 미세단서 보강."
               if adopt else
               f"🔴 **멀티레이어 무효** — 얕은/중간 블록을 더해도 global+L256 못 넘음 (best {best} Δ{d_best[0]:+}). "
               f"마지막층이 이미 q에서 필요한 정보를 담고 있고, 얕은층 텍스처는 부위내 정체성에 신호 안 됨.")
    report = f"""# 053 — 멀티레이어 DINO 융합 (블록 3/6/9/11을 q에서 풀링·concat)

- 날짜: {datetime.date.today().isoformat()} · 커밋 `main @ {_git_sha()}` · `scripts/multilayer_fusion.py`
- clean 502 (dev {len(dev)}/test {len(test)} 봉인), dev 10-seed paired. baseline = global+L256.
- 동기: 얕은/중간 블록 = 미세 텍스처(혈관벽·결), 마지막 = 의미(부위). 여러 층 결합이 미세단서 보강?

## 결과 (paired Δ vs global+L256)
| 변형 | dev-CV top1 | Δ | wins |
|---|---|---|---|
{rowmd}

![variants](fig1.png)
![delta](fig2_delta.png)

## 판정
{verdict}
"""
    explog.write(d, report, {
        "title": "멀티레이어 DINO 융합", "date": datetime.date.today().isoformat(),
        "headline": f"best={best} dev Δ{d_best[0]:+}({d_best[1]}/10){' ADOPT'+sealed if adopt else ' — 무효'}",
        "adopt": bool(adopt), "best": best,
        "devcv": {k: {"top1": table[k][0], "delta": table[k][1]} for k in variants}})
    print(f"\nwrote -> {d}  (2 figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
