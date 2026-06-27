"""Experiment 055 — test-time augmentation (TTA) of the local crop: multi-view embedding averaging.

A single deterministic crop embedding carries augmentation-sensitivity noise. Averaging embeddings over
a few views (flip, scale) is the classic variance-reduction trick; for retrieval it can sharpen both
gallery and query. We TTA the discriminative LOCAL crop (L256) over {center, h-flip, scale 0.8, 1.25}
and average (renorm), then fuse with the global embedding. dev 10-seed paired vs global+L256 (33.5).

    .venv/bin/python scripts/tta_local.py
"""

from __future__ import annotations

import collections
import datetime
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

import explog  # noqa: E402
from eval_appearance import _MEAN, _STD, _git_sha  # noqa: E402
from eval_merged import load, exemplar_eval, block_split, BASE  # noqa: E402
from split_devtest import get_split  # noqa: E402
from multiscale_local import crop_pad  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402

SEEDS = 10
VIEWS = [("c256", 256, False), ("flip256", 256, True), ("s205", 205, False), ("s320", 320, False)]


def unit(Z):
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)


def ms(v):
    v = [x for x in v if x == x]
    return (round(st.mean(v), 1), round(st.pstdev(v), 1)) if v else (float("nan"), 0.0)


@torch.no_grad()
def embed_tta(rows, core, bb, pool, centers, S, device):
    cache = BASE / "_tta_views.npz"
    if cache.exists():
        d = np.load(cache)
        if d["n"].item() == len(rows):
            print("  cached tta views")
            return {v[0]: d[v[0]] for v in VIEWS}
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    cq = torch.tensor([[259.0, 259.0]], device=device)
    by = collections.defaultdict(list)
    for i in core:
        by[rows[i]["image"]].append(i)
    out = {v[0]: [None] * len(rows) for v in VIEWS}
    for n, (img, idxs) in enumerate(by.items(), 1):
        arr = np.asarray(Image.open(BASE / img).convert("RGB"))
        for i in idxs:
            qx, qy = rows[i]["q"]
            for name, sz, flip in VIEWS:
                c = crop_pad(arr, qx, qy, sz)
                if flip:
                    c = c[:, ::-1]
                c = cv2.resize(np.ascontiguousarray(c), (S, S)).astype(np.float32) / 255.0
                x = torch.from_numpy(c).permute(2, 0, 1).unsqueeze(0).to(device)
                grid, _ = bb((x - mean) / std)
                out[name][i] = F.normalize(pool(grid, centers, cq)[0], dim=0).cpu().numpy()
        if n % 200 == 0:
            print(f"   tta {n} imgs")
    D = {}
    for name, _, _ in VIEWS:
        A = np.zeros((len(rows), len(out[name][core[0]])), np.float32)
        for i in core:
            A[i] = out[name][i]
        D[name] = A
    np.savez(cache, n=np.array(len(rows)), **D)
    return D


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size
    rows = load(); Y = [r["label"] for r in rows]
    split = get_split()
    cnt = collections.Counter(Y)
    core = [i for i in range(len(rows)) if cnt[Y[i]] >= 2]
    dev = [i for i in core if split[rows[i]["image"]] == "dev"]
    test = [i for i in core if split[rows[i]["image"]] == "test"]
    block = [json.loads((BASE / "_blocks.json").read_text())[r["image"]] for r in rows]
    zg = unit(np.load(BASE / "_dino_cache.npy").astype(np.float32))
    zl256 = unit(np.load(BASE / "_local256_cache.npy").astype(np.float32))
    print(f"core {len(core)} | dev {len(dev)} / test {len(test)}")

    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    pool = GaussianPool(cfg.point).to(device); centers = bb.patch_centers(device)
    print("embedding TTA views...")
    V = embed_tta(rows, core, bb, pool, centers, S, device)
    V = {k: unit(v) for k, v in V.items()}

    tta_noflip = unit(V["c256"] + V["s205"] + V["s320"])
    tta_all = unit(V["c256"] + V["flip256"] + V["s205"] + V["s320"])
    Zbase = unit(np.concatenate([zg, zl256], 1))
    variants = {
        "global+L256 (base)": Zbase,
        "global+TTA(noflip)": unit(np.concatenate([zg, tta_noflip], 1)),
        "global+TTA(all)": unit(np.concatenate([zg, tta_all], 1)),
        "global+TTA-local-only": unit(np.concatenate([zg, unit(V["c256"] + V["s205"])], 1)),
    }
    splits = [block_split(dev, block, s) for s in range(SEEDS)]
    res = {k: [exemplar_eval(v, Y, tr, te)[0] for tr, te in splits] for k, v in variants.items()}
    base = res["global+L256 (base)"]

    def paired(a):
        d = [x - y for x, y in zip(a, base)]
        return round(st.mean(d), 2), int(sum(x > 0 for x in d))

    print("\n== dev-CV 10-seed (paired vs global+L256) ==")
    table = {}
    for k in variants:
        dlt = paired(res[k]); table[k] = (ms(res[k]), dlt)
        print(f"  {k:24} top1 {ms(res[k])[0]}±{ms(res[k])[1]}  Δ {dlt[0]:+} ({dlt[1]}/10)")

    cand = [k for k in variants if "base" not in k]
    best = max(cand, key=lambda k: st.mean(res[k]))
    d_best = table[best][1]
    adopt = d_best[0] > 0 and d_best[1] >= 7
    sealed = ""
    if adopt:
        sb = exemplar_eval(variants[best], Y, dev, test)[0]; s0 = exemplar_eval(Zbase, Y, dev, test)[0]
        sealed = f" | SEALED base {round(s0,1)} → {best} {round(sb,1)}"
    print(f"\n  {'★ '+best+' ADOPT' if adopt else 'best '+best+' — no adoption'} (dev Δ{d_best[0]:+}, {d_best[1]}/10){sealed}")

    d = explog.EXP / "055-tta-local"; d.mkdir(parents=True, exist_ok=True)
    ks = list(variants)
    explog.bar(d / "fig1.png", [k.replace(" ", "\n") for k in ks], [table[k][0][0] for k in ks],
               "055 TTA local-crop averaging (dev-CV top1)", "%", ymax=40, errors=[table[k][0][1] for k in ks])

    rowmd = "\n".join(f"| {k} | {table[k][0][0]}±{table[k][0][1]} | {table[k][1][0]:+} | {table[k][1][1]}/10 |" for k in ks)
    verdict = (f"🟢 **{best}** 채택 (dev Δ{d_best[0]:+}, {d_best[1]}/10){sealed}."
               if adopt else
               f"🔴 **TTA 무효** — 다중뷰 평균이 global+L256 못 넘음 (best {best} Δ{d_best[0]:+}). "
               f"단일 크롭 임베딩이 이미 안정적이거나, flip이 laterality를 흐림.")
    report = f"""# 055 — TTA: 로컬 크롭 다중뷰 평균 (분산 감소)

- 날짜: {datetime.date.today().isoformat()} · 커밋 `main @ {_git_sha()}` · `scripts/tta_local.py`
- clean 502 (dev {len(dev)}/test {len(test)} 봉인), dev 10-seed paired. baseline = global+L256.
- 뷰: {{center256, hflip256, scale205, scale320}} 평균. flip은 laterality 흐릴 위험이라 noflip도 별도.

## 결과 (paired Δ vs global+L256)
| 변형 | dev-CV top1 | Δ | wins |
|---|---|---|---|
{rowmd}

![variants](fig1.png)

## 판정
{verdict}
"""
    explog.write(d, report, {
        "title": "TTA 로컬 다중뷰 평균", "date": datetime.date.today().isoformat(),
        "headline": f"best={best} dev Δ{d_best[0]:+}({d_best[1]}/10){' ADOPT'+sealed if adopt else ' — 무효'}",
        "adopt": bool(adopt), "best": best,
        "devcv": {k: {"top1": table[k][0], "delta": table[k][1]} for k in variants}})
    print(f"\nwrote -> {d}  (1 figure)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
