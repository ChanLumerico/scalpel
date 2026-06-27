"""Experiment 048 — M-rep0 refinement: make the SINGLE local crop better (extend the one win).

045 adopted global+L256 (+2.6 sealed) but found stacking MORE scales dilutes (global+L256+L512 32.3
< global+L256 33.5). So the lever isn't "more scales" — it's a better single local representation.
Four cheap probes on top of the 045 best:
  (a) α-weighted fusion  z = unit([z_global ; α·z_local])   — trust local more/less (FREE, cached)
  (b) L128 tighter zoom  — finer detail (vessel wall) at a smaller crop
  (c) fraction crop      — crop 0.25·min(H,W) not fixed px → consistent zoom across 39–2995 px images
  (d) local CLS pooling  — summarise the whole zoomed crop vs σ40 at its centre

Protocol §1.7: clean 502, dev 10-seed CV select, sealed test once. Baseline = global+L256 (045, 33.5).

    .venv/bin/python scripts/multiscale_refine.py
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
from multiscale_local import crop_pad  # noqa: E402  reuse the q-centred padded crop
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402

SEEDS = 10


@torch.no_grad()
def embed_local(rows, bb, pool, centers, S, device, tag, crop_size=None, frac=None, mode="center"):
    """Local embedding with a configurable crop + pooling. Cached by tag.
    crop_size: fixed px box; frac: box = frac*min(H,W); mode: 'center' σ40 | 'cls' token."""
    cache = BASE / f"_loc_{tag}.npy"
    if cache.exists() and np.load(cache, mmap_mode="r").shape[0] == len(rows):
        print(f"  cached {tag}")
        return np.load(cache).astype(np.float32)
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    cq = torch.tensor([[259.0, 259.0]], device=device)
    by = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by[r["image"]].append(i)
    Z = [None] * len(rows)
    for n, (img, idxs) in enumerate(by.items(), 1):
        arr = np.asarray(Image.open(BASE / img).convert("RGB"))
        H, W = arr.shape[:2]
        box = crop_size if crop_size else max(24, int(frac * min(H, W)))
        for i in idxs:
            qx, qy = rows[i]["q"]
            c = cv2.resize(crop_pad(arr, qx, qy, box), (S, S)).astype(np.float32) / 255.0
            x = torch.from_numpy(c).permute(2, 0, 1).unsqueeze(0).to(device)
            grid, cls = bb((x - mean) / std)
            if mode == "cls":
                z = cls[0] if cls is not None else grid.reshape(-1, grid.shape[-1]).mean(0)
            else:
                z = pool(grid, centers, cq)[0]
            Z[i] = F.normalize(z, dim=0).cpu().numpy()
        if n % 200 == 0:
            print(f"   {tag}: {n} imgs")
    Z = np.stack(Z).astype(np.float32)
    np.save(cache, Z)
    return Z


def unit(Z):
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)


def ms(v):
    v = [x for x in v if x == x]
    return (round(st.mean(v), 1), round(st.pstdev(v), 1)) if v else (float("nan"), 0.0)


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
    print(f"core {len(core)} | dev {len(dev)} / test {len(test)} | {device}")

    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    pool = GaussianPool(cfg.point).to(device); centers = bb.patch_centers(device)
    print("re-embedding local variants (cached)...")
    zl128 = unit(embed_local(rows, bb, pool, centers, S, device, "c128", crop_size=128))
    zlfrac = unit(embed_local(rows, bb, pool, centers, S, device, "frac25", frac=0.25))
    zl256cls = unit(embed_local(rows, bb, pool, centers, S, device, "c256cls", crop_size=256, mode="cls"))

    splits = [block_split(dev, block, s) for s in range(SEEDS)]

    def devcv(Z):
        t1, t5, cov = zip(*[exemplar_eval(Z, Y, tr, te) for tr, te in splits])
        return list(t1), list(t5), list(cov)

    def cat(*blocks, weights=None):
        ws = weights or [1.0] * len(blocks)
        return unit(np.concatenate([w * b for w, b in zip(ws, blocks)], 1))

    # baselines
    base_g = devcv(zg)[0]
    base_l256 = devcv(cat(zg, zl256))[0]            # 045 best ~33.5

    # (a) α-weighted fusion on global+L256 (FREE)
    alphas = [0.5, 0.7, 1.0, 1.4, 2.0, 3.0]
    alpha_res = {a: devcv(cat(zg, zl256, weights=[1.0, a]))[0] for a in alphas}
    best_a = max(alphas, key=lambda a: st.mean(alpha_res[a]))

    # (b/c/d) single-local variants (each fused with global, equal weight)
    variants = {
        "global+L256 (045)": cat(zg, zl256),
        "global+L128": cat(zg, zl128),
        "global+Lfrac25": cat(zg, zlfrac),
        "global+L256-CLS": cat(zg, zl256cls),
        f"global+α·L256 (α={best_a})": cat(zg, zl256, weights=[1.0, best_a]),
        "global+L128+L256": cat(zg, zl128, zl256),
    }
    res = {k: devcv(v) for k, v in variants.items()}

    def paired(a, b):
        d = [x - y for x, y in zip(a, b)]
        return round(st.mean(d), 2), int(sum(x > 0 for x in d))

    print(f"\n== dev-CV 10-seed (baseline global {ms(base_g)[0]}, global+L256 {ms(base_l256)[0]}) ==")
    print(f"  α-sweep on global+L256: " + " ".join(f"{a}:{round(st.mean(alpha_res[a]),1)}" for a in alphas)
          + f"  → best α={best_a}")
    rowtab = {}
    for k in variants:
        t1 = res[k][0]; d = paired(t1, base_l256)
        rowtab[k] = (ms(t1), ms(res[k][1]), d)
        print(f"  {k:28} top1 {ms(t1)[0]}±{ms(t1)[1]}  Δ vs L256 {d[0]:+} ({d[1]}/10)")

    # dev-select best variant (by mean top1), sealed test
    best = max(variants, key=lambda k: st.mean(res[k][0]))
    bt1, bt5, bcov = exemplar_eval(variants[best], Y, dev, test)
    g_test = exemplar_eval(zg, Y, dev, test)[0]
    l256_test = exemplar_eval(cat(zg, zl256), Y, dev, test)[0]
    # bootstrap CI
    labset = set(Y[i] for i in dev); cov_q = [q for q in test if Y[q] in labset]
    Zb = variants[best]; sims = Zb[cov_q] @ Zb[dev].T
    cols = collections.defaultdict(list); labs = sorted(labset); li = {l: j for j, l in enumerate(labs)}
    for j, i in enumerate(dev):
        cols[li[Y[i]]].append(j)
    sc = np.full((len(cov_q), len(labs)), -2.0, np.float32)
    for c, ix in cols.items():
        sc[:, c] = sims[:, ix].max(1)
    corr = np.array([labs[sc[r].argmax()] == Y[cov_q[r]] for r in range(len(cov_q))])
    rng = np.random.default_rng(0)
    boot = sorted(100 * corr[rng.integers(0, len(corr), len(corr))].mean() for _ in range(2000))
    ci = (round(boot[50], 1), round(boot[1950], 1))
    d_best = paired(res[best][0], base_l256)
    improved = d_best[0] > 0 and d_best[1] >= 7
    print(f"\n  ★ SEALED TEST: global {round(g_test,1)} → +L256 {round(l256_test,1)} → "
          f"best '{best}' {round(bt1,1)} (CI {ci[0]}–{ci[1]})  | dev Δ vs L256 {d_best[0]:+} ({d_best[1]}/10) → "
          f"{'🟢 추가 가산' if improved else '🟡 L256가 이미 최선 (추가 가산 없음)'}")

    # ===== figures =====
    d = explog.EXP / "048-multiscale-refine"; d.mkdir(parents=True, exist_ok=True)
    vk = list(variants)
    explog.bar(d / "fig1_variants.png", [k.replace(" ", "\n") for k in vk],
               [rowtab[k][0][0] for k in vk], "048 local-representation refinements (dev-CV top1)", "%",
               ymax=42, errors=[rowtab[k][0][1] for k in vk])
    explog.lineplot(d / "fig2_alpha.png",
                    [("global+α·L256", alphas, [round(st.mean(alpha_res[a]), 2) for a in alphas])],
                    "048 local-weight α sweep (dev-CV top1)", "α (local block weight)", "top1 %",
                    hline=(round(st.mean(base_l256), 2), "α=1 (045)"))
    explog.bar(d / "fig3_delta.png", [k.replace(" ", "\n") for k in vk],
               [rowtab[k][2][0] for k in vk], "048 paired Δtop1 vs global+L256 (045)", "Δ top1 pp")

    rowmd = "\n".join(f"| {k} | {rowtab[k][0][0]}±{rowtab[k][0][1]} | {rowtab[k][1][0]} | {rowtab[k][2][0]:+} | {rowtab[k][2][1]}/10 |"
                      for k in vk)
    verdict = (f"🟢 **{best}** 가 045 global+L256를 추가로 이김 (dev Δ{d_best[0]:+}, {d_best[1]}/10) → 채택."
               if improved else
               "🟡 **추가 가산 없음** — global+L256(045)가 이미 단일-로컬 최선. 해상도 레버는 045에서 포화. "
               "다음은 학습형 표현(M-rep1) 또는 데이터.")
    report = f"""# 048 — M-rep0 정제: 단일 로컬 표현 개선 (해상도 레버 확장)

- 날짜: {datetime.date.today().isoformat()} · 커밋 `main @ {_git_sha()}` · `scripts/multiscale_refine.py`
- clean 502 (dev {len(dev)}/test {len(test)} 봉인), dev 10-seed CV 선택 + 봉인 test 1회 (§1.7).
- 045 교훈: 스케일 쌓으면 희석(L256+L512<L256) → 단일 로컬을 더 잘. baseline = global+L256 (045).

## 결과 (paired Δ vs global+L256)
| variant | dev-CV top1 | top5 | Δ vs L256 | wins |
|---|---|---|---|---|
{rowmd}

- α-sweep(local 가중): {' '.join(f'{a}:{round(st.mean(alpha_res[a]),1)}' for a in alphas)} → best α={best_a}.
- **봉인 TEST: global {round(g_test,1)} → +L256 {round(l256_test,1)} → best({best}) {round(bt1,1)}** (CI {ci[0]}–{ci[1]}).
- 판정: {verdict}

![variants](fig1_variants.png)
![alpha](fig2_alpha.png)
![delta](fig3_delta.png)

## 핵심
- 더 타이트(L128)·fraction·CLS·α-가중 중 {('하나가 추가 이득' if improved else '어느 것도 L256를 못 넘음')}.
- 해상도 레버 상태: {('아직 열림 — 추가 정제 여지.' if improved else '045에서 포화 — 단일 고해상 로컬이 천장. 다음 축은 학습형 표현/데이터.')}
"""
    explog.write(d, report, {
        "title": "M-rep0 정제: 단일 로컬 표현 개선", "date": datetime.date.today().isoformat(),
        "headline": f"best={best} dev {ms(res[best][0])[0]} (Δ vs L256 {d_best[0]:+}, {d_best[1]}/10) | "
                    f"봉인 global {round(g_test,1)}→L256 {round(l256_test,1)}→best {round(bt1,1)} (CI {ci[0]}-{ci[1]}) "
                    f"{'🟢가산' if improved else '🟡포화'} | best α={best_a}",
        "best_variant": best, "improved": bool(improved), "best_alpha": best_a,
        "devcv": {k: {"top1": rowtab[k][0], "delta_vs_L256": rowtab[k][2]} for k in vk},
        "alpha_sweep": {str(a): round(st.mean(alpha_res[a]), 2) for a in alphas},
        "sealed": {"global": round(g_test, 1), "global_L256": round(l256_test, 1),
                   "best": round(bt1, 1), "ci": list(ci)}})
    print(f"\nwrote -> {d}  (3 figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
