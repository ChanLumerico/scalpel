"""Experiment 056 — M-bb0: backbone sweep (minimal setting, training-free).

The largest untried lever: 9 months on a single DINOv2 generation. 019 compared only DINOv2 *sizes*
(+1.1, marginal). This sweeps backbones under the MINIMAL fair setting (σ40 GaussianPool + exemplar
1-NN, no L256/CSLS) and selects on the metric that IS our readout — leave-one-fold k-NN top1 — with
three diagnostics (silhouette, hubness skew, tissue/region centroid sep) to explain *why*.

Backbones: DINOv2 {vitb14 baseline, vitl14, vitg14} now; DINOv3 {vitb16, vitl16} pending HF license
(facebook/dinov3-* is gated=manual — accept at the repo page, then add to BACKBONES). All training-free
re-embedding, clean 502, 10-seed photo-block split. Gate: best vs DINOv2-vitb14 Δ>0 AND ≥7/10.

    .venv/bin/python scripts/backbone_sweep.py vitl14            # one or more variants
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
from eval_merged import load, exemplar_eval, block_split, BASE  # noqa: E402
from split_devtest import get_split  # noqa: E402

SEEDS = 10
SIGMA = 40.0
TIS = {"artery": "artery", "arteries": "artery", "vein": "vein", "veins": "vein", "nerve": "nerve",
       "nerves": "nerve", "cn": "nerve", "muscle": "muscle", "muscles": "muscle", "bone": "bone", "joint": "bone"}

# variant -> (hub_repo, hub_name, image_size, patch)
DINOV2 = {
    "vitb14": ("facebookresearch/dinov2", "dinov2_vitb14", 518, 14),
    "vitl14": ("facebookresearch/dinov2", "dinov2_vitl14", 518, 14),
    "vitg14": ("facebookresearch/dinov2", "dinov2_vitg14", 518, 14),
}


def tissue(lab):
    t = lab.split()
    if "cn" in t:
        return "nerve"
    for w in reversed(t):
        if w in TIS:
            return TIS[w]
    return "other"


def unit(Z):
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)


def ms(v):
    v = [x for x in v if x == x]
    return (round(st.mean(v), 1), round(st.pstdev(v), 1)) if v else (float("nan"), 0.0)


@torch.no_grad()
def embed_dinov2(variant, rows, core, device):
    repo, name, S, patch = DINOV2[variant]
    cache = BASE / f"_bb_{variant}.npy"
    if cache.exists() and np.load(cache, mmap_mode="r").shape[0] == len(rows):
        print(f"  cached {variant}")
        return np.load(cache).astype(np.float32)
    print(f"  loading {name} ...")
    m = torch.hub.load(repo, name); m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    g = S // patch
    ys, xs = np.divmod(np.arange(g * g), g)
    centers = torch.tensor(np.stack([(xs + 0.5) * patch, (ys + 0.5) * patch], 1), dtype=torch.float32, device=device)
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    by = collections.defaultdict(list)
    for i in core:
        by[rows[i]["image"]].append(i)
    Z = [None] * len(rows)
    for n, (img, idxs) in enumerate(by.items(), 1):
        im = Image.open(BASE / img).convert("RGB"); w, h = im.size
        arr = np.asarray(im.resize((S, S)), np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        out = m.forward_features((x - mean) / std)
        tok = out["x_norm_patchtokens"][0]            # (g*g, D)
        for i in idxs:
            qx, qy = rows[i]["q"][0] * S / w, rows[i]["q"][1] * S / h
            d2 = ((centers - torch.tensor([qx, qy], device=device)) ** 2).sum(1)
            wts = torch.softmax(-d2 / (2 * SIGMA ** 2), 0)
            Z[i] = F.normalize((wts[:, None] * tok).sum(0), dim=0).cpu().numpy()
        if n % 200 == 0:
            print(f"   {variant}: {n} imgs")
    A = np.zeros((len(rows), len(Z[core[0]])), np.float32)
    for i in core:
        A[i] = Z[i]
    np.save(cache, A)
    del m
    if device == "mps":
        torch.mps.empty_cache()
    return A


def hubness_skew(Z, core, k=10):
    Zc = Z[core]
    sims = Zc @ Zc.T
    np.fill_diagonal(sims, -9)
    knn = np.argsort(-sims, axis=1)[:, :k]
    occ = np.bincount(knn.reshape(-1), minlength=len(core))
    m = occ.mean(); s = occ.std() + 1e-9
    return float(np.mean(((occ - m) / s) ** 3))         # skewness of k-occurrence


def centroid_sep(Z, Y, core, keyfn):
    groups = collections.defaultdict(list)
    for i in core:
        groups[Y[i]].append(i)
    cents = {c: unit(Z[ix].mean(0, keepdims=True))[0] for c, ix in groups.items() if len(ix) >= 2}
    cs = list(cents)
    same, diff = [], []
    for a in range(len(cs)):
        for b in range(a + 1, len(cs)):
            c = float(cents[cs[a]] @ cents[cs[b]])
            (same if keyfn(cs[a]) == keyfn(cs[b]) else diff).append(c)
    return round((st.mean(same) - st.mean(diff)) if same and diff else 0.0, 3)


def silhouette(Z, Y, core):
    try:
        from sklearn.metrics import silhouette_score
        labs = [Y[i] for i in core]
        return round(float(silhouette_score(Z[core], labs, metric="cosine")), 3)
    except Exception as e:
        print("  silhouette fail:", str(e)[:50]); return float("nan")


def metrics(Z, Y, core, splits, region):
    Z = unit(Z)
    t1 = [exemplar_eval(Z, Y, tr, te)[0] for tr, te in splits]
    return {
        "knn_top1": ms(t1), "knn_seeds": t1,
        "silhouette": silhouette(Z, Y, core),
        "hubness_skew": round(hubness_skew(Z, core), 3),
        "tissue_sep": centroid_sep(Z, Y, core, tissue),
        "region_sep": centroid_sep(Z, Y, core, lambda c: region.get(c, "?")),
    }


def main():
    variants = sys.argv[1:] or ["vitl14"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = load(); Y = [r["label"] for r in rows]
    split = get_split()
    cnt = collections.Counter(Y)
    core = [i for i in range(len(rows)) if cnt[Y[i]] >= 2]
    dev = [i for i in core if split[rows[i]["image"]] == "dev"]
    test = [i for i in core if split[rows[i]["image"]] == "test"]
    block = [json.loads((BASE / "_blocks.json").read_text())[r["image"]] for r in rows]
    cls_reg = {}
    for c in set(Y[i] for i in core):
        rs = [rows[i].get("region", "?") or "?" for i in core if Y[i] == c]
        cls_reg[c] = collections.Counter(rs).most_common(1)[0][0]
    splits = [block_split(dev, block, s) for s in range(SEEDS)]
    print(f"core {len(core)} | dev {len(dev)} | {device} | sweep: vitb14(base) + {variants}")

    # baseline vitb14 from existing σ40 global cache
    res = {}
    zb = np.load(BASE / "_dino_cache.npy").astype(np.float32)
    res["DINOv2-vitb14 (base)"] = metrics(zb, Y, core, splits, cls_reg)
    for v in variants:
        if v not in DINOV2:
            print(f"  skip unknown {v}"); continue
        Z = embed_dinov2(v, rows, core, device)
        res[f"DINOv2-{v}"] = metrics(Z, Y, core, splits, cls_reg)

    base_t1 = res["DINOv2-vitb14 (base)"]["knn_seeds"]

    def paired(a):
        d = [x - y for x, y in zip(a, base_t1)]
        return round(st.mean(d), 2), int(sum(x > 0 for x in d))

    print("\n== M-bb0 backbone sweep (minimal σ40 1-NN, 10-seed) ==")
    print(f"  {'backbone':24} {'k-NN top1':>12} {'Δ':>7} {'wins':>5} {'silh':>6} {'hub':>6} {'tisSep':>7} {'regSep':>7}")
    table = {}
    for k in res:
        r = res[k]; dlt = paired(r["knn_seeds"]) if "base" not in k else (0.0, 0)
        table[k] = dlt
        print(f"  {k:24} {r['knn_top1'][0]:>6}±{r['knn_top1'][1]:<4} {dlt[0]:>+7} {dlt[1]:>4}/10 "
              f"{r['silhouette']:>6} {r['hubness_skew']:>6} {r['tissue_sep']:>7} {r['region_sep']:>7}")

    cand = [k for k in res if "base" not in k]
    best = max(cand, key=lambda k: res[k]["knn_top1"][0]) if cand else None
    adopt = best and table[best][0] > 0 and table[best][1] >= 7
    sealed = ""
    if adopt:
        v = best.split("-")[1]
        Zb = unit(embed_dinov2(v, rows, core, device)); zb_ = unit(zb)
        sb = exemplar_eval(Zb, Y, dev, test)[0]; s0 = exemplar_eval(zb_, Y, dev, test)[0]
        sealed = f" | SEALED base {round(s0,1)} → {best} {round(sb,1)}"
    print(f"\n  {'★ '+best+' ADOPT (→ M-bb1 적층)' if adopt else 'best '+str(best)+' — 게이트 미달'}{sealed}")
    print("  ⚠️ DINOv3 (gated) 미포함 — HF 라이선스 수락 후 BACKBONES에 추가.")

    d = explog.EXP / "056-backbone-sweep"; d.mkdir(parents=True, exist_ok=True)
    ks = list(res)
    explog.bar(d / "fig1_knn.png", [k.replace(" ", "\n") for k in ks], [res[k]["knn_top1"][0] for k in ks],
               "056 backbone sweep: k-NN top1 (minimal σ40 1-NN)", "%", ymax=40,
               errors=[res[k]["knn_top1"][1] for k in ks])
    explog.grouped_bar(d / "fig2_diag.png", [k.replace(" ", "\n") for k in ks],
                       {"silhouette×100": [res[k]["silhouette"] * 100 for k in ks],
                        "tissue_sep×100": [res[k]["tissue_sep"] * 100 for k in ks],
                        "region_sep×100": [res[k]["region_sep"] * 100 for k in ks]},
                       "056 diagnostics (×100): cluster quality & axis separation", "")

    rowmd = "\n".join(
        f"| {k} | {res[k]['knn_top1'][0]}±{res[k]['knn_top1'][1]} | {table[k][0]:+} | {table[k][1]}/10 | "
        f"{res[k]['silhouette']} | {res[k]['hubness_skew']} | {res[k]['tissue_sep']} | {res[k]['region_sep']} |"
        for k in ks)
    bd = table[best][0] if best else 0.0
    verdict = (f"🟢 **{best}** 채택 (Δ{bd:+}, {table[best][1]}/10){sealed} → M-bb1 적층."
               if adopt else
               f"🟡 best {best} 게이트 미달 (Δ{bd:+}). DINOv2 세대내 크기로는 부족(019 재확인) — "
               f"**진짜 검증은 DINOv3** (gated, 라이선스 수락 대기).")
    report = f"""# 056 — M-bb0: 백본 sweep (최소 세팅, 학습 0)

- 날짜: {datetime.date.today().isoformat()} · 커밋 `main @ {_git_sha()}` · `scripts/backbone_sweep.py`
- clean 502 (dev {len(dev)}/test {len(test)} 봉인), 최소 세팅(σ40 GaussianPool + exemplar 1-NN, L256/CSLS 없음), 10-seed.
- 주지표 = k-NN top1(우리 readout 그 자체). 진단 = silhouette/hubness/tissue·region centroid sep.
- ⚠️ **DINOv3 (핵심 후보)는 HF gated=manual → 라이선스 수락 대기** (facebook/dinov3-vitb16/vitl16). 수락 후 BACKBONES에 추가하면 한 줄.

## 결과 (paired Δ vs DINOv2-vitb14)
| 백본 | k-NN top1 | Δ | wins | silhouette | hubness | tissue_sep | region_sep |
|---|---|---|---|---|---|---|---|
{rowmd}

![knn](fig1_knn.png)
![diag](fig2_diag.png)

## 판정
{verdict}

## 핵심
- 019(세대내 크기, +1.1)를 clean 502 + 메트릭 풀세트로 재검증. {'세대내 크기가 작은 이득' if not adopt else '큰 백본 채택'}.
- **진짜 프론티어 = 세대 교체(DINOv3)** — gated라 막힘. 라이선스 수락이 다음 차단해제.
"""
    explog.write(d, report, {
        "title": "백본 sweep (최소 세팅, 학습 0)", "date": datetime.date.today().isoformat(),
        "headline": f"sweep {list(res)} | best={best} Δ{table.get(best,(0,0))[0] if best else 0}{sealed} | "
                    f"DINOv3 gated 대기",
        "adopt": bool(adopt), "best": best,
        "metrics": {k: {mk: mv for mk, mv in res[k].items() if mk != "knn_seeds"} for k in res},
        "delta_vs_base": {k: table[k] for k in res}})
    print(f"\nwrote -> {d}  (2 figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
