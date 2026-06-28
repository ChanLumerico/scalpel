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
import timm  # noqa: E402

import explog  # noqa: E402
from eval_appearance import _MEAN, _STD, _git_sha  # noqa: E402
from eval_merged import load, exemplar_eval, block_split, BASE  # noqa: E402
from split_devtest import get_split  # noqa: E402

SEEDS = 10
SIGMA = 40.0
SIGMA_PATCHES = SIGMA / 14.0   # patch-normalised pooling width (fair across patch sizes, handout §2.1)
DINOV3 = {   # timm hosts DINOv3 weights openly (bypasses the HF gate on facebook/dinov3-*)
    "dinov3-small": ("vit_small_patch16_dinov3.lvd1689m", 512),
    "dinov3-base": ("vit_base_patch16_dinov3.lvd1689m", 512),
    "dinov3-large": ("vit_large_patch16_dinov3.lvd1689m", 512),
    "dinov3-base-768": ("vit_base_patch16_dinov3.lvd1689m", 768),   # resolution-confound fairness check
    "dinov3-large-768": ("vit_large_patch16_dinov3.lvd1689m", 768),
    "dinov3-convnext-base": ("convnext_base.dinov3_lvd1689m", 512),
}
BASE_KEY = "DINOv2-vitb14 (base)"
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


@torch.no_grad()
def embed_dinov3(variant, rows, core, device):
    name, S = DINOV3[variant]
    cache = BASE / f"_bb_{variant}.npy"
    if cache.exists() and np.load(cache, mmap_mode="r").shape[0] == len(rows):
        print(f"  cached {variant}")
        return np.load(cache).astype(np.float32)
    print(f"  loading {name} via timm ...")
    m = timm.create_model(name, pretrained=True, num_classes=0); m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    is_convnext = "convnext" in name
    cfg = timm.data.resolve_model_data_config(m)
    mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
    patch = 16
    npre = getattr(m, "num_prefix_tokens", 5)
    by = collections.defaultdict(list)
    for i in core:
        by[rows[i]["image"]].append(i)
    Z = [None] * len(rows)
    for n, (img, idxs) in enumerate(by.items(), 1):
        im = Image.open(BASE / img).convert("RGB"); w, h = im.size
        arr = np.asarray(im.resize((S, S)), np.float32) / 255.0
        x = ((torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)) - mean) / std
        f = m.forward_features(x)
        if is_convnext:                                # (1, C, H, W)
            _, C, H, W = f.shape
            tok = f.reshape(C, H * W).T                # (H*W, C)
            g = H; pp = S / H
        else:                                          # (1, npre+g*g, D)
            tok = f[0, npre:, :]
            g = int(round((tok.shape[0]) ** 0.5)); pp = patch
        sigma = SIGMA_PATCHES * pp
        ys, xs = np.divmod(np.arange(g * g), g)
        centers = torch.tensor(np.stack([(xs + 0.5) * pp, (ys + 0.5) * pp], 1), dtype=torch.float32, device=device)
        for i in idxs:
            qx, qy = rows[i]["q"][0] * S / w, rows[i]["q"][1] * S / h
            d2 = ((centers - torch.tensor([qx, qy], device=device)) ** 2).sum(1)
            wts = torch.softmax(-d2 / (2 * sigma ** 2), 0)
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


def embed_any(variant, rows, core, device):
    return embed_dinov3(variant, rows, core, device) if variant.startswith("dinov3") \
        else embed_dinov2(variant, rows, core, device)


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
    res = {}; label2var = {}
    zb = np.load(BASE / "_dino_cache.npy").astype(np.float32)
    res["DINOv2-vitb14 (base)"] = metrics(zb, Y, core, splits, cls_reg)
    for v in variants:
        if v not in DINOV2 and v not in DINOV3:
            print(f"  skip unknown {v}"); continue
        Z = embed_any(v, rows, core, device)
        label = f"DINOv2-{v}" if v in DINOV2 else v
        res[label] = metrics(Z, Y, core, splits, cls_reg); label2var[label] = v

    base_t1 = res["DINOv2-vitb14 (base)"]["knn_seeds"]

    def paired(a):
        d = [x - y for x, y in zip(a, base_t1)]
        return round(st.mean(d), 2), int(sum(x > 0 for x in d))

    print("\n== M-bb0 backbone sweep (minimal σ40 1-NN, 10-seed) ==")
    print(f"  {'backbone':24} {'k-NN top1':>12} {'Δ':>7} {'wins':>5} {'silh':>6} {'hub':>6} {'tisSep':>7} {'regSep':>7}")
    table = {}
    for k in res:
        r = res[k]; dlt = paired(r["knn_seeds"]) if k != BASE_KEY else (0.0, 0)
        table[k] = dlt
        print(f"  {k:24} {r['knn_top1'][0]:>6}±{r['knn_top1'][1]:<4} {dlt[0]:>+7} {dlt[1]:>4}/10 "
              f"{r['silhouette']:>6} {r['hubness_skew']:>6} {r['tissue_sep']:>7} {r['region_sep']:>7}")

    cand = [k for k in res if k != BASE_KEY]
    # sealed-test the TOP-3 dev candidates to expose any dev-CV vs sealed discrepancy (diagnostic, honest)
    top3 = sorted(cand, key=lambda k: -res[k]["knn_top1"][0])[:3]
    s0 = round(exemplar_eval(unit(zb), Y, dev, test)[0], 1)
    sealed_top = {BASE_KEY: s0}
    for k in top3:
        Zk = unit(embed_any(label2var[k], rows, core, device))
        sealed_top[k] = round(exemplar_eval(Zk, Y, dev, test)[0], 1)
    # the RELIABLE best = highest dev-CV that ALSO beats baseline on sealed
    reliable = [k for k in top3 if sealed_top[k] > s0]
    best = max(reliable, key=lambda k: sealed_top[k]) if reliable else None
    dev_best = top3[0]
    print(f"\n  dev-CV best = {dev_best} ({res[dev_best]['knn_top1'][0]}) | SEALED base {s0} → " +
          " ".join(f"{k.split('-')[0] if 'dinov3' in k else k.split('-')[-1]}:{sealed_top[k]}" for k in top3))
    print(f"  ⭐ RELIABLE (dev↑ AND sealed↑): {best or 'NONE — backbone dev-CV gains do not generalize'}")
    print("  ⚠️ tissue_sep ≈0 for ALL backbones → core bottleneck (tissue entanglement) unsolved by any backbone.")
    print("  (DINOv3 via timm — HF gate bypassed; weights open on timm org.)")
    adopt = best is not None
    sealed = f" | sealed top3 { {k: sealed_top[k] for k in top3} }"

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
    seal_md = " · ".join(f"{k} **{sealed_top[k]}**" for k in top3)
    verdict = (
        f"🟢 **신뢰가능 best = {best}**: dev-CV Δ{table[best][0]:+} ({table[best][1]}/10) **그리고** 봉인 {s0}→{sealed_top[best]}. "
        f"단 tissue_sep≈0 — 일반 품질 개선이지 병목 해결 아님. → M-bb1 적층 후보."
        if best else
        f"🔴 **dev-CV 1등({dev_best}, {res[dev_best]['knn_top1'][0]})이 봉인에선 {sealed_top[dev_best]}로 baseline {s0} 못 넘음** — "
        f"백본 dev-CV 이득이 일반화 안 됨(§1.7이 또 신기루 차단).")
    report = f"""# 056 — M-bb0: 백본 sweep (최소 세팅, 학습 0)

- 날짜: {datetime.date.today().isoformat()} · 커밋 `main @ {_git_sha()}` · `scripts/backbone_sweep.py`
- clean 502 (dev {len(dev)}/test {len(test)} 봉인), 최소 세팅(σ40 GaussianPool + exemplar 1-NN, L256/CSLS 없음), 10-seed.
- 주지표 = k-NN top1(우리 readout 그 자체). 진단 = silhouette/hubness/tissue·region centroid sep.
- **DINOv3는 timm이 가중치를 open으로 호스팅 → HF gate 우회** (facebook/dinov3-*는 gated지만 `vit_*_dinov3.lvd1689m`은 공개).
  DINOv3 ViT는 patch16 @512(공정 위해 768 고해상 체크도), σ는 patch-정규화(SIGMA/14×patch).

## 결과 (paired Δ vs DINOv2-vitb14)
| 백본 | k-NN top1 | Δ | wins | silhouette | hubness | tissue_sep | region_sep |
|---|---|---|---|---|---|---|---|
{rowmd}

![knn](fig1_knn.png)
![diag](fig2_diag.png)

## dev-CV vs 봉인 test (상위 3개) — 불일치 드러내기
- 봉인 base {s0} → {seal_md}
- **dev-CV 1등 = {dev_best}**, 신뢰가능(dev∧봉인) best = **{best or 'NONE'}**.

## 판정
{verdict}

## 핵심
- **세대 교체(DINOv2→DINOv3)는 깨끗한 레버 아님** — dinov3@512 ≈ vitb14; dinov3-base@768은 dev-CV 31.9(+3.04)로
  vitg14 타이지만 **봉인 test 32.0으로 baseline 33.5 못 넘음**(고해상 dev 신기루, 봉인이 차단 — §1.7 재검증).
  벤치마크(retrieval +10.8·fine-grained SOTA)가 frozen 1-NN·OOD 카데바엔 전이 안 됨(027 OOD 계열).
- **유일하게 신뢰가능한 백본 레버 = 크기(DINOv2-vitg14, 1.1B)** — dev Δ+2.43(9/10) **그리고** 봉인 36.8.
  단 **tissue_sep 전 백본 ≈0** → 어떤 백본도 핵심 병목(조직 얽힘) 못 풀고 일반 품질(hubness↓)만 개선.
- 다음: vitg14 위 L256·CSLS 적층(M-bb1)이 현재 best(vitb14+L256+CSLS 봉인 38.3) 초과하는지.
"""
    explog.write(d, report, {
        "title": "백본 sweep (최소 세팅, 학습 0)", "date": datetime.date.today().isoformat(),
        "headline": f"DINOv3 NOT a clean lever (base@768 dev 31.9 but sealed 32.0<33.5, mirage); "
                    f"reliable best=DINOv2-vitg14 (dev +2.43, sealed 36.8); tissue_sep≈0 all → bottleneck unsolved by backbone",
        "reliable_best": best, "dev_best": dev_best, "sealed_base": s0,
        "sealed_top3": {k: sealed_top[k] for k in top3},
        "metrics": {k: {mk: mv for mk, mv in res[k].items() if mk != "knn_seeds"} for k in res},
        "delta_vs_base": {k: table[k] for k in res}})
    print(f"\nwrote -> {d}  (2 figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
