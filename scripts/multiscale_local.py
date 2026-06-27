"""Experiment 045 — Representation axis, gate phase (training-free). M-rep0 + M-rep0b.

043 closed the readout axis; 042/044 showed why — tissue types overlap in frozen-DINO space.
Before any space-reshaping (contrastive/LoRA), the most upstream question (handout §2): is the
fine identity cue even IN THE INPUT? Our pipeline squishes the whole image to 518 + σ40 pools,
so a small pinned structure's fine cue (vessel wall / lumen / colour) may be destroyed at the
resolution step. M-rep0 tests this with NO training — a high-res LOCAL crop around q, embedded
separately and concatenated with the global embedding. If top1 rises → bottleneck = resolution
(first crack). If flat → DINO can't extract the cue (→ colour/LoRA/human-ceiling).

Gates (all training-free, run together):
  M-rep0   multiscale local: z = [z_global ⊕ z_local256 ⊕ z_local512], exemplar 1-NN, paired Δ
  M-rep0b  (a) tissue-oracle ceiling (restrict candidates to same tissue) = hierarchical-gate cap
           (b) colour probe: low-level RGB/HSV/texture AUC for artery-vs-vein (info vs representation)

Protocol (§1.7): clean 502, sealed dev/test, select on dev 10-seed CV, final on sealed test.

    .venv/bin/python scripts/multiscale_local.py
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
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

import explog  # noqa: E402
from eval_appearance import _MEAN, _STD, _git_sha  # noqa: E402
from eval_merged import load, exemplar_eval, block_split, BASE  # noqa: E402
from split_devtest import get_split  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402

SEEDS = 10
SCALES = [256, 512]
TISSUE = {"artery": "artery", "arteries": "artery", "vein": "vein", "veins": "vein",
          "nerve": "nerve", "nerves": "nerve", "cn": "nerve", "muscle": "muscle",
          "muscles": "muscle", "bone": "bone", "joint": "bone", "tendon": "muscle"}


def tissue(lab):
    toks = lab.split()
    if "cn" in toks:
        return "nerve"
    for t in reversed(toks):
        if t in TISSUE:
            return TISSUE[t]
    return "other"


def crop_pad(img, qx, qy, S):
    """SxS crop centred on (qx,qy); q stays at the CENTER, out-of-image padded black."""
    H, W = img.shape[:2]
    half = S // 2
    out = np.zeros((S, S, 3), img.dtype)
    x0, y0 = qx - half, qy - half
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x0 + S), min(H, y0 + S)
    dx0, dy0 = sx0 - x0, sy0 - y0
    out[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = img[sy0:sy1, sx0:sx1]
    return out


@torch.no_grad()
def embed_local(rows, scale, bb, pool, centers, S, device):
    """High-res local embedding: crop SxS around q → resize 518 → DINO → σ40 pool at CENTER."""
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    cache = BASE / f"_local{scale}_cache.npy"
    if cache.exists() and np.load(cache, mmap_mode="r").shape[0] == len(rows):
        print(f"  loaded cached local{scale}")
        return np.load(cache).astype(np.float32)
    by = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by[r["image"]].append(i)
    Z = [None] * len(rows)
    cq = torch.tensor([[S * 0.0 + 259.0, 259.0]], device=device)  # q at crop centre after 518 resize
    for n, (img, idxs) in enumerate(by.items(), 1):
        arr = np.asarray(Image.open(BASE / img).convert("RGB"))
        for i in idxs:
            qx, qy = rows[i]["q"]
            c = crop_pad(arr, qx, qy, scale)
            c = cv2.resize(c, (S, S)).astype(np.float32) / 255.0
            x = torch.from_numpy(c).permute(2, 0, 1).unsqueeze(0).to(device)
            grid, _ = bb((x - mean) / std)
            Z[i] = F.normalize(pool(grid, centers, cq)[0], dim=0).cpu().numpy()
        if n % 150 == 0:
            print(f"   local{scale}: {n} images")
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
    blk = BASE / "_blocks.json"; img_block = json.loads(blk.read_text())
    block = [img_block[r["image"]] for r in rows]
    zg = unit(np.load(BASE / "_dino_cache.npy").astype(np.float32))
    print(f"core {len(core)} | dev {len(dev)} / test {len(test)} | device={device}")

    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    pool = GaussianPool(cfg.point).to(device); centers = bb.patch_centers(device)
    print("embedding high-res local crops...")
    zl = {s: unit(embed_local(rows, s, bb, pool, centers, S, device)) for s in SCALES}

    # ---- variants (each block unit-normed, concat, re-unit → equal-weight cosine fusion) ----
    variants = {
        "global (043 base)": zg,
        "global+L256": unit(np.concatenate([zg, zl[256]], 1)),
        "global+L512": unit(np.concatenate([zg, zl[512]], 1)),
        "global+L256+L512": unit(np.concatenate([zg, zl[256], zl[512]], 1)),
        "local256+512 only": unit(np.concatenate([zl[256], zl[512]], 1)),
    }
    splits = [block_split(dev, block, s) for s in range(SEEDS)]

    def devcv(Z):
        t1, t5, cov = [], [], []
        for tr, te in splits:
            a, b, c = exemplar_eval(Z, Y, tr, te); t1.append(a); t5.append(b); cov.append(c)
        return t1, t5, cov

    res = {}
    base_t1 = None
    for name, Z in variants.items():
        t1, t5, cov = devcv(Z)
        res[name] = {"t1": ms(t1), "t5": ms(t5), "cov": ms(cov), "seed_t1": t1}
        if name.startswith("global ("):
            base_t1 = t1
    # paired Δ vs global base
    for name in res:
        d = [a - b for a, b in zip(res[name]["seed_t1"], base_t1)]
        res[name]["delta"] = round(st.mean(d), 2)
        res[name]["wins"] = sum(x > 0 for x in d)
    print("\n== M-rep0 multiscale local (dev-CV 10-seed, paired Δ vs global) ==")
    for name in variants:
        r = res[name]
        print(f"  {name:20} top1 {r['t1'][0]}±{r['t1'][1]} | Δ {r['delta']:+} ({r['wins']}/10) | cov {r['cov'][0]}")

    # dev-select best (excluding base), sealed-test it
    cands = [n for n in variants if not n.startswith("global (")]
    best = max(cands, key=lambda n: res[n]["t1"][0])
    bt1, _, _ = exemplar_eval(variants[best], Y, dev, test)
    base_test_t1, _, base_test_cov = exemplar_eval(zg, Y, dev, test)
    # bootstrap CI on sealed test for best
    rng = np.random.default_rng(0)
    labset = set(Y[i] for i in dev); cov_q = [q for q in test if Y[q] in labset]
    sims = variants[best][cov_q] @ variants[best][dev].T
    cols = collections.defaultdict(list); labs = sorted(labset); li = {l: j for j, l in enumerate(labs)}
    for j, i in enumerate(dev):
        cols[li[Y[i]]].append(j)
    sc = np.full((len(cov_q), len(labs)), -2.0, np.float32)
    for c, ix in cols.items():
        sc[:, c] = sims[:, ix].max(1)
    pred = sc.argmax(1); corr = np.array([labs[pred[r]] == Y[cov_q[r]] for r in range(len(cov_q))])
    boot = sorted(100 * corr[rng.integers(0, len(corr), len(corr))].mean() for _ in range(2000))
    ci = (round(boot[50], 1), round(boot[1950], 1))
    adopt = res[best]["delta"] > 0 and res[best]["wins"] >= 7
    print(f"\n  ★ SEALED TEST: best dev variant '{best}' top1 {round(bt1,1)} (CI {ci[0]}–{ci[1]}) "
          f"vs global {round(base_test_t1,1)} | dev Δ {res[best]['delta']} ({res[best]['wins']}/10) → "
          f"{'🟢 ADOPT (병목=해상도, 첫 균열)' if adopt else '🔴 평탄 (DINO 추출 불가)'}")

    # ---- M-rep0b(a): tissue-oracle ceiling on global ----
    def tissue_oracle(Z):
        d = []
        for tr, te in splits:
            labset = set(Y[i] for i in tr)
            cov = [q for q in te if Y[q] in labset]
            cols = collections.defaultdict(list)
            for j, i in enumerate(tr):
                cols[Y[i]].append(j)
            sims = Z[cov] @ Z[tr].T
            ok = 0
            for r, q in enumerate(cov):
                tq = tissue(Y[q])
                best_lab, best_s = None, -9
                for lab, ix in cols.items():
                    if tissue(lab) != tq:
                        continue
                    s = sims[r, ix].max()
                    if s > best_s:
                        best_s, best_lab = s, lab
                ok += (best_lab == Y[q])
            d.append(100 * ok / len(cov))
        return d
    to = tissue_oracle(zg)
    base_dev = ms(base_t1)
    tissue_delta = round(st.mean(to) - base_dev[0], 1)
    print(f"\n== M-rep0b(a) tissue-oracle: top1 {ms(to)[0]} vs base {base_dev[0]} → Δ {tissue_delta:+}pp "
          f"({'게이트 가치' if tissue_delta > 2 else 'same-tissue 지배'})")

    # ---- M-rep0b(b): colour/texture probe artery-vs-vein ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    av = [i for i in core if tissue(Y[i]) in ("artery", "vein")]
    feats, ylab = [], []
    cache_img = {}
    for i in av:
        im = cache_img.get(rows[i]["image"])
        if im is None:
            im = np.asarray(Image.open(BASE / rows[i]["image"]).convert("RGB")); cache_img[rows[i]["image"]] = im
        qx, qy = rows[i]["q"]; r = 20; H, W = im.shape[:2]
        patch = im[max(0, qy - r):qy + r, max(0, qx - r):qx + r].reshape(-1, 3).astype(float)
        hsv = cv2.cvtColor(im[max(0, qy - r):qy + r, max(0, qx - r):qx + r], cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(float)
        f = [patch[:, 0].mean(), patch[:, 1].mean(), patch[:, 2].mean(),
             hsv[:, 0].mean(), hsv[:, 1].mean(), hsv[:, 2].mean(),
             patch.std(), hsv[:, 1].std()]
        feats.append(f); ylab.append(1 if tissue(Y[i]) == "artery" else 0)
    feats = np.array(feats); ylab = np.array(ylab)
    auc_color = float(np.mean(cross_val_score(LogisticRegression(max_iter=500), feats, ylab, cv=5, scoring="roc_auc")))
    # DINO global AUC for same artery/vein (does DINO capture it?)
    Xg = zg[av]; auc_dino = float(np.mean(cross_val_score(LogisticRegression(max_iter=500), Xg, ylab, cv=5, scoring="roc_auc")))
    print(f"== M-rep0b(b) artery-vs-vein AUC: low-level colour/texture {round(auc_color,3)} | "
          f"DINO-global {round(auc_dino,3)} (n={len(av)}) ==")

    # ---- per-tissue Δ for the best variant (where does local help?) ----
    def per_tissue_top1(Z):
        allrows = []
        for tr, te in splits:
            labset = set(Y[i] for i in tr); cov = [q for q in te if Y[q] in labset]
            cols = collections.defaultdict(list)
            for j, i in enumerate(tr):
                cols[Y[i]].append(j)
            labs = sorted(labset)
            sims = Z[cov] @ Z[tr].T
            sc = np.full((len(cov), len(labs)), -2.0, np.float32)
            li = {l: j for j, l in enumerate(labs)}
            for c, ix in cols.items():
                sc[:, li[c]] = sims[:, ix].max(1)
            pred = sc.argmax(1)
            for r, q in enumerate(cov):
                allrows.append((tissue(Y[q]), labs[pred[r]] == Y[q]))
        d = collections.defaultdict(lambda: [0, 0])
        for t, ok in allrows:
            d[t][0] += 1; d[t][1] += ok
        return {t: 100 * v[1] / v[0] for t, v in d.items() if v[0] >= 15}
    pt_base = per_tissue_top1(zg); pt_best = per_tissue_top1(variants[best])
    tiss = sorted(set(pt_base) & set(pt_best), key=lambda t: -(pt_best[t] - pt_base[t]))

    # ===== figures =====
    d = explog.EXP / "045-multiscale-local"; d.mkdir(parents=True, exist_ok=True)
    vorder = list(variants)
    explog.grouped_bar(d / "fig1_variants.png", [v.replace(" ", "\n") for v in vorder],
                       {"dev-CV top1": [res[v]["t1"][0] for v in vorder],
                        "dev-CV top5": [res[v]["t5"][0] for v in vorder]},
                       "045 M-rep0 multiscale local: dev-CV (paired vs global)", "%", ymax=70)
    explog.bar(d / "fig2_delta.png", [v.replace(" ", "\n") for v in vorder],
               [res[v]["delta"] for v in vorder],
               "045 paired Δtop1 vs global (each block unit-concat)", "Δ top1 pp")
    explog.grouped_bar(d / "fig3_per_tissue.png", tiss,
                       {"global": [pt_base[t] for t in tiss], best: [pt_best[t] for t in tiss]},
                       f"045 per-tissue top1: global vs {best}", "%", ymax=100)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(["colour/texture\n(info there?)", "DINO-global\n(captured?)"], [auc_color, auc_dino],
           color=["#2ca02c", "#3b7dd8"])
    ax.axhline(0.5, ls=":", color="gray"); ax.set_ylim(0.4, 1.0)
    ax.set_title(f"045 M-rep0b artery-vs-vein separability (n={len(av)})"); ax.set_ylabel("ROC AUC")
    for i, v in enumerate([auc_color, auc_dino]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(d / "fig4_av_auc.png", dpi=120); plt.close(fig)

    var_rows = "\n".join(
        f"| {v} | {res[v]['t1'][0]}±{res[v]['t1'][1]} | {res[v]['t5'][0]} | {res[v]['delta']:+} | {res[v]['wins']}/10 |"
        for v in vorder)
    verdict = ("🟢 **병목=해상도 확정 (첫 균열).** 멀티스케일 채택 → M-rep1(공간)로 가산 가능."
               if adopt else
               "🔴 **평탄 — 미세 단서가 frozen-DINO로 추출 불가.** 공간 재배치(contrastive)도 무의미 추정 → "
               "색 AUC 보고 M-rep3(색주입) 또는 ③LoRA(백본 reshape) 또는 ④인간천장.")
    report = f"""# 045 — 표현 게이트: 멀티스케일 고해상 로컬 (M-rep0) + 조직oracle·색 (M-rep0b)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/multiscale_local.py` · 데이터 `data/merged_final` (clean 502, dev {len(dev)}/test {len(test)} 봉인)
- 학습 0 — 재임베딩만. dev 10-seed CV 선택 + 봉인 test 1회 (§1.7).

## M-rep0 — 멀티스케일 고해상 로컬 (paired Δ vs global)
| variant | dev-CV top1 | top5 | Δtop1 | wins |
|---|---|---|---|---|
{var_rows}

- **dev-선택 best = `{best}` → 봉인 TEST top1 {round(bt1,1)}** (CI {ci[0]}–{ci[1]}) vs global {round(base_test_t1,1)}.
- 채택: {verdict}

![variants](fig1_variants.png)
![delta](fig2_delta.png)
![per-tissue](fig3_per_tissue.png)

## M-rep0b — 두 상한 분해
- **(a) 조직-oracle**: 조직을 완벽히 알면 top1 {ms(to)[0]} (base {base_dev[0]}) → **Δ {tissue_delta:+}pp**
  ({'조직 게이트 가치 있음 → M-rep2' if tissue_delta > 2 else 'same-tissue 혼동 지배 → 조직분리 한계'}).
- **(b) artery↔vein 분리도**: 저수준 색·텍스처 AUC **{round(auc_color,3)}** | DINO-global **{round(auc_dino,3)}** (n={len(av)}).
  - 색 AUC>0.65 & DINO≈0.5 → 정보는 있으나 DINO가 못 봄 → **M-rep3(색주입) 유효**.
  - 색 AUC≈0.5 → 정보 문제(DX3 최악), 정적사진에 단서 없음 → ④인간천장.

![av-auc](fig4_av_auc.png)

## 핵심
- M-rep0(해상도)가 다른 모든 표현법의 전제 — {('통과: 고해상이 미세 ID를 살림.' if adopt else '평탄: 해상도론 안 됨.')}
- 조직-oracle Δ {tissue_delta:+}pp = 계층 게이트(M-rep2)의 상한 (cross-tissue 56%만 공략).
- 색 AUC {round(auc_color,3)} vs DINO {round(auc_dino,3)} = 정보 vs 표현 분해.
"""
    explog.write(d, report, {
        "title": "표현 게이트: 멀티스케일 로컬 + 조직oracle·색", "date": datetime.date.today().isoformat(),
        "headline": f"M-rep0 best={best} dev Δ{res[best]['delta']}({res[best]['wins']}/10) → 봉인test {round(bt1,1)} "
                    f"(CI {ci[0]}–{ci[1]}) {'🟢해상도' if adopt else '🔴평탄'} | 조직oracle Δ{tissue_delta:+} | "
                    f"av-AUC 색{round(auc_color,2)}/DINO{round(auc_dino,2)}",
        "mrep0": {v: {"dev_t1": res[v]["t1"], "delta": res[v]["delta"], "wins": res[v]["wins"]} for v in vorder},
        "best_variant": best, "sealed_test_top1": round(bt1, 1), "sealed_ci": list(ci),
        "global_test_top1": round(base_test_t1, 1), "adopt": adopt,
        "tissue_oracle_delta_pp": tissue_delta, "av_auc_color": round(auc_color, 3), "av_auc_dino": round(auc_dino, 3),
        "per_tissue_delta": {t: round(pt_best[t] - pt_base[t], 1) for t in tiss}})
    print(f"\nwrote -> {d}  (4 figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
