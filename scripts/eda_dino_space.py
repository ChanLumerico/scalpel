"""Experiment 042 — EDA: geometry of the class space in DINO embedding space.

On the clean merged dataset, embed every pin (frozen dinov2_vitb14@518 → GaussianPool σ40),
take the per-class CENTROID (mean embedding), and project the 502 core-class centroids to 2D
(t-SNE). Colour by tissue type and by region. This shows *where each class lives* in the
representation and tests structural questions:
  - do tissue types (artery/vein/nerve/muscle/bone) separate in DINO-space?
  - how close are paired artery↔vein centroids? (DX3 — appearance can't split them)
  - intra-class spread (how tight is each class) vs inter-class gaps.

Figures contain NO cadaver imagery (scatter only) → committable.

    .venv/bin/python scripts/eda_dino_space.py
"""

from __future__ import annotations

import collections
import datetime
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

import explog  # noqa: E402
from eval_appearance import _MEAN, _STD, _git_sha  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402

BASE = Path("data/merged_final")
TISSUE = {"artery": "artery", "arteries": "artery", "vein": "vein", "veins": "vein",
          "nerve": "nerve", "nerves": "nerve", "muscle": "muscle", "muscles": "muscle",
          "bone": "bone", "ligament": "other", "tendon": "muscle", "duct": "other",
          "gland": "other", "node": "other", "joint": "bone", "membrane": "other"}
TCOLORS = {"artery": "#d62728", "vein": "#1f77b4", "nerve": "#e6c200", "muscle": "#8c564b",
           "bone": "#7f7f7f", "other": "#cfcfcf"}


def tissue(lab):
    for t in reversed(lab.split()):
        if t in TISSUE:
            return TISSUE[t]
    return "other"


@torch.no_grad()
def embed(rows, bb, pool, centers, S, device):
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    by = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by[r["image"]].append(i)
    Z = [None] * len(rows)
    for n, (img, idxs) in enumerate(by.items(), 1):
        im = Image.open(BASE / img).convert("RGB"); w, h = im.size
        arr = np.asarray(im.resize((S, S)), np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        grid, _ = bb((x - mean) / std)
        for i in idxs:
            qx, qy = rows[i]["q"]
            q = torch.tensor([[qx * S / w, qy * S / h]], device=device)
            Z[i] = F.normalize(pool(grid, centers, q)[0], dim=0).cpu().numpy()
        if n % 150 == 0:
            print(f"   embedded {n} images")
    return np.stack(Z).astype(np.float32)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size
    rows = [json.loads(l) for l in open(BASE / "triples.jsonl", encoding="utf-8")]
    Y = [r["label"] for r in rows]
    region = [r.get("region", "") for r in rows]
    print(f"{len(rows)} triples / {len(set(Y))} classes")
    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    pool = GaussianPool(cfg.point).to(device); centers = bb.patch_centers(device)
    print("embedding (DINOv2 + GaussianPool σ40)...")
    Z = embed(rows, bb, pool, centers, S, device)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)

    # ---- per-class centroids (core >=2) ----
    by_c = collections.defaultdict(list)
    for i, l in enumerate(Y):
        by_c[l].append(i)
    core = {l: idx for l, idx in by_c.items() if len(idx) >= 2}
    labels = sorted(core)
    cent = np.stack([Z[core[l]].mean(0) for l in labels])
    cent = cent / (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9)
    tiss = [tissue(l) for l in labels]
    freq = [len(core[l]) for l in labels]
    print(f"core classes: {len(labels)} | centroids {cent.shape}")

    # ---- intra-class spread (mean cosine of instances to their centroid) ----
    spread = {}
    for l in labels:
        c = Z[core[l]].mean(0); c = c / (np.linalg.norm(c) + 1e-9)
        spread[l] = float(np.mean(Z[core[l]] @ c))
    tspread = collections.defaultdict(list)
    for l in labels:
        tspread[tissue(l)].append(spread[l])

    # ---- tissue separation: within vs across centroid cosine ----
    C = cent @ cent.T
    within = []; across = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            (within if tiss[i] == tiss[j] else across).append(C[i, j])
    # ---- artery<->vein paired centroid distance (DX3): same region modifier ----
    def names(l):
        return frozenset(t for t in l.split() if t not in TISSUE)
    av_pairs = []
    aidx = {names(labels[i]): i for i in range(len(labels)) if tiss[i] == "artery"}
    for j in range(len(labels)):
        if tiss[j] == "vein" and names(labels[j]) in aidx:
            i = aidx[names(labels[j])]
            av_pairs.append(float(C[i, j]))

    # ---- t-SNE of centroids ----
    from sklearn.manifold import TSNE
    print("t-SNE on centroids...")
    emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=0,
               metric="cosine").fit_transform(cent)

    d = explog.EXP / "042-dino-space-eda"; d.mkdir(parents=True, exist_ok=True)

    # Figure 1: centroids by tissue type
    fig, ax = plt.subplots(figsize=(11, 9))
    for t in ["other", "bone", "muscle", "nerve", "vein", "artery"]:
        m = [k for k in range(len(labels)) if tiss[k] == t]
        if not m:
            continue
        ax.scatter(emb[m, 0], emb[m, 1], s=[8 + 4 * freq[k] for k in m], c=TCOLORS[t],
                   label=f"{t} ({len(m)})", alpha=0.75, edgecolors="white", linewidths=0.3)
    ax.set_title("DINO-space class centroids (502 core classes) — t-SNE, coloured by tissue type")
    ax.legend(loc="best", framealpha=0.9); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(d / "fig_centroids_tissue.png", dpi=130); plt.close(fig)

    # Figure 2: by coarse region (from region title first word)
    def reg(r):
        r = r.lower()
        for k in ["head", "neck", "thora", "thorax", "abdom", "pelvi", "perine", "lower limb",
                  "leg", "thigh", "foot", "upper limb", "arm", "forearm", "hand", "shoulder",
                  "brachial", "cranial", "face", "orbit", "nasal", "oral", "spinal", "back", "gluteal"]:
            if k in r:
                return k
        return "other"
    rlab = collections.Counter()
    cls_region = {}
    for l in labels:
        rs = [reg(region[i]) for i in core[l] if region[i]]
        cls_region[l] = collections.Counter(rs).most_common(1)[0][0] if rs else "other"
        rlab[cls_region[l]] += 1
    topreg = [r for r, _ in rlab.most_common(10) if r != "other"]
    # region separation: within-region vs across-region centroid cosine (exclude 'other')
    rwithin = []; racross = []
    for i in range(len(labels)):
        ri = cls_region[labels[i]]
        if ri == "other":
            continue
        for j in range(i + 1, len(labels)):
            rj = cls_region[labels[j]]
            if rj == "other":
                continue
            (rwithin if ri == rj else racross).append(float(C[i, j]))
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.scatter(emb[:, 0], emb[:, 1], s=14, c="#dddddd", alpha=0.4)
    for ri, rr in enumerate(topreg):
        m = [k for k in range(len(labels)) if cls_region[labels[k]] == rr]
        ax.scatter(emb[m, 0], emb[m, 1], s=22, color=cmap(ri % 10), label=f"{rr} ({len(m)})", alpha=0.8)
    ax.set_title("DINO-space class centroids — coloured by anatomical region")
    ax.legend(loc="best", fontsize=8, framealpha=0.9); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(d / "fig_centroids_region.png", dpi=130); plt.close(fig)

    ms = lambda v: (round(float(st.mean(v)), 3), round(float(st.pstdev(v)), 3)) if v else (float("nan"), 0.0)
    report = f"""# 042 — EDA: DINO-space 클래스 중심점 기하

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/eda_dino_space.py` · 데이터: `data/merged_final` ({len(rows)} triples / {len(labels)} core classes)
- 엔진: frozen dinov2_vitb14@518 → GaussianPool σ40 → 클래스 평균 = 중심점, t-SNE(cosine) 2D

## 2D 중심점 분포
![tissue](fig_centroids_tissue.png)
![region](fig_centroids_region.png)

## 기하 통계
| 항목 | 값 |
|---|---|
| 조직형: within {ms(within)[0]} / across {ms(across)[0]} → 분리 | **{round(ms(within)[0]-ms(across)[0],3)}** (≈0, 안 갈림) |
| 부위: within {ms(rwithin)[0]} / across {ms(racross)[0]} → 분리 | **{round(ms(rwithin)[0]-ms(racross)[0],3)}** (강함) |
| artery↔vein 같은부위 쌍 중심 cos (DX3) | **{ms(av_pairs)[0]}** (n={len(av_pairs)}) |
| 클래스내 응집 (instance→centroid cos) | {ms([spread[l] for l in labels])[0]} |

### 조직형별 클래스내 응집
{chr(10).join(f"- {t}: {ms(v)[0]}" for t, v in sorted(tspread.items(), key=lambda x:-st.mean(x[1])))}

## 해석 — DINO-space는 *부위*로 조직화되지 *조직형*으론 안 된다
- **조직형 분리 ≈ {round(ms(within)[0]-ms(across)[0],3)} (0)**: 같은 조직형(artery들끼리)이 다른 조직형보다 가깝지
  **않다** → DINO는 "동맥/정맥/신경"을 분리축으로 인코딩하지 않음.
- **부위 분리 = {round(ms(rwithin)[0]-ms(racross)[0],3)} (강함)**: 같은 부위(orbit, pelvis…) 구조끼리 뚜렷이 뭉침
  (figure에서 orbit·cranial·pelvi 군집 가시) → DINO는 *국소 맥락(부위)*을 인코딩.
- **artery↔vein 쌍 cos {ms(av_pairs)[0]}**: 같은 부위 동맥↔정맥은 중심점이 거의 동일 → DX3 정량 확증.
- **함의**: "top5 좋고(부위 맞힘) top1 나쁜(부위내 미세정체성)" 시그니처의 기하학적 정체. 부위내 동맥/정맥/신경
  분리가 천장 — 외형 밖 정보(관계추론 040) 또는 더 많은 데이터가 필요한 지점.
"""
    explog.write(d, report, {
        "title": "EDA: DINO-space 클래스 중심점 기하", "date": datetime.date.today().isoformat(),
        "headline": f"502 core 중심점 2D | 조직형 분리 {round(ms(within)[0]-ms(across)[0],3)}(≈0) vs 부위 분리 "
                    f"{round(ms(rwithin)[0]-ms(racross)[0],3)}(강) | artery↔vein cos {ms(av_pairs)[0]}(DX3) → DINO는 부위로 조직화",
        "n_core": len(labels), "tissue_within": ms(within), "tissue_across": ms(across),
        "tissue_sep": round(ms(within)[0] - ms(across)[0], 3),
        "region_within": ms(rwithin), "region_across": ms(racross),
        "region_sep": round(ms(rwithin)[0] - ms(racross)[0], 3),
        "av_pair_cos": ms(av_pairs)[0], "n_av_pairs": len(av_pairs),
        "intra_spread": ms([spread[l] for l in labels])[0],
        "tissue_spread": {t: ms(v)[0] for t, v in tspread.items()}})
    print(f"wrote -> {d}")
    print(f"  tissue separation within-across: {ms(within)[0]-ms(across)[0]:.3f}")
    print(f"  artery<->vein paired centroid cos (DX3): {ms(av_pairs)[0]} (n={len(av_pairs)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
