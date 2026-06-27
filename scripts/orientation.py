"""Experiment — local-orientation / texture descriptor at the pin (fibre grain).

DINO's semantic features wash out micro-texture. But anatomical "grain" is
discriminative locally: a vessel wall is smooth, a nerve is striated/cable-like, a
muscle is anisotropic fibre. We compute a hand-built orientation descriptor on a
grayscale patch AT the pin — multi-scale HOG-style orientation histogram (gradient
angle mod 180°, magnitude-weighted) + structure-tensor coherence/anisotropy — and
fuse it with the DINO appearance score:  score = sim_dino + λ · sim_orient.

Both are exemplar class-max sims; λ swept at FIXED values (not tuned on test).
10-seed, paired vs DINO-only (λ=0). Also report orientation-only.

    .venv/bin/python scripts/orientation.py
"""

from __future__ import annotations

import collections
import datetime
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import explog  # noqa: E402
from eval_appearance import load_core, split_indices, embed, _git_sha  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402

LAMBDAS = [0.0, 0.2, 0.4, 0.7, 1.0]
NBINS = 9
RADII = [24, 48]


def _block(gray, cx, cy, r):
    h, w = gray.shape
    x0, x1 = max(0, cx - r), min(w, cx + r)
    y0, y1 = max(0, cy - r), min(h, cy + r)
    p = gray[y0:y1, x0:x1]
    if p.size < 16:
        return np.zeros(NBINS + 3, np.float32)
    gx = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx) % np.pi)                       # 0..π (orientation, not direction)
    bins = np.minimum((ang / np.pi * NBINS).astype(int), NBINS - 1)
    hog = np.zeros(NBINS, np.float32)
    for b in range(NBINS):
        hog[b] = mag[bins == b].sum()
    hog /= (hog.sum() + 1e-6)
    # structure tensor → coherence (anisotropy) + dominant orientation
    Jxx, Jyy, Jxy = (gx * gx).sum(), (gy * gy).sum(), (gx * gy).sum()
    tr = Jxx + Jyy + 1e-6
    coh = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy * Jxy) / tr     # 0..1
    dom = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)               # dominant orientation
    return np.concatenate([hog, [coh, np.cos(2 * dom), np.sin(2 * dom)]]).astype(np.float32)


def orient_descriptors(core, base):
    by = collections.defaultdict(list)
    for i, r in enumerate(core):
        by[r["image"]].append(i)
    Z = [None] * len(core)
    for img, idxs in by.items():
        im = Image.open(base / img).convert("L")
        gray = np.asarray(im, np.float32)
        for i in idxs:
            qx, qy = core[i]["q"]
            desc = np.concatenate([_block(gray, int(qx), int(qy), r) for r in RADII])
            Z[i] = desc
    Z = np.stack(Z).astype(np.float32)
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)


def class_max(sims, cols, C):
    out = np.full((sims.shape[0], C), -2.0, np.float32)
    for c, idx in cols.items():
        out[:, c] = sims[:, idx].max(1)
    return out


def main():
    base = Path("data/triples")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size
    core = load_core("data/triples/triples.jsonl", 2)
    Y = [r["label"] for r in core]
    print(f"core {len(core)}/{len(set(Y))} | device={device}")
    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    pool = GaussianPool(cfg.point).to(device)
    print("DINO embed..."); Zd = embed(core, base, bb, pool, S, device).numpy().astype(np.float32)
    Zd = Zd / (np.linalg.norm(Zd, axis=1, keepdims=True) + 1e-9)
    print("orientation descriptors..."); Zo = orient_descriptors(core, base)
    print(f"orient dim={Zo.shape[1]}")

    out = {lam: ([], []) for lam in LAMBDAS}
    oonly = ([], [])
    for seed in range(10):
        tr, te = split_indices(core, 0.3, seed)
        ytr = [Y[i] for i in tr]
        labels = sorted(set(ytr)); lidx = {l: j for j, l in enumerate(labels)}
        cols = collections.defaultdict(list)
        for j, l in enumerate(ytr):
            cols[lidx[l]].append(j)
        cv = [k for k in te if Y[k] in lidx]
        scd = class_max(Zd[cv] @ Zd[tr].T, cols, len(labels))
        sco = class_max(Zo[cv] @ Zo[tr].T, cols, len(labels))
        for lam in LAMBDAS:
            sc = scd + lam * sco
            o = np.argsort(-sc, axis=1)
            n1 = sum(int(labels[o[r, 0]] == Y[k]) for r, k in enumerate(cv))
            n5 = sum(int(Y[k] in [labels[o[r, t]] for t in range(5)]) for r, k in enumerate(cv))
            out[lam][0].append(100 * n1 / len(cv)); out[lam][1].append(100 * n5 / len(cv))
        o = np.argsort(-sco, axis=1)
        oonly[0].append(100 * sum(int(labels[o[r, 0]] == Y[k]) for r, k in enumerate(cv)) / len(cv))
        oonly[1].append(100 * sum(int(Y[k] in [labels[o[r, t]] for t in range(5)]) for r, k in enumerate(cv)) / len(cv))

    ms = lambda v: (round(st.mean(v), 1), round(st.pstdev(v), 1))
    base1 = out[0.0][0]
    rows = []
    for lam in LAMBDAS:
        t1, t5 = out[lam]
        dd = [a - b for a, b in zip(t1, base1)]
        rows.append((lam, ms(t1), ms(t5), round(st.mean(dd), 1), sum(x > 0 for x in dd)))
    best = max(rows[1:], key=lambda r: r[3])
    verdict = ("방향 기술자가 외형 보완 (paired)" if (best[3] > 0 and best[4] >= 8)
               else "방향 기술자 추가 이득 없음")
    print(f"  orient-only top1 {ms(oonly[0])[0]}±{ms(oonly[0])[1]}")
    for lam, t1, t5, dd, w in rows:
        print(f"  λ={lam}: top1 {t1[0]}±{t1[1]}  top5 {t5[0]}  Δ{dd:+}({w}/10)")
    print(f"\n== best λ={best[0]} Δtop1 {best[3]:+}({best[4]}/10) -> {verdict} ==")

    d = explog.next_dir("orientation")
    explog.bar(d / "fig_orient.png", [f"λ{r[0]}" for r in rows] + ["orient\nonly"],
               [r[1][0] for r in rows] + [ms(oonly[0])[0]],
               "Orientation/texture fusion: top1 (10-seed)", "%", ymax=100,
               errors=[r[1][1] for r in rows] + [ms(oonly[0])[1]])
    tab = "\n".join(f"| {lam} | {t1[0]}±{t1[1]}% | {t5[0]}% | {dd:+} ({w}/10) |" for lam, t1, t5, dd, w in rows)
    report = f"""# 국소 방향/질감 기술자 (orientation)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/orientation.py`

## 목적
DINO 의미특징이 뭉개는 *미세 결(grain)* — 혈관벽(매끈)·신경(케이블 줄무늬)·근육(이방성 섬유) — 을
핀 패치에서 직접 기술. 다중스케일 HOG식 방향 히스토그램 + 구조텐서 coherence/이방성. 외형점수와
`score = sim_dino + λ·sim_orient` 융합(고정 λ). exemplar 1-NN, 10-seed.

## 결과 (paired vs λ=0)
| λ | top1 | top5 | Δtop1 |
|---|---|---|---|
{tab}

- 방향 기술자 단독: top1 {ms(oonly[0])[0]}±{ms(oonly[0])[1]}% (≈무작위 0.47% 대비)

![orient](fig_orient.png)

## 판정
- 베스트 λ={best[0]}: Δtop1 {best[3]:+}%p ({best[4]}/10) → **{verdict}**

## 해석
- 융합이 도우면 → 미세 결은 외형에 *없던* 보완 신호. 무효면 → DINO가 이미 그 질감을 흡수했거나,
  핀 한 점의 국소 결이 같은-부위 판별엔 부족(=데이터 천장 재확인).
"""
    explog.write(d, report, {
        "title": "국소 방향/질감 기술자", "date": datetime.date.today().isoformat(),
        "headline": f"best λ={best[0]} Δtop1 {best[3]:+}({best[4]}/10) → {verdict} | orient-only {ms(oonly[0])[0]}",
        "lambdas": {str(lam): {"top1": t1, "top5": t5, "dtop1": dd} for lam, t1, t5, dd, w in rows},
        "orient_only_top1": ms(oonly[0])})
    print(f"wrote -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
