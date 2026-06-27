"""Experiment 034 — Visual prompting: inject q at the BACKBONE INPUT, not the readout.

The last unexplored axis of the model. Every prior method conditions on q at readout
time (Gaussian/SAM pooling of a grid the backbone computed from the CLEAN image). Here
q enters the backbone INPUT: we draw a marker (red dot / ring) at q on the image, run
DINO on the MARKED image, and read out an embedding. This is orthogonal to the whole
pooling plane — the backbone itself "sees" where the pin is.

Readouts on the marked image:
  cls    : global CLS token (location-aware via the marker — canonical visual prompting)
  gpool  : GaussianPool at q on the marked grid (does the marker sharpen local pooling?)
Baseline: GaussianPool on the CLEAN image (the canonical 46.6 model).

Markers: red dot r8 (filled), red ring r18, red ring r30 (hollow → no occlusion),
lime ring r18 (colour variant). exemplar 1-NN, 10-seed, PAIRED vs baseline.
Pre-registered: ADOPT iff the best VP variant has Δtop1 > 0 AND ≥8/10 seeds (strict,
since ~9 variants are compared — Holm-Bonferroni noted in the report).

    .venv/bin/python scripts/visual_prompt.py
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
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

import explog  # noqa: E402
from eval_appearance import load_core, split_indices, _MEAN, _STD, _git_sha  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402

# (name, color RGB or None=clean, radius px @S, thickness (-1 filled))
MARKERS = [
    ("clean", None, 0, 0),
    ("red-dot8", (255, 0, 0), 8, -1),
    ("red-ring18", (255, 0, 0), 18, 3),
    ("red-ring30", (255, 0, 0), 30, 3),
    ("lime-ring18", (0, 255, 0), 18, 3),
]


def _norm(Z):
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)


@torch.no_grad()
def embed_marker(core, cache, bb, pool, centers, S, device, color, radius, thick, batch=16):
    """Return (Zcls, Zgpool) over all triples for one marker style."""
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    Zc = [None] * len(core); Zg = [None] * len(core)
    buf_x, buf_q, buf_i = [], [], []

    def flush():
        if not buf_x:
            return
        x = torch.stack(buf_x).to(device)
        grid, cls = bb((x - mean) / std)
        q = torch.tensor(np.stack(buf_q), dtype=torch.float32, device=device)
        for k, idx in enumerate(buf_i):
            Zc[idx] = F.normalize(cls[k], dim=0).cpu().numpy()
            Zg[idx] = F.normalize(pool(grid[k:k + 1], centers, q[k:k + 1])[0], dim=0).cpu().numpy()
        buf_x.clear(); buf_q.clear(); buf_i.clear()

    for i, r in enumerate(core):
        clean, w, h = cache[r["image"]]
        img = clean.copy()
        qx, qy = r["q"]; sx, sy = int(qx * S / w), int(qy * S / h)
        if color is not None:
            cv2.circle(img, (sx, sy), radius, color, thick, lineType=cv2.LINE_AA)
        buf_x.append(torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1))
        buf_q.append([sx, sy]); buf_i.append(i)
        if len(buf_x) == batch:
            flush()
    flush()
    return _norm(np.stack(Zc).astype(np.float32)), _norm(np.stack(Zg).astype(np.float32))


def exemplar(Z, core, Y):
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
        t1.append(float(100 * sum(int(labels[o[r, 0]] == Y[k]) for r, k in enumerate(cv)) / len(cv)))
        t5.append(float(100 * sum(int(Y[k] in [labels[o[r, t]] for t in range(5)]) for r, k in enumerate(cv)) / len(cv)))
    return t1, t5


def main():
    base = Path("data/triples")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size
    core = load_core("data/triples/triples.jsonl", 2)
    Y = [r["label"] for r in core]
    print(f"core {len(core)}/{len(set(Y))} | device={device}")
    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    pool = GaussianPool(cfg.point).to(device); centers = bb.patch_centers(device)

    # cache clean resized uint8 image per path
    cache = {}
    for r in core:
        if r["image"] not in cache:
            im = Image.open(base / r["image"]).convert("RGB"); w, h = im.size
            cache[r["image"]] = (np.asarray(im.resize((S, S)), np.uint8).copy(), w, h)

    series = {}            # method -> (t1, t5)
    for name, color, radius, thick in MARKERS:
        print(f"  embedding marker={name} ...", flush=True)
        Zc, Zg = embed_marker(core, cache, bb, pool, centers, S, device, color, radius, thick)
        if name == "clean":
            series["base (gpool)"] = exemplar(Zg, core, Y)
            series["clean-cls"] = exemplar(Zc, core, Y)
        else:
            series[f"{name}-cls"] = exemplar(Zc, core, Y)
            series[f"{name}-gpool"] = exemplar(Zg, core, Y)

    ms = lambda v: (float(round(st.mean(v), 1)), float(round(st.pstdev(v), 1)))
    base1 = series["base (gpool)"][0]
    rows = []
    for name, (t1, t5) in series.items():
        d = [a - b for a, b in zip(t1, base1)]
        rows.append((name, ms(t1), ms(t5), float(round(st.mean(d), 1)), int(sum(x > 1e-9 for x in d))))
    cand = [r for r in rows if r[0] != "base (gpool)"]
    best = max(cand, key=lambda r: r[3])
    adopt = bool(best[3] > 0 and best[4] >= 8)
    verdict = ("채택 — visual prompting이 GaussianPool을 능가 (모델 축 미소진!)" if adopt
               else "기각 — backbone 입력 주입도 무효, 모델 축 전체 소진 → 천장은 데이터")

    print("\n  method                 top1        top5     Δtop1 (n/10)")
    for name, t1, t5, dd, w in rows:
        print(f"  {name:22s} {t1[0]:5.1f}±{t1[1]:<4.1f} {t5[0]:5.1f}    {dd:+.1f} ({w}/10)")
    print(f"\n  best VP = {best[0]}  Δtop1 {best[3]:+} ({best[4]}/10)  ==> {verdict}")

    d = explog.next_dir("visual-prompt")
    explog.bar(d / "fig_vp.png", [r[0].replace("-", "\n", 1) for r in rows], [r[1][0] for r in rows],
               "Visual prompting vs GaussianPool: top1 (10-seed)", "%", ymax=100, errors=[r[1][1] for r in rows])
    tab = "\n".join(f"| {name} | {t1[0]}±{t1[1]}% | {t5[0]}% | {dd:+} ({w}/10) |" for name, t1, t5, dd, w in rows)
    report = f"""# 034 — Visual prompting (backbone 입력단 q 주입)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/visual_prompt.py`

## 목적
지금껏 q는 항상 readout 단계(깨끗한 이미지의 grid를 Gaussian/SAM 풀링)에서만 조건화됐다. 여기선 q를
**backbone 입력**에 주입 — 핀 위치에 마커(빨간 점/고리)를 그려 DINO에 통과 → CLS/국소 토큰 readout.
공간 풀링 평면과 직교한 마지막 모델 축. 학습 0, exemplar 1-NN, 10-seed paired vs GaussianPool.

## 결과 (paired vs base gpool)
| 방법 | top1 | top5 | Δtop1 |
|---|---|---|---|
{tab}

![vp](fig_vp.png)

## 판정 (사전 등록: 최선 VP Δ>0 & ≥8/10)
- 최선 VP = **{best[0]}** Δtop1 {best[3]:+}%p ({best[4]}/10) → **{verdict}**
- 다중비교: 비-baseline {len(cand)}개 비교 → Holm-Bonferroni 하에서도 ≥8/10 + 유효크기여야 채택.

## 해석
- 음성이면 → backbone 입력 주입조차 무효 = **모델 축 전체(풀링 안+밖) 소진**, "데이터 한계" 결론 완성.
- 양성이면 → 천장이 사실 *q 조건화 위치* 문제였다는 반전, 9개 phase 해석 재검토.
"""
    explog.write(d, report, {
        "title": "Visual prompting", "date": datetime.date.today().isoformat(),
        "headline": f"best {best[0]} Δtop1 {best[3]:+}({best[4]}/10) → {'채택' if adopt else '기각'}",
        "methods": {name: {"top1": t1, "top5": t5, "dtop1": dd, "wins": w} for name, t1, t5, dd, w in rows},
        "adopt": adopt})
    print(f"wrote -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
