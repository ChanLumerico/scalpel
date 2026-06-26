"""Experiment — data-scaling (learning curve): is DATA the binding constraint?

Model tweaks (exemplar +8, learned head +2.6) gave only modest top1 gains, hinting
the limit is data scale (953 triples, ~1.9 cadavers/class, 33% singletons). Decide
it directly: subsample the gallery to 25/50/75/100% of its specimens and plot top1
(on covered) and coverage vs gallery size. If the curve is still rising at 100%,
more data is the lever; if it plateaus, structure/model is.

Reuses frozen exemplar retrieval; test set fixed per seed; 10-seed mean±std.

    .venv/bin/python scripts/learning_curve.py
"""

from __future__ import annotations

import collections
import datetime
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import explog  # noqa: E402
from eval_appearance import load_core, split_indices, embed, _git_sha  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402

FRACS = [0.25, 0.5, 0.75, 1.0]


def exemplar_eval(Ztr, ytr, Zte, yte):
    labels = sorted(set(ytr)); lidx = {l: i for i, l in enumerate(labels)}
    cols = collections.defaultdict(list)
    for j, l in enumerate(ytr):
        cols[lidx[l]].append(j)
    cov = [k for k, l in enumerate(yte) if l in lidx]
    if not cov:
        return 0.0, 0.0
    sims = Zte[cov] @ Ztr.T
    sc = np.full((len(cov), len(labels)), -2.0, np.float32)
    for c, idx in cols.items():
        sc[:, c] = sims[:, idx].max(1)
    pred = sc.argmax(1)
    n1 = sum(int(labels[pred[r]] == yte[k]) for r, k in enumerate(cov))
    return 100 * n1 / len(cov), 100 * len(cov) / len(yte)


def main():
    base = Path("data/triples")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size
    core = load_core("data/triples/triples.jsonl", 2)
    ncls = len({r["label"] for r in core})
    print(f"core {len(core)}/{ncls} | device={device}")
    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    pool = GaussianPool(cfg.point).to(device); centers = bb.patch_centers(device)
    print("embedding once..."); Z = embed(core, base, bb, pool, S, device).numpy().astype(np.float32)

    top1 = {f: [] for f in FRACS}
    cov = {f: [] for f in FRACS}
    ntri = {f: [] for f in FRACS}
    for seed in range(10):
        tr, te = split_indices(core, 0.3, seed)
        yte = [core[i]["label"] for i in te]
        spec = collections.defaultdict(list)
        for i in tr:
            spec[f'{core[i]["src"]}#{core[i]["page"]}'].append(i)
        specs = sorted(spec)
        rng = np.random.default_rng(seed)
        rng.shuffle(specs)
        for f in FRACS:
            keep = specs[:max(1, math.ceil(f * len(specs)))]
            gi = [i for s in keep for i in spec[s]]
            ytr = [core[i]["label"] for i in gi]
            a, c = exemplar_eval(Z[gi], ytr, Z[te], yte)
            top1[f].append(a); cov[f].append(c); ntri[f].append(len(gi))
    ms = lambda v: (round(st.mean(v), 1), round(st.pstdev(v), 1))
    xs = [round(st.mean(ntri[f])) for f in FRACS]
    t1m = [ms(top1[f]) for f in FRACS]
    cvm = [ms(cov[f]) for f in FRACS]
    slope = round(t1m[-1][0] - t1m[-2][0], 1)        # top1 gain over last 25% data
    rising = slope >= 1.0
    verdict = ("데이터가 레버 — 100%에서도 top1 상승 중 (더 모으면 오름)" if rising
               else "포화 — 데이터만으론 top1 한계 (구조/모델 필요)")
    for f, x, a, c in zip(FRACS, xs, t1m, cvm):
        print(f"  gallery {int(f*100):3d}% (~{x} tri): top1 {a[0]}±{a[1]}  coverage {c[0]}±{c[1]}")
    print(f"  last-25% slope Δtop1={slope} -> {verdict}")

    d = explog.next_dir("learning-curve")
    explog.lineplot(d / "fig_curve.png",
                    [("top1 (covered)", [int(f * 100) for f in FRACS], [a[0] for a in t1m], [a[1] for a in t1m]),
                     ("coverage", [int(f * 100) for f in FRACS], [c[0] for c in cvm], [c[1] for c in cvm])],
                    "Data-scaling: accuracy & coverage vs gallery size",
                    "gallery size (% of specimens)", "%", xlim=(20, 105), ylim=(0, 100))
    table = "\n".join(f"| {int(f*100)}% | ~{x} | {a[0]}±{a[1]}% | {c[0]}±{c[1]}% |"
                      for f, x, a, c in zip(FRACS, xs, t1m, cvm))
    report = f"""# 데이터 스케일링 곡선 (learning-curve)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/learning_curve.py`  (frozen exemplar, 10-seed)

## 목적
top1의 병목이 **데이터 규모**인지 직접 판정. 갤러리를 표본 단위로 25→100%로 줄여가며 top1(covered)
과 coverage를 측정. 100%에서도 상승 중이면 더 모을수록 오른다 = 데이터가 레버.

## 결과
| 갤러리 | ~트리플 | top1(covered) | coverage |
|---|---|---|---|
{table}

![curve](fig_curve.png)

## 판정
- 마지막 25%(75→100%) 구간 Δtop1 = {'+' if slope>=0 else ''}{slope}%p → **{verdict}**
- coverage는 갤러리와 함께 단조 증가 → 새 구조물 커버는 *명백히* 데이터에 비례.

## 해석 / 다음
- top1 곡선이 100%에서도 오르면 → **데이터 확장이 최우선 레버**(coverage + 정확도 동시 개선).
  모델 레버(학습형 풀러/M6')는 그 위에서 추가 이득.
- 평탄하면 → 데이터만으론 부족, **구조적 관계(M6')** 가 천장 돌파의 핵심.
"""
    explog.write(d, report, {
        "title": "데이터 스케일링 곡선", "date": datetime.date.today().isoformat(),
        "headline": f"top1 {t1m[0][0]}→{t1m[-1][0]}% as gallery 25→100%; last-25% Δ{slope} → {verdict}",
        "top1": {int(f*100): a for f, a in zip(FRACS, t1m)},
        "coverage": {int(f*100): c for f, c in zip(FRACS, cvm)}})
    print(f"wrote -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
