"""Experiment — pin-noise robustness (deployment): how off can the tag be?

A real 땡시 tag / user click is imperfect, and the backbone can't absorb a
mislocated pin (it changes WHERE we pool). Perturb the test pin by Gaussian noise
of growing radius and measure top1 degradation; then test whether a JITTER-
AUGMENTED gallery (exemplars pooled at jittered q) widens the tolerance. DINO
grids are cached once and re-pooled at the shifted q (cheap). 10-seed mean±std.

    .venv/bin/python scripts/pin_robustness.py
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

import explog  # noqa: E402
from eval_appearance import split_indices, _git_sha  # noqa: E402
from learned_pooler import cache_grids, gather, gauss_pool  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone  # noqa: E402

RADII = [0, 10, 20, 40, 80]      # pin jitter std (original-image px)
JIT_AUG = 30.0                   # gallery jitter-aug std
J = 3                            # jitter-aug copies per gallery pin


def qframe(core, wh, S, jitter=0.0, rng=None):
    out = np.zeros((len(core), 2), np.float32)
    for i, r in enumerate(core):
        w, h = wh[r["image"]]
        x, y = r["q"]
        if jitter > 0:
            dx, dy = rng.normal(0, jitter, 2)
            x = np.clip(x + dx, 0, w - 1); y = np.clip(y + dy, 0, h - 1)
        out[i] = [x * S / w, y * S / h]
    return out


@torch.no_grad()
def pool(core, idx, grids, qf, centers, device, chunk=128):
    Z = []
    for s in range(0, len(idx), chunk):
        ch = idx[s:s + chunk]
        g = gather(grids, core, ch, device)
        q = torch.tensor(qf[ch], device=device)
        Z.append(gauss_pool(g, centers, q).cpu())
    return torch.cat(Z).numpy()


def top1(galZ, galY, teZ, teY):
    labels = sorted(set(galY)); lidx = {l: i for i, l in enumerate(labels)}
    cols = collections.defaultdict(list)
    for j, l in enumerate(galY):
        cols[lidx[l]].append(j)
    cv = [k for k, l in enumerate(teY) if l in lidx]
    sims = teZ[cv] @ galZ.T
    sc = np.full((len(cv), len(labels)), -2.0, np.float32)
    for c, ix in cols.items():
        sc[:, c] = sims[:, ix].max(1)
    p = sc.argmax(1)
    return 100 * sum(int(labels[p[r]] == teY[k]) for r, k in enumerate(cv)) / len(cv)


def main():
    base = Path("data/triples")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size
    rows = [json.loads(l) for l in open("data/triples/triples.jsonl", encoding="utf-8") if l.strip()]
    cnt = collections.Counter(r["label"] for r in rows)
    core = [r for r in rows if cnt[r["label"]] >= 2]
    Y = [r["label"] for r in core]
    print(f"core {len(core)} | device={device}")
    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    centers = bb.patch_centers(device)
    print("caching grids..."); grids, wh = cache_grids(core, base, bb, S, device)
    qf0 = qframe(core, wh, S)

    clean = {r: [] for r in RADII}
    augg = {r: [] for r in RADII}
    for seed in range(10):
        tr, te = split_indices(core, 0.3, seed)
        ytr = [Y[i] for i in tr]; yte = [Y[i] for i in te]
        idx = np.array(tr); tei = np.array(te)
        Ztr = pool(core, idx, grids, qf0, centers, device)
        # jitter-augmented gallery
        rng = np.random.default_rng(seed)
        galZ = [Ztr]; galY = list(ytr)
        for _ in range(J):
            qfj = qframe(core, wh, S, JIT_AUG, rng)
            galZ.append(pool(core, idx, grids, qfj, centers, device)); galY += ytr
        galZ = np.concatenate(galZ)
        for r in RADII:
            qte = qframe(core, wh, S, r, rng) if r > 0 else qf0
            Zte = pool(core, tei, grids, qte, centers, device)
            clean[r].append(top1(Ztr, ytr, Zte, yte))
            augg[r].append(top1(galZ, galY, Zte, yte))
        print(f"  seed {seed}: clean " + "/".join(f"{clean[r][-1]:.0f}" for r in RADII)
              + "  aug " + "/".join(f"{augg[r][-1]:.0f}" for r in RADII))

    ms = lambda v: (round(st.mean(v), 1), round(st.pstdev(v), 1))
    cm = [ms(clean[r]) for r in RADII]; am = [ms(augg[r]) for r in RADII]
    drop = round(cm[0][0] - cm[RADII.index(40)][0], 1)     # drop at 40px jitter
    print("\n== clean: " + " ".join(f"{r}px {cm[i][0]}" for i, r in enumerate(RADII))
          + " | aug-gal: " + " ".join(f"{r}px {am[i][0]}" for i, r in enumerate(RADII)) + " ==")

    d = explog.next_dir("pin-robustness")
    explog.lineplot(d / "fig_pinrobust.png",
                    [("clean gallery", RADII, [c[0] for c in cm], [c[1] for c in cm]),
                     ("jitter-aug gallery", RADII, [a[0] for a in am], [a[1] for a in am])],
                    "Pin-noise robustness", "pin jitter std (px)", "top1 (%)",
                    xlim=(-3, 83), ylim=(0, 60))
    tab = "\n".join(f"| {r}px | {cm[i][0]}±{cm[i][1]}% | {am[i][0]}±{am[i][1]}% |"
                    for i, r in enumerate(RADII))
    report = f"""# 핀 노이즈 강건성 (pin-robustness)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/pin_robustness.py`  (frozen exemplar, 10-seed)

## 목적
실전 태그/클릭은 부정확 → 핀 위치를 Gaussian noise로 흔들며 top1 저하 측정. **jitter-augmented
gallery**(갤러리 exemplar를 흔든 q로도 풀링, J={J}, σ={JIT_AUG:.0f}px)가 허용오차를 넓히는지.
참고: 풀링이 σ=40px 가우시안이라 어느 정도 공간 평균 효과가 있음.

## 결과 (top1, mean±std)
| 핀 jitter | clean gallery | jitter-aug gallery |
|---|---|---|
{tab}

![pinrobust](fig_pinrobust.png)

## 해석
- 무jitter {cm[0][0]}% → 40px jitter에서 clean이 {cm[RADII.index(40)][0]}% (−{drop}%p). 핀 오차에 대한 허용도.
- jitter-aug 갤러리가 흔들린 핀에서 clean보다 높으면 → **배포시 부정확한 태그에 강건** (채택 가치).

## 다음
val 기반 운영점 고정(정확도 보장), open-set 기권 스트레스 테스트.
"""
    explog.write(d, report, {
        "title": "핀 노이즈 강건성", "date": datetime.date.today().isoformat(),
        "headline": f"top1 {cm[0][0]}%(0px)→{cm[RADII.index(40)][0]}%(40px); aug-gal {am[RADII.index(40)][0]}% @40px",
        "clean": {r: cm[i] for i, r in enumerate(RADII)}, "aug": {r: am[i] for i, r in enumerate(RADII)}})
    print(f"wrote -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
