"""Experiment — diverse-backbone ensemble (DINO ⊕ BiomedCLIP-image).

bmc-img alone (36.9) is weaker than DINO (46.6), but it was pretrained on a totally
different distribution (medical figures) — its ERRORS may be complementary, and a
fused score can beat either alone even when one is weaker. Fuse the two exemplar
class-max sims:  score = sim_dino + λ · sim_bmc.  Fixed λ sweep (not tuned on test),
10-seed, paired vs DINO-only.

    .venv/bin/python scripts/ensemble.py
"""

from __future__ import annotations

import collections
import datetime
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import open_clip  # noqa: E402
import torch  # noqa: E402

import explog  # noqa: E402
from biomedclip import BMC, bmc_images, ex_scores, topk  # noqa: E402
from eval_appearance import load_core, split_indices, embed, _git_sha  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402

LAMBDAS = [0.0, 0.3, 0.6, 1.0, 1.5]


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
    del bb
    if device == "mps":
        torch.mps.empty_cache()

    print("BiomedCLIP embed..."); model, prep = open_clip.create_model_from_pretrained(BMC)
    model = model.to(device).eval()
    Zb = bmc_images(core, base, model, prep, device)

    out = {lam: ([], []) for lam in LAMBDAS}
    for seed in range(10):
        tr, te = split_indices(core, 0.3, seed)
        ytr = [Y[i] for i in tr]
        labels = sorted(set(ytr)); lidx = {l: j for j, l in enumerate(labels)}
        cols = collections.defaultdict(list)
        for j, l in enumerate(ytr):
            cols[lidx[l]].append(j)
        cv = [k for k in te if Y[k] in lidx]
        scd = ex_scores(Zd[tr], ytr, Zd[cv], labels, lidx, cols)
        scb = ex_scores(Zb[tr], ytr, Zb[cv], labels, lidx, cols)
        for lam in LAMBDAS:
            a = topk(scd + lam * scb, labels, Y, cv)
            out[lam][0].append(a[0]); out[lam][1].append(a[1])
        print(f"  seed {seed}: " + " ".join(f"λ{lam} {out[lam][0][-1]:.0f}" for lam in LAMBDAS))

    ms = lambda v: (round(st.mean(v), 1), round(st.pstdev(v), 1))
    base1 = out[0.0][0]
    rows = []
    for lam in LAMBDAS:
        t1, t5 = out[lam]
        dd = [a - b for a, b in zip(t1, base1)]
        rows.append((lam, ms(t1), ms(t5), round(st.mean(dd), 1), sum(x > 0 for x in dd)))
    best = max(rows[1:], key=lambda r: r[3])
    verdict = ("상보적 — 앙상블이 향상 (paired)" if (best[3] > 0 and best[4] >= 8)
               else "상보성 부족 — 앙상블 무효")
    for lam, t1, t5, dd, w in rows:
        print(f"  λ={lam}: top1 {t1[0]}±{t1[1]}  top5 {t5[0]}  Δ{dd:+}({w}/10)")
    print(f"\n== best λ={best[0]} Δtop1 {best[3]:+}({best[4]}/10) -> {verdict} ==")

    d = explog.next_dir("ensemble")
    explog.bar(d / "fig_ens.png", [f"λ{r[0]}" for r in rows], [r[1][0] for r in rows],
               "DINO ⊕ BiomedCLIP ensemble: top1 (10-seed)", "%", ymax=100, errors=[r[1][1] for r in rows])
    tab = "\n".join(f"| {lam} | {t1[0]}±{t1[1]}% | {t5[0]}% | {dd:+} ({w}/10) |" for lam, t1, t5, dd, w in rows)
    report = f"""# 다양백본 앙상블 (ensemble)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/ensemble.py`

## 목적
bmc-img 단독(36.9)은 약하지만 *다른 분포(의료 figure)* 로 학습돼 오류가 상보적일 수 있음 →
`score = sim_dino + λ·sim_bmc` 융합이 DINO 단독을 넘는지. 고정 λ, 10-seed, paired.

## 결과 (paired vs λ=0=DINO)
| λ | top1 | top5 | Δtop1 |
|---|---|---|---|
{tab}

![ens](fig_ens.png)

## 판정
- 베스트 λ={best[0]}: Δtop1 {best[3]:+}%p ({best[4]}/10) → **{verdict}**

## 해석
- 향상하면 → 두 백본이 상보적, 값싼 앙상블이 레버. 무효면 → bmc-img가 박리 사진엔 약해 노이즈만 추가.
"""
    explog.write(d, report, {
        "title": "다양백본 앙상블", "date": datetime.date.today().isoformat(),
        "headline": f"best λ={best[0]} Δtop1 {best[3]:+}({best[4]}/10) → {verdict}",
        "lambdas": {str(lam): {"top1": t1, "top5": t5, "dtop1": dd} for lam, t1, t5, dd, w in rows}})
    print(f"wrote -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
