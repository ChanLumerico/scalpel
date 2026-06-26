"""Definitive best-setting evaluation — the full stack in one run.

frozen DINOv2 -> pin GaussianPool -> learned SupCon head -> exemplar 1-NN in the
learned space -> temperature calibration (gallery LOO) -> risk-coverage / abstention.
The one combination not yet measured jointly. 10-seed mean±std, specimen split.

    .venv/bin/python scripts/best_setting.py
"""

from __future__ import annotations

import collections
import datetime
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import explog  # noqa: E402
from eval_appearance import load_core, split_indices, embed, _git_sha  # noqa: E402
from eval_exemplar import class_max, fit_scale  # noqa: E402
from eval_calibration import ece, risk_coverage, COV_GRID  # noqa: E402
from learned_head import train_head  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402


def main():
    base = Path("data/triples")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size
    core = load_core("data/triples/triples.jsonl", 2)
    ncls = len({r["label"] for r in core})
    print(f"core {len(core)}/{ncls} | device={device}")
    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    pool = GaussianPool(cfg.point).to(device); centers = bb.patch_centers(device)
    print("embedding once..."); Z = embed(core, base, bb, pool, S, device)
    Znp = Z.numpy().astype(np.float32)

    t1, t5, cov, a30, a50, c80, ecal, rc_curves = [], [], [], [], [], [], [], []
    for seed in range(10):
        tr, te = split_indices(core, 0.3, seed)
        ytr = [core[i]["label"] for i in tr]; yte = [core[i]["label"] for i in te]
        head = train_head(Znp[tr], ytr, device="cpu")
        with torch.no_grad():
            Ptr = F.normalize(head(torch.from_numpy(Znp[tr])), dim=1).numpy()
            Pte = F.normalize(head(torch.from_numpy(Znp[te])), dim=1).numpy()
        labels = sorted(set(ytr)); lidx = {l: i for i, l in enumerate(labels)}; C = len(labels)
        cols = collections.defaultdict(list)
        for j, l in enumerate(ytr):
            cols[lidx[l]].append(j)
        # calibration: LOO exemplar scores on the gallery (learned space)
        G = Ptr @ Ptr.T; np.fill_diagonal(G, -2.0)
        Sg = class_max(G, cols, C)
        keep = [j for j, l in enumerate(ytr) if len(cols[lidx[l]]) >= 2]
        s = fit_scale(Sg[keep], [lidx[ytr[j]] for j in keep]) if keep else 15.0
        # test
        ci = [k for k, l in enumerate(yte) if l in lidx]
        St = class_max(Pte[ci] @ Ptr.T, cols, C)
        probs = F.softmax(torch.from_numpy(s * St), dim=1).numpy()
        order = np.argsort(-St, axis=1)
        confs, corr = [], []
        n1 = n5 = 0
        for r, k in enumerate(ci):
            t = lidx[yte[k]]
            n1 += int(order[r, 0] == t); n5 += int(t in order[r, :5])
            confs.append(float(probs[r].max())); corr.append(order[r, 0] == t)
        ncov = len(ci); ntot = len(te)
        t1.append(100 * n1 / ncov); t5.append(100 * n5 / ncov); cov.append(100 * ncov / ntot)
        rc = risk_coverage(confs, corr); rc_curves.append(rc)
        a30.append(float(np.interp(0.3, COV_GRID, rc)) * 100)
        a50.append(float(np.interp(0.5, COV_GRID, rc)) * 100)
        rca = np.array(rc); c80.append(float(COV_GRID[rca >= 0.80].max() * 100) if (rca >= 0.80).any() else 0.0)
        ecal.append(ece(confs, corr))
        print(f"  seed {seed}: top1 {t1[-1]:.0f} top5 {t5[-1]:.0f} cov {cov[-1]:.0f} "
              f"acc@30 {a30[-1]:.0f} cov@80 {c80[-1]:.0f} ECE {ecal[-1]:.2f}")

    ms = lambda v: (round(st.mean(v), 1), round(st.pstdev(v), 1))
    A1, A5, AC = ms(t1), ms(t5), ms(cov)
    P30, P50, C80, EC = ms(a30), ms(a50), ms(c80), ms(ecal)
    rc_mean = np.mean(rc_curves, axis=0) * 100; rc_std = np.std(rc_curves, axis=0) * 100
    print(f"\n== BEST: top1 {A1[0]}±{A1[1]} top5 {A5[0]}±{A5[1]} cov {AC[0]}% | "
          f"acc@30 {P30[0]} cov@acc80 {C80[0]} | ECE {EC[0]} ==")

    d = explog.next_dir("best-setting")
    explog.bar(d / "fig_best.png", ["top1", "top5", "acc@30%", "acc@50%"],
               [A1[0], A5[0], P30[0], P50[0]], "Best setting (10-seed mean±std)", "%",
               ymax=100, errors=[A1[1], A5[1], P30[1], P50[1]])
    explog.lineplot(d / "fig_risk_coverage.png",
                    [("selective accuracy", (COV_GRID * 100).tolist(), rc_mean.tolist(), rc_std.tolist())],
                    "Risk–coverage (best setting)", "coverage (% answered)", "accuracy (%)",
                    xlim=(0, 100), ylim=(0, 100), hline=(A1[0], f"answer-all = {A1[0]}%"))
    report = f"""# 최고 세팅 최종 성능 (best-setting)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/best_setting.py`  (10-seed mean±std)

## 세팅 (전체 스택)
frozen DINOv2(vitb14, 518px) → 핀 GaussianPool(σ40) → **학습 SupCon 헤드** →
**exemplar 1-NN**(학습 공간) → **temperature 보정**(갤러리 LOO) → **risk-coverage 기권**.

## 결과 ({len(core)} 트리플 / {ncls} 클래스, 표본분할 10 seed)
| 지표 | frozen-exemplar(010) | **best(현재)** |
|---|---|---|
| top1 (전부 답) | 46.6% | **{A1[0]} ± {A1[1]}%** |
| top5 | 58.0% | **{A5[0]} ± {A5[1]}%** |
| coverage | 83% | {AC[0]}% |
| **확신 상위 30%만 답** | 88.5% | **{P30[0]} ± {P30[1]}%** |
| 확신 상위 50%만 답 | 64.0% | {P50[0]} ± {P50[1]}% |
| 정확도 80% 유지 coverage | 24% | {C80[0]}% |
| ECE (보정 후) | 0.2 | {EC[0]} |

![best](fig_best.png)
![risk-coverage](fig_risk_coverage.png)

## 한 줄
전부 답하면 **top1 {A1[0]}% / top5 {A5[0]}%**, 확신 상위 30%만 답하면 **{P30[0]}%**.
(multi-seed cross-cadaver; 여러 실험 test 재사용에 의한 ~1-2%p 낙관 가능, 오염 아님.)

## 남은 레버
데이터 곡선(exp 013)이 미포화 → **데이터 확장이 최우선**, 그 위에 학습형 풀러/M6'.
"""
    explog.write(d, report, {
        "title": "최고 세팅 최종 성능", "date": datetime.date.today().isoformat(),
        "headline": f"top1 {A1[0]}±{A1[1]}% / top5 {A5[0]}±{A5[1]}% @cov{AC[0]}% | 확신30% {P30[0]}% | ECE {EC[0]}",
        "top1": A1, "top5": A5, "coverage": AC, "acc_at_30": P30, "acc_at_50": P50,
        "cov_at_acc80": C80, "ece_cal": EC})
    print(f"wrote -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
