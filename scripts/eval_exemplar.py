"""Canonical model with nearest-EXEMPLAR retrieval (exp 009 win) + calibration.

exp 009 showed nearest single gallery exemplar beats the mean prototype by ~+8%p
(averaging washes out detail). Adopt it as the retrieval rule and re-measure the
full profile: accuracy (top1/top5/coverage) AND the M5' selective-prediction
numbers (temperature-calibrated risk-coverage), 10-seed mean±std.

Score for a class = max cosine(z_q, gallery-exemplar-of-class); temperature s fit
by leave-one-out on the gallery (exclude self); confidence = max softmax prob.

    .venv/bin/python scripts/eval_exemplar.py
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
from eval_calibration import ece, risk_coverage, COV_GRID  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402


def class_max(sims, cols_by_class, C):
    """sims (M, Ntr) -> (M, C): per-class max similarity (1-NN per class)."""
    out = np.full((sims.shape[0], C), -2.0, dtype=np.float32)
    for c, cols in cols_by_class.items():
        out[:, c] = sims[:, cols].max(1)
    return out


def fit_scale(scores_loo, targets):
    best = (1e9, 15.0)
    S = torch.from_numpy(scores_loo)
    tgt = torch.tensor(targets)
    for s in np.arange(2, 40.5, 1.0):
        logp = F.log_softmax(float(s) * S, dim=1)
        nll = -logp[torch.arange(len(tgt)), tgt].mean().item()
        if nll < best[0]:
            best = (nll, float(s))
    return best[1]


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

    t1, t5, cov = [], [], []
    acc100, acc50, acc30, c70, c80 = [], [], [], [], []
    ece_raw, ece_cal, scales, rc_curves = [], [], [], []
    for seed in range(10):
        tr, te = split_indices(core, 0.3, seed)
        Ztr = Znp[tr]; ytr = [core[i]["label"] for i in tr]
        labels = sorted(set(ytr)); lidx = {l: i for i, l in enumerate(labels)}
        C = len(labels)
        cols = collections.defaultdict(list)
        for j, l in enumerate(ytr):
            cols[lidx[l]].append(j)

        # calibration: LOO scores on the gallery (exclude self), classes with >=2
        G = Ztr @ Ztr.T
        np.fill_diagonal(G, -2.0)
        Sg = class_max(G, cols, C)                      # (Ntr, C)
        keep = [j for j, l in enumerate(ytr) if len(cols[lidx[l]]) >= 2]
        s_scale = fit_scale(Sg[keep], [lidx[ytr[j]] for j in keep]) if keep else 15.0
        scales.append(s_scale)

        # test
        cov_idx = [j for j in te if core[j]["label"] in lidx]
        if not cov_idx:
            continue
        Zte = Znp[cov_idx]
        St = class_max(Zte @ Ztr.T, cols, C)            # (Ncov, C)
        probs = F.softmax(torch.from_numpy(s_scale * St), dim=1).numpy()
        probs_raw = F.softmax(torch.from_numpy(1.0 * St), dim=1).numpy()
        order = np.argsort(-St, axis=1)
        confs, corr = [], []
        n1 = n5 = 0
        for r, j in enumerate(cov_idx):
            true = lidx[core[j]["label"]]
            top = order[r, :5]
            n1 += int(top[0] == true); n5 += int(true in top)
            confs.append(float(probs[r].max())); corr.append(top[0] == true)
        ncov = len(cov_idx); ntot = len(te)
        t1.append(100 * n1 / ncov); t5.append(100 * n5 / ncov); cov.append(100 * ncov / ntot)
        rc = risk_coverage(confs, corr); rc_curves.append(rc)
        acc100.append(rc[-1] * 100)
        acc50.append(float(np.interp(0.5, COV_GRID, rc)) * 100)
        acc30.append(float(np.interp(0.3, COV_GRID, rc)) * 100)
        rca = np.array(rc)
        c70.append(float(COV_GRID[rca >= 0.70].max() * 100) if (rca >= 0.70).any() else 0.0)
        c80.append(float(COV_GRID[rca >= 0.80].max() * 100) if (rca >= 0.80).any() else 0.0)
        ece_cal.append(ece(confs, corr))
        ece_raw.append(ece([float(p.max()) for p in probs_raw], corr))
        print(f"  seed {seed}: top1 {t1[-1]:.0f} top5 {t5[-1]:.0f} cov {cov[-1]:.0f} "
              f"acc@30 {acc30[-1]:.0f} s={s_scale:.0f} ECE {ece_raw[-1]:.2f}->{ece_cal[-1]:.2f}")

    ms = lambda v: (round(st.mean(v), 1), round(st.pstdev(v), 1))
    a1, a5, ac = ms(t1), ms(t5), ms(cov)
    p100, p50, p30 = ms(acc100), ms(acc50), ms(acc30)
    cc70, cc80 = ms(c70), ms(c80); er, ecal = ms(ece_raw), ms(ece_cal)
    rc_mean = (np.mean(rc_curves, axis=0) * 100); rc_std = (np.std(rc_curves, axis=0) * 100)
    print(f"\n== EXEMPLAR top1 {a1[0]}±{a1[1]} top5 {a5[0]}±{a5[1]} cov {ac[0]}% "
          f"| acc@30 {p30[0]} | ECE {er[0]}->{ecal[0]} ==")

    d = explog.next_dir("baseline-exemplar")
    explog.bar(d / "fig_topk.png", ["top1", "top5"], [a1[0], a5[0]],
               "Exemplar retrieval (10-seed mean±std)", "%", ymax=100, errors=[a1[1], a5[1]])
    explog.lineplot(d / "fig_risk_coverage.png",
                    [("selective accuracy", (COV_GRID * 100).tolist(), rc_mean.tolist(), rc_std.tolist())],
                    "Risk–coverage (exemplar + calibration)", "coverage (% answered)", "accuracy (%)",
                    xlim=(0, 100), ylim=(0, 100), hline=(p100[0], f"answer-all = {p100[0]}%"))
    report = f"""# 정식 모델: exemplar retrieval + 보정 (baseline-exemplar)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/eval_exemplar.py`  (10-seed mean±std)

## 무엇이 바뀌었나
exp 009에서 **최근접 단일 exemplar(1-NN) > 평균 프로토타입 +8%p**를 확인 → retrieval 규칙을
exemplar로 교체. 클래스 점수 = max cosine(z_q, 갤러리 exemplar), 나머지(보정/기권)는 동일.

## 결과 ({len(core)} 트리플 / {ncls} 클래스)
| 지표 | proto(005/007) | **exemplar(현재)** |
|---|---|---|
| top1 | 38.8±3.4% | **{a1[0]}±{a1[1]}%** |
| top5 | 55.8±4.0% | **{a5[0]}±{a5[1]}%** |
| coverage | 83% | {ac[0]}% |
| 확신 상위 30%만 답 | 78.4% | **{p30[0]}±{p30[1]}%** |
| 확신 상위 50%만 답 | 64.0% | {p50[0]}±{p50[1]}% |
| 정확도 80% 유지 coverage | 24% | {cc80[0]}% |
| ECE (보정 후) | 0.2 | {ecal[0]} |

![topk](fig_topk.png)
![risk-coverage](fig_risk_coverage.png)

## 해석
- 학습 0으로 top1 {a1[0]}%, top5 {a5[0]}% — 평균 프로토타입의 디테일 손실을 제거한 결과.
- 기권을 붙이면 확신 상위 30%에서 **{p30[0]}%** → 신뢰 구간 정확도가 더 올라감.
- 여전히 무학습. 다음 레버는 **학습형 판별 헤드**(look-alike 분리)와 **데이터(coverage)**.

## 다음
exemplar를 기본 retrieval로 채택(코드 반영). 이후 학습형 metric 헤드 실험 → top1 추가 상승 시도.
"""
    explog.write(d, report, {
        "title": "정식 모델: exemplar + 보정", "date": datetime.date.today().isoformat(),
        "headline": f"exemplar top1 {a1[0]}±{a1[1]}% / top5 {a5[0]}±{a5[1]}% @cov{ac[0]}% | 상위30% {p30[0]}%",
        "top1": a1, "top5": a5, "coverage": ac, "acc_at_30": p30, "acc_at_50": p50,
        "cov_at_acc80": cc80, "ece_cal": ecal, "scale": round(st.mean(scales))})
    print(f"wrote -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
