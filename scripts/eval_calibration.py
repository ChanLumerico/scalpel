"""M5' — calibration + abstention (selective prediction) for the appearance MVP.

The raw model answers every covered pin at ~39% top1. For a 땡시 assistant the
useful question is instead: *if it only answers when confident, how accurate is
it, and how much can it answer?* This adds:

1. **Temperature calibration** — the appearance expert scores a pin by cosine
   similarity to class prototypes; ``softmax(s * sims)`` turns that into a
   probability. The scale ``s`` (inverse temperature) is fit by minimizing NLL via
   leave-one-out on the GALLERY (no test leakage), so the confidence value means
   something (low ECE).
2. **Abstention / risk-coverage** — rank covered test pins by confidence; answering
   only the most-confident fraction trades coverage for accuracy. We report the
   risk-coverage curve + operating points ("answer the top X% → accuracy Y%").

Multi-seed (embed once, N seeds) -> mean±std. Logged via ``explog``.

    .venv/bin/python scripts/eval_calibration.py
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
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402

COV_GRID = np.linspace(0.05, 1.0, 20)   # fraction of covered pins answered


def build_protos(rows, Z):
    sums = collections.defaultdict(lambda: torch.zeros(Z.shape[1]))
    cnt = collections.Counter()
    by = collections.defaultdict(list)
    for i, r in enumerate(rows):
        sums[r["label"]] += Z[i]; cnt[r["label"]] += 1; by[r["label"]].append(i)
    labels = list(sums)
    P = torch.stack([F.normalize(sums[l], dim=0) for l in labels])
    return labels, P, sums, cnt, by


def fit_scale(rows, Z, labels, P, sums, cnt, by):
    """Inverse-temperature s minimizing leave-one-out NLL on the gallery."""
    idx = {l: i for i, l in enumerate(labels)}
    sims_rows, targets = [], []
    for l in labels:
        if cnt[l] < 2:
            continue
        for i in by[l]:
            loo = F.normalize(sums[l] - Z[i], dim=0)      # prototype without item i
            s = (P @ Z[i]).clone()
            s[idx[l]] = torch.dot(Z[i], loo)
            sims_rows.append(s); targets.append(idx[l])
    if not sims_rows:
        return 15.0
    S = torch.stack(sims_rows)
    tgt = torch.tensor(targets)
    best = (1e9, 15.0)
    for s in np.arange(2, 40.5, 1.0):
        logp = F.log_softmax(float(s) * S, dim=1)
        nll = -logp[torch.arange(len(tgt)), tgt].mean().item()
        if nll < best[0]:
            best = (nll, float(s))
    return best[1]


def ece(confs, corr, bins=10):
    confs, corr = np.asarray(confs), np.asarray(corr, dtype=float)
    n, e = len(confs), 0.0
    for b in range(bins):
        m = (confs > b / bins) & (confs <= (b + 1) / bins)
        if m.sum():
            e += abs(corr[m].mean() - confs[m].mean()) * m.sum() / n
    return e


def risk_coverage(confs, corr):
    """Accuracy at each coverage level (most-confident-first)."""
    order = np.argsort(-np.asarray(confs))
    c = np.asarray(corr, dtype=float)[order]
    n = len(c)
    return [float(c[: max(1, int(round(g * n)))].mean()) for g in COV_GRID]


def main():
    jsonl = "data/triples/triples.jsonl"
    base = Path(jsonl).parent
    n_seeds, test_frac = 10, 0.3
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg()
    S = cfg.backbone.image_size

    core = load_core(jsonl, 2)
    ncls = len({r["label"] for r in core})
    print(f"core {len(core)}/{ncls} | device={device} | seeds={n_seeds}")
    backbone = DinoBackbone(cfg.backbone); backbone.ensure_loaded(); backbone.to(device)
    pool = GaussianPool(cfg.point).to(device)
    print("embedding once..."); Z = embed(core, base, backbone, pool, S, device)

    rc_curves, scales = [], []
    ece_raw, ece_cal = [], []
    acc_all, acc50, acc30 = [], [], []
    cov_at_70, cov_at_80 = [], []
    for seed in range(n_seeds):
        tr, te = split_indices(core, test_frac, seed)
        trr, ter = [core[i] for i in tr], [core[i] for i in te]
        Ztr, Zte = Z[tr], Z[te]
        labels, P, sums, cnt, by = build_protos(trr, Ztr)
        s = fit_scale(trr, Ztr, labels, P, sums, cnt, by)
        scales.append(s)
        idx = {l: i for i, l in enumerate(labels)}
        confs, corr, confs_raw = [], [], []
        for j, r in enumerate(ter):
            if r["label"] not in idx:
                continue                                   # OOV -> abstain (excluded)
            sims = P @ Zte[j]
            p = F.softmax(s * sims, dim=0)
            p1 = F.softmax(1.0 * sims, dim=0)              # uncalibrated reference
            k = int(torch.argmax(sims))
            confs.append(float(p.max())); confs_raw.append(float(p1.max()))
            corr.append(labels[k] == r["label"])
        if not corr:
            continue
        rc = risk_coverage(confs, corr)
        rc_curves.append(rc)
        ece_cal.append(ece(confs, corr)); ece_raw.append(ece(confs_raw, corr))
        acc_all.append(rc[-1] * 100)                       # coverage 100%
        acc50.append(np.interp(0.5, COV_GRID, rc) * 100)
        acc30.append(np.interp(0.3, COV_GRID, rc) * 100)
        # max coverage achieving >= target accuracy (most-confident-first)
        rc_arr = np.array(rc)
        cov_at_70.append(float(COV_GRID[rc_arr >= 0.70].max() * 100) if (rc_arr >= 0.70).any() else 0.0)
        cov_at_80.append(float(COV_GRID[rc_arr >= 0.80].max() * 100) if (rc_arr >= 0.80).any() else 0.0)
        print(f"  seed {seed}: s={s:.0f} acc@100={rc[-1]*100:.0f} acc@50={acc50[-1]:.0f} "
              f"acc@30={acc30[-1]:.0f} ECE {ece_raw[-1]:.2f}->{ece_cal[-1]:.2f}")

    rc_mean = np.mean(rc_curves, axis=0) * 100
    rc_std = np.std(rc_curves, axis=0) * 100
    ms = lambda v: (round(st.mean(v), 1), round(st.pstdev(v), 1))
    a100, a50, a30 = ms(acc_all), ms(acc50), ms(acc30)
    c70, c80 = ms(cov_at_70), ms(cov_at_80)
    er, ec = ms(ece_raw), ms(ece_cal)
    s_mean = round(st.mean(scales))
    print(f"\n== acc@100={a100[0]}±{a100[1]} acc@50={a50[0]}±{a50[1]} acc@30={a30[0]}±{a30[1]} "
          f"| cov@acc70={c70[0]}% cov@acc80={c80[0]}% | ECE {er[0]}->{ec[0]} | s~{s_mean} ==")

    # ---- experiment folder ---------------------------------------------------
    d = explog.next_dir("calibration")
    explog.lineplot(
        d / "fig_risk_coverage.png",
        [("selective accuracy", (COV_GRID * 100).tolist(), rc_mean.tolist(), rc_std.tolist())],
        "Risk–coverage (answer most-confident first)", "coverage (% of covered pins answered)",
        "accuracy (%)", xlim=(0, 100), ylim=(0, 100), hline=(a100[0], f"answer-all = {a100[0]}%"))
    explog.bar(d / "fig_operating.png",
               ["acc@100%", "acc@50%", "acc@30%"], [a100[0], a50[0], a30[0]],
               "Accuracy vs how much it answers", "%", ymax=100,
               errors=[a100[1], a50[1], a30[1]])

    report = f"""# M5' — 보정 + 기권 (calibration / abstention)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/eval_calibration.py`  ({n_seeds}-seed mean±std)

## 목적
"전부 {a100[0]}%로 답하기"보다 **확신하는 것만 답하고 나머진 기권**하는 게 땡시 보조로 유용하다.
외형 전문가의 cosine 유사도를 `softmax(s·sims)` 확률로 바꾸고(s는 갤러리 LOO로 NLL 최소화 →
보정), 확신도 순으로 답하며 coverage–정확도 trade-off(risk–coverage)를 측정한다.

## 결과 ({n_seeds}-seed, 코어 {len(core)}/{ncls} 클래스)
| 운영점 | 정확도 |
|---|---|
| 전부 답 (coverage 100%) | {a100[0]} ± {a100[1]}% |
| 확신 상위 50%만 답 | **{a50[0]} ± {a50[1]}%** |
| 확신 상위 30%만 답 | **{a30[0]} ± {a30[1]}%** |
| 정확도 70% 유지하며 답할 수 있는 비율 | {c70[0]}% |
| 정확도 80% 유지하며 답할 수 있는 비율 | {c80[0]}% |

보정 전/후 ECE: **{er[0]} → {ec[0]}** (낮을수록 확신도가 실제 정확도에 가까움), 보정 scale s≈{s_mean}.

![risk-coverage](fig_risk_coverage.png)
![operating](fig_operating.png)

## 해석
- 확신 구간만 고르면 정확도가 {a100[0]}% → 상위30% 기준 **{a30[0]}%**로 올라간다 → "**모르면 기권**"이
  실제로 작동. 땡시 보조로는 전체를 낮게 답하는 것보다 이게 가치.
- 보정으로 ECE가 {er[0]}→{ec[0]}로 줄어 확신도 임계값(τ)을 실제 정확도로 해석 가능.
- 단, 여기 coverage는 *프로토타입이 있는* 핀 기준. 새 시신의 미지 구조물(OOV)은 별도로 항상 기권.

## 한계 / 다음
정확도 자체의 상한은 외형 전문가(top1 ~{a100[0]}%)가 결정 → 더 올리려면 **M6' 관계추론**(인접
구조물 혼동 분리) 또는 데이터 확장. M5'는 그 위에 "신뢰" 레이어를 얹어 지금 모델을 쓸 수 있게 만든다.
"""
    explog.write(d, report, {
        "title": "보정 + 기권 (M5')", "date": datetime.date.today().isoformat(),
        "headline": f"acc@cov100={a100[0]}% → 상위30%만 답 {a30[0]}% | ECE {er[0]}→{ec[0]} ({n_seeds} seeds)",
        "acc_at_100": a100, "acc_at_50": a50, "acc_at_30": a30,
        "cov_at_acc70": c70, "cov_at_acc80": c80, "ece_raw": er, "ece_cal": ec, "scale": s_mean})
    print(f"wrote experiment -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
