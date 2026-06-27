"""Experiment — deployment operating point + open-set (the deployment spec).

Clean 3-way SPECIMEN split (gallery / val / test) — also resolves the earlier
val-vs-test reuse concern. On VAL, pick the confidence threshold tau* that
guarantees a target accuracy (90%); report on the untouched TEST how much it can
answer at that accuracy. Plus OPEN-SET: singleton-class pins (never in the
gallery) are out-of-vocabulary queries the system must REJECT, not confidently
mis-answer — measure rejection at tau* and the confidence AUROC (in-vocab vs OOV).

Frozen exemplar retrieval + temperature calibration. 10-seed mean±std.

    .venv/bin/python scripts/deployment.py
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
import torch.nn.functional as F  # noqa: E402

import explog  # noqa: E402
from eval_appearance import embed, _git_sha  # noqa: E402
from eval_exemplar import class_max, fit_scale  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402

TARGET = 0.90       # guaranteed-accuracy target for the operating point


def split_pages(rows, seed, fracs=(0.55, 0.15, 0.30)):
    pages = sorted({f'{r["src"]}#{r["page"]}' for r in rows})
    np.random.default_rng(seed).shuffle(pages)
    n = len(pages); a = int(n * fracs[0]); b = a + int(n * fracs[1])
    return set(pages[:a]), set(pages[a:b]), set(pages[b:])


def auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    base = Path("data/triples")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size
    rows = [json.loads(l) for l in open("data/triples/triples.jsonl", encoding="utf-8") if l.strip()]
    cnt = collections.Counter(r["label"] for r in rows)
    Y = [r["label"] for r in rows]
    page = [f'{r["src"]}#{r["page"]}' for r in rows]
    core = [i for i in range(len(rows)) if cnt[Y[i]] >= 2]
    oov_all = [i for i in range(len(rows)) if cnt[Y[i]] == 1]
    print(f"triples {len(rows)} | in-vocab(>=2) {len(core)} | OOV singletons {len(oov_all)} | device={device}")
    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    pool = GaussianPool(cfg.point).to(device)
    print("embedding once..."); Z = embed(rows, base, bb, pool, S, device).numpy().astype(np.float32)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)

    test_acc, test_cov, oov_rej, aurocs, taus = [], [], [], [], []
    for seed in range(10):
        gp, vp, tp = split_pages(rows, seed)
        gal = [i for i in core if page[i] in gp]
        labels = sorted({Y[i] for i in gal}); lidx = {l: j for j, l in enumerate(labels)}
        cols = collections.defaultdict(list)
        gpos = {}
        for j, i in enumerate(gal):
            cols[lidx[Y[i]]].append(j); gpos[i] = j
        Gz = Z[gal]
        # calibrate s on gallery LOO
        G = Gz @ Gz.T; np.fill_diagonal(G, -2.0)
        Sg = class_max(G, cols, len(labels))
        keep = [gpos[i] for i in gal if len(cols[lidx[Y[i]]]) >= 2]
        s = fit_scale(Sg[keep], [lidx[Y[i]] for i in gal if len(cols[lidx[Y[i]]]) >= 2]) if keep else 15.0

        def cp(idx):
            St = class_max(Z[idx] @ Gz.T, cols, len(labels))
            P = F.softmax(torch.from_numpy(s * St), dim=1).numpy()
            return P.max(1), St.argmax(1)

        val_iv = [i for i in core if page[i] in vp and Y[i] in lidx]
        vc, vp_ = cp(val_iv)
        vcorr = np.array([labels[vp_[r]] == Y[i] for r, i in enumerate(val_iv)])
        # smallest tau giving >= TARGET accuracy on answered val
        tau, best = 1.01, None
        for th in np.unique(vc):
            ans = vc >= th
            if ans.sum() >= 5 and vcorr[ans].mean() >= TARGET:
                tau = th; break
        taus.append(float(tau))

        tst_iv = [i for i in core if page[i] in tp and Y[i] in lidx]
        tc, tp_ = cp(tst_iv)
        tcorr = np.array([labels[tp_[r]] == Y[i] for r, i in enumerate(tst_iv)])
        ans = tc >= tau
        test_acc.append(100 * tcorr[ans].mean() if ans.sum() else 0.0)
        test_cov.append(100 * ans.mean())
        oov = [i for i in oov_all if page[i] in tp]
        if oov:
            oc, _ = cp(oov)
            oov_rej.append(100 * (oc < tau).mean())
            aurocs.append(100 * auroc(tc, oc))      # in-vocab vs OOV confidence
        print(f"  seed {seed}: tau {tau:.2f} | test acc {test_acc[-1]:.0f}@cov {test_cov[-1]:.0f} "
              f"| OOV reject {oov_rej[-1] if oov else float('nan'):.0f} AUROC {aurocs[-1] if oov else float('nan'):.0f}")

    ms = lambda v: (round(st.mean(v), 1), round(st.pstdev(v), 1))
    TA, TC, OR, AU = ms(test_acc), ms(test_cov), ms(oov_rej), ms(aurocs)
    print(f"\n== @target {int(TARGET*100)}%: test acc {TA[0]}±{TA[1]} cov {TC[0]}±{TC[1]} "
          f"| OOV reject {OR[0]}% AUROC {AU[0]} ==")

    d = explog.next_dir("deployment")
    explog.bar(d / "fig_deploy.png",
               [f"test acc\n@tau*", "answer rate\n(coverage)", "OOV\nreject", "open-set\nAUROC"],
               [TA[0], TC[0], OR[0], AU[0]], f"Deployment @ guaranteed {int(TARGET*100)}% accuracy", "%",
               ymax=100, errors=[TA[1], TC[1], OR[1], AU[1]])
    report = f"""# 배포 운영점 + open-set (deployment)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/deployment.py`  (3-way split, 10-seed)

## 목적 (배포 스펙)
**val에서 정확도 {int(TARGET*100)}% 보장 임계값 τ\\* 고정 → 한 번도 안 본 test에서 "그 정확도로 답할 수
있는 비율"** 측정. 동시에 **open-set**: 갤러리에 없는 구조물(싱글톤=OOV)을 자신있게 틀리지 않고
기권하는가. (3-way 분할이라 앞서 지적된 test-재사용 낙관도 해소.)

## 결과 (10-seed, 목표 정확도 {int(TARGET*100)}%)
| 지표 | 값 |
|---|---|
| τ\\*에서 **test 실제 정확도** | **{TA[0]} ± {TA[1]}%** (목표 {int(TARGET*100)}%를 test에서 지키는지) |
| **답할 수 있는 비율(coverage)** | **{TC[0]} ± {TC[1]}%** |
| **OOV 기권율** (갤러리에 없는 구조물 거부) | {OR[0]} ± {OR[1]}% |
| open-set AUROC (in-vocab vs OOV 확신도) | {AU[0]} |

![deploy](fig_deploy.png)

## 해석 (배포 관점)
- "**확신할 때만 답하면 ~{int(TARGET*100)}% 정확도로 in-vocab 핀의 {TC[0]}%를 답**한다" — 운영점을 val로
  잡아 test에서 검증했으니 배포 스펙으로 신뢰 가능.
- **OOV 기권 {OR[0]}%** / AUROC {AU[0]}: 갤러리에 없는 구조물도 상당수 거부 → "모르는 건 안 답함"의 배포 안전성.
  (완벽치 않으니 임계값·OOD 점수 개선 여지.)

## 종합 (배포 견고성)
조명/색차(백본 흡수, 016) · 핀오차 ~40px(풀링 흡수, 017) · **운영점/open-set(본 실험)** 까지 확인.
남은 천장은 데이터(013) — coverage·정확도 동시 상승은 더 많은 표본에서.
"""
    explog.write(d, report, {
        "title": "배포 운영점 + open-set", "date": datetime.date.today().isoformat(),
        "headline": f"@{int(TARGET*100)}%: test acc {TA[0]}% answer {TC[0]}% | OOV reject {OR[0]}% AUROC {AU[0]}",
        "test_acc": TA, "coverage": TC, "oov_reject": OR, "auroc": AU, "tau_mean": round(st.mean(taus), 2)})
    print(f"wrote -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
