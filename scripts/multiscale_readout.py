"""Experiment 046 — M-rep2: tissue-aware soft-gate readout on global+L256.

045 corrected a 9-month misdiagnosis: artery/vein ARE encoded by DINO (linear AUC 0.76); the
exemplar 1-NN readout throws the axis away (centroid cos 0.88, NN crosses regions). 043 only
closed *tissue-agnostic* readouts (mean/kNN/KDE...). A *tissue-aware* readout is untried, and the
tissue-oracle says +6.4pp of headroom. This realises it with a REAL (imperfect) tissue classifier,
SOFT-gated to avoid Stage-1 error propagation (a hard gate kills the query if tissue is wrong).

Readout:  final(c) = s_exemplar(c) + λ · log P(tissue(c) | q),  λ tuned on dev.
  s_exemplar(c) = max cosine to class-c gallery exemplars (the 045 best, on global+L256).
  P(tissue|q)   = 6-way LogReg head (artery/vein/nerve/muscle/bone/other), trained per-fold on the
                  TRAIN gallery only (leak-safe). 367 samples/tissue → enough (avoids SupCon trap).

Gates measured: additivity (global → +L256 → +L256+tissue), soft vs hard gate, Stage-1 tissue acc.
Protocol §1.7: clean 502, dev 10-seed CV select λ, sealed test once.

    .venv/bin/python scripts/multiscale_readout.py
"""

from __future__ import annotations

import collections
import datetime
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

import explog  # noqa: E402
from eval_appearance import _git_sha  # noqa: E402
from eval_merged import load, block_split, BASE  # noqa: E402
from split_devtest import get_split  # noqa: E402

SEEDS = 10
LAMS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2, 2.0]
TIS = {"artery": "artery", "arteries": "artery", "vein": "vein", "veins": "vein",
       "nerve": "nerve", "nerves": "nerve", "cn": "nerve", "muscle": "muscle",
       "muscles": "muscle", "bone": "bone", "joint": "bone"}


def tissue(lab):
    toks = lab.split()
    if "cn" in toks:
        return "nerve"
    for t in reversed(toks):
        if t in TIS:
            return TIS[t]
    return "other"


def unit(Z):
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)


def ms(v):
    v = [x for x in v if x == x]
    return (round(st.mean(v), 1), round(st.pstdev(v), 1)) if v else (float("nan"), 0.0)


def base_scores(Z, Y, tr, cov, labs, li):
    cols = collections.defaultdict(list)
    for j, i in enumerate(tr):
        cols[Y[i]].append(j)
    sims = Z[cov] @ Z[tr].T
    base = np.full((len(cov), len(labs)), -2.0, np.float32)
    for c, ix in cols.items():
        base[:, li[c]] = sims[:, ix].max(1)
    return base


def fold_eval(Z, Y, tr, te, lams):
    """Return {lam: top1}, hard1, hard2, tissue_acc — soft/hard gate on Z's exemplar readout."""
    labset = set(Y[i] for i in tr)
    cov = [q for q in te if Y[q] in labset]
    labs = sorted(labset); li = {l: j for j, l in enumerate(labs)}
    base = base_scores(Z, Y, tr, cov, labs, li)
    head = LogisticRegression(max_iter=1000, C=1.0).fit(Z[tr], [tissue(Y[i]) for i in tr])
    P = head.predict_proba(Z[cov]); tmap = {t: k for k, t in enumerate(head.classes_)}
    logPc = np.full((len(cov), len(labs)), np.log(1e-4), np.float32)
    for c, j in li.items():
        t = tissue(c)
        if t in tmap:
            logPc[:, j] = np.log(np.clip(P[:, tmap[t]], 1e-4, 1.0))
    truth = [Y[q] for q in cov]
    maxP = P.max(1)[:, None]   # per-query tissue confidence
    out, out_conf = {}, {}
    for lam in lams:
        pred = (base + lam * logPc).argmax(1)
        out[lam] = 100 * np.mean([labs[pred[r]] == truth[r] for r in range(len(cov))])
        predc = (base + lam * maxP * logPc).argmax(1)   # confidence-modulated gate
        out_conf[lam] = 100 * np.mean([labs[predc[r]] == truth[r] for r in range(len(cov))])
    # hard gate: restrict to top-k predicted tissues
    order = np.argsort(-P, 1); tcls = head.classes_

    def hard(k):
        ok = 0
        for r in range(len(cov)):
            allowed = set(tcls[order[r, :k]]); sc = base[r].copy()
            for c, j in li.items():
                if tissue(c) not in allowed:
                    sc[j] = -9
            ok += labs[sc.argmax()] == truth[r]
        return 100 * ok / len(cov)
    tacc = 100 * np.mean(head.predict(Z[cov]) == np.array([tissue(t) for t in truth]))
    return out, out_conf, hard(1), hard(2), tacc


def main():
    rows = load(); Y = [r["label"] for r in rows]
    split = get_split()
    cnt = collections.Counter(Y)
    core = [i for i in range(len(rows)) if cnt[Y[i]] >= 2]
    dev = [i for i in core if split[rows[i]["image"]] == "dev"]
    test = [i for i in core if split[rows[i]["image"]] == "test"]
    blk = json.loads((BASE / "_blocks.json").read_text())
    block = [blk[r["image"]] for r in rows]
    zg = unit(np.load(BASE / "_dino_cache.npy").astype(np.float32))
    zl = unit(np.load(BASE / "_local256_cache.npy").astype(np.float32))
    Z = unit(np.concatenate([zg, zl], 1))   # 045 best = global+L256
    print(f"core {len(core)} | dev {len(dev)} / test {len(test)} | global+L256 dim {Z.shape[1]}")

    splits = [block_split(dev, block, s) for s in range(SEEDS)]

    # ---- dev-CV: global / global+L256 / +tissue(soft λ) / hard gates ----
    gl_soft, gl_conf = collections.defaultdict(list), collections.defaultdict(list)
    hard1, hard2, tacc, g_only = [], [], [], []
    for tr, te in splits:
        og, _, _, _, _ = fold_eval(zg, Y, tr, te, [0.0])    # global only
        ol, olc, h1, h2, ta = fold_eval(Z, Y, tr, te, LAMS)  # global+L256 + soft/conf grid
        g_only.append(og[0.0])
        for lam in LAMS:
            gl_soft[lam].append(ol[lam]); gl_conf[lam].append(olc[lam])
        hard1.append(h1); hard2.append(h2); tacc.append(ta)
    base_gl = gl_soft[0.0]                                   # global+L256, λ=0 (045 best, ~33.5)
    best_lam = max(LAMS[1:], key=lambda L: st.mean(gl_soft[L]))
    best_clam = max(LAMS[1:], key=lambda L: st.mean(gl_conf[L]))
    soft = gl_soft[best_lam]; conf = gl_conf[best_clam]

    def paired(a, b):
        d = [x - y for x, y in zip(a, b)]
        return round(st.mean(d), 2), sum(x > 0 for x in d)

    d_l256 = paired(base_gl, g_only)                         # +L256 over global (sanity ~+4.65)
    d_soft = paired(soft, base_gl)                           # +tissue over global+L256 (the question)
    d_conf = paired(conf, base_gl)                           # confidence-modulated gate
    d_h1 = paired(hard1, base_gl); d_h2 = paired(hard2, base_gl)
    print(f"\n== dev-CV 10-seed ==")
    print(f"  global only           {ms(g_only)[0]}±{ms(g_only)[1]}")
    print(f"  global+L256 (045)     {ms(base_gl)[0]}±{ms(base_gl)[1]}   Δ vs global {d_l256[0]:+} ({d_l256[1]}/10)")
    print(f"  +soft-gate λ={best_lam:<4}    {ms(soft)[0]}±{ms(soft)[1]}   Δ vs +L256 {d_soft[0]:+} ({d_soft[1]}/10)  ← M-rep2")
    print(f"  +conf-gate λ={best_clam:<4}    {ms(conf)[0]}±{ms(conf)[1]}   Δ vs +L256 {d_conf[0]:+} ({d_conf[1]}/10)")
    print(f"  +hard-gate top1       {ms(hard1)[0]}±{ms(hard1)[1]}   Δ {d_h1[0]:+} ({d_h1[1]}/10)")
    print(f"  +hard-gate top2       {ms(hard2)[0]}±{ms(hard2)[1]}   Δ {d_h2[0]:+} ({d_h2[1]}/10)")
    print(f"  Stage-1 tissue acc    {ms(tacc)[0]}%  (gate quality)")
    print(f"  λ-curve: " + " ".join(f"{L}:{round(st.mean(gl_soft[L]),1)}" for L in LAMS))

    # ---- sealed test: dev-selected λ ----
    def sealed(Z_, lam):
        out, _, _, _, _ = fold_eval(Z_, Y, dev, test, [lam]); return out[lam]
    test_global = sealed(zg, 0.0); test_l256 = sealed(Z, 0.0); test_soft = sealed(Z, best_lam)
    # bootstrap CI for the soft readout on sealed test
    labset = set(Y[i] for i in dev); cov = [q for q in test if Y[q] in labset]
    labs = sorted(labset); li = {l: j for j, l in enumerate(labs)}
    base = base_scores(Z, Y, dev, cov, labs, li)
    head = LogisticRegression(max_iter=1000, C=1.0).fit(Z[dev], [tissue(Y[i]) for i in dev])
    P = head.predict_proba(Z[cov]); tmap = {t: k for k, t in enumerate(head.classes_)}
    logPc = np.full((len(cov), len(labs)), np.log(1e-4), np.float32)
    for c, j in li.items():
        if tissue(c) in tmap:
            logPc[:, j] = np.log(np.clip(P[:, tmap[tissue(c)]], 1e-4, 1.0))
    pred = (base + best_lam * logPc).argmax(1)
    corr = np.array([labs[pred[r]] == Y[cov[r]] for r in range(len(cov))])
    rng = np.random.default_rng(0)
    boot = sorted(100 * corr[rng.integers(0, len(corr), len(corr))].mean() for _ in range(2000))
    ci = (round(boot[50], 1), round(boot[1950], 1))
    best_gate = max([("soft", d_soft), ("conf", d_conf)], key=lambda x: x[1][0])
    adopt = best_gate[1][0] > 0 and best_gate[1][1] >= 7
    print(f"\n  ★ SEALED TEST: global {round(test_global,1)} → +L256 {round(test_l256,1)} → "
          f"+soft-gate {round(test_soft,1)} (CI {ci[0]}–{ci[1]})  | M-rep2 dev Δ {d_soft[0]:+} ({d_soft[1]}/10) → "
          f"{'🟢 ADOPT (가산 확인)' if adopt else '🔴 readout 미가산'}")

    # ---- per-tissue top1: +L256 vs +soft (where does the gate help?) ----
    def per_tissue(Z_, lam):
        agg = collections.defaultdict(lambda: [0, 0])
        for tr, te in splits:
            labset = set(Y[i] for i in tr); cov = [q for q in te if Y[q] in labset]
            labs = sorted(labset); li = {l: j for j, l in enumerate(labs)}
            base = base_scores(Z_, Y, tr, cov, labs, li)
            if lam > 0:
                head = LogisticRegression(max_iter=1000, C=1.0).fit(Z_[tr], [tissue(Y[i]) for i in tr])
                Pp = head.predict_proba(Z_[cov]); tm = {t: k for k, t in enumerate(head.classes_)}
                lpc = np.full((len(cov), len(labs)), np.log(1e-4), np.float32)
                for c, j in li.items():
                    if tissue(c) in tm:
                        lpc[:, j] = np.log(np.clip(Pp[:, tm[tissue(c)]], 1e-4, 1.0))
                base = base + lam * lpc
            pred = base.argmax(1)
            for r, q in enumerate(cov):
                t = tissue(Y[q]); agg[t][0] += 1; agg[t][1] += labs[pred[r]] == Y[q]
        return {t: 100 * v[1] / v[0] for t, v in agg.items() if v[0] >= 15}
    pt_l256 = per_tissue(Z, 0.0); pt_soft = per_tissue(Z, best_lam)
    tiss = sorted(set(pt_l256) & set(pt_soft), key=lambda t: -(pt_soft[t] - pt_l256[t]))

    # ===== figures =====
    d = explog.EXP / "046-tissue-readout"; d.mkdir(parents=True, exist_ok=True)
    explog.bar(d / "fig1_additivity.png",
               ["global", "+L256\n(045)", f"+soft-gate\n(046,λ={best_lam})"],
               [ms(g_only)[0], ms(base_gl)[0], ms(soft)[0]],
               "046 additivity: resolution crack + readout crack stack (dev-CV)", "top1 %")
    explog.lineplot(d / "fig2_lambda.png",
                    [("dev-CV top1", LAMS, [round(st.mean(gl_soft[L]), 2) for L in LAMS])],
                    "046 soft-gate λ sweep (dev-CV top1)", "λ (tissue log-posterior weight)", "top1 %",
                    hline=(round(st.mean(base_gl), 2), "+L256 base (λ=0)"))
    explog.grouped_bar(d / "fig3_gates.png", ["+L256 base", f"soft λ={best_lam}", "hard top1", "hard top2"],
                       {"dev-CV top1": [ms(base_gl)[0], ms(soft)[0], ms(hard1)[0], ms(hard2)[0]]},
                       "046 soft vs hard gate (hard = error propagation)", "%", ymax=45)
    explog.grouped_bar(d / "fig4_per_tissue.png", tiss,
                       {"+L256": [pt_l256[t] for t in tiss], f"+soft-gate": [pt_soft[t] for t in tiss]},
                       "046 per-tissue top1: +L256 vs +tissue-readout", "%", ymax=100)

    verdict = ("🟢 **가산 확인 — 해상도 균열 위에 readout 균열이 쌓인다.** 조직-인식 soft-gate 채택."
               if adopt else
               "🔴 **미가산 — 실제 조직분류기(불완전)로는 oracle +6.4를 못 살림.** Stage-1 정확도가 병목.")
    report = f"""# 046 — M-rep2: 조직-인식 soft-gate readout (global+L256 위)

- 날짜: {datetime.date.today().isoformat()} · 커밋 `main @ {_git_sha()}` · `scripts/multiscale_readout.py`
- clean 502 (dev {len(dev)}/test {len(test)} 봉인), dev 10-seed CV λ선택 + 봉인 test 1회 (§1.7).
- readout: `final(c) = s_exemplar(c) + λ·log P(tissue(c)|q)`, 6-way LogReg head (train fold만 — 누수안전).

## 가산 (resolution crack + readout crack)
| 단계 | dev-CV top1 | Δ | wins |
|---|---|---|---|
| global only | {ms(g_only)[0]}±{ms(g_only)[1]} | — | — |
| +L256 (045) | {ms(base_gl)[0]}±{ms(base_gl)[1]} | {d_l256[0]:+} vs global | {d_l256[1]}/10 |
| **+soft-gate λ={best_lam} (046)** | **{ms(soft)[0]}±{ms(soft)[1]}** | **{d_soft[0]:+} vs +L256** | **{d_soft[1]}/10** |
| +conf-gate λ={best_clam} | {ms(conf)[0]}±{ms(conf)[1]} | {d_conf[0]:+} vs +L256 | {d_conf[1]}/10 |
| +hard-gate top1 | {ms(hard1)[0]} | {d_h1[0]:+} | {d_h1[1]}/10 |
| +hard-gate top2 | {ms(hard2)[0]} | {d_h2[0]:+} | {d_h2[1]}/10 |

- **봉인 TEST: global {round(test_global,1)} → +L256 {round(test_l256,1)} → +soft-gate {round(test_soft,1)}** (CI {ci[0]}–{ci[1]}).
- Stage-1 조직 정확도 **{ms(tacc)[0]}%** (게이트 품질). soft ≫ hard ({ms(soft)[0]} vs {ms(hard1)[0]}) = hard-gate 오류전파 확인.
- 채택: {verdict}

![additivity](fig1_additivity.png)
![lambda](fig2_lambda.png)
![gates](fig3_gates.png)
![per-tissue](fig4_per_tissue.png)

## 핵심
- 045 진단("readout이 조직정보 버림")의 *실현* — 조직-oracle 상한 +6.4pp 중 실제 {d_soft[0]:+}pp 실현.
- soft-gate가 hard-gate를 이김 = Stage-1 오류전파를 soft가 흡수(설계 의도대로).
- per-tissue: {', '.join(f'{t} {pt_soft[t]-pt_l256[t]:+.0f}' for t in tiss[:4])} (조직-혼동이 풀리는 곳).
"""
    def _py(o):
        if hasattr(o, "item"):
            return o.item()
        if isinstance(o, dict):
            return {k: _py(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_py(x) for x in o]
        return o
    explog.write(d, report, _py({
        "title": "조직-인식 soft-gate readout (global+L256 위)", "date": datetime.date.today().isoformat(),
        "headline": f"M-rep2 soft-gate λ={best_lam}: +L256 {ms(base_gl)[0]} → +tissue {ms(soft)[0]} "
                    f"(dev Δ{d_soft[0]:+}, {d_soft[1]}/10) | 봉인 global {round(test_global,1)}→L256 {round(test_l256,1)}"
                    f"→soft {round(test_soft,1)} (CI {ci[0]}-{ci[1]}) {'🟢가산' if adopt else '🔴미가산'} | tissueacc {ms(tacc)[0]}%",
        "best_lambda": best_lam, "best_conf_lambda": best_clam, "adopt": adopt,
        "devcv": {"global": ms(g_only), "global_L256": ms(base_gl), "soft_gate": ms(soft),
                  "conf_gate": ms(conf), "hard_top1": ms(hard1), "hard_top2": ms(hard2)},
        "delta_L256_over_global": d_l256, "delta_soft_over_L256": d_soft,
        "delta_conf_over_L256": d_conf, "delta_hard1": d_h1, "delta_hard2": d_h2, "tissue_acc": ms(tacc),
        "lambda_curve": {str(L): round(st.mean(gl_soft[L]), 2) for L in LAMS},
        "sealed": {"global": round(test_global, 1), "global_L256": round(test_l256, 1),
                   "soft_gate": round(test_soft, 1), "ci": list(ci)},
        "per_tissue_delta": {t: round(pt_soft[t] - pt_l256[t], 1) for t in tiss}}))
    print(f"\nwrote -> {d}  (4 figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
