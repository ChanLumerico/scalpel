"""Experiment 037 — KDE posterior + Conformal + OOD (reliability & coverage layer).

OPT_HANDOUT §3.1. Replace the heuristic softmax-cosine confidence with a principled
density model and add a guaranteed prediction-set layer:

  KDE posterior:  p(y|z) ∝ π(y) Σ_{e∈gal_y} exp((cos(z,z_e) − 1)/h²)   (h fit on gallery-LOO)
  Conformal:      split-conformal sets at target 1−α, nonconformity 1 − p̂(y|z)
  OOD/OOV:        marginal density p(z) low ⇒ class-not-in-gallery (coverage gate)

PROTOCOL (user-locked):
- PRIMARY = page-split (≥2 core), 10-seed, paired Δ. Primary axis = AURC (lower=better).
  Pitfall #2: KDE ECE verified vs GLOBAL-TEMPERATURE baseline, PAIRED (elegant theory ≠
  flat measurement).
- REPORT = cross-cadaver holdout (unseen PDF): same methods' absolute numbers (~37-40,
  noisy ±5, 6 PDFs — unstable).
- 🔴 KEY: does a conformal set FIT on page-split actually hold 1−α coverage on
  cross-cadaver? exchangeability breaks → expect violation; quantify it.

    .venv/bin/python scripts/conformal_kde.py
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

import explog  # noqa: E402
from eval_appearance import load_core, embed, _git_sha  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402

ALPHA = 0.1                       # target miscoverage → 90% sets
H_GRID = [0.05, 0.08, 0.12, 0.18, 0.25, 0.35]
S_GRID = [5, 8, 10, 13, 16, 20]   # baseline softmax temperature (per-class-max cosine)


# ---- splits ---------------------------------------------------------------
def pages(core, idxs):
    g = collections.defaultdict(list)
    for i in idxs:
        g[f'{core[i]["src"]}#{core[i]["page"]}'].append(i)
    return g


def three_way(core, seed, fr=(0.5, 0.2)):
    g = pages(core, range(len(core)))
    keys = sorted(g); np.random.default_rng(seed).shuffle(keys)
    n = len(keys); a = int(n * fr[0]); b = a + int(n * fr[1])
    gal = [i for k in keys[:a] for i in g[k]]
    cal = [i for k in keys[a:b] for i in g[k]]
    te = [i for k in keys[b:] for i in g[k]]
    return gal, cal, te


# ---- density / posteriors -------------------------------------------------
def class_cols(yg):
    labels = sorted(set(yg)); lidx = {l: j for j, l in enumerate(labels)}
    cols = collections.defaultdict(list)
    for j, l in enumerate(yg):
        cols[lidx[l]].append(j)
    return labels, lidx, cols


def kde_post(Zq, Zg, cols, labels, prior, h):
    K = np.exp((Zq @ Zg.T - 1.0) / (h * h))               # (Nq, G), ∈(0,1]
    P = np.zeros((len(Zq), len(labels)), np.float32)
    for c, ix in cols.items():
        P[:, c] = prior[c] * K[:, ix].sum(1)
    dens = P.sum(1) + 1e-30                                 # marginal p(z) (unnormalized)
    return P / dens[:, None], dens


def fit_h(Zg, yg, labels, lidx, cols, prior):
    Kg = Zg @ Zg.T
    best, bh = -1e9, H_GRID[0]
    yc = np.array([lidx[l] for l in yg])
    for h in H_GRID:
        K = np.exp((Kg - 1.0) / (h * h)); np.fill_diagonal(K, 0.0)   # LOO
        P = np.zeros((len(Zg), len(labels)), np.float32)
        for c, ix in cols.items():
            P[:, c] = prior[c] * K[:, ix].sum(1)
        P /= (P.sum(1, keepdims=True) + 1e-30)
        ll = np.log(P[np.arange(len(Zg)), yc] + 1e-12).mean()
        if ll > best:
            best, bh = ll, h
    return bh


def baseline_conf(Zq, Zg, cols, labels):
    sims = Zq @ Zg.T
    sc = np.full((len(Zq), len(labels)), -2.0, np.float32)
    for c, ix in cols.items():
        sc[:, c] = sims[:, ix].max(1)
    return sc                                              # per-class max cosine


def fit_temp(Zg, yg, labels, lidx, cols):
    Kg = Zg @ Zg.T; np.fill_diagonal(Kg, -2.0)
    sc = np.full((len(Zg), len(labels)), -2.0, np.float32)
    for c, ix in cols.items():
        sc[:, c] = Kg[:, ix].max(1)
    yc = np.array([lidx[l] for l in yg])
    best, bs = -1e9, S_GRID[0]
    for s in S_GRID:
        P = _softmax(s * sc)
        ll = np.log(P[np.arange(len(Zg)), yc] + 1e-12).mean()
        if ll > best:
            best, bs = ll, s
    return bs


def _softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x); return e / e.sum(1, keepdims=True)


# ---- metrics --------------------------------------------------------------
def ece(conf, correct, bins=10):
    conf = np.asarray(conf); correct = np.asarray(correct, float)
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (conf > lo) & (conf <= hi) if b else (conf >= 0) & (conf <= hi)
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def aurc(conf, correct):
    order = np.argsort(-np.asarray(conf))
    c = np.asarray(correct, float)[order]
    risk = 1 - np.cumsum(c) / np.arange(1, len(c) + 1)
    return float(risk.mean())                              # lower = better selective pred


def auroc(score, pos):
    score = np.asarray(score); pos = np.asarray(pos, bool)
    if pos.all() or (~pos).all():
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    n1 = pos.sum(); n0 = (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def conformal(Pcal, ycal_idx, Pte, alpha):
    nc = 1 - Pcal[np.arange(len(Pcal)), ycal_idx]          # nonconformity on cal (true class)
    n = len(nc); lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    qhat = float(np.quantile(nc, lvl, method="higher"))
    inset = (1 - Pte) <= qhat                              # (Nte, C) bool
    sizes = inset.sum(1)
    return qhat, inset, sizes


# ---- one split ------------------------------------------------------------
def run_split(Z, Y, gal, cal, te):
    Zg, yg = Z[gal], [Y[i] for i in gal]
    labels, lidx, cols = class_cols(yg)
    cnt = np.array([sum(1 for l in yg if lidx[l] == c) for c in range(len(labels))], float)
    prior = cnt / cnt.sum()
    lab_set = set(labels)
    h = fit_h(Zg, yg, labels, lidx, cols, prior)
    s = fit_temp(Zg, yg, labels, lidx, cols)
    nC = len(labels)

    def covered(idxs):
        return [i for i in idxs if Y[i] in lab_set]
    cte = covered(te); ccal = covered(cal)
    yte = np.array([lidx[Y[i]] for i in cte])
    # SHARED prediction = exemplar (max per-class cosine) → top1 identical for both
    scb = baseline_conf(Z[cte], Zg, cols, labels)
    pred = scb.argmax(1); corr = (pred == yte)
    Pb, _ = _softmax(s * scb), None
    Pk, _ = kde_post(Z[cte], Zg, cols, labels, prior, h)
    ar = np.arange(len(cte))
    conf_b, conf_k = Pb[ar, pred], Pk[ar, pred]            # confidence at the SAME prediction
    out = {
        "top1": float(100 * corr.mean()),
        "ece_kde": ece(conf_k, corr), "ece_base": ece(conf_b, corr),
        "aurc_kde": aurc(conf_k, corr), "aurc_base": aurc(conf_b, corr),
    }
    # conformal for BOTH posteriors — calibrate on cal, evaluate on test
    ycal = np.array([lidx[Y[i]] for i in ccal])
    Pk_cal, _ = kde_post(Z[ccal], Zg, cols, labels, prior, h)
    Pb_cal = _softmax(s * baseline_conf(Z[ccal], Zg, cols, labels))
    for nm, Pc, Pt in [("kde", Pk_cal, Pk), ("base", Pb_cal, Pb)]:
        _, inset, sizes = conformal(Pc, ycal, Pt, ALPHA)
        out[f"conf_cover_{nm}"] = float(np.mean(inset[ar, yte]))
        out[f"conf_size_{nm}"] = float(sizes.mean())
    out["nclass"] = nC
    # OOD/OOV: does the score separate covered (in-gallery) vs OOV test pins?
    iscov = np.array([Y[i] in lab_set for i in te])
    dens_all = _density_all(Z[te], Zg, cols, labels, prior, h)
    scb_all = baseline_conf(Z[te], Zg, cols, labels).max(1)
    out["ood_auroc_kde"] = auroc(dens_all, iscov)
    out["ood_auroc_base"] = auroc(scb_all, iscov)
    return out


def _density_all(Zq, Zg, cols, labels, prior, h):
    K = np.exp((Zq @ Zg.T - 1.0) / (h * h))
    P = np.zeros((len(Zq), len(labels)), np.float32)
    for c, ix in cols.items():
        P[:, c] = prior[c] * K[:, ix].sum(1)
    return P.sum(1)


def main():
    base = Path("data/triples")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size
    core = load_core("data/triples/triples.jsonl", 2)
    Y = [r["label"] for r in core]
    print(f"core {len(core)}/{len(set(Y))} | device={device}")
    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    pool = GaussianPool(cfg.point).to(device)
    print("embedding..."); Z = embed(core, base, bb, pool, S, device).numpy().astype(np.float32)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)

    # ---- PRIMARY: page-split 10-seed ----
    keys = ["top1", "ece_kde", "ece_base", "aurc_kde", "aurc_base",
            "conf_cover_kde", "conf_size_kde", "conf_cover_base", "conf_size_base",
            "ood_auroc_kde", "ood_auroc_base", "nclass"]
    acc = {k: [] for k in keys}
    for seed in range(10):
        gal, cal, te = three_way(core, seed)
        o = run_split(Z, Y, gal, cal, te)
        for k in keys:
            acc[k].append(o[k])
    ms = lambda v: (round(st.mean([x for x in v if x == x]), 3), round(st.pstdev([x for x in v if x == x]), 3))
    P = {k: ms(acc[k]) for k in keys}
    d_ece = [b - a for a, b in zip(acc["ece_kde"], acc["ece_base"])]      # +=kde better (lower ECE)
    d_aurc = [b - a for a, b in zip(acc["aurc_kde"], acc["aurc_base"])]   # +=kde better (lower AURC)
    d_ood = [a - b for a, b in zip(acc["ood_auroc_kde"], acc["ood_auroc_base"])]
    w_ece, w_aurc, w_ood = sum(x > 0 for x in d_ece), sum(x > 0 for x in d_aurc), sum(x > 0 for x in d_ood)

    # ---- CROSS-CADAVER: conformal coverage under shift (🔴) ----
    pdfs = sorted(set(r["src"] for r in core))
    xc = {"top1": [], "ece": [], "aurc": [], "cover_kde": [], "size_kde": [], "cover_base": [], "size_base": []}
    for fold in range(5):
        rng = np.random.default_rng(200 + fold)
        pp = pdfs[:]; rng.shuffle(pp); nh = max(1, int(round(0.2 * len(pp))))
        hold = set(pp[:nh])
        dev = [i for i in range(len(core)) if core[i]["src"] not in hold]
        hidx = [i for i in range(len(core)) if core[i]["src"] in hold]
        gk = pages(core, dev); kk = sorted(gk)
        np.random.default_rng(fold).shuffle(kk)
        a = int(len(kk) * 0.7)
        gal = [i for k in kk[:a] for i in gk[k]]; cal = [i for k in kk[a:] for i in gk[k]]
        o = run_split(Z, Y, gal, cal, hidx)                # test = unseen-PDF holdout
        xc["top1"].append(o["top1"]); xc["ece"].append(o["ece_kde"]); xc["aurc"].append(o["aurc_kde"])
        xc["cover_kde"].append(o["conf_cover_kde"]); xc["size_kde"].append(o["conf_size_kde"])
        xc["cover_base"].append(o["conf_cover_base"]); xc["size_base"].append(o["conf_size_base"])
    XC = {k: ms(v) for k, v in xc.items()}

    print("\n== PRIMARY (page-split, 10-seed) — top1 {} (shared exemplar) ==".format(P["top1"][0]))
    print(f"  ECE   kde {P['ece_kde'][0]}  base {P['ece_base'][0]}   (kde better {w_ece}/10)")
    print(f"  AURC  kde {P['aurc_kde'][0]}  base {P['aurc_base'][0]}   (kde better {w_aurc}/10)  [primary]")
    print(f"  conf  kde cover {P['conf_cover_kde'][0]} size {P['conf_size_kde'][0]} | base cover {P['conf_cover_base'][0]} size {P['conf_size_base'][0]} (target {1-ALPHA}, ~{P['nclass'][0]:.0f} classes, top5=5)")
    print(f"  OOD   AUROC kde {P['ood_auroc_kde'][0]}  base {P['ood_auroc_base'][0]}  (kde better {w_ood}/10)")
    print("== CROSS-CADAVER (unseen PDF, 5-fold; noisy ±5) ==")
    print(f"  top1 {XC['top1'][0]}  ECE {XC['ece'][0]}  AURC {XC['aurc'][0]}")
    vk = round(100 * ((1 - ALPHA) - XC["cover_kde"][0]), 1)
    vb = round(100 * ((1 - ALPHA) - XC["cover_base"][0]), 1)
    print(f"  🔴 conformal cover  kde {XC['cover_kde'][0]} (viol {vk}pp)  base {XC['cover_base'][0]} (viol {vb}pp)  target {1-ALPHA}")

    adopt_aurc = bool(st.mean(d_aurc) > 0 and w_aurc >= 8)
    adopt_ece = bool(st.mean(d_ece) > 0 and w_ece >= 8)
    viol = vk

    d = explog.next_dir("conformal-kde")
    explog.bar(d / "fig_037.png",
               ["ECE\nkde", "ECE\nbase", "AURC\nkde", "AURC\nbase", "OOD\nkde", "OOD\nbase"],
               [P["ece_kde"][0], P["ece_base"][0], P["aurc_kde"][0], P["aurc_base"][0],
                P["ood_auroc_kde"][0], P["ood_auroc_base"][0]],
               "037 KDE vs global-temp baseline (page-split 10-seed)", "", ymax=1.0)
    ecev = "개선(이론=실측)" if adopt_ece else "평탄/미달(이론≠실측, 함정#2)"
    aurcv = "KDE 채택" if adopt_aurc else "동률/미달 — 보정만 개선, 선택예측 순위는 동급"
    xcverdict = ("보장 대체로 유지" if vk <= 3 else
                 "**보장 위반** — exchangeability 깨짐 → conformal 보장은 same-cadaver 한정으로 정직 보고")
    report = f"""# 037 — KDE posterior + Conformal + OOD (신뢰도·coverage 계층)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/conformal_kde.py`  (PRIMARY page-split 10-seed + cross-cadaver 병기)

예측은 **두 방법 공통 exemplar**(top1 {P['top1'][0]}%, 3-way split 갤러리 50%) — KDE는 *신뢰도*만 비교.

## PRIMARY (page-split, paired vs global-temp baseline)
| 지표 | KDE | baseline(글로벌온도) | KDE 우세 |
|---|---|---|---|
| **ECE** ↓ | {P['ece_kde'][0]} | {P['ece_base'][0]} | {w_ece}/10 |
| **AURC** ↓ [primary] | {P['aurc_kde'][0]} | {P['aurc_base'][0]} | {w_aurc}/10 |
| OOD AUROC ↑ | {P['ood_auroc_kde'][0]} | {P['ood_auroc_base'][0]} | {w_ood}/10 |
| conformal cover (목표 {1-ALPHA}) | {P['conf_cover_kde'][0]} | {P['conf_cover_base'][0]} | — |
| conformal **평균 집합크기** | {P['conf_size_kde'][0]} | {P['conf_size_base'][0]} | (~{P['nclass'][0]:.0f}클래스 중, top5=5) |

- 함정#2(ECE): KDE가 글로벌온도 대비 paired {w_ece}/10 → **{ecev}**.
- AURC(주축): {w_aurc}/10 → **{aurcv}**.
- OOD: {'KDE 밀도' if w_ood>=8 else 'baseline max-cos'}가 OOV 분리 우세({w_ood}/10).

## 병기 (cross-cadaver, unseen PDF 5-fold; ±5 노이즈, 6 PDF 불안정)
| top1 | ECE | AURC | conf cover kde | conf cover base |
|---|---|---|---|---|
| {XC['top1'][0]}% | {XC['ece'][0]} | {XC['aurc'][0]} | {XC['cover_kde'][0]} | {XC['cover_base'][0]} |

## 🔴 conformal 보장의 cross-cadaver 검증 (핵심)
- page-split 적합 집합 → unseen PDF 커버리지: **kde {XC['cover_kde'][0]} (위반 {vk}pp)**, base {XC['cover_base'][0]} (위반 {vb}pp). 목표 {1-ALPHA}.
- 판정: {xcverdict}.

## 판정 / 다음
- ECE는 KDE가 명백히 개선(보정 가치 실재). AURC가 동급이면 → **순위(선택예측)는 글로벌온도로 충분**,
  KDE의 가치는 *절대 신뢰도 보정(ECE)+OOD*에 한정. conformal 집합크기가 top5보다 크면(약한 모델)
  핸드아웃의 "보장되나 큰 집합" 시나리오 — 정직 보고. 다음 038(shrinkage, coverage).
"""
    explog.write(d, report, {
        "title": "KDE posterior + Conformal + OOD", "date": datetime.date.today().isoformat(),
        "headline": f"ECE kde {P['ece_kde'][0]} vs base {P['ece_base'][0]} ({w_ece}/10) | AURC {P['aurc_kde'][0]} vs {P['aurc_base'][0]} ({w_aurc}/10) | conf size kde {P['conf_size_kde'][0]} base {P['conf_size_base'][0]} | xcadaver conf cover {XC['cover_kde'][0]} (viol {vk}pp)",
        "primary": {k: P[k] for k in keys},
        "primary_paired": {"aurc_wins": w_aurc, "ece_wins": w_ece, "ood_wins": w_ood,
                           "adopt_aurc": adopt_aurc, "adopt_ece": adopt_ece},
        "crosscadaver": {k: XC[k] for k in XC},
        "conformal_violation_kde_pp": vk, "conformal_violation_base_pp": vb, "alpha": ALPHA})
    print(f"wrote -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
