"""Experiment 038 — Cross-cadaver gap: decomposition (A3) → conditional cadaver-invariant
normalization (A1). PRIMARY axis = cross-cadaver (unseen-PDF), exceptionally.

M-opt0 found a ~6.5pp gap (page-split ~44 → unseen-PDF ~37-40): exemplar 1-NN exploits
same-cadaver appearance (stain/lighting/cut). Closing it raises the *deployment-honest*
number. But whether it is closeable depends on the gap's CAUSE, so decompose first.

A3 (first, decisive):
  A3-1  colour contribution — re-embed Reinhard colour-normalized images (per-cadaver
        colour removed toward a global LAB reference) and measure how much of the gap
        that recovers. ≥ half ⇒ colour is the culprit (A1 can help). < half ⇒ anatomical
        variation (a 953 limit) ⇒ PRE-REGISTERED STOP of A1.
  A3-2  same-cadaver matching — fraction of test pins whose nearest gallery exemplar is
        the SAME cadaver, and accuracy same- vs different-cadaver matched.

A1 (only meaningful if A3-1 passes): z' = z − λ(μ_cadaver − μ_global), μ_cadaver =
  CLASS-UNIFORM mean of that PDF's exemplars (★037 lesson: NO frequency weighting —
  it distorts ranking). λ∈{0,.3,.5,.7,1}. Applied to gallery AND query, renormalized.

Eval (★ this experiment's primary is cross-cadaver — A1 trades same-cadaver for unseen):
  PRIMARY = cross-cadaver unseen-PDF (5-fold, ±5 noisy).  REPORT = page-split (10-seed).
  ADOPT iff cross-cadaver Δtop1>0 AND ≥4/5 folds AND page-split loss < cross gain.

    .venv/bin/python scripts/cadaver_invariant.py
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
from eval_appearance import load_core, _MEAN, _STD, _git_sha  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402

LAMBDAS = [0.0, 0.3, 0.5, 0.7, 1.0]


def reinhard(rgb, ref_m, ref_s):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    m = lab.reshape(-1, 3).mean(0); s = lab.reshape(-1, 3).std(0) + 1e-6
    lab = (lab - m) / s * ref_s + ref_m
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


@torch.no_grad()
def embed(core, base, bb, pool, centers, S, device, ref=None):
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    by = collections.defaultdict(list)
    for i, r in enumerate(core):
        by[r["image"]].append(i)
    Z = [None] * len(core)
    for img, idxs in by.items():
        im = Image.open(base / img).convert("RGB"); w, h = im.size
        rgb = np.asarray(im.resize((S, S)), np.uint8)
        if ref is not None:
            rgb = reinhard(rgb, ref[0], ref[1])
        x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        grid, _ = bb((x - mean) / std)
        for i in idxs:
            qx, qy = core[i]["q"]
            q = torch.tensor([[qx * S / w, qy * S / h]], device=device)
            Z[i] = F.normalize(pool(grid, centers, q)[0], dim=0).cpu().numpy()
    return np.stack(Z).astype(np.float32)


def lab_ref(core, base, S):
    acc_m = np.zeros(3); acc_s = np.zeros(3); n = 0
    seen = set()
    for r in core:
        if r["image"] in seen:
            continue
        seen.add(r["image"])
        im = Image.open(base / r["image"]).convert("RGB").resize((S, S))
        lab = cv2.cvtColor(np.asarray(im, np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3)
        acc_m += lab.mean(0); acc_s += lab.std(0); n += 1
    return acc_m / n, acc_s / n


def _norm(Z):
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)


def class_uniform_mean(Z, idxs, Y):
    by_c = collections.defaultdict(list)
    for i in idxs:
        by_c[Y[i]].append(i)
    cms = [Z[v].mean(0) for v in by_c.values()]
    return np.mean(cms, 0) if cms else np.zeros(Z.shape[1])


def cadaver_center(Z, Y, src, lam, mu_global):
    out = Z.copy()
    by_src = collections.defaultdict(list)
    for i in range(len(Z)):
        by_src[src[i]].append(i)
    for p, idxs in by_src.items():
        mu = class_uniform_mean(Z, idxs, Y)
        out[idxs] = Z[idxs] - lam * (mu - mu_global)
    return _norm(out)


def exemplar_eval(Z, Y, gal, qry):
    yg = [Y[i] for i in gal]; labset = set(yg)
    labels = sorted(labset); lidx = {l: j for j, l in enumerate(labels)}
    cols = collections.defaultdict(list)
    for j, i in enumerate(gal):
        cols[lidx[Y[i]]].append(j)
    cov = [q for q in qry if Y[q] in labset]
    if not cov:
        return float("nan"), 0.0
    sims = Z[cov] @ Z[gal].T
    sc = np.full((len(cov), len(labels)), -2.0, np.float32)
    for c, ix in cols.items():
        sc[:, c] = sims[:, ix].max(1)
    pred = sc.argmax(1)
    top1 = float(100 * np.mean([labels[pred[r]] == Y[cov[r]] for r in range(len(cov))]))
    return top1, float(100 * len(cov) / len(qry))


def page_split(core, seed, frac=0.3):
    g = collections.defaultdict(list)
    for i in range(len(core)):
        g[f'{core[i]["src"]}#{core[i]["page"]}'].append(i)
    keys = sorted(g); np.random.default_rng(seed).shuffle(keys)
    nt = max(1, int(round(len(keys) * frac)))
    tk = set(keys[:nt])
    return [i for k in keys if k not in tk for i in g[k]], [i for k in keys if k in tk for i in g[k]]


def pagesplit_top1(Z, Y, core, seeds=10):
    return [exemplar_eval(Z, Y, *page_split(core, s))[0] for s in range(seeds)]


def crosscadaver(Z, Y, core, src, folds=5):
    pdfs = sorted(set(src)); t1, cov = [], []
    for f in range(folds):
        rng = np.random.default_rng(300 + f); pp = pdfs[:]; rng.shuffle(pp)
        nh = max(1, int(round(0.2 * len(pp)))); hold = set(pp[:nh])
        gal = [i for i in range(len(core)) if src[i] not in hold]
        qry = [i for i in range(len(core)) if src[i] in hold]
        a, c = exemplar_eval(Z, Y, gal, qry); t1.append(a); cov.append(c)
    return t1, cov


def same_cadaver_analysis(Z, Y, core, src, seeds=10):
    fs, asame, adiff = [], [], []
    for s in range(seeds):
        gal, qry = page_split(core, s)
        yg = set(Y[i] for i in gal)
        cov = [q for q in qry if Y[q] in yg]
        if not cov:
            continue
        sims = Z[cov] @ Z[gal].T
        nn = sims.argmax(1)
        same = np.array([src[gal[nn[r]]] == src[cov[r]] for r in range(len(cov))])
        corr = np.array([Y[gal[nn[r]]] == Y[cov[r]] for r in range(len(cov))])  # 1-NN correct
        fs.append(float(same.mean()))
        if same.any():
            asame.append(float(100 * corr[same].mean()))
        if (~same).any():
            adiff.append(float(100 * corr[~same].mean()))
    m = lambda v: round(st.mean(v), 1) if v else float("nan")
    return m(fs), m(asame), m(adiff)


def main():
    base = Path("data/triples")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size
    core = load_core("data/triples/triples.jsonl", 2)
    Y = [r["label"] for r in core]; src = [r["src"] for r in core]
    print(f"core {len(core)}/{len(set(Y))} | {len(set(src))} PDFs | device={device}")
    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    pool = GaussianPool(cfg.point).to(device); centers = bb.patch_centers(device)
    print("embedding (original)..."); Z = _norm(embed(core, base, bb, pool, centers, S, device))
    print("LAB ref + embedding (Reinhard color-normalized)...")
    ref = lab_ref(core, base, S)
    Zct = _norm(embed(core, base, bb, pool, centers, S, device, ref=ref))

    ms = lambda v: (round(st.mean([x for x in v if x == x]), 1), round(st.pstdev([x for x in v if x == x]), 1))
    # ---- A3-1 colour decomposition ----
    ps_base = ms(pagesplit_top1(Z, Y, core)); xc_base = ms(crosscadaver(Z, Y, core, src)[0])
    ps_ct = ms(pagesplit_top1(Zct, Y, core)); xc_ct = ms(crosscadaver(Zct, Y, core, src)[0])
    gap = round(ps_base[0] - xc_base[0], 1)
    color_recovery = round(xc_ct[0] - xc_base[0], 1)
    a3_pass = color_recovery >= 0.5 * gap and gap > 0
    # ---- A3-2 same-cadaver matching ----
    frac_same, acc_same, acc_diff = same_cadaver_analysis(Z, Y, core, src)

    print(f"\n== A3-1 colour: page-split {ps_base[0]} → cross-cadaver {xc_base[0]} (gap {gap}) ==")
    print(f"   Reinhard: page-split {ps_ct[0]} → cross-cadaver {xc_ct[0]} | colour recovery {color_recovery}pp "
          f"({'≥' if a3_pass else '<'} half gap) → A1 {'진행' if a3_pass else '사전 중단'}")
    print(f"== A3-2 same-cadaver: nearest-exemplar same-PDF {frac_same}% | acc same {acc_same} vs diff {acc_diff} ==")

    # ---- A1 cadaver-invariant normalization (run for completeness; adopt gated on A3) ----
    mu_global = class_uniform_mean(Z, list(range(len(core))), Y)
    rows = []
    for lam in LAMBDAS:
        Zn = Z if lam == 0 else cadaver_center(Z, Y, src, lam, mu_global)
        ps = pagesplit_top1(Zn, Y, core)
        xt1, xcov = crosscadaver(Zn, Y, core, src)
        rows.append((lam, ms(ps), ms(xt1), ms(xcov)))
    base_xc = rows[0][2][0]; base_ps = rows[0][1][0]
    best = max(rows[1:], key=lambda r: r[2][0])
    # paired folds vs λ=0 for the best λ
    Zb = cadaver_center(Z, Y, src, best[0], mu_global)
    xt_best, _ = crosscadaver(Zb, Y, core, src); xt_base, _ = crosscadaver(Z, Y, core, src)
    fold_wins = sum(b > a for a, b in zip(xt_base, xt_best))
    cross_gain = round(best[2][0] - base_xc, 1)
    page_loss = round(base_ps - best[1][0], 1)
    net = round(cross_gain - max(0, page_loss), 1)
    adopt = bool(a3_pass and cross_gain > 0 and fold_wins >= 4 and cross_gain > page_loss)

    print("\n== A1 cadaver-invariant (λ sweep) ==")
    for lam, ps, xt, xc in rows:
        print(f"  λ={lam}: page-split {ps[0]}±{ps[1]} | cross-cadaver {xt[0]}±{xt[1]} (cov {xc[0]})")
    print(f"  best λ={best[0]}: cross gain {cross_gain} ({fold_wins}/5 fold), page loss {page_loss}, net {net}")
    if not a3_pass:
        verdict = "🔴 A3 사전중단 (색조 기여 < 갭절반 = 해부변이, 953 한계) — A1 결과는 탐색용, 미채택"
    elif adopt:
        verdict = f"🟢 채택 — cross-cadaver +{cross_gain}({fold_wins}/5), 순이득 +{net} (배포 성능 첫 상승)"
    else:
        verdict = "🟡 A3 통과했으나 A1 cross-cadaver 이득 미달/page 손해 큼 — 재해석"
    print(f"  ==> {verdict}")

    d = explog.next_dir("cadaver-invariant")
    explog.bar(d / "fig_038.png", [f"λ{r[0]}\npage" for r in rows] + [f"λ{r[0]}\ncross" for r in rows],
               [r[1][0] for r in rows] + [r[2][0] for r in rows],
               "038 cadaver-invariant: page-split vs cross-cadaver top1", "%", ymax=70)
    tab = "\n".join(f"| {lam} | {ps[0]}±{ps[1]}% | {xt[0]}±{xt[1]}% | {xc[0]}% |" for lam, ps, xt, xc in rows)
    report = f"""# 038 — Cross-cadaver 갭 분해(A3) → 조건부 cadaver-invariant 정규화(A1)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/cadaver_invariant.py`  (★PRIMARY = cross-cadaver)

## A3-1 — 색조 기여 분해 (먼저, 결정적)
- 원본: page-split **{ps_base[0]}** → cross-cadaver **{xc_base[0]}** (갭 **{gap}pp**)
- Reinhard 색정규화: page-split {ps_ct[0]} → cross-cadaver {xc_ct[0]}
- **색조 recovery = {color_recovery}pp** ({'≥' if a3_pass else '<'} 갭 절반) → A1 {'진행' if a3_pass else '**사전 중단**'}

## A3-2 — same-cadaver 매칭 (갭의 정체)
- 최근접 exemplar가 같은 PDF인 비율: **{frac_same}%**
- 1-NN 정확도: same-cadaver 매칭 **{acc_same}%** vs diff-cadaver **{acc_diff}%** (차 = same-cadaver 이득)

## A1 — cadaver-invariant 정규화 (클래스균등 μ, λ 스윕)
| λ | page-split top1 | cross-cadaver top1 | cross cov |
|---|---|---|---|
{tab}

- 베스트 λ={best[0]}: cross-cadaver **+{cross_gain}pp** ({fold_wins}/5 fold), page-split 손해 −{page_loss}pp, **순이득 {net}pp**

![038](fig_038.png)

## 판정 (사전등록)
{verdict}

## 해석
- frac_same·acc 차이가 same-cadaver 외형 leakage를 정량화. 색조 recovery가 갭을 설명하면 정규화로
  배포 성능을 올릴 수 있고, 못 하면 해부 변이(953 한계)다. cross-cadaver를 primary로 올려야만 보이는 레버.
"""
    explog.write(d, report, {
        "title": "Cross-cadaver 갭 분해 → cadaver-invariant", "date": datetime.date.today().isoformat(),
        "headline": f"갭 {gap}pp | 색조recovery {color_recovery}pp (A3 {'pass' if a3_pass else 'stop'}) | A1 best λ{best[0]} cross +{cross_gain}({fold_wins}/5) net {net} → {verdict[:2]}",
        "gap_pp": gap, "color_recovery_pp": color_recovery, "a3_pass": a3_pass,
        "same_cadaver": {"frac_same_pct": frac_same, "acc_same": acc_same, "acc_diff": acc_diff},
        "A1": {str(lam): {"page": ps, "cross": xt, "cross_cov": xc} for lam, ps, xt, xc in rows},
        "best_lambda": best[0], "cross_gain_pp": cross_gain, "page_loss_pp": page_loss, "net_pp": net,
        "fold_wins": fold_wins, "adopt": adopt})
    print(f"wrote -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
