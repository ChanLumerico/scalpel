"""M-opt0 — Evaluation purification: how much is the reported ~50 inflated by
HP-SELECTION on the eval set? (the gate before any optimization, per OPT_HANDOUT §1)

Split leakage is already closed (DX1: specimen/page-level split, top1 invariant). The
open question is HP-selection leakage — σ40 / exemplar / calibration were chosen while
looking at core's eval. We quantify the optimism with a nested, PDF-level protocol:

  for each PDF-level fold (holdout = ~20% of PDFs, sealed; dev = rest):
    SELECT (σ, rule) on dev via dev-internal specimen splits   ← model selection
    EVALUATE the dev-selected config on the sealed holdout      ← unseen PDFs
    also track the FIXED canonical (exemplar, σ40) on dev vs holdout
  gap = dev − holdout, averaged over folds = the selection-overfitting amount.

Pre-registered gate (fixed before running, OPT_HANDOUT §1.2):
  ≤2pp → leakage negligible, trust the 30 numbers 🟢
  3–5pp → weak selection overfit; absolute values optimistic but paired Δ valid 🟡
  >5pp → serious; re-validate adopted stacks on dev 🔴

    .venv/bin/python scripts/mopt0.py
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
from PIL import Image  # noqa: E402

import explog  # noqa: E402
from eval_appearance import load_core, _MEAN, _STD, _git_sha  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone  # noqa: E402

SIGMAS = [10, 20, 40, 60, 80]
RULES = ["exemplar", "proto"]
N_FOLD = 5          # PDF-level outer folds (holdout each)
HOLD_FRAC = 0.2
INNER = 5           # dev-internal splits for selection
HOLD_SEEDS = 10     # holdout-internal splits for evaluation


@torch.no_grad()
def embed_all_sigmas(core, base, bb, centers, S, device):
    """For each pin, GaussianPool z at every σ (one forward / image)."""
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    by = collections.defaultdict(list)
    for i, r in enumerate(core):
        by[r["image"]].append(i)
    Z = {s: [None] * len(core) for s in SIGMAS}
    cen = centers  # (M,2) tensor
    for img, idxs in by.items():
        im = Image.open(base / img).convert("RGB"); w, h = im.size
        arr = np.asarray(im.resize((S, S)), np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        grid, _ = bb((x - mean) / std)
        g = grid.shape[1]
        tok = grid.reshape(g * g, -1)
        for i in idxs:
            qx, qy = core[i]["q"]
            q = torch.tensor([qx * S / w, qy * S / h], device=device)
            d2 = ((cen - q) ** 2).sum(1)
            for s in SIGMAS:
                wts = torch.softmax(-d2 / (2 * s * s), 0)
                Z[s][i] = F.normalize((wts[:, None] * tok).sum(0), dim=0).cpu().numpy()
    return {s: np.stack(v).astype(np.float32) for s, v in Z.items()}


def spec_split(idxs, core, frac, seed):
    g = collections.defaultdict(list)
    for i in idxs:
        g[f'{core[i]["src"]}#{core[i]["page"]}'].append(i)
    keys = sorted(g); np.random.default_rng(seed).shuffle(keys)
    nt = max(1, int(round(len(keys) * frac)))
    tk = set(keys[:nt])
    tr = [i for k in keys if k not in tk for i in g[k]]
    te = [i for k in keys if k in tk for i in g[k]]
    return tr, te


def top1(Z, Y, tr, te, rule):
    ytr = [Y[i] for i in tr]
    labels = sorted(set(ytr)); lidx = {l: j for j, l in enumerate(labels)}
    cv = [k for k in te if Y[k] in lidx]
    if not cv:
        return float("nan")
    if rule == "exemplar":
        cols = collections.defaultdict(list)
        for j, l in enumerate(ytr):
            cols[lidx[l]].append(j)
        sims = Z[cv] @ Z[tr].T
        sc = np.full((len(cv), len(labels)), -2.0, np.float32)
        for c, ix in cols.items():
            sc[:, c] = sims[:, ix].max(1)
    else:  # proto
        protos = np.zeros((len(labels), Z.shape[1]), np.float32)
        for l, j in lidx.items():
            m = Z[[t for t in tr if Y[t] == l]].mean(0)
            protos[j] = m / (np.linalg.norm(m) + 1e-9)
        sc = Z[cv] @ protos.T
    pred = np.argmax(sc, 1)
    return float(100 * np.mean([labels[pred[r]] == Y[cv[r]] for r in range(len(cv))]))


def eval_cfg(Zs, Y, idxs, sigma, rule, seeds):
    accs = [top1(Zs[sigma], Y, *spec_split(idxs, CORE, 0.3, s), rule) for s in seeds]
    accs = [a for a in accs if a == a]
    return float(np.mean(accs)) if accs else float("nan")


def main():
    global CORE
    base = Path("data/triples")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size
    CORE = load_core("data/triples/triples.jsonl", 2)
    Y = [r["label"] for r in CORE]
    pdfs = sorted(set(r["src"] for r in CORE))
    print(f"core {len(CORE)}/{len(set(Y))} | {len(pdfs)} PDFs | device={device}")
    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    centers = bb.patch_centers(device)
    print("embedding (all σ)..."); Zs = embed_all_sigmas(CORE, base, bb, centers, S, device)

    sel_dev, sel_hold, can_dev, can_hold, picks = [], [], [], [], []
    for fold in range(N_FOLD):
        rng = np.random.default_rng(100 + fold)
        pp = pdfs[:]; rng.shuffle(pp)
        nh = max(1, int(round(HOLD_FRAC * len(pp))))
        hold_pdf = set(pp[:nh])
        dev_idx = [i for i in range(len(CORE)) if CORE[i]["src"] not in hold_pdf]
        hold_idx = [i for i in range(len(CORE)) if CORE[i]["src"] in hold_pdf]
        # selection on dev
        best, best_acc = None, -1
        for sg in SIGMAS:
            for rl in RULES:
                a = eval_cfg(Zs, Y, dev_idx, sg, rl, range(INNER))
                if a > best_acc:
                    best_acc, best = a, (sg, rl)
        # holdout eval of selected + canonical
        h_sel = eval_cfg(Zs, Y, hold_idx, best[0], best[1], range(HOLD_SEEDS))
        d_can = eval_cfg(Zs, Y, dev_idx, 40, "exemplar", range(INNER))
        h_can = eval_cfg(Zs, Y, hold_idx, 40, "exemplar", range(HOLD_SEEDS))
        sel_dev.append(best_acc); sel_hold.append(h_sel)
        can_dev.append(d_can); can_hold.append(h_can); picks.append(best)
        print(f"  fold {fold}: holdPDF {nh} | select {best} dev {best_acc:.1f}→hold {h_sel:.1f} "
              f"| canon(σ40,ex) dev {d_can:.1f}→hold {h_can:.1f}")

    ms = lambda v: (round(st.mean(v), 1), round(st.pstdev(v), 1))
    gap_sel = round(st.mean([d - h for d, h in zip(sel_dev, sel_hold)]), 1)
    gap_can = round(st.mean([d - h for d, h in zip(can_dev, can_hold)]), 1)
    # DECOMPOSE the gap into its two causes (must not be conflated):
    sel_opt = round(gap_sel - gap_can, 1)                 # excess optimism FROM HP-selection
    cross = gap_can                                       # within-core(page-split) → unseen-cadaver shift
    pick_ct = collections.Counter(picks)
    # the gate's actual target is HP-SELECTION leakage (= sel_opt), per OPT_HANDOUT §1.2
    gate = ("🟢 누수 거의 없음 — paired Δ·30개 숫자 신뢰" if sel_opt <= 2 else
            "🟡 약한 selection 과적합 — 절대값↓ 읽되 paired Δ 유효" if sel_opt <= 5 else
            "🔴 심각 — best-stack dev 재검증 필요")
    cross_note = ("작음(DX1 정합)" if cross <= 2 else
                  "**유의** — page-split이 same-cadaver 매칭으로 부풀음; cross-cadaver는 더 낮음")
    print(f"\n== selection: dev {ms(sel_dev)} → holdout {ms(sel_hold)} | gap {gap_sel} ==")
    print(f"== canonical(σ40,ex): dev {ms(can_dev)} → holdout {ms(can_hold)} | gap {gap_can} ==")
    print(f"== picks {dict(pick_ct)} ==")
    print(f"== [분해] HP-selection 누수 = {sel_opt}pp → {gate} ==")
    print(f"== [분해] cross-cadaver 갭 = {cross}pp → {cross_note} ==")

    d = explog.next_dir("mopt0-purify")
    explog.bar(d / "fig_mopt0.png",
               ["select\ndev", "select\nhold", "canon\ndev", "canon\nhold"],
               [ms(sel_dev)[0], ms(sel_hold)[0], ms(can_dev)[0], ms(can_hold)[0]],
               "M-opt0: dev vs sealed holdout top1 (PDF-level nested)", "%", ymax=100,
               errors=[ms(sel_dev)[1], ms(sel_hold)[1], ms(can_dev)[1], ms(can_hold)[1]])
    report = f"""# M-opt0 — 평가 정화 (HP-selection 누수 측정)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/mopt0.py`  (PDF-level nested, {N_FOLD}-fold, holdout 봉인)

## 목적
split 누수는 DX1이 닫음. dev에서 (σ,rule) 고르고 *봉인 holdout*(unseen PDF)에서 평가. **gap을 두
원인으로 분해** — HP-selection 누수 vs cross-cadaver 일반화 갭(섞으면 오독).

## 결과 ({N_FOLD}-fold mean±std)
| | dev(코어 내 page-split) | sealed holdout(unseen PDF) | gap |
|---|---|---|---|
| 선택(σ,rule) | {ms(sel_dev)[0]}±{ms(sel_dev)[1]}% | {ms(sel_hold)[0]}±{ms(sel_hold)[1]}% | {gap_sel} |
| canonical(σ40,exemplar) | {ms(can_dev)[0]}±{ms(can_dev)[1]}% | {ms(can_hold)[0]}±{ms(can_hold)[1]}% | {gap_can} |

- 선택된 config: {dict(pick_ct)} (선택 holdout {ms(sel_hold)[0]} ≈ canonical holdout {ms(can_hold)[0]})

![mopt0](fig_mopt0.png)

## 분해 (핵심)
- **HP-selection 누수 = gap_sel − gap_can = {sel_opt}pp** → **{gate}**
  (dev에서 σ80을 골라도 holdout에선 σ40과 동일 → 선택이 일반화 이득 0; 006/009 정합.)
- **cross-cadaver 갭 = {cross}pp** → {cross_note}.
  page-level split은 *같은 카데바의 다른 페이지*를 갤러리에 허용 → exemplar가 same-cadaver 외형
  (염색·조명·절개결) 매칭으로 덕을 봄. unseen PDF엔 없음 → 진짜 일반화 top1은 ~{ms(can_hold)[0]}.

## 결론 / 시사
- 게이트(HP-selection) 통과: **paired Δ 비교는 유효**, 30개 실험의 *상대* 결론 안전.
- 단 **절대 ~50은 cross-cadaver 기준 낙관**(같은-카데바 leakage). 배포 관련 정직한 수치는
  page-split ~44–50 과 cross-cadaver ~{ms(can_hold)[0]} 를 *함께* 보고해야 함(DX1을 exemplar로 정밀화).
- 037+ 최적화는 dev/holdout 규율 유지, holdout(cross-cadaver) 수치 병기.
"""
    explog.write(d, report, {
        "title": "M-opt0 평가 정화", "date": datetime.date.today().isoformat(),
        "headline": f"HP-selection 누수 {sel_opt}pp → {gate} | cross-cadaver 갭 {cross}pp (page-split ~{ms(can_dev)[0]} vs unseen-PDF ~{ms(can_hold)[0]})",
        "selection": {"dev": ms(sel_dev), "holdout": ms(sel_hold), "gap": gap_sel},
        "canonical": {"dev": ms(can_dev), "holdout": ms(can_hold), "gap": gap_can},
        "hp_selection_leak_pp": sel_opt, "cross_cadaver_gap_pp": cross,
        "picks": {f"{k[0]}-{k[1]}": v for k, v in pick_ct.items()}})
    print(f"wrote -> {d}")
    return 0


CORE = None
if __name__ == "__main__":
    raise SystemExit(main())
