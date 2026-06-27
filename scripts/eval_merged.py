"""Experiment 041 — Precise re-evaluation on the clean merged dataset (data/merged_final).

The data-expansion payoff, measured under the iron rules (§1): 10-seed mean±std, photo-
level leak-safe split, paired Δ, random baseline. The 49% QuizLink↔BlueLink photo overlap
makes a naive split LEAK (same photo as gallery-BlueLink and query-QuizLink), so the split
unit is a PHOTO-TWIN BLOCK (exact hash ∪ corr≥0.90) — twins never cross train/test. This
block grouping is for SPLIT SAFETY only (it does NOT merge labels/pins; data stays strict).

Comparisons:
  1. merged pooled        — new operating point (top1/top5/coverage/end-to-end) on 502-way
  2. quizlink-only        — original baseline (~215-way) reproduced, leak-safe
  3. ⭐ expanded gallery   — query = QuizLink test (fixed); gallery = QuizLink-train vs
                            QuizLink-train+BlueLink. Does BlueLink data improve the
                            deployment-target (QuizLink) recognition? (paired, leak-safe)
  4. coverage gain        — singleton→core promotion ⇒ fewer OOV (coverable %)

    .venv/bin/python scripts/eval_merged.py
"""

from __future__ import annotations

import collections
import datetime
import hashlib
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

import explog  # noqa: E402
from eval_appearance import _MEAN, _STD, _git_sha  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402

BASE = Path("data/merged_final")
SEEDS = 10


def load():
    import json
    return [json.loads(l) for l in open(BASE / "triples.jsonl", encoding="utf-8")]


# ---- photo-twin blocks (leak-safe split unit): exact hash ∪ corr>=0.90 ----
def photo_blocks(images):
    def dhash(p):
        a = np.asarray(Image.open(p).convert("L").resize((9, 8)), np.int16)
        bits = (a[:, 1:] > a[:, :-1]).flatten(); h = 0
        for b in bits:
            h = (h << 1) | int(b)
        return h
    H = {im: dhash(BASE / im) for im in images}
    sha = {im: hashlib.sha256((BASE / im).read_bytes()).hexdigest() for im in images}
    g64 = {im: np.asarray(Image.open(BASE / im).convert("L").resize((64, 64)), float).flatten() for im in images}
    pc = lambda x: bin(int(x)).count("1")
    par = {im: im for im in images}

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    fl = list(images)
    bysha = collections.defaultdict(list)
    for im in fl:
        bysha[sha[im]].append(im)
    for g in bysha.values():
        for x in g[1:]:
            par[find(x)] = find(g[0])
    for i in range(len(fl)):
        for j in range(i + 1, len(fl)):
            a, b = fl[i], fl[j]
            if find(a) != find(b) and pc(H[a] ^ H[b]) <= 12 and float(np.corrcoef(g64[a], g64[b])[0, 1]) >= 0.90:
                par[find(a)] = find(b)
    return {im: find(im) for im in images}


@torch.no_grad()
def embed(rows, bb, pool, centers, S, device):
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    by = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by[r["image"]].append(i)
    Z = [None] * len(rows)
    for n, (img, idxs) in enumerate(by.items(), 1):
        im = Image.open(BASE / img).convert("RGB"); w, h = im.size
        arr = np.asarray(im.resize((S, S)), np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        grid, _ = bb((x - mean) / std)
        for i in idxs:
            qx, qy = rows[i]["q"]
            q = torch.tensor([[qx * S / w, qy * S / h]], device=device)
            Z[i] = F.normalize(pool(grid, centers, q)[0], dim=0).cpu().numpy()
        if n % 100 == 0:
            print(f"   embedded {n} images")
    return (np.stack(Z) / (np.linalg.norm(np.stack(Z), axis=1, keepdims=True) + 1e-9)).astype(np.float32)


def block_split(idxs, block, seed, frac=0.3):
    blocks = sorted(set(block[i] for i in idxs))
    np.random.default_rng(seed).shuffle(blocks)
    nt = max(1, int(round(len(blocks) * frac)))
    test_blocks = set(blocks[:nt])
    tr = [i for i in idxs if block[i] not in test_blocks]
    te = [i for i in idxs if block[i] in test_blocks]
    return tr, te


def exemplar_eval(Z, Y, gal, qry):
    """class-max cosine. Returns (top1|covered, top5|covered, coverage%)."""
    labset = sorted(set(Y[i] for i in gal)); lidx = {l: j for j, l in enumerate(labset)}
    cols = collections.defaultdict(list)
    for j, i in enumerate(gal):
        cols[lidx[Y[i]]].append(j)
    cov = [q for q in qry if Y[q] in set(labset)]
    if not cov:
        return float("nan"), float("nan"), 0.0
    sims = Z[cov] @ Z[gal].T
    sc = np.full((len(cov), len(labset)), -2.0, np.float32)
    for c, ix in cols.items():
        sc[:, c] = sims[:, ix].max(1)
    order = np.argsort(-sc, axis=1)
    t1 = t5 = 0
    for r in range(len(cov)):
        true = Y[cov[r]]
        top5 = [labset[order[r, k]] for k in range(min(5, len(labset)))]
        t1 += top5[0] == true; t5 += true in top5
    return 100 * t1 / len(cov), 100 * t5 / len(cov), 100 * len(cov) / len(qry)


def ms(v):
    v = [x for x in v if x == x]
    return (round(st.mean(v), 1), round(st.pstdev(v), 1)) if v else (float("nan"), 0.0)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size
    rows = load()
    Y = [r["label"] for r in rows]
    src = [r["source"] for r in rows]
    images = sorted(set(r["image"] for r in rows))
    print(f"merged_final: {len(rows)} triples / {len(set(Y))} classes / {len(images)} photos | device={device}")
    print("computing photo-twin blocks (leak-safe split unit)...")
    img_block = photo_blocks(images)
    block = [img_block[r["image"]] for r in rows]
    print(f"  split-blocks: {len(set(block))} (vs {len(images)} photos — {len(images)-len(set(block))} twins grouped)")

    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    pool = GaussianPool(cfg.point).to(device); centers = bb.patch_centers(device)
    print("embedding (DINOv2 + GaussianPool σ40)...")
    Z = embed(rows, bb, pool, centers, S, device)

    # ★ CORE (>=2) filter for comparability with the established protocol (load_core min_count=2)
    cnt_all = collections.Counter(Y)
    cnt_ql = collections.Counter(Y[i] for i in range(len(rows)) if src[i] == "quizlink")
    all_idx = [i for i in range(len(rows)) if cnt_all[Y[i]] >= 2]                       # merged core
    ql_idx = [i for i in range(len(rows)) if src[i] == "quizlink" and cnt_ql[Y[i]] >= 2]  # quizlink core
    ncls_merged = len(set(Y[i] for i in all_idx)); ncls_ql = len(set(Y[i] for i in ql_idx))
    print(f"core(>=2): merged {len(all_idx)} triples/{ncls_merged} cls | quizlink {len(ql_idx)}/{ncls_ql} cls")

    # ---- 1. merged pooled (core) ----
    m1 = [exemplar_eval(Z, Y, *block_split(all_idx, block, s)) for s in range(SEEDS)]
    # ---- 2. quizlink-only (core) — should reproduce the established ~46-49 baseline ----
    m2 = [exemplar_eval(Z, Y, *block_split(ql_idx, block, s)) for s in range(SEEDS)]
    # ---- 3. expanded gallery: query = QuizLink-core test; gallery A=QL-core-train, B=+BlueLink ----
    ql_core_set = set(ql_idx)
    a_t1 = []; b_t1 = []; a_cov = []; b_cov = []
    for s in range(SEEDS):
        tr, te = block_split(all_idx, block, s)         # split merged-core (blocks leak-safe)
        q_te = [i for i in te if src[i] == "quizlink" and i in ql_core_set]  # query = QL-core test
        gA = [i for i in tr if src[i] == "quizlink"]     # gallery A: quizlink train
        gB = tr                                          # gallery B: quizlink+bluelink train
        ra = exemplar_eval(Z, Y, gA, q_te); rb = exemplar_eval(Z, Y, gB, q_te)
        a_t1.append(ra[0]); a_cov.append(ra[2]); b_t1.append(rb[0]); b_cov.append(rb[2])
    # paired Δ (gallery B - A) on top1 and coverage
    d_t1 = [b - a for a, b in zip(a_t1, b_t1)]; d_cov = [b - a for a, b in zip(a_cov, b_cov)]
    wins_t1 = sum(d > 0 for d in d_t1); wins_cov = sum(d > 0 for d in d_cov)

    def fmt(m):
        t1 = ms([x[0] for x in m]); t5 = ms([x[1] for x in m]); cv = ms([x[2] for x in m])
        ee = ms([x[0] * x[2] / 100 for x in m])
        return t1, t5, cv, ee

    f1 = fmt(m1); f2 = fmt(m2)
    print(f"\n== 1. MERGED pooled ({ncls_merged}-way, random {100/ncls_merged:.2f}%) ==")
    print(f"   top1 {f1[0][0]}±{f1[0][1]} | top5 {f1[1][0]}±{f1[1][1]} | cov {f1[2][0]}% | end-to-end {f1[3][0]}")
    print(f"== 2. QUIZLINK-only ({ncls_ql}-way, random {100/ncls_ql:.2f}%) ==")
    print(f"   top1 {f2[0][0]}±{f2[0][1]} | top5 {f2[1][0]}±{f2[1][1]} | cov {f2[2][0]}% | end-to-end {f2[3][0]}")
    print(f"== 3. ⭐ EXPANDED GALLERY (query=QuizLink test, paired) ==")
    at1 = ms(a_t1); bt1 = ms(b_t1); acov = ms(a_cov); bcov = ms(b_cov)
    print(f"   gallery A (QL only):   top1 {at1[0]}±{at1[1]} | cov {acov[0]}%")
    print(f"   gallery B (QL+BlueL):  top1 {bt1[0]}±{bt1[1]} | cov {bcov[0]}%")
    print(f"   Δtop1 {round(st.mean(d_t1),2)} ({wins_t1}/{SEEDS}) | Δcov {round(st.mean(d_cov),2)} ({wins_cov}/{SEEDS})")

    # ---- log ----
    d = explog.EXP / "041-merged-eval"; d.mkdir(parents=True, exist_ok=True)
    explog.bar(d / "fig_041.png",
               ["merged\ntop1", "merged\ncov", "QL\ntop1", "QL\ncov", "expGalA\ntop1", "expGalB\ntop1"],
               [f1[0][0], f1[2][0], f2[0][0], f2[2][0], at1[0], bt1[0]],
               "041 merged-dataset eval (10-seed, leak-safe blocks)", "%", ymax=100)
    headline = (f"merged {ncls_merged}-way top1 {f1[0][0]}±{f1[0][1]} cov {f1[2][0]} ee {f1[3][0]} | "
                f"QL {ncls_ql}-way top1 {f2[0][0]} cov {f2[2][0]} | "
                f"+BlueL gallery Δtop1 {round(st.mean(d_t1),1)}({wins_t1}/{SEEDS}) Δcov {round(st.mean(d_cov),1)}({wins_cov}/{SEEDS})")
    report = f"""# 041 — 정밀 재평가 (clean merged dataset)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/eval_merged.py` · 데이터: `data/merged_final` ({len(rows)} triples / {ncls_merged} classes / {len(images)} photos)
- 엔진: frozen dinov2_vitb14@518 → GaussianPool σ40 → exemplar class-max cosine, {SEEDS}-seed
- ★ 누수안전: photo-twin 블록(exact∪corr≥0.90)을 분할단위로 — QuizLink 49% BlueLink 중복 누수 차단

## 결과 ({SEEDS}-seed mean±std)
| 비교 | top1 | top5 | coverage | end-to-end |
|---|---|---|---|---|
| **1. Merged pooled** ({ncls_merged}-way) | **{f1[0][0]}±{f1[0][1]}** | {f1[1][0]}±{f1[1][1]} | {f1[2][0]}% | {f1[3][0]} |
| 2. QuizLink-only ({ncls_ql}-way) | {f2[0][0]}±{f2[0][1]} | {f2[1][0]}±{f2[1][1]} | {f2[2][0]}% | {f2[3][0]} |

random baseline: merged {round(100/ncls_merged,2)}% / QL {round(100/ncls_ql,2)}%

### 3. ⭐ 확장갤러리 효과 (쿼리=QuizLink test 고정, paired, 누수안전)
| 갤러리 | top1 | coverage |
|---|---|---|
| A: QuizLink-train only | {at1[0]}±{at1[1]} | {acov[0]}% |
| B: QuizLink-train + BlueLink | {bt1[0]}±{bt1[1]} | {bcov[0]}% |

- **Δtop1 {round(st.mean(d_t1),2)}pp ({wins_t1}/{SEEDS})** | **Δcoverage {round(st.mean(d_cov),2)}pp ({wins_cov}/{SEEDS})**

![041](fig_041.png)

## 해석
- merged는 {ncls_merged}-way(난도↑)임에도 random({round(100/ncls_merged,2)}%)의 {round(f1[0][0]/(100/ncls_merged))}배.
- 확장갤러리: BlueLink 추가가 QuizLink 인식 top1/coverage를 {'올림' if st.mean(d_t1)>0 else '못 올림'}(Δtop1 {round(st.mean(d_t1),2)}).
- coverage가 데이터 확장의 직접 레버(exp 038/039 가설) — merged cov {f1[2][0]}% vs QL {f2[2][0]}%.
"""
    explog.write(d, report, {
        "title": "정밀 재평가 (clean merged)", "date": datetime.date.today().isoformat(), "headline": headline,
        "merged": {"ncls": ncls_merged, "top1": f1[0], "top5": f1[1], "cov": f1[2], "ee": f1[3]},
        "quizlink": {"ncls": ncls_ql, "top1": f2[0], "top5": f2[1], "cov": f2[2], "ee": f2[3]},
        "expanded_gallery": {"A_top1": at1, "B_top1": bt1, "A_cov": acov, "B_cov": bcov,
                             "d_top1": round(st.mean(d_t1), 2), "wins_t1": wins_t1,
                             "d_cov": round(st.mean(d_cov), 2), "wins_cov": wins_cov},
        "blocks": len(set(block)), "photos": len(images)})
    print(f"\nwrote -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
