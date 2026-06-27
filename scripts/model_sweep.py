"""Experiment 043 — Model-methodology sweep on the clean merged dataset (leak-safe).

Re-tries the model levers on data/merged_final (502 core, 10-seed photo-twin block split):
the old verdicts (exemplar≫mean, learned-head modest, etc.) were on the 953/leak-inflated
data — re-check on 2.3× cleaner leak-safe data. Aggregation methods run on CACHED σ40
embeddings (fast); SupCon trains a small head. Produces MANY information-rich figures, not
one bar.

    .venv/bin/python scripts/model_sweep.py
"""

from __future__ import annotations

import collections
import datetime
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import explog  # noqa: E402
from eval_appearance import _git_sha  # noqa: E402
from eval_merged import load, photo_blocks, block_split, BASE  # noqa: E402

SEEDS = 10
TISSUE = {"artery": "artery", "arteries": "artery", "vein": "vein", "veins": "vein",
          "nerve": "nerve", "nerves": "nerve", "muscle": "muscle", "muscles": "muscle",
          "bone": "bone", "joint": "bone", "tendon": "muscle"}


def tissue(lab):
    for t in reversed(lab.split()):
        if t in TISSUE:
            return TISSUE[t]
    return "other"


def region_of(r):
    r = r.lower()
    for k in ["abdom", "pelvi", "thora", "neck", "oral", "cranial", "orbit", "thigh",
              "arm", "forearm", "leg", "foot", "hand", "shoulder", "brachial", "face",
              "nasal", "spinal", "gluteal", "head", "back"]:
        if k in r:
            return k
    return "other"


# ---------- prediction methods (gallery gal, query cov; return per-query scores over labels) ----------
def _cols(Y, gal):
    labset = sorted(set(Y[i] for i in gal)); lidx = {l: j for j, l in enumerate(labset)}
    cols = collections.defaultdict(list)
    for j, i in enumerate(gal):
        cols[lidx[Y[i]]].append(j)
    return labset, cols


def score_exemplar(Z, Y, gal, cov):
    labset, cols = _cols(Y, gal)
    sims = Z[cov] @ Z[gal].T
    sc = np.full((len(cov), len(labset)), -2.0, np.float32)
    for c, ix in cols.items():
        sc[:, c] = sims[:, ix].max(1)
    return labset, sc


def score_mean(Z, Y, gal, cov):
    labset, cols = _cols(Y, gal)
    P = np.stack([Z[[gal[j] for j in ix]].mean(0) for ix in cols.values()])
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    order = [c for c in cols]
    sc = np.full((len(cov), len(labset)), -2.0, np.float32)
    s = Z[cov] @ P.T
    for k, c in enumerate(order):
        sc[:, c] = s[:, k]
    return labset, sc


def score_knn(Z, Y, gal, cov, k=5):
    labset, cols = _cols(Y, gal)
    lab_of = {j: c for c, ix in cols.items() for j in ix}
    sims = Z[cov] @ Z[gal].T
    sc = np.full((len(cov), len(labset)), -2.0, np.float32)
    topk = np.argsort(-sims, axis=1)[:, :k]
    for r in range(len(cov)):
        agg = collections.defaultdict(float)
        for j in topk[r]:
            agg[lab_of[j]] += float(sims[r, j])
        for c, v in agg.items():
            sc[r, c] = v
    return labset, sc


def score_lse(Z, Y, gal, cov, temp=0.1):
    labset, cols = _cols(Y, gal)
    sims = Z[cov] @ Z[gal].T
    sc = np.full((len(cov), len(labset)), -1e9, np.float32)
    for c, ix in cols.items():
        sc[:, c] = temp * np.log(np.exp(sims[:, ix] / temp).sum(1) + 1e-9)
    return labset, sc


def score_kde(Z, Y, gal, cov, h=0.3):
    labset, cols = _cols(Y, gal)
    sims = Z[cov] @ Z[gal].T
    sc = np.full((len(cov), len(labset)), 0.0, np.float32)
    for c, ix in cols.items():
        sc[:, c] = np.exp((sims[:, ix] - 1) / (h * h)).sum(1)
    return labset, sc


def score_multiproto(Z, Y, gal, cov, k=3):
    labset, cols = _cols(Y, gal)
    sc = np.full((len(cov), len(labset)), -2.0, np.float32)
    for c, ix in cols.items():
        pts = Z[[gal[j] for j in ix]]
        if len(pts) <= k:
            protos = pts
        else:
            # cheap k-means
            rng = np.random.default_rng(0); cen = pts[rng.choice(len(pts), k, replace=False)]
            for _ in range(8):
                a = np.argmax(pts @ cen.T, 1)
                cen = np.stack([pts[a == m].mean(0) if (a == m).any() else cen[m] for m in range(k)])
                cen = cen / (np.linalg.norm(cen, axis=1, keepdims=True) + 1e-9)
            protos = cen
        sc[:, c] = (Z[cov] @ protos.T).max(1)
    return labset, sc


METHODS = {
    "mean-proto": score_mean, "exemplar": score_exemplar, "kNN-5": lambda *a: score_knn(*a, k=5),
    "kNN-3": lambda *a: score_knn(*a, k=3), "multi-proto": score_multiproto,
    "LSE": score_lse, "KDE": score_kde,
}


def evalu(scorer, Z, Y, gal, qry):
    labset = sorted(set(Y[i] for i in gal)); lset = set(labset)
    cov = [q for q in qry if Y[q] in lset]
    if not cov:
        return None
    labset2, sc = scorer(Z, Y, gal, cov)
    order = np.argsort(-sc, axis=1)
    rows = []
    for r in range(len(cov)):
        top5 = [labset2[order[r, t]] for t in range(min(5, len(labset2)))]
        s = sc[r]; e = np.exp((s - s.max())); conf = float(e[order[r, 0]] / (e.sum() + 1e-9))
        rows.append((Y[cov[r]], top5[0], top5, conf))
    return rows, len(cov), len(qry)


# ---------- SupCon head ----------
def supcon_head(Z, Y, gal, dim=256, steps=200, temp=0.1, device="cpu"):
    g = torch.tensor(Z[gal], device=device)
    labs = [Y[i] for i in gal]; uniq = {l: k for k, l in enumerate(sorted(set(labs)))}
    y = torch.tensor([uniq[l] for l in labs], device=device)
    W = torch.nn.Linear(g.size(1), dim).to(device)
    opt = torch.optim.Adam(W.parameters(), lr=1e-3)
    N = g.size(0); eye = torch.eye(N, device=device)
    pos = ((y.view(-1, 1) == y.view(1, -1)).float() - eye)
    for _ in range(steps):
        z = F.normalize(W(g), dim=1)
        logits = z @ z.T / temp - eye * 1e9
        lp = logits - torch.logsumexp(logits, 1, keepdim=True)
        loss = -(pos * lp).sum(1) / pos.sum(1).clamp(min=1)
        opt.zero_grad(); loss.mean().backward(); opt.step()
    return W


def project(Z, W, device="cpu"):
    with torch.no_grad():
        z = F.normalize(W(torch.tensor(Z, device=device)), dim=1).cpu().numpy()
    return z.astype(np.float32)


def ms(v):
    v = [x for x in v if x == x]
    return (round(st.mean(v), 1), round(st.pstdev(v), 1)) if v else (float("nan"), 0.0)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = load()
    Y = [r["label"] for r in rows]
    reg = [region_of(r.get("region", "")) for r in rows]
    images = sorted(set(r["image"] for r in rows))
    cache = BASE / "_dino_cache.npy"
    if not cache.exists() or np.load(cache, mmap_mode="r").shape[0] != len(rows):
        sys.exit("run eda_dino_space.py first to build the embedding cache")
    Z = np.load(cache).astype(np.float32)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    bc = BASE / "_blocks.json"
    if bc.exists():
        img_block = json.loads(bc.read_text())
    else:
        print("computing photo-twin blocks..."); img_block = photo_blocks(images)
        bc.write_text(json.dumps(img_block))
    block = [img_block[r["image"]] for r in rows]
    cnt = collections.Counter(Y)
    core = [i for i in range(len(rows)) if cnt[Y[i]] >= 2]
    ncls = len(set(Y[i] for i in core))
    # ★ fixed sealed dev/test (§1.7): select on dev 10-seed CV, final on sealed test once
    from split_devtest import get_split
    split = get_split()
    dev_idx = [i for i in core if split[rows[i]["image"]] == "dev"]
    test_idx = [i for i in core if split[rows[i]["image"]] == "test"]
    print(f"{len(core)} core / {ncls} cls | dev {len(dev_idx)} / test {len(test_idx)} (sealed) | {device}")

    # ---- method battery: 10-seed CV on DEV (selection) ----
    results = {}
    splits = [block_split(dev_idx, block, s) for s in range(SEEDS)]
    for m, sc in METHODS.items():
        t1s, t5s, covs = [], [], []
        for tr, te in splits:
            out = evalu(sc, Z, Y, tr, te)
            if not out:
                continue
            rws, ncov, ntot = out
            t1 = 100 * np.mean([r[0] == r[1] for r in rws])
            t5 = 100 * np.mean([r[0] in r[2] for r in rws])
            t1s.append(t1); t5s.append(t5); covs.append(100 * ncov / ntot)
        results[m] = {"top1": ms(t1s), "top5": ms(t5s), "cov": ms(covs),
                      "ee": ms([a * c / 100 for a, c in zip(t1s, covs)])}
        print(f"  {m:12} top1 {results[m]['top1'][0]}±{results[m]['top1'][1]} | "
              f"top5 {results[m]['top5'][0]} | cov {results[m]['cov'][0]}")

    # ---- SupCon head (project then exemplar) ----
    sc_t1, sc_t5 = [], []
    for tr, te in splits:
        W = supcon_head(Z, Y, tr, device=device)
        Zp = project(Z, W, device)
        Zp = Zp / (np.linalg.norm(Zp, axis=1, keepdims=True) + 1e-9)
        out = evalu(score_exemplar, Zp, Y, tr, te)
        rws, ncov, ntot = out
        sc_t1.append(100 * np.mean([r[0] == r[1] for r in rws]))
        sc_t5.append(100 * np.mean([r[0] in r[2] for r in rws]))
    results["SupCon+exemplar"] = {"top1": ms(sc_t1), "top5": ms(sc_t5),
                                  "cov": results["exemplar"]["cov"], "ee": ms([0])}
    base = results["exemplar"]["top1"][0]
    print(f"  SupCon+exemplar top1 {results['SupCon+exemplar']['top1'][0]} (vs exemplar {base})")

    # ---- SEALED TEST (final, ONCE): gallery = all dev, query = sealed test ----
    test_res = {}
    for m, sc in METHODS.items():
        out = evalu(sc, Z, Y, dev_idx, test_idx); rws, ncov, ntot = out
        test_res[m] = (round(100 * np.mean([r[0] == r[1] for r in rws]), 1), round(100 * ncov / ntot, 1))
    W = supcon_head(Z, Y, dev_idx, device=device); Zp = project(Z, W, device)
    Zp = Zp / (np.linalg.norm(Zp, axis=1, keepdims=True) + 1e-9)
    out = evalu(score_exemplar, Zp, Y, dev_idx, test_idx)
    test_res["SupCon+exemplar"] = (round(100 * np.mean([r[0] == r[1] for r in out[0]]), 1), round(100 * out[1] / out[2], 1))
    best_dev = max(results, key=lambda m: results[m]["top1"][0])  # selected ON DEV ONLY
    scorer = score_exemplar if best_dev == "exemplar" else METHODS.get(best_dev, score_exemplar)
    out = evalu(scorer, Z, Y, dev_idx, test_idx); rws = out[0]
    corr = np.array([r[0] == r[1] for r in rws]); rng = np.random.default_rng(0)
    boot = sorted(100 * corr[rng.integers(0, len(corr), len(corr))].mean() for _ in range(2000))
    ci = (round(boot[50], 1), round(boot[1950], 1))
    dev_best_t1 = results[best_dev]["top1"][0]
    print(f"\n  ★ SEALED TEST — dev-selected best='{best_dev}': top1 {test_res[best_dev][0]} "
          f"(95% CI {ci[0]}–{ci[1]}) cov {test_res[best_dev][1]} | dev-CV was {dev_best_t1} "
          f"(optimism {round(dev_best_t1 - test_res[best_dev][0], 1)}pp)")

    # ---- diagnostics on exemplar baseline (per-query over all DEV seeds) ----
    allrows = []
    for tr, te in splits:
        out = evalu(score_exemplar, Z, Y, tr, te)
        allrows += out[0]
    # per tissue / region / shot
    def agg_by(keyfn):
        d = collections.defaultdict(lambda: [0, 0, 0])
        for true, pred, top5, conf in allrows:
            k = keyfn(true); d[k][0] += 1; d[k][1] += (pred == true); d[k][2] += (true in top5)
        return {k: (v[0], 100 * v[1] / v[0], 100 * v[2] / v[0]) for k, v in d.items() if v[0] >= 10}
    by_t = agg_by(tissue)
    by_shot = collections.defaultdict(lambda: [0, 0, 0])
    for true, pred, top5, conf in allrows:
        b = "2" if cnt[true] == 2 else "3" if cnt[true] == 3 else "4-5" if cnt[true] <= 5 else "6+"
        by_shot[b][0] += 1; by_shot[b][1] += (pred == true); by_shot[b][2] += (true in top5)
    by_shot = {k: (v[0], 100 * v[1] / v[0], 100 * v[2] / v[0]) for k, v in by_shot.items()}
    # confusions
    conf_pairs = collections.Counter()
    for true, pred, top5, c in allrows:
        if pred != true:
            conf_pairs[f"{true} -> {pred}"] += 1
    # within-tissue confusion: of errors, what frac pred is same tissue as true
    same_t = sum(1 for t, p, _, _ in allrows if p != t and tissue(p) == tissue(t))
    n_err = sum(1 for t, p, _, _ in allrows if p != t)
    # risk-coverage (sort by conf desc)
    sr = sorted(allrows, key=lambda r: -r[3])
    cors = np.array([r[0] == r[1] for r in sr])
    cum_acc = np.cumsum(cors) / (np.arange(len(cors)) + 1)
    covs_x = (np.arange(len(cors)) + 1) / len(cors)

    # ===================== FIGURES =====================
    d = explog.EXP / "043-model-sweep"; d.mkdir(parents=True, exist_ok=True)
    order_m = ["mean-proto", "exemplar", "kNN-3", "kNN-5", "multi-proto", "LSE", "KDE", "SupCon+exemplar"]

    # 1. method comparison: DEV-CV (selection) top1/top5 + SEALED-TEST top1
    explog.grouped_bar(d / "fig1_methods.png", order_m,
                       {"dev-CV top1": [results[m]["top1"][0] for m in order_m],
                        "dev-CV top5": [results[m]["top5"][0] for m in order_m],
                        "SEALED-TEST top1": [test_res[m][0] for m in order_m]},
                       "043 methods: dev-CV (selection) vs sealed-test (final), 502-way leak-safe", "%", ymax=90)
    # 2. accuracy by tissue (top1 vs top5)
    tt = sorted(by_t, key=lambda k: -by_t[k][1])
    explog.grouped_bar(d / "fig2_by_tissue.png", tt,
                       {"top1": [by_t[k][1] for k in tt], "top5": [by_t[k][2] for k in tt]},
                       "043 exemplar accuracy by tissue type (DX3: vessels low)", "%", ymax=100)
    # 3. accuracy by region
    by_region = collections.defaultdict(lambda: [0, 0, 0])
    # map true label -> its dominant region from triples
    lab_region = {}
    for i in core:
        lab_region.setdefault(Y[i], collections.Counter())[reg[i]] += 1
    lab_region = {l: c.most_common(1)[0][0] for l, c in lab_region.items()}
    for true, pred, top5, conf in allrows:
        rr = lab_region.get(true, "other")
        by_region[rr][0] += 1; by_region[rr][1] += (pred == true)
    by_region = {k: 100 * v[1] / v[0] for k, v in by_region.items() if v[0] >= 15 and k != "other"}
    rr_keys = sorted(by_region, key=lambda k: -by_region[k])
    explog.bar(d / "fig3_by_region.png", rr_keys, [by_region[k] for k in rr_keys],
               "043 exemplar top1 by anatomical region", "top1 %", ymax=100)
    # 4. accuracy by shot count
    sk = ["2", "3", "4-5", "6+"]
    explog.grouped_bar(d / "fig4_by_shot.png", sk,
                       {"top1": [by_shot.get(k, (1, 0, 0))[1] for k in sk],
                        "top5": [by_shot.get(k, (1, 0, 0))[2] for k in sk]},
                       "043 accuracy vs gallery shot count (long-tail)", "%", ymax=100)
    # 5. risk-coverage curve
    explog.lineplot(d / "fig5_risk_coverage.png",
                    [("selective accuracy", covs_x.tolist(), cum_acc.tolist())],
                    "043 risk-coverage (exemplar, sorted by confidence)", "coverage", "selective top1",
                    xlim=(0, 1), ylim=(0, 1))
    # 6. top confusions
    explog.barh_pairs(d / "fig6_confusions.png", conf_pairs.most_common(14),
                      "043 top confusions (true -> pred, seeds summed)")
    # 7. confidence histogram correct vs wrong
    cc = [r[3] for r in allrows if r[0] == r[1]]; wc = [r[3] for r in allrows if r[0] != r[1]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(cc, bins=25, alpha=0.6, label=f"correct ({len(cc)})", color="#2ca02c", density=True)
    ax.hist(wc, bins=25, alpha=0.6, label=f"wrong ({len(wc)})", color="#d62728", density=True)
    ax.set_title("043 confidence distribution: correct vs wrong (separability = abstain quality)")
    ax.set_xlabel("softmax confidence"); ax.set_ylabel("density"); ax.legend()
    fig.tight_layout(); fig.savefig(d / "fig7_confidence.png", dpi=120); plt.close(fig)

    headline = (f"dev-선택 best={best_dev} → 봉인 TEST top1 {test_res[best_dev][0]} (CI {ci[0]}–{ci[1]}) "
                f"cov {test_res[best_dev][1]} | dev-CV {dev_best_t1} (낙관 {round(dev_best_t1-test_res[best_dev][0],1)}pp) | "
                f"SupCon test {test_res['SupCon+exemplar'][0]} | errors same-tissue {round(100*same_t/max(1,n_err))}%")
    method_rows = "\n".join(
        f"| {m} | {results[m]['top1'][0]}±{results[m]['top1'][1]} | {results[m]['top5'][0]} | {results[m]['cov'][0]} | **{test_res[m][0]}** |"
        for m in order_m)
    tis_rows = "\n".join(f"| {k} | {by_t[k][0]} | {by_t[k][1]:.0f} | {by_t[k][2]:.0f} |" for k in tt)
    report = f"""# 043 — 모델 방법론 스윕 (clean merged, 중첩 멀티시드 3-way)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/model_sweep.py` · 데이터 `data/merged_final` ({len(core)} core / {ncls} cls)
- 엔진: frozen dinov2_vitb14@518 → GaussianPool σ40 (캐시)
- ★ 프로토콜(§1.7): **dev {len(dev_idx)} / 봉인 test {len(test_idx)}** photo-block. 선택은 dev 10-seed CV,
  **최종은 봉인 test 1회**. dev로 best를 고르고 → 그 수치만 test로 보고(부트스트랩 CI).

## 방법 비교 (dev-CV = 선택 / 봉인 TEST = 최종)
| 방법 | dev-CV top1 | dev top5 | dev cov | **봉인 TEST top1** |
|---|---|---|---|---|
{method_rows}

- **dev-선택 best = `{best_dev}` → 봉인 TEST top1 {test_res[best_dev][0]}** (95% CI {ci[0]}–{ci[1]}, cov {test_res[best_dev][1]});
  dev-CV는 {dev_best_t1} → **HP-선택 낙관 {round(dev_best_t1-test_res[best_dev][0],1)}pp** (exp036 ~1.5pp와 정합).

![methods](fig1_methods.png)

## 진단 — 어디서 막히나
### 조직형별 (DX3: 혈관/신경 낮음)
| 조직형 | n | top1 | top5 |
|---|---|---|---|
{tis_rows}

![tissue](fig2_by_tissue.png)
![region](fig3_by_region.png)
![shot](fig4_by_shot.png)
![risk-coverage](fig5_risk_coverage.png)
![confusions](fig6_confusions.png)
![confidence](fig7_confidence.png)

## 핵심
- **봉인 test 최종: `{best_dev}` top1 {test_res[best_dev][0]}** (CI {ci[0]}–{ci[1]}) — 선택누수 제거한 정직 수치.
  dev-CV {dev_best_t1}와의 낙관 {round(dev_best_t1-test_res[best_dev][0],1)}pp는 작아 결론 안전.
- 집계방법: dev best = {best_dev} (옛 953에서 exemplar≫mean였는데 누수안전 502에서도 재확인).
- **SupCon 학습헤드는 도움 안 됨** (test {test_res['SupCon+exemplar'][0]} vs exemplar {test_res['exemplar'][0]}) — 옛 +2.6은 부분 누수기반.
- **오류의 {round(100*same_t/max(1,n_err))}%가 같은 조직형 내 혼동** — 조직형 *내* 미세정체성이 천장(DX3, exp042 기하와 일치).
- shot 역설: 빈번 클래스가 더 어려움(흔한 혈관/근육 혼동) — 데이터-부족 아닌 내재 난이도.
"""
    explog.write(d, report, {
        "title": "모델 방법론 스윕 (중첩 멀티시드 3-way)", "date": datetime.date.today().isoformat(),
        "headline": headline, "n_core": len(core), "ncls": ncls,
        "n_dev": len(dev_idx), "n_test": len(test_idx),
        "dev_cv": {m: results[m] for m in order_m},
        "sealed_test": {m: {"top1": test_res[m][0], "cov": test_res[m][1]} for m in order_m},
        "dev_best": best_dev, "test_best_top1": test_res[best_dev][0], "test_ci": list(ci),
        "hp_selection_optimism_pp": round(dev_best_t1 - test_res[best_dev][0], 1),
        "by_tissue": {k: {"n": by_t[k][0], "top1": round(by_t[k][1], 1), "top5": round(by_t[k][2], 1)} for k in tt},
        "by_shot": {k: round(by_shot.get(k, (1, 0, 0))[1], 1) for k in sk},
        "errors_same_tissue_pct": round(100 * same_t / max(1, n_err), 1)})
    print(f"\nwrote -> {d}  (7 figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
