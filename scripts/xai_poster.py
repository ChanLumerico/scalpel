"""Experiment 058 — XAI poster: a 4x4 (16-panel) deep visual dissection of the BEST configuration.

Best config (exp 057): frozen DINOv2-vitb14 + σ40 GaussianPool + [global ⊕ L256] + exemplar 1-NN + CSLS.
This probes it from 16 angles / many viz types:
  Row 1 representation geometry | Row 2 pin mechanism (what it sees) | Row 3 retrieval/readout | Row 4 error/operating
Panels with cadaver imagery → the whole poster is saved as poster.private.png (gitignored, §3); the
script + metrics are committable. Run: .venv/bin/python scripts/xai_poster.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

from eval_merged import load, block_split, BASE  # noqa: E402
from split_devtest import get_split  # noqa: E402
from multiscale_local import crop_pad  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402

TIS = {"artery": "artery", "arteries": "artery", "vein": "vein", "veins": "vein", "nerve": "nerve",
       "nerves": "nerve", "cn": "nerve", "muscle": "muscle", "muscles": "muscle", "bone": "bone", "joint": "bone"}
TCOL = {"artery": "#d62728", "vein": "#1f77b4", "nerve": "#fdb813", "muscle": "#8c564b",
        "bone": "#7f7f7f", "other": "#2ca02c"}


def tissue(lab):
    t = lab.split()
    if "cn" in t:
        return "nerve"
    for w in reversed(t):
        if w in TIS:
            return TIS[w]
    return "other"


def unit(Z):
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)


def csls_scores(Zq, Zg, k=5):
    sqg = Zq @ Zg.T; gg = Zg @ Zg.T; np.fill_diagonal(gg, -9)
    rg = np.sort(gg, 1)[:, -k:].mean(1); rq = np.sort(sqg, 1)[:, -k:].mean(1)
    return 2 * sqg - rq[:, None] - rg[None, :]


def classmax(S, Y, tr, cov, labs, li):
    cols = collections.defaultdict(list)
    for j, i in enumerate(tr):
        cols[li[Y[i]]].append(j)
    sc = np.full((len(cov), len(labs)), -9.0, np.float32)
    for c, ix in cols.items():
        sc[:, c] = S[:, ix].max(1)
    return sc


def overlay(ax, img_rgb, heat, title, q=None):
    ax.imshow(img_rgb)
    h = cv2.resize(heat.astype(np.float32), (img_rgb.shape[1], img_rgb.shape[0]))
    ax.imshow(h, cmap="jet", alpha=0.45)
    if q is not None:
        ax.plot(q[0], q[1], "o", mfc="none", mec="white", mew=2, ms=14)
    ax.set_title(title, fontsize=11); ax.axis("off")


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size; g = cfg.backbone.grid_size
    rows = load(); Y = [r["label"] for r in rows]
    reg = {i: (rows[i].get("region") or "?") for i in range(len(rows))}
    split = get_split()
    cnt = collections.Counter(Y)
    core = [i for i in range(len(rows)) if cnt[Y[i]] >= 2]
    dev = [i for i in core if split[rows[i]["image"]] == "dev"]
    block = [json.loads((BASE / "_blocks.json").read_text())[r["image"]] for r in rows]
    zg = unit(np.load(BASE / "_dino_cache.npy").astype(np.float32))
    zl = unit(np.load(BASE / "_local256_cache.npy").astype(np.float32))
    Z = unit(np.concatenate([zg, zl], 1))   # the BEST embedding
    Ty = {i: tissue(Y[i]) for i in core}
    print(f"core {len(core)} | dev {len(dev)} | building 16-panel poster...")

    # ---- aggregate dev-CV predictions (plain + CSLS) for diagnostics ----
    splits = [block_split(dev, block, s) for s in range(10)]
    rec = []   # (true, pred_csls, rank_plain, rank_csls, tissue, region, margin, correct)
    confus = collections.Counter()
    kocc = np.zeros(len(core)); coremap = {ci: k for k, ci in enumerate(core)}
    for tr, te in splits:
        labset = set(Y[i] for i in tr); cov = [q for q in te if Y[q] in labset]
        if not cov:
            continue
        labs = sorted(labset); li = {l: j for j, l in enumerate(labs)}
        plain = classmax(Z[cov] @ Z[tr].T, Y, tr, cov, labs, li)
        cs = classmax(csls_scores(Z[cov], Z[tr]), Y, tr, cov, labs, li)
        # hubness: k-occurrence of gallery in queries' top-10
        knn = np.argsort(-(Z[cov] @ Z[tr].T), 1)[:, :10]
        for gi in knn.reshape(-1):
            kocc[coremap[tr[gi]]] += 1
        op = np.argsort(-plain, 1); oc = np.argsort(-cs, 1)
        for r, q in enumerate(cov):
            tl = Y[q]
            rp = list(op[r]).index(li[tl]) + 1; rc = list(oc[r]).index(li[tl]) + 1
            pred = labs[oc[r, 0]]
            srt = np.sort(cs[r])[::-1]; margin = float(srt[0] - srt[1])
            rec.append((tl, pred, rp, rc, Ty[q], reg[q], margin, int(pred == tl)))
            if pred != tl:
                confus[tuple(sorted((tl, pred)))] += 1

    acc = np.mean([r[7] for r in rec]) * 100

    # ---- t-SNE on dev core (for panels 1-3) ----
    dc = [i for i in dev]
    print("  t-SNE...")
    XY = TSNE(n_components=2, metric="cosine", init="pca", perplexity=30, random_state=0).fit_transform(Z[dc])
    dt = np.array([Ty[i] for i in dc]); dr = np.array([reg[i] for i in dc])

    # ---- model for cadaver panels ----
    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    pool = GaussianPool(cfg.point).to(device); centers = bb.patch_centers(device)
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def grid_of(idx):
        im = Image.open(BASE / rows[idx]["image"]).convert("RGB"); w, h = im.size
        arr = np.asarray(im.resize((S, S)), np.float32) / 255.0
        x = (torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device) - MEAN) / STD
        with torch.no_grad():
            grid, _ = bb(x)
        qx, qy = rows[idx]["q"][0] * S / w, rows[idx]["q"][1] * S / h
        return (arr * 255).astype(np.uint8), grid[0], (qx, qy), (w, h)

    # choose a clean sample (artery, decent resolution) for panels 5-7
    cand = [i for i in core if Ty[i] == "artery" and Image.open(BASE / rows[i]["image"]).size[0] >= 700]
    s_idx = cand[0] if cand else core[0]
    s_img, s_grid, s_q, s_wh = grid_of(s_idx)
    gp = s_grid.reshape(g * g, -1)
    qpatch = int(np.argmin(((centers.cpu().numpy() - np.array(s_q)) ** 2).sum(1)))
    sal = F.cosine_similarity(gp, gp[qpatch:qpatch + 1], dim=1).cpu().numpy().reshape(g, g)
    d2 = ((centers.cpu().numpy() - np.array(s_q)) ** 2).sum(1)
    gw = torch.softmax(torch.tensor(-d2 / (2 * 40.0 ** 2)), 0).numpy().reshape(g, g)

    def crop_disp(idx, sz=240):
        arr = np.asarray(Image.open(BASE / rows[idx]["image"]).convert("RGB"))
        return crop_pad(arr, rows[idx]["q"][0], rows[idx]["q"][1], sz)

    # correct & wrong retrieval examples (use last fold's gallery)
    tr, te = splits[0]
    labset = set(Y[i] for i in tr); cov = [q for q in te if Y[q] in labset]
    labs = sorted(labset); li = {l: j for j, l in enumerate(labs)}
    cs = csls_scores(Z[cov], Z[tr])
    sc = classmax(cs, Y, tr, cov, labs, li)
    pred0 = [labs[a] for a in sc.argmax(1)]
    correct_q = next((r for r, q in enumerate(cov) if pred0[r] == Y[q] and Ty[q] in ("artery", "vein")), 0)
    wrong_q = next((r for r, q in enumerate(cov) if pred0[r] != Y[q] and Ty[q] in ("artery", "vein")),
                   next(r for r, q in enumerate(cov) if pred0[r] != Y[q]))

    def topk_gallery(qi, k=4):
        sims = Z[cov[qi]] @ Z[tr].T
        return [tr[j] for j in np.argsort(-sims)[:k]]

    # ================= build 4x4 =================
    fig, A = plt.subplots(4, 4, figsize=(26, 26))
    fig.suptitle(f"SCALPEL — XAI dissection of the best config (vitb14 + σ40 + global⊕L256 + exemplar-1NN + CSLS)  |  "
                 f"dev-CV top1 {acc:.1f}%  ·  sealed 38.3  ·  502-way", fontsize=17, y=0.995)

    # P1 t-SNE by region
    ax = A[0, 0]
    topreg = [r for r, _ in collections.Counter(dr).most_common(8)]
    for r in topreg:
        m = dr == r; ax.scatter(XY[m, 0], XY[m, 1], s=8, label=r[:14], alpha=0.7, edgecolors="#222", linewidths=0.3)
    ax.legend(fontsize=6, markerscale=1.5, ncol=2); ax.set_title("1. DINO-space t-SNE — colored by REGION\n(organizes by region)", fontsize=11); ax.set_xticks([]); ax.set_yticks([])

    # P2 t-SNE by tissue
    ax = A[0, 1]
    for t in ["other", "muscle", "nerve", "artery", "vein", "bone"]:
        m = dt == t; ax.scatter(XY[m, 0], XY[m, 1], s=8, c=TCOL[t], label=t, alpha=0.7, edgecolors="#222", linewidths=0.3)
    ax.legend(fontsize=7, markerscale=1.5); ax.set_title("2. same t-SNE — colored by TISSUE\n(artery/vein/nerve ENTANGLED = the bottleneck)", fontsize=11); ax.set_xticks([]); ax.set_yticks([])

    # P3 KDE density artery vs vein vs nerve
    ax = A[0, 2]
    ax.scatter(XY[:, 0], XY[:, 1], s=3, c="#ddd")
    for t in ["artery", "vein", "nerve"]:
        m = dt == t
        if m.sum() > 10:
            try:
                kde = gaussian_kde(XY[m].T)
                xx, yy = np.mgrid[XY[:, 0].min():XY[:, 0].max():80j, XY[:, 1].min():XY[:, 1].max():80j]
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                ax.contour(xx, yy, zz, levels=4, colors=TCOL[t], linewidths=1.4)
            except Exception:
                pass
    ax.set_title("3. tissue density (KDE) — artery∩vein∩nerve\noverlap in the SAME subspace", fontsize=11); ax.set_xticks([]); ax.set_yticks([])

    # P4 class-centroid cosine heatmap (ordered by region)
    ax = A[0, 3]
    cls = sorted(set(Y[i] for i in dc), key=lambda c: collections.Counter(reg[i] for i in dc if Y[i] == c).most_common(1)[0][0])
    cents = unit(np.stack([Z[[i for i in dc if Y[i] == c]].mean(0) for c in cls]))
    Cm = cents @ cents.T
    im = ax.imshow(Cm, cmap="magma", vmin=0, vmax=1); ax.set_title("4. class-centroid cosine (ordered by region)\nblock structure = region dominates", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im, ax=ax, fraction=0.046)

    # P5 GaussianPool σ40 footprint  [cadaver]
    overlay(A[1, 0], s_img, gw, "5. GaussianPool σ40 footprint at the pin\n(what the embedding averages)", s_q)
    # P6 self-similarity saliency  [cadaver]
    overlay(A[1, 1], s_img, (sal - sal.min()) / (sal.max() - sal.min() + 1e-9), "6. self-similarity saliency from q-patch\n(DINO 'attends' to same-structure pixels)", s_q)
    # P7 global vs L256 crop  [cadaver]
    ax = A[1, 2]
    comp = np.concatenate([cv2.resize(s_img, (220, 220)),
                           cv2.resize(crop_pad(np.asarray(Image.open(BASE / rows[s_idx]['image']).convert('RGB')),
                                               rows[s_idx]['q'][0], rows[s_idx]['q'][1], 256), (220, 220))], 1)
    ax.imshow(comp); ax.axvline(220, color="w", lw=2); ax.set_title("7. global-518 (L) vs high-res L256 crop (R)\n045: the zoom adds the fine cue", fontsize=11); ax.axis("off")
    # P8 per-tissue: where L256 helps (chart)
    ax = A[1, 3]
    g_only = {}; gl = {}
    for t in ["artery", "vein", "nerve", "muscle", "other"]:
        pass
    # compute per-tissue top1 for global vs global+L256 across folds
    def per_tissue_top1(emb):
        agg = collections.defaultdict(lambda: [0, 0])
        for tr2, te2 in splits:
            ls = set(Y[i] for i in tr2); cov2 = [q for q in te2 if Y[q] in ls]
            lb = sorted(ls); lim = {l: j for j, l in enumerate(lb)}
            scm = classmax(emb[cov2] @ emb[tr2].T, Y, tr2, cov2, lb, lim)
            pr = scm.argmax(1)
            for r, q in enumerate(cov2):
                agg[Ty[q]][0] += 1; agg[Ty[q]][1] += lb[pr[r]] == Y[q]
        return {k: 100 * v[1] / v[0] for k, v in agg.items() if v[0] >= 15}
    pg = per_tissue_top1(zg); pgl = per_tissue_top1(Z)
    tk = [t for t in ["artery", "vein", "nerve", "muscle", "bone", "other"] if t in pg and t in pgl]
    xx = np.arange(len(tk))
    ax.bar(xx - 0.2, [pg[t] for t in tk], 0.4, label="global", color="#bbb")
    ax.bar(xx + 0.2, [pgl[t] for t in tk], 0.4, label="global+L256", color="#d62728")
    ax.set_xticks(xx); ax.set_xticklabels(tk, fontsize=8); ax.legend(fontsize=8)
    ax.set_title("8. per-tissue top1: global vs +L256\n(where the high-res crop pays off)", fontsize=11); ax.set_ylabel("%")

    # P9 correct retrieval montage  [cadaver]
    def montage(ax, qi, title):
        q_idx = cov[qi]; gal = topk_gallery(qi)
        tiles = [cv2.resize(crop_disp(q_idx), (150, 150))] + [cv2.resize(crop_disp(g_), (150, 150)) for g_ in gal]
        strip = np.concatenate(tiles, 1)
        ax.imshow(strip)
        for k in range(5):
            ax.axvline(k * 150, color="w", lw=1)
        lbls = [Y[q_idx]] + [Y[g_] for g_ in gal]
        for k, l in enumerate(lbls):
            ok = (k == 0) or (l == Y[q_idx])
            ax.text(k * 150 + 75, 165, ("Q: " if k == 0 else "") + l[:16], ha="center", fontsize=6.5,
                    color=("black" if k == 0 else ("#1a9850" if l == Y[q_idx] else "#d62728")))
        ax.set_title(title, fontsize=11); ax.axis("off")
    montage(A[2, 0], correct_q, "9. retrieval ✓ — query + 4 nearest exemplars\n(green=same class)")
    montage(A[2, 1], wrong_q, "10. retrieval ✗ — failure mode\n(red=wrong, nearest crosses region/tissue)")

    # P11 hubness histogram
    ax = A[2, 2]
    from scipy.stats import skew
    ax.hist(kocc, bins=40, color="#7b68ee", edgecolor="#333")
    ax.axvline(kocc.mean(), color="k", ls="--", lw=1)
    ax.set_title(f"11. gallery hubness (k=10 occurrence)\nskew={skew(kocc):.2f} → CSLS corrects hubs", fontsize=11)
    ax.set_xlabel("# times an exemplar is a top-10 neighbor"); ax.set_ylabel("count")

    # P12 CSLS rank shift
    ax = A[2, 3]
    shift = [rp - rc for _, _, rp, rc, *_ in rec]
    ax.hist(np.clip(shift, -8, 8), bins=np.arange(-8.5, 9.5), color="#2ca02c", edgecolor="#333")
    up = np.mean([s > 0 for s in shift]) * 100
    ax.set_title(f"12. CSLS rank shift of the TRUTH (plain−CSLS)\n>0 = CSLS moves truth up ({up:.0f}% of pins)", fontsize=11)
    ax.set_xlabel("rank improvement"); ax.axvline(0, color="k", lw=1)

    # P13 confusion graph
    ax = A[3, 0]
    top = confus.most_common(14)
    nodes = list({n for (a, b), _ in top for n in (a, b)})
    pos = {n: (np.cos(2 * np.pi * k / len(nodes)), np.sin(2 * np.pi * k / len(nodes))) for k, n in enumerate(nodes)}
    mx = max(c for _, c in top)
    for (a, b), c in top:
        x1, y1 = pos[a]; x2, y2 = pos[b]
        ax.plot([x1, x2], [y1, y2], "-", color=plt.cm.Reds(0.3 + 0.7 * c / mx), lw=0.5 + 3 * c / mx, alpha=0.8)
    for n, (x, y) in pos.items():
        ax.plot(x, y, "o", ms=6, color=TCOL[tissue(n)])
        ax.text(x * 1.12, y * 1.12, n[:14], fontsize=6, ha="center", va="center", rotation=np.degrees(np.arctan2(y, x)))
    ax.set_title("13. confusion graph (top-14 pairs)\nedge=∝confusions, node color=tissue", fontsize=11)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.axis("off")

    # P14 per-tissue + per-region accuracy
    ax = A[3, 1]
    byt = collections.defaultdict(lambda: [0, 0]); byr = collections.defaultdict(lambda: [0, 0])
    for tl, pr, rp, rc, t, rg_, m, ok in rec:
        byt[t][0] += 1; byt[t][1] += ok; byr[rg_][0] += 1; byr[rg_][1] += ok
    tt = sorted([t for t in byt if byt[t][0] >= 15], key=lambda t: byt[t][1] / byt[t][0])
    ax.barh(range(len(tt)), [100 * byt[t][1] / byt[t][0] for t in tt], color=[TCOL[t] for t in tt])
    ax.set_yticks(range(len(tt))); ax.set_yticklabels(tt, fontsize=8)
    ax.axvline(acc, color="k", ls="--", lw=1, label=f"overall {acc:.0f}%"); ax.legend(fontsize=7)
    ax.set_title("14. top1 by tissue (vein hardest — DX3)", fontsize=11); ax.set_xlabel("%")

    # P15 risk-coverage
    ax = A[3, 2]
    order = sorted(rec, key=lambda r: -r[6])
    covx, accy = [], []
    for frac in np.linspace(0.05, 1.0, 40):
        k = max(1, int(frac * len(order))); sub = order[:k]
        covx.append(100 * frac); accy.append(100 * np.mean([s[7] for s in sub]))
    ax.plot(covx, accy, "-", color="#1f77b4", lw=2)
    ax.axhline(acc, color="gray", ls=":", lw=1)
    sel30 = np.interp(30, covx, accy)
    ax.axvline(30, color="crimson", ls=":", lw=1)
    ax.set_title(f"15. risk–coverage (abstention, dev-CV)\nconfident 30% → {sel30:.0f}% selective top1 (sealed op. pt. ~52%)", fontsize=11)
    ax.set_xlabel("coverage %"); ax.set_ylabel("selective top1 %"); ax.grid(alpha=0.3)

    # P16 rank-of-truth distribution
    ax = A[3, 3]
    rk = collections.Counter(min(r[3], 6) for r in rec)
    bars = [rk.get(i, 0) for i in range(1, 7)]
    cols = ["#1a9850", "#91cf60", "#d9ef8b", "#fee08b", "#fc8d59", "#d73027"]
    ax.bar(["1✓", "2", "3", "4", "5", ">5"], bars, color=cols, edgecolor="#333")
    resc = 100 * sum(rk.get(i, 0) for i in [2, 3]) / max(1, sum(bars) - rk.get(1, 0))
    ax.set_title(f"16. rank of the TRUTH (CSLS)\nof misses, {resc:.0f}% have truth in top-3 (rescuable)", fontsize=11)
    ax.set_ylabel("count")

    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = Path("experiments/058-xai-poster"); out.mkdir(parents=True, exist_ok=True)
    p = out / "poster.private.png"
    fig.savefig(p, dpi=105); plt.close(fig)
    (out / "metrics.json").write_text(json.dumps({
        "title": "XAI 16-panel poster of best config", "devcv_top1": round(float(acc), 1),
        "n_eval": len(rec), "csls_moves_truth_up_pct": round(float(up), 1),
        "rescuable_top3_pct": round(float(resc), 1),
        "per_tissue_top1": {t: round(100 * byt[t][1] / byt[t][0], 1) for t in tt},
        "note": "poster.private.png contains cadaver imagery -> gitignored per CLAUDE.md §3"}, ensure_ascii=False, indent=2))
    print(f"wrote -> {p}  (16 panels, PRIVATE)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
