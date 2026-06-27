"""Experiment 033 — thin-gated SAM pooling: the FINAL SAM verdict.

Converged design (from inspection + exp 032):
  - thin (artery/vein/nerve/duct): a distinct object SAM can mask → pool DINO tokens
    with weight = feather(SAM mask) × pin-Gaussian. If even SAM's smallest mask is
    loose (>6% of image, e.g. a fat vein blob), the mask is unreliable → fall back to
    the plain Gaussian for that item.
  - bulk (muscle/bone/brain/gland): uniform tissue, no internal boundary for SAM →
    plain pin-Gaussian (== baseline). SAM not used.

So 'treatment' differs from 'baseline' ONLY on thin items. Pre-registered decision
rule: ADOPT iff the thin subset is paired-positive in ≥7/10 seeds (bulk is identical
by construction; overall is reported only as a sanity check). exemplar 1-NN, 10 seed.

    .venv/bin/python scripts/sam_thingate.py
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
from PIL import Image  # noqa: E402
from segment_anything import SamPredictor, sam_model_registry  # noqa: E402

import explog  # noqa: E402
from eval_appearance import load_core, split_indices, _MEAN, _STD, _git_sha  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone  # noqa: E402

SAM_CKPT = ".cache/sam/sam_vit_b.pth"
THIN_KW = ("artery", "arteria", "vein", "vena", "nerve", "nervus", "vessel", "duct", "ductus")
TIGHT_MAX = 0.06          # thin mask must be ≤6% of image, else fall back to Gaussian


def coarse(label):
    s = label.lower()
    return "thin" if any(k in s for k in THIN_KW) else "bulk"


def _norm(v):
    return v / (np.linalg.norm(v) + 1e-9)


def gauss_w(centers, qS, sigma):
    d2 = ((centers - qS) ** 2).sum(1)
    w = np.exp(-d2 / (2 * sigma ** 2)); return w / (w.sum() + 1e-12)


def thin_mask(masks):
    area = masks.reshape(len(masks), -1).mean(1)
    j = int(np.argsort(area)[0])               # smallest
    return masks[j], float(area[j])


@torch.no_grad()
def build(core, base, bb, centers_t, S, sigma, device, predictor, ct):
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    centers = centers_t.cpu().numpy()
    by = collections.defaultdict(list)
    for i, r in enumerate(core):
        by[r["image"]].append(i)
    Zb = [None] * len(core); Zt = [None] * len(core)
    n_mask = n_fall = 0
    for n, (img, idxs) in enumerate(by.items(), 1):
        im = Image.open(base / img).convert("RGB"); w, h = im.size
        rgb = np.asarray(im)
        arr = np.asarray(im.resize((S, S)), np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        grid, _ = bb((x - mean) / std)
        g = grid.shape[1]
        tokens = grid.reshape(g * g, -1).cpu().numpy()
        has_thin = any(ct[i] == "thin" for i in idxs)
        if has_thin:
            predictor.set_image(rgb)
        for i in idxs:
            qx, qy = core[i]["q"]
            qS = np.array([qx * S / w, qy * S / h], np.float32)
            gw = gauss_w(centers, qS, sigma)
            Zb[i] = _norm((gw[:, None] * tokens).sum(0))           # baseline = plain Gaussian
            if ct[i] != "thin":
                Zt[i] = Zb[i]; continue
            masks, _, _ = predictor.predict(
                point_coords=np.array([[qx, qy]], np.float32),
                point_labels=np.array([1], np.int32), multimask_output=True)
            m, area = thin_mask(masks)
            if area > TIGHT_MAX:
                Zt[i] = Zb[i]; n_fall += 1; continue               # loose mask → fallback
            soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigmaX=max(2.0, 0.010 * w))
            cov = cv2.resize(soft, (g, g), interpolation=cv2.INTER_AREA).reshape(-1)
            wv = gw * cov
            if wv.sum() < 1e-8:
                Zt[i] = Zb[i]; n_fall += 1; continue
            wv = wv / wv.sum()
            Zt[i] = _norm((wv[:, None] * tokens).sum(0)); n_mask += 1
        if n % 60 == 0:
            print(f"   {n}/{len(by)} images  (masked {n_mask}, fallback {n_fall})", flush=True)
    print(f"  thin items masked={n_mask}  fallback(loose/empty)={n_fall}")
    return np.stack(Zb).astype(np.float32), np.stack(Zt).astype(np.float32), n_mask, n_fall


def exemplar_hits(Ztr, ytr, Zte_all, te, Y):
    labels = sorted(set(ytr)); lidx = {l: j for j, l in enumerate(labels)}
    cols = collections.defaultdict(list)
    for j, l in enumerate(ytr):
        cols[lidx[l]].append(j)
    cv = [k for k in te if Y[k] in lidx]
    sims = Zte_all[cv] @ Ztr.T
    sc = np.full((len(cv), len(labels)), -2.0, np.float32)
    for c, ix in cols.items():
        sc[:, c] = sims[:, ix].max(1)
    pred = np.argmax(sc, 1)
    hit = np.array([labels[pred[r]] == Y[cv[r]] for r in range(len(cv))])
    return np.array(cv), hit


def main():
    base = Path("data/triples")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size; sigma = cfg.point.gauss_sigma_px
    core = load_core("data/triples/triples.jsonl", 2)
    Y = [r["label"] for r in core]
    ct = np.array([coarse(l) for l in Y])
    print(f"core {len(core)}/{len(set(Y))} | thin {int((ct=='thin').sum())} bulk {int((ct=='bulk').sum())} | σ={sigma} device={device}")

    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    centers_t = bb.patch_centers(device)
    try:
        sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT).to(device)
    except Exception as e:  # noqa: BLE001
        print(f"  SAM {device} failed ({type(e).__name__}); cpu"); sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT).to("cpu")
    predictor = SamPredictor(sam)
    print("embedding (baseline Gaussian + thin-gated)...")
    Zb, Zt, n_mask, n_fall = build(core, base, bb, centers_t, S, sigma, device, predictor, ct)

    # accuracy, split by coarse type of the test item
    res = {"base": {"all": [], "thin": [], "bulk": []}, "treat": {"all": [], "thin": [], "bulk": []}}
    for seed in range(10):
        tr, te = split_indices(core, 0.3, seed)
        ytr = [Y[i] for i in tr]
        for name, Zall in [("base", Zb), ("treat", Zt)]:
            cv, hit = exemplar_hits(Zall[tr], ytr, Zall, te, Y)
            ctcv = ct[cv]
            res[name]["all"].append(100 * hit.mean())
            for t in ("thin", "bulk"):
                m = ctcv == t
                res[name][t].append(100 * hit[m].mean() if m.any() else float("nan"))

    ms = lambda v: (float(round(st.mean(v), 1)), float(round(st.pstdev(v), 1)))
    cnt = {"thin": int((ct == "thin").sum()), "bulk": int((ct == "bulk").sum()), "all": len(core)}
    rows = []
    for grp in ("thin", "bulk", "all"):
        b, t = res["base"][grp], res["treat"][grp]
        d = [a - c for a, c in zip(t, b)]
        wins = int(sum(x > 1e-9 for x in d))
        rows.append((grp, ms(t), ms(b), float(round(st.mean(d), 1)), wins))
    thin_row = rows[0]
    adopt = bool(thin_row[3] > 0 and thin_row[4] >= 7)
    verdict = ("채택 — thin에서 SAM 게이팅이 처음으로 양성" if adopt
               else "기각 — thin에서도 신호 없음, SAM 방향 종결")

    print("\n             feather×Gauss      pure Gauss      paired Δ (n/10)")
    for grp, t, b, dd, w in rows:
        print(f"  {grp:5s} ({cnt[grp]})   "
              f"{t[0]:5.1f}±{t[1]:<4.1f}      {b[0]:5.1f}±{b[1]:<4.1f}      {dd:+.1f} ({w}/10)")
    print(f"\n  thin masked={n_mask} fallback={n_fall} | PRE-REGISTERED: adopt iff thin Δ>0 & ≥7/10")
    print(f"  ==> {verdict}")

    d = Path("experiments/033-sam-thingate"); d.mkdir(parents=True, exist_ok=True)
    explog.bar(
        d / "fig_thingate.png", ["thin\ntreat", "thin\nbase", "bulk\ntreat", "bulk\nbase", "all\ntreat", "all\nbase"],
        [rows[0][1][0], rows[0][2][0], rows[1][1][0], rows[1][2][0], rows[2][1][0], rows[2][2][0]],
        "Thin-gated SAM pooling vs Gaussian (10-seed top1)", "%", ymax=100)
    tab = "\n".join(f"| {grp} ({cnt[grp]}) | "
                    f"{t[0]}±{t[1]}% | {b[0]}±{b[1]}% | {dd:+} ({w}/10) |" for grp, t, b, dd, w in rows)
    report = f"""# 033 — thin-게이팅 SAM 풀링 (SAM 최종 판정)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/sam_thingate.py`

## 설계 (검수 + exp 032에서 수렴)
- **thin**(동맥/정맥/신경/관): SAM이 잡는 독립 객체 → 가중 = feather(SAM 마스크)×핀-Gaussian.
  SAM 최소 마스크가 {int(TIGHT_MAX*100)}% 초과로 헐거우면(예: 굵은 정맥) 그 항목은 Gaussian fallback.
- **bulk**(근육/뼈/뇌/샘): 내부 경계 없음 → 순수 핀-Gaussian(=baseline). SAM 미사용.
- 따라서 treatment는 thin 항목에서만 baseline과 다름. **사전 등록 채택 기준: thin paired Δ>0 & ≥7/10.**

## 결과 (exemplar 1-NN, 10-seed, paired)
| 그룹 | feather×Gauss | pure Gauss | paired Δtop1 |
|---|---|---|---|
{tab}

(thin 마스크 적용 {n_mask}건 / 헐거워 fallback {n_fall}건)

![thingate](fig_thingate.png)

## 판정
- thin Δtop1 {thin_row[3]:+}%p ({thin_row[4]}/10) → **{verdict}**

## 해석
- thin에서 양성이면 → SAM이 처음으로 가치를 더한 것(혈관 추적 풀링이 인접 번짐 차단). 음성이면 →
  예쁜 마스크조차 임베딩을 못 바꿈(008/024/026 패턴 반복), **SAM 방향 종결**, 천장은 데이터.
"""
    explog.write(d, report, {
        "title": "thin-게이팅 SAM 풀링 (최종 판정)", "date": datetime.date.today().isoformat(),
        "headline": f"thin Δtop1 {thin_row[3]:+}({thin_row[4]}/10) → {verdict}",
        "groups": {grp: {"treat": t, "base": b, "dtop1": dd, "wins": w} for grp, t, b, dd, w in rows},
        "thin_masked": n_mask, "thin_fallback": n_fall, "adopt": adopt})
    print(f"wrote -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
