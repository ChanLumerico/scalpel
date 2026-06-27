"""Experiment — class-aware adaptive SAM masking for point pooling.

DX4 killed SAM at a SINGLE setting. New variant (user's idea): make the mask SCALE
class-aware. A thin tubular structure (artery / vein / nerve / duct) wants a small,
tight mask that follows the vessel; a bulk tissue (muscle / gland / organ / bone)
wants a large mask covering the whole structure. SAM's multimask_output gives 3
masks at 3 scales per pin — we pick small-vs-large by the structure type and pool
DINO patch tokens over the chosen mask, instead of an isotropic Gaussian.

CHEAP PROBE FIRST: route by the ORACLE coarse type (true thin/bulk) — the best
possible class-aware masking. If even the oracle can't beat the Gaussian baseline,
the idea is dead and no router is worth building.

Policies (all exemplar 1-NN, 10-seed, PAIRED vs gauss):
  gauss        : GaussianPool σ40 (current baseline)
  sam-best     : SAM's own highest-IoU mask
  sam-small    : always the smallest of the 3 masks
  sam-large    : always the largest
  class-aware  : thin→small, bulk→large  (oracle routing — the user's idea)

Also reports a thin-vs-bulk accuracy breakdown (WHERE, if anywhere, it helps).

    .venv/bin/python scripts/sam_classaware.py
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
from segment_anything import SamPredictor, sam_model_registry  # noqa: E402

import explog  # noqa: E402
from eval_appearance import load_core, split_indices, _MEAN, _STD, _git_sha  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402

SAM_CKPT = ".cache/sam/sam_vit_b.pth"
THIN_KW = ("artery", "arteria", "arterial", "vein", "vena", "venous", "nerve",
           "nervus", "vessel", "vasa", "duct", "ductus", "trunk", "truncus", "ramus")
POLICIES = ["gauss", "sam-best", "sam-small", "sam-large", "class-aware"]


def coarse_type(label):
    s = label.lower()
    return "thin" if any(k in s for k in THIN_KW) else "bulk"


def _norm(v):
    return v / (np.linalg.norm(v) + 1e-9)


def mask_pool(tokens_g, mask_orig, g, seed_idx):
    """Masked-mean pool of (g*g, D) tokens over a mask; back off to the pin patch."""
    cov = cv2.resize(mask_orig.astype(np.float32), (g, g), interpolation=cv2.INTER_AREA)
    sel = cov.reshape(-1) > 0.5
    if not sel.any():
        sel = np.zeros(g * g, bool); sel[seed_idx] = True        # back off to pin patch
    return _norm(tokens_g[sel].mean(0))


@torch.no_grad()
def build(core, base, bb, pool, centers, S, device, predictor):
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    by = collections.defaultdict(list)
    for i, r in enumerate(core):
        by[r["image"]].append(i)
    Z = {p: [None] * len(core) for p in ["gauss", "best", "small", "large"]}
    for n, (img, idxs) in enumerate(by.items(), 1):
        im = Image.open(base / img).convert("RGB"); w, h = im.size
        rgb = np.asarray(im)                                      # HxWx3 uint8
        arr = np.asarray(im.resize((S, S)), np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        grid, _ = bb((x - mean) / std)                           # (1,g,g,D)
        g = grid.shape[1]
        tokens_g = grid.reshape(g * g, -1).cpu().numpy()
        predictor.set_image(rgb)
        for i in idxs:
            qx, qy = core[i]["q"]
            qS = torch.tensor([[qx * S / w, qy * S / h]], device=device)
            Z["gauss"][i] = _norm(F.normalize(pool(grid, centers, qS)[0], dim=0).cpu().numpy())
            seed_idx = int(((centers.cpu().numpy() - np.array([qx * S / w, qy * S / h])) ** 2).sum(1).argmin())
            masks, scores, _ = predictor.predict(
                point_coords=np.array([[qx, qy]], np.float32),
                point_labels=np.array([1], np.int32), multimask_output=True)
            areas = masks.reshape(masks.shape[0], -1).sum(1)
            order = np.argsort(areas)                             # small -> large
            Z["small"][i] = mask_pool(tokens_g, masks[order[0]], g, seed_idx)
            Z["large"][i] = mask_pool(tokens_g, masks[order[-1]], g, seed_idx)
            Z["best"][i] = mask_pool(tokens_g, masks[int(scores.argmax())], g, seed_idx)
        if n % 40 == 0:
            print(f"   {n}/{len(by)} images", flush=True)
    return {k: np.stack(v).astype(np.float32) for k, v in Z.items()}


def policy_vec(Z, ct, policy):
    if policy == "gauss":
        return Z["gauss"]
    if policy == "sam-best":
        return Z["best"]
    if policy == "sam-small":
        return Z["small"]
    if policy == "sam-large":
        return Z["large"]
    # class-aware: thin -> small, bulk -> large
    out = np.where((ct == "thin")[:, None], Z["small"], Z["large"])
    return out.astype(np.float32)


def exemplar(Ztr, ytr, Zte, yte_idx, Y):
    labels = sorted(set(ytr)); lidx = {l: j for j, l in enumerate(labels)}
    cols = collections.defaultdict(list)
    for j, l in enumerate(ytr):
        cols[lidx[l]].append(j)
    cv = [k for k in yte_idx if Y[k] in lidx]
    sims = Zte[cv] @ Ztr.T
    sc = np.full((len(cv), len(labels)), -2.0, np.float32)
    for c, ix in cols.items():
        sc[:, c] = sims[:, ix].max(1)
    o = np.argsort(-sc, axis=1)
    hit1 = np.array([labels[o[r, 0]] == Y[cv[r]] for r in range(len(cv))])
    hit5 = np.array([Y[cv[r]] in [labels[o[r, t]] for t in range(5)] for r in range(len(cv))])
    return cv, hit1, hit5


def main():
    base = Path("data/triples")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size
    core = load_core("data/triples/triples.jsonl", 2)
    Y = [r["label"] for r in core]
    ct = np.array([coarse_type(l) for l in Y])
    print(f"core {len(core)}/{len(set(Y))} | thin {int((ct=='thin').sum())} bulk {int((ct=='bulk').sum())} | device={device}")

    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    pool = GaussianPool(cfg.point).to(device); centers = bb.patch_centers(device)
    try:
        sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT).to(device)
    except Exception as e:  # noqa: BLE001
        print(f"  SAM on {device} failed ({type(e).__name__}); cpu"); sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT).to("cpu")
    predictor = SamPredictor(sam)
    print("embedding DINO + SAM masks (3 scales/pin)...")
    Z = build(core, base, bb, pool, centers, S, device, predictor)

    acc = {p: ([], []) for p in POLICIES}
    # per-coarse-type top1 for gauss vs class-aware
    split_acc = {"gauss": {"thin": [], "bulk": []}, "class-aware": {"thin": [], "bulk": []}}
    for seed in range(10):
        tr, te = split_indices(core, 0.3, seed)
        ytr = [Y[i] for i in tr]
        for p in POLICIES:
            Zp = policy_vec(Z, ct, p)
            cv, h1, h5 = exemplar(Zp[tr], ytr, Zp, te, Y)
            acc[p][0].append(100 * h1.mean()); acc[p][1].append(100 * h5.mean())
            if p in ("gauss", "class-aware"):
                ctcv = ct[cv]
                for t in ("thin", "bulk"):
                    m = ctcv == t
                    if m.any():
                        split_acc[p][t].append(100 * h1[m].mean())
        print(f"  seed {seed}: " + " ".join(f"{p} {acc[p][0][-1]:.0f}" for p in POLICIES))

    ms = lambda v: (round(st.mean(v), 1), round(st.pstdev(v), 1))
    base1 = acc["gauss"][0]
    rows = []
    for p in POLICIES:
        t1, t5 = acc[p]
        dd = [a - b for a, b in zip(t1, base1)]
        rows.append((p, ms(t1), ms(t5), round(st.mean(dd), 1), sum(x > 0 for x in dd)))
    best = max([r for r in rows if r[0] != "gauss"], key=lambda r: r[3])
    verdict = ("클래스-인지 마스킹이 향상 (paired)" if (best[3] > 0 and best[4] >= 8)
               else "마스킹 무효 — Gaussian이 최선 (DX4 재확인)")
    gt, gb = ms(split_acc["gauss"]["thin"]), ms(split_acc["gauss"]["bulk"])
    at, ab = ms(split_acc["class-aware"]["thin"]), ms(split_acc["class-aware"]["bulk"])
    print("\n== " + " | ".join(f"{p} {ms(acc[p][0])[0]}" for p in POLICIES) +
          f" | best {best[0]} Δ{best[3]:+}({best[4]}/10) -> {verdict} ==")
    print(f"   thin: gauss {gt[0]} vs class-aware {at[0]} | bulk: gauss {gb[0]} vs class-aware {ab[0]}")

    d = explog.next_dir("sam-classaware")
    explog.bar(d / "fig_sam.png", [r[0] for r in rows], [r[1][0] for r in rows],
               "Class-aware SAM masking: top1 (10-seed)", "%", ymax=100, errors=[r[1][1] for r in rows])
    tab = "\n".join(f"| {p} | {t1[0]}±{t1[1]}% | {t5[0]}% | {dd:+} ({w}/10) |" for p, t1, t5, dd, w in rows)
    report = f"""# 클래스-인지 적응 SAM 마스킹 (sam-classaware)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/sam_classaware.py`

## 목적
DX4는 SAM을 한 설정으로만 써서 기각. 이번엔 **마스크 스케일을 구조종류에 따라 가변**:
가는 구조(동맥/정맥/신경/관)→small 마스크, 큰 조직(근육/샘/뼈)→large 마스크. SAM multimask 3스케일
중 선택해 DINO 패치를 masked-pool. **싼 프로브 = 오라클 라우팅(참 thin/bulk)** — 상한조차 못 넘으면 기각.

## 결과 (exemplar 1-NN, 10-seed, paired vs gauss)
| 정책 | top1 | top5 | Δtop1 |
|---|---|---|---|
{tab}

![sam](fig_sam.png)

## 구조종류별 top1 (gauss vs class-aware)
| 종류 | gauss | class-aware |
|---|---|---|
| 가는 구조(thin) | {gt[0]}±{gt[1]}% | {at[0]}±{at[1]}% |
| 큰 조직(bulk) | {gb[0]}±{gb[1]}% | {ab[0]}±{ab[1]}% |

## 판정
- 베스트(비-gauss): **{best[0]}** Δtop1 {best[3]:+}%p ({best[4]}/10) → **{verdict}**

## 해석
- 오라클 라우팅조차 못 넘으면 → 마스크-풀링은 본질적으로 영역 평균이라 디테일을 뭉갬(008/015/020/024/030
  과 동일 서명). thin에서만 미세 이득이 보이면 → 가는 구조 한정 추적 마스크는 추후 가치 있을 수 있음.
"""
    explog.write(d, report, {
        "title": "클래스-인지 SAM 마스킹", "date": datetime.date.today().isoformat(),
        "headline": f"best {best[0]} Δtop1 {best[3]:+}({best[4]}/10) → {verdict} | thin g{gt[0]}/ca{at[0]} bulk g{gb[0]}/ca{ab[0]}",
        "policies": {p: {"top1": t1, "top5": t5, "dtop1": dd} for p, t1, t5, dd, w in rows},
        "split_thin": {"gauss": gt, "class_aware": at}, "split_bulk": {"gauss": gb, "class_aware": ab}})
    print(f"wrote -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
