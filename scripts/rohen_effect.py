"""Experiment 062 / M-rohen0 STEP 1d — Rohen gallery-expansion effect (raw q, cheap directional probe).

User-agreed order: measure Δtop1 with RAW (unverified) q FIRST; if positive, hand-verify to clean up.
Build (I,q,y) triples from the 287 full-book candidates (vitb14 global ⊕ L256), normalize+match labels to
our classes, add the matchable ones to the dev gallery, and measure sealed Δtop1 (041-style expanded
gallery) with the best readout (global+L256 + CSLS). Rohen photos kept local (copyrighted).

    .venv/bin/python scripts/rohen_effect.py
"""

from __future__ import annotations

import collections
import datetime
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from pypdf import PdfReader  # noqa: E402

import explog  # noqa: E402
from eval_appearance import _MEAN, _STD, _git_sha  # noqa: E402
from eval_merged import load, BASE  # noqa: E402
from split_devtest import get_split  # noqa: E402
from multiscale_local import crop_pad  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402

PDF = "data/color_atlas_of_anatomy.pdf"


def unit(Z):
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)


def norm(s):
    s = s.lower().strip()
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"\b(muscle|musculus|m)\b", "muscle", s)
    s = re.sub(r"\b(nerve|nervus|n)\b", "nerve", s)
    s = re.sub(r"\b(artery|arteria|a)\b", "artery", s)
    s = re.sub(r"\b(vein|vena|v)\b", "vein", s)
    s = re.sub(r"[^a-z ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def csls_top1(Zq, Zg, Yg, truth, k=5):
    """top1 + coverage of test queries (truth) vs gallery (Zg,Yg) with CSLS readout."""
    labs = sorted(set(Yg)); li = {l: j for j, l in enumerate(labs)}
    cover = [i for i in range(len(truth)) if truth[i] in li]
    if not cover:
        return 0.0, 0.0, []
    sqg = Zq[cover] @ Zg.T; gg = Zg @ Zg.T; np.fill_diagonal(gg, -9)
    rg = np.sort(gg, 1)[:, -k:].mean(1); rq = np.sort(sqg, 1)[:, -k:].mean(1)
    Smat = 2 * sqg - rq[:, None] - rg[None, :]
    cols = collections.defaultdict(list)
    for j, y in enumerate(Yg):
        cols[li[y]].append(j)
    sc = np.full((len(cover), len(labs)), -9.0, np.float32)
    for c, ix in cols.items():
        sc[:, c] = Smat[:, ix].max(1)
    pred = sc.argmax(1)
    nn = sc.argmax(1)  # for source tracking we'd need argmax exemplar; skip
    correct = [labs[pred[r]] == truth[cover[r]] for r in range(len(cover))]
    return 100 * np.mean(correct), 100 * len(cover) / len(truth), cover


@torch.no_grad()
def embed_rohen(cands, bb, pool, centers, S, device):
    """global σ40@q ⊕ L256 crop, with vitb14 — same as our pipeline. Groups candidates by page."""
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    cq = torch.tensor([[259.0, 259.0]], device=device)
    reader = PdfReader(PDF)
    by = collections.defaultdict(list)
    for i, c in enumerate(cands):
        by[c["page"]].append(i)
    zg = [None] * len(cands); zl = [None] * len(cands)
    photo_cache = {}
    for npg, (pg, idxs) in enumerate(by.items(), 1):
        page = reader.pages[pg]
        imgs = sorted(page.images, key=lambda im: len(im.data), reverse=True)
        arr = np.asarray(imgs[0].image.convert("RGB")); H, W = arr.shape[:2]
        g = np.asarray(Image.fromarray(arr).resize((S, S)), np.float32) / 255.0
        x = (torch.from_numpy(g).permute(2, 0, 1).unsqueeze(0).to(device) - mean) / std
        grid, _ = bb(x)
        for i in idxs:
            qx, qy = cands[i]["q"]
            q = torch.tensor([[qx * S / W, qy * S / H]], device=device)
            zg[i] = F.normalize(pool(grid, centers, q)[0], dim=0).cpu().numpy()
            c = cv2.resize(crop_pad(arr, qx, qy, 256), (S, S)).astype(np.float32) / 255.0
            xl = (torch.from_numpy(c).permute(2, 0, 1).unsqueeze(0).to(device) - mean) / std
            gl, _ = bb(xl)
            zl[i] = F.normalize(pool(gl, centers, cq)[0], dim=0).cpu().numpy()
        if npg % 15 == 0:
            print(f"   rohen embed: {npg} pages")
    return unit(np.stack(zg)), unit(np.stack(zl))


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size
    rows = load(); Y = [r["label"] for r in rows]
    split = get_split(); cnt = collections.Counter(Y)
    core = [i for i in range(len(rows)) if cnt[Y[i]] >= 2]
    dev = [i for i in core if split[rows[i]["image"]] == "dev"]
    test = [i for i in core if split[rows[i]["image"]] == "test"]
    zg = unit(np.load(BASE / "_dino_cache.npy").astype(np.float32))
    zl = unit(np.load(BASE / "_local256_cache.npy").astype(np.float32))
    Zours = unit(np.concatenate([zg, zl], 1))

    our_norm = {}
    for c in set(Y):
        our_norm.setdefault(norm(c), c)
    cands = json.loads(Path("data/rohen/candidates.json").read_text())
    # label matching: normalized exact → our class; else a NEW class (its normalized name)
    matched = 0
    for c in cands:
        nn = norm(c["name"])
        c["label"] = our_norm.get(nn, "ROHEN::" + nn)
        c["matched"] = nn in our_norm
        matched += c["matched"]
    print(f"Rohen candidates {len(cands)} | matched to our class {matched} | dev {len(dev)} / test {len(test)}")

    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    pool = GaussianPool(cfg.point).to(device); centers = bb.patch_centers(device)
    print("embedding Rohen (vitb14 global+L256)...")
    rzg, rzl = embed_rohen(cands, bb, pool, centers, S, device)
    Zrohen = unit(np.concatenate([rzg, rzl], 1))
    Yrohen = [c["label"] for c in cands]

    truth = [Y[i] for i in test]
    # baseline: dev gallery only
    base_t1, base_cov, _ = csls_top1(Zours[test], Zours[dev], [Y[i] for i in dev], truth)
    # with Rohen: dev gallery + Rohen exemplars
    Zg = np.concatenate([Zours[dev], Zrohen]); Yg = [Y[i] for i in dev] + Yrohen
    roh_t1, roh_cov, _ = csls_top1(Zours[test], Zg, Yg, truth)
    # only-matched variant (add just the ones matching an existing class)
    mi = [i for i, c in enumerate(cands) if c["matched"]]
    Zgm = np.concatenate([Zours[dev], Zrohen[mi]]); Ygm = [Y[i] for i in dev] + [Yrohen[i] for i in mi]
    rohm_t1, rohm_cov, _ = csls_top1(Zours[test], Zgm, Ygm, truth)

    d_all = round(roh_t1 - base_t1, 1); dc_all = round(roh_cov - base_cov, 1)
    d_m = round(rohm_t1 - base_t1, 1); dc_m = round(rohm_cov - base_cov, 1)
    print(f"\n==== M-rohen0 STEP-1d effect (RAW q, sealed test) ====")
    print(f"  baseline (dev only)        top1 {base_t1:.1f}  cov {base_cov:.1f}")
    print(f"  + Rohen matched-only ({len(mi)})   top1 {rohm_t1:.1f} (Δ{d_m:+})  cov {rohm_cov:.1f} (Δ{dc_m:+})")
    print(f"  + Rohen all ({len(cands)})        top1 {roh_t1:.1f} (Δ{d_all:+})  cov {roh_cov:.1f} (Δ{dc_all:+})")
    verdict = ("🟢 Rohen 추가가 top1↑ → 도메인·라벨 매칭 작동, 검증 정제로 확대 가치 (041 재현)."
               if d_m > 1 or d_all > 1 else
               "🟡 raw q로는 약함 — q 노이즈/라벨 매칭이 효과를 누르거나, 신규구조 위주. 검증 정제 후 재측정 가치."
               if d_m > -0.5 else
               "🔴 raw q Rohen 추가가 top1을 떨어뜨림 — q 노이즈가 해롭거나 도메인 미세갭. 검증 정제 필수.")
    print(f"\n  → {verdict}")

    d = explog.EXP / "062-rohen-effect"; d.mkdir(parents=True, exist_ok=True)
    explog.grouped_bar(d / "fig1.png", ["top1", "coverage"],
                       {"dev only (base)": [round(base_t1, 1), round(base_cov, 1)],
                        f"+Rohen matched ({len(mi)})": [round(rohm_t1, 1), round(rohm_cov, 1)],
                        f"+Rohen all ({len(cands)})": [round(roh_t1, 1), round(roh_cov, 1)]},
                       "062 Rohen gallery expansion (raw q, sealed test)", "%", ymax=85)
    report = f"""# 062 / M-rohen0 STEP-1d — Rohen gallery-expansion effect (RAW q, cheap probe)

- 날짜: {datetime.date.today().isoformat()} · 커밋 `main @ {_git_sha()}` · `scripts/rohen_effect.py`
- 사용자 합의 순서: **raw q로 Δtop1 먼저** → 효과 입증 시 손수 정제(틀린 q 제거) → 재측정.
- 287 후보를 vitb14 global+L256로 임베딩, 라벨 정규화 매칭({matched} 매칭), dev gallery에 추가, sealed test(CSLS).

## 결과 (sealed test, RAW q)
| gallery | top1 | Δ | coverage | Δ |
|---|---|---|---|---|
| dev only (baseline) | {base_t1:.1f} | — | {base_cov:.1f} | — |
| + Rohen matched-only ({len(mi)}) | {rohm_t1:.1f} | {d_m:+} | {rohm_cov:.1f} | {dc_m:+} |
| + Rohen all ({len(cands)}) | {roh_t1:.1f} | {d_all:+} | {roh_cov:.1f} | {dc_all:+} |

![fig](fig1.png)

## 판정
{verdict}

## 다음
- {'🟢 효과 양성 → 사용자 손수 검증(틀린 q 제거)으로 정제 → 깨끗한 데이터로 재측정·확대.' if (d_m>1 or d_all>1) else '검증 정제 후 재측정 (raw q 노이즈 제거 시 효과 드러날 수 있음).'}
"""
    explog.write(d, report, {
        "title": "Rohen gallery 확장 효과 (raw q)", "date": datetime.date.today().isoformat(),
        "headline": f"baseline {base_t1:.1f} → +Rohen matched {rohm_t1:.1f}(Δ{d_m:+}) / all {roh_t1:.1f}(Δ{d_all:+}) "
                    f"cov Δ{dc_all:+} (RAW q, n_match {len(mi)})",
        "baseline_top1": round(base_t1, 1), "baseline_cov": round(base_cov, 1),
        "rohen_matched_top1": round(rohm_t1, 1), "delta_matched": d_m,
        "rohen_all_top1": round(roh_t1, 1), "delta_all": d_all,
        "cov_delta_all": dc_all, "n_candidates": len(cands), "n_matched": len(mi)})
    print(f"\nwrote -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
