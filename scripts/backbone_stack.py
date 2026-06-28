"""Experiment 057 — M-bb1: stack our techniques (L256 + CSLS) on the best backbone (DINOv2-vitg14).

056 found DINOv2-vitg14 is the only reliable backbone lever (minimal sealed 36.8 vs vitb14 33.5), but
benefits only general quality (tissue_sep ~0). Question: does it STACK with the two validated technique
levers — high-res local (045: +2.6) and CSLS re-ranking (051: +2.2) — to beat the current best
(vitb14 + L256 + CSLS, sealed 38.3)? Or are the gains redundant (vitg14 already captures that signal)?

We embed the L256 local crop with vitg14 too (consistent backbone), then compare vitb14-stack vs
vitg14-stack head-to-head on the same splits. dev 10-seed CV + sealed test.

    .venv/bin/python scripts/backbone_stack.py
"""

from __future__ import annotations

import collections
import datetime
import json
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
from eval_appearance import _MEAN, _STD, _git_sha  # noqa: E402
from eval_merged import load, block_split, BASE  # noqa: E402
from split_devtest import get_split  # noqa: E402
from multiscale_local import crop_pad  # noqa: E402

SEEDS = 10


def unit(Z):
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)


def ms(v):
    v = [x for x in v if x == x]
    return (round(st.mean(v), 1), round(st.pstdev(v), 1)) if v else (float("nan"), 0.0)


@torch.no_grad()
def embed_vitg14_local(rows, core, device, crop=256, S=518):
    cache = BASE / "_bb_vitg14_L256.npy"
    if cache.exists() and np.load(cache, mmap_mode="r").shape[0] == len(rows):
        print("  cached vitg14-L256")
        return np.load(cache).astype(np.float32)
    print("  loading dinov2_vitg14 (re-download) ...")
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14"); m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    patch = 14; g = S // patch
    ys, xs = np.divmod(np.arange(g * g), g)
    centers = torch.tensor(np.stack([(xs + 0.5) * patch, (ys + 0.5) * patch], 1), dtype=torch.float32, device=device)
    cq = torch.tensor([S / 2.0, S / 2.0], device=device)
    sigma = 40.0
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    by = collections.defaultdict(list)
    for i in core:
        by[rows[i]["image"]].append(i)
    Z = [None] * len(rows)
    for n, (img, idxs) in enumerate(by.items(), 1):
        arr = np.asarray(Image.open(BASE / img).convert("RGB"))
        for i in idxs:
            qx, qy = rows[i]["q"]
            c = cv2.resize(crop_pad(arr, qx, qy, crop), (S, S)).astype(np.float32) / 255.0
            x = ((torch.from_numpy(c).permute(2, 0, 1).unsqueeze(0).to(device)) - mean) / std
            tok = m.forward_features(x)["x_norm_patchtokens"][0]
            d2 = ((centers - cq) ** 2).sum(1)
            wts = torch.softmax(-d2 / (2 * sigma ** 2), 0)
            Z[i] = F.normalize((wts[:, None] * tok).sum(0), dim=0).cpu().numpy()
        if n % 150 == 0:
            print(f"   vitg14-L256: {n} imgs")
    A = np.zeros((len(rows), len(Z[core[0]])), np.float32)
    for i in core:
        A[i] = Z[i]
    np.save(cache, A); del m
    if device == "mps":
        torch.mps.empty_cache()
    return A


def csls_top1(Z, Y, tr, te, k=5):
    labset = set(Y[i] for i in tr); cov = [q for q in te if Y[q] in labset]
    if not cov:
        return float("nan")
    labs = sorted(labset); li = {l: j for j, l in enumerate(labs)}
    cols = collections.defaultdict(list)
    for j, i in enumerate(tr):
        cols[li[Y[i]]].append(j)
    sqg = Z[cov] @ Z[tr].T; gg = Z[tr] @ Z[tr].T; np.fill_diagonal(gg, -9)
    rg = np.sort(gg, 1)[:, -k:].mean(1); rq = np.sort(sqg, 1)[:, -k:].mean(1)
    S = 2 * sqg - rq[:, None] - rg[None, :]
    sc = np.full((len(cov), len(labs)), -9.0, np.float32)
    for c, ix in cols.items():
        sc[:, c] = S[:, ix].max(1)
    pred = sc.argmax(1)
    return 100 * np.mean([labs[pred[r]] == Y[cov[r]] for r in range(len(cov))])


def plain_top1(Z, Y, tr, te):
    labset = set(Y[i] for i in tr); cov = [q for q in te if Y[q] in labset]
    if not cov:
        return float("nan")
    labs = sorted(labset); li = {l: j for j, l in enumerate(labs)}
    cols = collections.defaultdict(list)
    for j, i in enumerate(tr):
        cols[li[Y[i]]].append(j)
    sims = Z[cov] @ Z[tr].T
    sc = np.full((len(cov), len(labs)), -9.0, np.float32)
    for c, ix in cols.items():
        sc[:, c] = sims[:, ix].max(1)
    pred = sc.argmax(1)
    return 100 * np.mean([labs[pred[r]] == Y[cov[r]] for r in range(len(cov))])


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = load(); Y = [r["label"] for r in rows]
    split = get_split()
    cnt = collections.Counter(Y)
    core = [i for i in range(len(rows)) if cnt[Y[i]] >= 2]
    dev = [i for i in core if split[rows[i]["image"]] == "dev"]
    test = [i for i in core if split[rows[i]["image"]] == "test"]
    block = [json.loads((BASE / "_blocks.json").read_text())[r["image"]] for r in rows]
    # vitb14 (current best stack)
    b_g = unit(np.load(BASE / "_dino_cache.npy").astype(np.float32))
    b_l = unit(np.load(BASE / "_local256_cache.npy").astype(np.float32))
    # vitg14
    g_g = unit(np.load(BASE / "_bb_vitg14.npy").astype(np.float32))
    print(f"core {len(core)} | dev {len(dev)} / test {len(test)} | embedding vitg14-L256...")
    g_l = unit(embed_vitg14_local(rows, core, device))

    variants = {
        "vitb14 G+L256 (045)": unit(np.concatenate([b_g, b_l], 1)),
        "vitg14 G only": g_g,
        "vitg14 G+L256": unit(np.concatenate([g_g, g_l], 1)),
    }
    splits = [block_split(dev, block, s) for s in range(SEEDS)]

    res = {}
    for name, Z in variants.items():
        res[name + " | plain"] = [plain_top1(Z, Y, tr, te) for tr, te in splits]
        res[name + " | CSLS"] = [csls_top1(Z, Y, tr, te) for tr, te in splits]
    base = res["vitb14 G+L256 (045) | CSLS"]   # current best readout (≈ sealed 38.3)

    def paired(a):
        d = [x - y for x, y in zip(a, base)]
        return round(st.mean(d), 2), int(sum(x > 0 for x in d))

    print("\n== dev-CV 10-seed (paired vs vitb14 G+L256 CSLS = current best) ==")
    table = {}
    for k in res:
        dlt = paired(res[k]); table[k] = (ms(res[k]), dlt)
        print(f"  {k:34} top1 {ms(res[k])[0]}±{ms(res[k])[1]}  Δ {dlt[0]:+} ({dlt[1]}/10)")

    # sealed: current best vs vitg14 best
    best_g = "vitg14 G+L256 | CSLS"
    sb_cur = csls_top1(variants["vitb14 G+L256 (045)"], Y, dev, test)
    sb_g = csls_top1(variants["vitg14 G+L256"], Y, dev, test)
    d_g = paired(res[best_g])
    adopt = sb_g > sb_cur and d_g[0] > 0 and d_g[1] >= 7
    print(f"\n  ★ SEALED: vitb14+L256+CSLS {round(sb_cur,1)} (current best) vs vitg14+L256+CSLS {round(sb_g,1)} → "
          f"{'🟢 vitg14 stack 새 best' if adopt else '🟡 vitg14 stack 미초과 (백본 이득이 L256/CSLS와 중복)'}")

    d = explog.EXP / "057-backbone-stack"; d.mkdir(parents=True, exist_ok=True)
    ks = list(res)
    explog.bar(d / "fig1.png", [k.replace(" | ", "\n") for k in ks], [table[k][0][0] for k in ks],
               "057 M-bb1: stack L256+CSLS on vitg14 vs vitb14 (dev-CV top1)", "%", ymax=40,
               errors=[table[k][0][1] for k in ks])

    rowmd = "\n".join(f"| {k} | {table[k][0][0]}±{table[k][0][1]} | {table[k][1][0]:+} | {table[k][1][1]}/10 |" for k in ks)
    verdict = (f"🟢 **vitg14+L256+CSLS = 새 best** (봉인 {round(sb_g,1)} > vitb14 {round(sb_cur,1)})."
               if adopt else
               f"🟡 **vitg14 stack가 vitb14 stack 미초과** (봉인 {round(sb_g,1)} vs {round(sb_cur,1)}). "
               f"vitg14의 일반-품질 이득이 L256(해상도)·CSLS(hubness)와 *중복* — 적층 시 가산 안 됨. 현재 best 유지.")
    report = f"""# 057 — M-bb1: 베스트 백본(vitg14) 위 L256·CSLS 적층

- 날짜: {datetime.date.today().isoformat()} · 커밋 `main @ {_git_sha()}` · `scripts/backbone_stack.py`
- clean 502 (dev {len(dev)}/test {len(test)} 봉인), dev 10-seed paired + 봉인 test.
- 질문: 056의 신뢰 백본(vitg14, 최소 봉인 36.8)이 검증된 기법(L256 045·CSLS 051)과 *가산*돼 현재 best(vitb14+L256+CSLS 봉인 38.3) 초과?

## 결과 (paired Δ vs vitb14 G+L256 CSLS = 현재 best)
| 구성 | dev-CV top1 | Δ | wins |
|---|---|---|---|
{rowmd}

- **봉인 TEST: vitb14+L256+CSLS {round(sb_cur,1)} vs vitg14+L256+CSLS {round(sb_g,1)}.**

![bar](fig1.png)

## 판정
{verdict}

## 핵심
- vitg14 최소(36.8)는 vitb14 최소(33.5)보다 높지만, L256·CSLS를 얹으면 {'추가 가산' if adopt else '중복(가산 안 됨)'}.
- {'백본 크기가 새 레버 — 적층 best 갱신.' if adopt else 'vitg14의 일반-품질 이득과 L256/CSLS가 같은 신호를 공유 → 백본 교체가 기법을 대체하지 추가하지 않음. 현재 best(vitb14 스택) 유지.'}
"""
    explog.write(d, report, {
        "title": "M-bb1: vitg14 위 L256·CSLS 적층", "date": datetime.date.today().isoformat(),
        "headline": f"vitg14+L256+CSLS 봉인 {round(sb_g,1)} vs vitb14+L256+CSLS {round(sb_cur,1)} "
                    f"({'🟢새 best' if adopt else '🟡미초과·중복'}) | dev Δ{d_g[0]:+}({d_g[1]}/10)",
        "adopt": bool(adopt), "sealed_vitb14_stack": round(sb_cur, 1), "sealed_vitg14_stack": round(sb_g, 1),
        "devcv": {k: {"top1": table[k][0], "delta_vs_best": table[k][1]} for k in res}})
    print(f"\nwrote -> {d}  (1 figure)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
