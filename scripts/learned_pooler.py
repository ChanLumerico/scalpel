"""Experiment — learned point pooler (PinCrossAttention) vs GaussianPool.

GaussianPool weights patches by DISTANCE only (content-blind). PinCrossAttention
LEARNS which patches to attend to (cross-attention seeded at the pin). Train it on
the gallery (frozen DINO grids, SupCon loss), evaluate exemplar 1-NN cross-cadaver,
compare PAIRED to GaussianPool on the same splits. Also blends each pooler's patch
weights onto the photo for a visual comparison (correct samples).

    .venv/bin/python scripts/learned_pooler.py
"""

from __future__ import annotations

import collections
import datetime
import statistics as st
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

import explog  # noqa: E402
from eval_appearance import load_core, split_indices, _MEAN, _STD, _git_sha  # noqa: E402
from learned_head import supcon  # noqa: E402
from scalpel import perception as perc  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, PinCrossAttention  # noqa: E402

VIZ = "/private/tmp/claude-501/-Users-chanlee-Desktop-Programming-scalpel/320890f6-79e3-48e4-86bc-86ffdb842a81/scratchpad/pooler_attn.png"


@torch.no_grad()
def cache_grids(core, base, backbone, S, device):
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    grids, wh = {}, {}
    for n, img in enumerate(sorted({r["image"] for r in core}), 1):
        im = Image.open(base / img).convert("RGB"); w, h = im.size
        arr = np.asarray(im.resize((S, S)), dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        grids[img] = backbone((x - mean) / std)[0][0].cpu().half()
        wh[img] = (w, h)
        if n % 60 == 0:
            print(f"   cached {n}")
    return grids, wh


def qscaled(core, wh, S):
    out = []
    for r in core:
        w, h = wh[r["image"]]
        out.append([r["q"][0] * S / w, r["q"][1] * S / h])
    return torch.tensor(out, dtype=torch.float32)


def gather(grids, core, idx, device):
    return torch.stack([grids[core[i]["image"]].float() for i in idx]).to(device)


def gauss_pool(g, centers, q, sigma=40.0):
    B, gg, _, D = g.shape
    tok = g.reshape(B, gg * gg, D)
    d2 = ((centers[None] - q[:, None]) ** 2).sum(-1)
    w = torch.softmax(-d2 / (2 * sigma ** 2), dim=1)
    return F.normalize(torch.einsum("bm,bmd->bd", w, tok), dim=1)


@torch.no_grad()
def pool_all(fn, grids, core, idx, centers, qs, device, chunk=128):
    Z = []
    for s in range(0, len(idx), chunk):
        ch = idx[s:s + chunk]
        g = gather(grids, core, ch, device)
        Z.append(fn(g, centers, qs[ch].to(device)).cpu())
    return torch.cat(Z)


def exemplar(Ztr, ytr, Zte, yte):
    Ztr, Zte = Ztr.numpy(), Zte.numpy()
    labels = sorted(set(ytr)); lidx = {l: i for i, l in enumerate(labels)}
    cols = collections.defaultdict(list)
    for j, l in enumerate(ytr):
        cols[lidx[l]].append(j)
    cv = [k for k, l in enumerate(yte) if l in lidx]
    sims = Zte[cv] @ Ztr.T
    sc = np.full((len(cv), len(labels)), -2.0, np.float32)
    for c, ix in cols.items():
        sc[:, c] = sims[:, ix].max(1)
    o = np.argsort(-sc, axis=1)
    n1 = sum(int(labels[o[r, 0]] == yte[k]) for r, k in enumerate(cv))
    n5 = sum(int(yte[k] in [labels[o[r, t]] for t in range(5)]) for r, k in enumerate(cv))
    return 100 * n1 / len(cv), 100 * n5 / len(cv)


def train_pooler(grids, core, tr, centers, qs, S, device, steps=250):
    cfg = replace(PipelineCfg().point, dropout=0.1)
    pooler = PinCrossAttention(cfg, 768, S).to(device)
    opt = torch.optim.Adam(pooler.parameters(), lr=1e-3, weight_decay=1e-4)
    by = collections.defaultdict(list)
    for j in tr:
        by[core[j]["label"]].append(j)
    multi = [l for l, v in by.items() if len(v) >= 2]
    rng = np.random.default_rng(0)
    pooler.train()
    for _ in range(steps):
        cls = rng.choice(multi, size=min(24, len(multi)), replace=False)
        idx = [int(rng.choice(by[c])) for c in cls for _ in range(2)]
        g = gather(grids, core, idx, device)
        z = pooler(g, centers, qs[idx].to(device))
        z = F.normalize(z, dim=1)
        y = torch.tensor([hash(core[i]["label"]) % (10 ** 8) for i in idx], device=device)
        loss = supcon(z, y)
        opt.zero_grad(); loss.backward(); opt.step()
    pooler.eval()
    return pooler


@torch.no_grad()
def attn_weights(pooler, grid, q, S):
    """Recompute the learned pooler's per-patch attention (g,g), head-averaged."""
    g = grid.unsqueeze(0)
    b, gg, _, d = g.shape
    tokens = g.reshape(b, gg * gg, d)
    seed = pooler._seed(g, q)
    gamma = perc._fourier_encode(2.0 * (q / S) - 1.0, pooler.n_pos_freqs)
    qq = pooler.q_proj(torch.cat([seed, gamma], -1)).view(b, pooler.n_heads, 1, pooler.head_dim)
    k = pooler.k_proj(tokens).view(b, -1, pooler.n_heads, pooler.head_dim).transpose(1, 2)
    a = torch.softmax((qq @ k.transpose(-2, -1)) / pooler.head_dim ** 0.5, dim=-1)
    return a.mean(1).reshape(gg, gg).cpu().numpy()


def blend(img_bgr, heat, q_xy):
    h, w = img_bgr.shape[:2]
    hm = cv2.resize((np.clip((heat - heat.min()) / (np.ptp(heat) + 1e-9), 0, 1) * 255).astype(np.uint8),
                    (w, h), interpolation=cv2.INTER_CUBIC)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    out = cv2.addWeighted(img_bgr, 0.6, hm, 0.45, 0)
    cv2.circle(out, (int(q_xy[0]), int(q_xy[1])), max(7, w // 70), (255, 255, 255), -1)
    cv2.circle(out, (int(q_xy[0]), int(q_xy[1])), max(7, w // 70), (0, 0, 0), 2)
    return out


def main():
    base = Path("data/triples")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size
    core = load_core("data/triples/triples.jsonl", 2)
    ncls = len({r["label"] for r in core})
    print(f"core {len(core)}/{ncls} | device={device}")
    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    centers = bb.patch_centers(device)
    print("caching grids..."); grids, wh = cache_grids(core, base, bb, S, device)
    qs = qscaled(core, wh, S)

    g1, g5, p1, p5 = [], [], [], []
    pooler0 = split0 = None
    for seed in range(10):
        tr, te = split_indices(core, 0.3, seed)
        ytr = [core[i]["label"] for i in tr]; yte = [core[i]["label"] for i in te]
        Ztr_g = pool_all(gauss_pool, grids, core, tr, centers, qs, device)
        Zte_g = pool_all(gauss_pool, grids, core, te, centers, qs, device)
        a1, a5 = exemplar(Ztr_g, ytr, Zte_g, yte)
        pooler = train_pooler(grids, core, tr, centers, qs, S, device)
        pl = lambda g, c, q: F.normalize(pooler(g, c, q), dim=1)
        Ztr_p = pool_all(pl, grids, core, tr, centers, qs, device)
        Zte_p = pool_all(pl, grids, core, te, centers, qs, device)
        b1, b5 = exemplar(Ztr_p, ytr, Zte_p, yte)
        g1.append(a1); g5.append(a5); p1.append(b1); p5.append(b5)
        if seed == 0:
            pooler0, split0 = pooler, (tr, te)
        print(f"  seed {seed}: gauss {a1:.0f}/{a5:.0f}  learned {b1:.0f}/{b5:.0f}")

    ms = lambda v: (round(st.mean(v), 1), round(st.pstdev(v), 1))
    G1, G5, P1, P5 = ms(g1), ms(g5), ms(p1), ms(p5)
    d1 = [a - b for a, b in zip(p1, g1)]; d5 = [a - b for a, b in zip(p5, g5)]
    md1, sd1, w1 = round(st.mean(d1), 1), round(st.pstdev(d1), 1), sum(x > 0 for x in d1)
    md5, sd5, w5 = round(st.mean(d5), 1), round(st.pstdev(d5), 1), sum(x > 0 for x in d5)
    verdict = ("학습형 풀러가 일관되게 도움" if (md1 > 0 and w1 >= 8) else "효과 불명확/노이즈")
    print(f"\n== gauss {G1[0]}±{G1[1]} | learned {P1[0]}±{P1[1]} | "
          f"PAIRED Δtop1 {md1}±{sd1} ({w1}/10)  Δtop5 {md5} ({w5}/10) -> {verdict} ==")

    # ---- attention comparison viz (correct learned-pooler samples) -----------
    tr, te = split0
    ytr = [core[i]["label"] for i in tr]; yte = [core[i]["label"] for i in te]
    Ztr_p = pool_all(lambda g, c, q: F.normalize(pooler0(g, c, q), dim=1), grids, core, tr, centers, qs, device)
    labels = sorted(set(ytr)); lidx = {l: i for i, l in enumerate(labels)}
    cols = collections.defaultdict(list)
    for j, l in enumerate(ytr):
        cols[lidx[l]].append(j)
    picks, seen = [], set()
    for k in te:
        lab = core[k]["label"]
        if lab not in lidx or lab in seen:
            continue
        with torch.no_grad():
            zk = F.normalize(pooler0(gather(grids, core, [k], device), centers, qs[[k]].to(device)), dim=1)[0].cpu().numpy()
        sims = Ztr_p.numpy() @ zk
        sc = {c: sims[ix].max() for c, ix in cols.items()}
        if max(sc, key=sc.get) == lidx[lab]:
            seen.add(lab); picks.append(k)
        if len(picks) >= 6:
            break
    rows = []
    for k in picks:
        r = core[k]
        im = cv2.cvtColor(np.asarray(Image.open(base / r["image"]).convert("RGB")), cv2.COLOR_RGB2BGR)
        gr = grids[r["image"]].float().to(device); q = qs[[k]].to(device)
        d2 = ((centers - q[0]) ** 2).sum(-1)
        gw = torch.softmax(-d2 / (2 * 40.0 ** 2), 0).reshape(37, 37).cpu().numpy()
        aw = attn_weights(pooler0, gr, q, S)
        pan_g = cv2.resize(blend(im.copy(), gw, r["q"]), (300, 300))
        pan_a = cv2.resize(blend(im.copy(), aw, r["q"]), (300, 300))
        cv2.putText(pan_g, "Gaussian", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(pan_a, "Learned", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        strip = np.hstack([pan_g, pan_a])
        cv2.putText(strip, r["label"][:30], (8, 292), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        rows.append(strip)
    cv2.imwrite(VIZ, np.vstack(rows))
    print("saved attention compare ->", VIZ)

    d = explog.next_dir("learned-pooler")
    explog.bar(d / "fig_pooler.png", ["gauss\ntop1", "learned\ntop1", "gauss\ntop5", "learned\ntop5"],
               [G1[0], P1[0], G5[0], P5[0]], "Learned pooler vs GaussianPool (10-seed)", "%",
               ymax=100, errors=[G1[1], P1[1], G5[1], P5[1]])
    report = f"""# 학습형 풀러 (PinCrossAttention) vs GaussianPool

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/learned_pooler.py`  (10-seed, seed별 풀러 재학습)

## 목적
거리 기반 고정 풀러(GaussianPool) 대신 **어느 패치를 볼지 학습**하는 cross-attention 풀러가
핀 구조물에 집중해 미세 판별을 올리는지. SupCon 학습, exemplar 1-NN, cross-cadaver, paired 비교.

## 설정
| 항목 | 값 |
|---|---|
| 풀러 | PinCrossAttention (4-head, attn_dim 256, dropout 0.1) |
| 학습 | SupCon, class-balanced 48/step, Adam 1e-3, 250 step, seed별 재학습 |
| 평가 | exemplar 1-NN, 표본분할 10 seed |

## 결과 (selective top1/top5, mean±std)
| | GaussianPool | learned pooler |
|---|---|---|
| top1 | {G1[0]}±{G1[1]}% | {P1[0]}±{P1[1]}% |
| top5 | {G5[0]}±{G5[1]}% | {P5[0]}±{P5[1]}% |

![pooler](fig_pooler.png)

## 판정 (paired)
- Δtop1 = {'+' if md1>=0 else ''}{md1}±{sd1}%p ({w1}/10 승), Δtop5 = {'+' if md5>=0 else ''}{md5}%p ({w5}/10)
- → **{verdict}**

## 시각 비교 (attention)
`outputs/pooler_attn.png` — 각 정답 샘플에 [GaussianPool | learned] 가중치 블렌딩. 가우시안은 핀 주변
원형, 학습형은 구조물 모양을 따라가는지 눈으로 확인.

## 해석 / 다음
- 도움되면 → 정식 풀러 채택, 데이터 늘려 추가 학습.
- 미미하면 → 작은 데이터에선 학습형 풀러도 포화 → **데이터가 결정적**(exp 013과 일관).
"""
    explog.write(d, report, {
        "title": "학습형 풀러 vs GaussianPool", "date": datetime.date.today().isoformat(),
        "headline": f"gauss {G1[0]} vs learned {P1[0]} | paired Δtop1 {md1}({w1}/10) Δtop5 {md5}({w5}/10) → {verdict}",
        "gauss_top1": G1, "learned_top1": P1, "gauss_top5": G5, "learned_top5": P5})
    print(f"wrote -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
