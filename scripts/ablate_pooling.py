"""Experiment — point-pooling ablation (training-free, multi-seed).

How should we turn the DINO patch grid into a single point embedding z_q? The
baseline uses GaussianPool with sigma=40px. Here we sweep the pooling *width*
(sigma) and compare against a single-point **bilinear** sample at the pin. DINO
grids are computed ONCE per image and cached, then re-pooled per config; for each
config we evaluate over many seeds and report mean +/- std (the test set is small,
so a single split is noisy).

Note: with L2-normalized embeddings, nearest-neighbour by cosine and by euclidean
are identical, so we don't ablate the metric.

    .venv/bin/python scripts/ablate_pooling.py
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
from eval_appearance import (load_core, split_indices, prototypes,  # noqa: E402
                             evaluate, _git_sha, _MEAN, _STD)
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone  # noqa: E402


@torch.no_grad()
def cache_grids(rows, base, backbone, S, device):
    """DINO patch grid per unique image (CPU float16), + original (w,h)."""
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    imgs = {}
    uniq = {r["image"] for r in rows}
    for n, img in enumerate(sorted(uniq), 1):
        im = Image.open(base / img).convert("RGB")
        w, h = im.size
        arr = np.asarray(im.resize((S, S)), dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        grid, _ = backbone((x - mean) / std)
        imgs[img] = (grid[0].cpu().half(), (w, h))
        if n % 50 == 0:
            print(f"   cached {n}/{len(uniq)} grids")
    return imgs


def _gauss(grid, centers, q, sigma):
    g, _, d = grid.shape
    tok = grid.reshape(g * g, d)
    d2 = ((centers - q) ** 2).sum(-1)
    w = torch.softmax(-d2 / (2.0 * sigma ** 2), dim=0)
    return (w[:, None] * tok).sum(0)


def _bilinear(grid, q, S):
    g, _, d = grid.shape
    feat = grid.permute(2, 0, 1).unsqueeze(0)
    qn = (2.0 * (q / S) - 1.0).view(1, 1, 1, 2)
    z = F.grid_sample(feat, qn, mode="bilinear", align_corners=False, padding_mode="border")
    return z.view(d)


def pool_all(rows, imgs, centers, S, device, mode, sigma):
    Z = []
    for r in rows:
        grid_cpu, (w, h) = imgs[r["image"]]
        grid = grid_cpu.float().to(device)
        qx, qy = r["q"]
        q = torch.tensor([qx * S / w, qy * S / h], device=device)
        z = _bilinear(grid, q, S) if mode == "bilinear" else _gauss(grid, centers, q, sigma)
        Z.append(F.normalize(z, dim=0).cpu())
    return torch.stack(Z)


def eval_seeds(core, Z, test_frac, n_seeds):
    """Mean/std selective top1/top5 over seeds for one embedding set."""
    t1, t5 = [], []
    for seed in range(n_seeds):
        tr, te = split_indices(core, test_frac, seed)
        protos, support = prototypes([core[i] for i in tr], Z[tr])
        m, _ = evaluate([core[i] for i in te], Z[te], protos, support)
        t1.append(m["sel_top1"]); t5.append(m["sel_top5"])
    return (round(st.mean(t1), 1), round(st.pstdev(t1), 1),
            round(st.mean(t5), 1), round(st.pstdev(t5), 1))


def main():
    jsonl = "data/triples/triples.jsonl"
    base = Path(jsonl).parent
    n_seeds, test_frac = 10, 0.3
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg()
    S = cfg.backbone.image_size

    core = load_core(jsonl, 2)
    ncls = len({r["label"] for r in core})
    print(f"core {len(core)}/{ncls} | device={device} | seeds={n_seeds}")

    backbone = DinoBackbone(cfg.backbone)
    backbone.ensure_loaded()
    backbone.to(device)
    centers = backbone.patch_centers(device)
    print("caching DINO grids (one pass)...")
    grids = cache_grids(core, base, backbone, S, device)

    configs = [("bilinear", None), ("gauss", 10), ("gauss", 20),
               ("gauss", 40), ("gauss", 60), ("gauss", 80)]
    rows_out = []
    for mode, sigma in configs:
        name = "bilinear" if mode == "bilinear" else f"sigma{sigma}"
        Z = pool_all(core, grids, centers, S, device, mode, sigma)
        m1, s1, m5, s5 = eval_seeds(core, Z, test_frac, n_seeds)
        rows_out.append((name, m1, s1, m5, s5))
        print(f"  {name:10s} top1 {m1:5.1f}±{s1:<4.1f} top5 {m5:5.1f}±{s5:.1f}")

    best = max(rows_out, key=lambda r: r[1])
    base_top1 = next(r[1] for r in rows_out if r[0] == "sigma40")

    d = explog.next_dir("pooling-ablation")
    explog.bar(d / "fig_pooling.png", [r[0] for r in rows_out], [r[1] for r in rows_out],
               f"Point pooling: top1 ({n_seeds}-seed mean±std)", "%", ymax=100,
               errors=[r[2] for r in rows_out])
    table = "\n".join(f"| {n} | {m1}±{s1}% | {m5}±{s5}% |" for n, m1, s1, m5, s5 in rows_out)
    report = f"""# 점 풀링 애블레이션 (pooling) — multi-seed

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/ablate_pooling.py`

## 목적
DINO 패치 격자를 핀 임베딩 z_q로 만드는 **풀링 방식/폭**의 영향 측정. 더 좁게(구조물 집중)
vs 넓게(맥락), bilinear 단일점 비교. DINO 격자는 1회 캐시 후 재풀링, **{n_seeds}-seed mean±std**.

## 설정
| 항목 | 값 |
|---|---|
| 백본 | dinov2_vitb14, 518px, frozen, {device} |
| 데이터 | ≥2 코어 {len(core)} 트리플 / {ncls} 클래스 |
| 분할 | 표본 단위 test_frac={test_frac}, seeds=0..{n_seeds - 1} |
| 비교 | bilinear · GaussianPool σ∈{{10,20,40,60,80}}px |

## 결과 (selective accuracy, mean±std)
| 풀링 | top1 | top5 |
|---|---|---|
{table}

![pooling](fig_pooling.png)

- **베스트 top1: `{best[0]}` = {best[1]}±{best[2]}%** (σ40 기준 {base_top1}%; 차이가 std 안이면 무의미).

## 해석
- σ가 좁을수록 핀 구조물 집중, 넓을수록 맥락 포함. std를 넘는 차이만 유의미.

## 다음
유의미하게 나은 풀링이 있으면 기본값 채택 → 모달리티 분석 / M5'.
"""
    explog.write(d, report, {
        "title": "점 풀링 애블레이션", "date": datetime.date.today().isoformat(),
        "headline": f"best={best[0]} top1 {best[1]}±{best[2]}% (σ40={base_top1}%)",
        "configs": {n: {"top1": m1, "top1_std": s1, "top5": m5} for n, m1, s1, m5, s5 in rows_out}})
    print(f"\nbest: {best[0]} {best[1]}±{best[2]}%  ->  {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
