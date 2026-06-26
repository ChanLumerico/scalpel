"""Experiment 002 — point-pooling ablation (training-free).

How should we turn the DINO patch grid into a single point embedding z_q? The
baseline uses GaussianPool with sigma=40px. Here we sweep the pooling *width*
(sigma) and compare against a single-point **bilinear** sample at the pin. DINO
grids are computed ONCE per image and cached, then re-pooled per config, so the
whole sweep costs one embedding pass.

Note: with L2-normalized embeddings, nearest-neighbour by cosine and by euclidean
are identical (argmin ||a-b||^2 = argmax a.b), so we don't ablate the metric.

    .venv/bin/python scripts/ablate_pooling.py
"""

from __future__ import annotations

import collections
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

import explog  # noqa: E402
from eval_appearance import (load_core, split_specimen, prototypes,  # noqa: E402
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
    feat = grid.permute(2, 0, 1).unsqueeze(0)            # (1,D,g,g)
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


def main():
    jsonl = "data/triples/triples.jsonl"
    base = Path(jsonl).parent
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg()
    S = cfg.backbone.image_size

    core = load_core(jsonl, 2)
    tr, te = split_specimen(core, 0.3, 0)
    ncls = len({r["label"] for r in core})
    print(f"core {len(core)}/{ncls} | gallery {len(tr)} / test {len(te)} | device={device}")

    backbone = DinoBackbone(cfg.backbone)
    backbone.ensure_loaded()
    backbone.to(device)
    centers = backbone.patch_centers(device)
    print("caching DINO grids (one pass)...")
    grids = cache_grids(tr + te, base, backbone, S, device)

    configs = [("bilinear", None), ("gauss", 10), ("gauss", 20),
               ("gauss", 40), ("gauss", 60), ("gauss", 80)]
    rows_out, results = [], {}
    for mode, sigma in configs:
        name = "bilinear" if mode == "bilinear" else f"sigma{sigma}"
        Ztr = pool_all(tr, grids, centers, S, device, mode, sigma)
        Zte = pool_all(te, grids, centers, S, device, mode, sigma)
        protos, support = prototypes(tr, Ztr)
        m, _ = evaluate(te, Zte, protos, support)
        results[name] = m
        rows_out.append((name, m["sel_top1"], m["sel_top3"], m["sel_top5"]))
        print(f"  {name:10s} top1 {m['sel_top1']:5.1f}  top3 {m['sel_top3']:5.1f}  top5 {m['sel_top5']:5.1f}")

    best = max(rows_out, key=lambda r: r[1])
    labels = [r[0] for r in rows_out]

    # ---- experiment folder --------------------------------------------------
    d = explog.next_dir("pooling-ablation")
    explog.grouped_bar(
        d / "fig_pooling.png", labels,
        {"top1": [r[1] for r in rows_out], "top5": [r[3] for r in rows_out]},
        "Point pooling: accuracy vs config", "%", ymax=100)
    table = "\n".join(f"| {n} | {t1}% | {t3}% | {t5}% |" for n, t1, t3, t5 in rows_out)
    base_top1 = results["sigma40"]["sel_top1"]
    delta = round(best[1] - base_top1, 1)
    report = f"""# 실험 002 — 점 풀링 애블레이션 (pooling)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/ablate_pooling.py`

## 목적
DINO 패치 격자를 핀 자리 임베딩 z_q로 만드는 **풀링 방식/폭**이 정확도에 주는 영향 측정.
베이스라인은 GaussianPool σ=40px. 더 좁게(구조물에 집중) vs 더 넓게(맥락 포함), 그리고
핀 한 점만 뽑는 bilinear를 비교한다. (학습 없음, DINO 격자는 한 번만 계산해 재풀링.)

## 설정
| 항목 | 값 |
|---|---|
| 백본 | dinov2_vitb14, 518px, frozen, {device} |
| 데이터 | ≥2 코어 {len(core)} 트리플 / {ncls} 클래스 |
| 분할 | 표본 단위 test_frac=0.3 seed=0 (갤러리 {len(tr)} / 테스트 {len(te)}) |
| 거리 | cosine (정규화 임베딩이라 L2와 순위 동일 → 메트릭 애블레이션 생략) |
| 비교 | bilinear(단일점) · GaussianPool σ∈{{10,20,40,60,80}}px |

## 결과 (selective accuracy @ coverage)
| 풀링 | top1 | top3 | top5 |
|---|---|---|---|
{table}

![pooling](fig_pooling.png)

- **베스트: `{best[0]}` — top1 {best[1]}%** (σ40 베이스라인 {base_top1}% 대비 {'+' if delta>=0 else ''}{delta}%p)

## 해석
- σ가 정확도에 주는 영향으로 "핀 주변 얼마나 좁게 봐야 하는가"를 알 수 있다.
  좁을수록(σ작음/bilinear) 핀 구조물에 집중, 넓을수록 맥락을 섞는다.
- 패치=14px이므로 σ≈14는 약 1패치, σ40은 약 3패치 범위.

## 다음
베스트 풀링을 기본값으로 → 모달리티 분리 분석, 이후 M5' 보정+기권.
"""
    m_idx = {"title": "점 풀링 애블레이션", "date": datetime.date.today().isoformat(),
             "headline": f"best={best[0]} top1 {best[1]}% (σ40={base_top1}%)",
             "configs": {n: results[n] for n in results}}
    explog.write(d, report, m_idx)
    print(f"\nbest: {best[0]} top1 {best[1]}%  ->  {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
