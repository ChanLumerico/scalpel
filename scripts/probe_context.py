"""Experiment — cheap relational/context probe (training-free).

Does neighbourhood/region CONTEXT raise the accuracy ceiling? The dominant errors
are adjacent / look-alike structures, and the handout's whole thesis is that
topological context disambiguates them. Before building the heavy M6'
(segmentation + R-GCN), test the cheapest possible version: augment the point
embedding with context and see if top1 moves beyond the ±3%p split-noise band.

Variants of the per-pin embedding (each sub-vector L2-normalized, concatenated,
then the whole re-normalized -> cosine = average of per-part cosines):

    local   = pool(sigma40)                              # baseline (current MVP)
    ms      = [ pool(sigma20) ; pool(sigma80) ]          # local detail + wide context
    cls     = [ pool(sigma40) ; CLS ]                    # local + global region token
    ms_cls  = [ pool(sigma20) ; pool(sigma80) ; CLS ]    # both

DINO grid + CLS are computed ONCE per image; each variant re-pools, then we
evaluate 10 seeds -> mean±std. Logged via explog.

    .venv/bin/python scripts/probe_context.py
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
def cache(rows, base, backbone, S, device):
    """Per unique image: (patch grid f16, CLS f16, (w,h))."""
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    out = {}
    uniq = sorted({r["image"] for r in rows})
    for n, img in enumerate(uniq, 1):
        im = Image.open(base / img).convert("RGB")
        w, h = im.size
        arr = np.asarray(im.resize((S, S)), dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        grid, cls = backbone((x - mean) / std)
        out[img] = (grid[0].cpu().half(), cls[0].cpu().half(), (w, h))
        if n % 50 == 0:
            print(f"   cached {n}/{len(uniq)}")
    return out


def _gauss(grid, centers, q, sigma):
    g, _, d = grid.shape
    tok = grid.reshape(g * g, d)
    w = torch.softmax(-((centers - q) ** 2).sum(-1) / (2.0 * sigma ** 2), dim=0)
    return F.normalize((w[:, None] * tok).sum(0), dim=0)


def build_emb(grid, cls, q, centers, variant):
    if variant == "local":
        return _gauss(grid, centers, q, 40)
    if variant == "ms":
        return torch.cat([_gauss(grid, centers, q, 20), _gauss(grid, centers, q, 80)])
    if variant == "cls":
        return torch.cat([_gauss(grid, centers, q, 40), F.normalize(cls, dim=0)])
    if variant == "ms_cls":
        return torch.cat([_gauss(grid, centers, q, 20), _gauss(grid, centers, q, 80),
                          F.normalize(cls, dim=0)])
    raise ValueError(variant)


def embed_variant(core, cached, centers, S, device, variant):
    Z = []
    for r in core:
        grid_cpu, cls_cpu, (w, h) = cached[r["image"]]
        grid = grid_cpu.float().to(device)
        cls = cls_cpu.float().to(device)
        qx, qy = r["q"]
        q = torch.tensor([qx * S / w, qy * S / h], device=device)
        Z.append(F.normalize(build_emb(grid, cls, q, centers, variant), dim=0).cpu())
    return torch.stack(Z)


def eval_seeds(core, Z, n_seeds, test_frac):
    t1, t5 = [], []
    for seed in range(n_seeds):
        tr, te = split_indices(core, test_frac, seed)
        protos, sup = prototypes([core[i] for i in tr], Z[tr])
        m, _ = evaluate([core[i] for i in te], Z[te], protos, sup)
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
    backbone = DinoBackbone(cfg.backbone); backbone.ensure_loaded(); backbone.to(device)
    centers = backbone.patch_centers(device)
    print("caching grids + CLS (one pass)...")
    cached = cache(core, base, backbone, S, device)

    variants = ["local", "ms", "cls", "ms_cls"]
    rows_out = []
    for v in variants:
        Z = embed_variant(core, cached, centers, S, device, v)
        m1, s1, m5, s5 = eval_seeds(core, Z, n_seeds, test_frac)
        rows_out.append((v, m1, s1, m5, s5))
        print(f"  {v:8s} top1 {m1:5.1f}±{s1:<4.1f} top5 {m5:5.1f}±{s5:.1f}")

    base_t1, base_std = rows_out[0][1], rows_out[0][2]
    best = max(rows_out, key=lambda r: r[1])
    delta = round(best[1] - base_t1, 1)
    # "real" only if the gain clears the baseline noise band
    verdict = "맥락이 도움 (M6' 정당화)" if delta > base_std else "노이즈 안 — 단순 맥락 concat은 무효"

    d = explog.next_dir("context-probe")
    explog.bar(d / "fig_context.png", [r[0] for r in rows_out], [r[1] for r in rows_out],
               f"Context probe: top1 ({n_seeds}-seed mean±std)", "%", ymax=100,
               errors=[r[2] for r in rows_out])
    table = "\n".join(f"| {v} | {m1}±{s1}% | {m5}±{s5}% |" for v, m1, s1, m5, s5 in rows_out)
    report = f"""# 관계/맥락 프로브 (context-probe) — multi-seed

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/probe_context.py`

## 목적
"이웃/맥락 정보가 외형 천장(top1 {base_t1}%)을 올리는가?"를 **학습 없이, 분할 없이** 싸게 검증.
무거운 M6'(scene graph + R-GCN) 투자 전 de-risk. 핀 임베딩에 다중 스케일 풀링 + 전역 CLS를
붙여 본다. (cosine = 부분별 cosine 평균이 되도록 부분 정규화 후 concat.)

## 설정
| 항목 | 값 |
|---|---|
| 백본 | dinov2_vitb14, 518px, frozen, {device} |
| 데이터 | ≥2 코어 {len(core)} 트리플 / {ncls} 클래스 |
| 분할 | 표본 단위 test_frac={test_frac}, {n_seeds} seeds |
| 변형 | local(σ40) · ms(σ20+σ80) · cls(σ40+CLS) · ms_cls |

## 결과 (selective accuracy, mean±std)
| 변형 | top1 | top5 |
|---|---|---|
{table}

![context](fig_context.png)

## 판정
- 베스트: **{best[0]}** top1 {best[1]}±{best[2]}% (local {base_t1}±{base_std}, Δ{'+' if delta>=0 else ''}{delta}%p)
- 베이스라인 노이즈 폭 ±{base_std}%p 기준 → **{verdict}**

## 해석 / 다음
- 맥락이 노이즈 밖으로 도움되면 → **풀 M6'(관계추론)** 투자 정당화.
- 도움 안되면 → 단순 맥락 concat으론 부족(천장이 외형 자체 또는 *구조화된* 관계 필요) → 데이터 확장
  또는 학습형 관계 모델로 선회.
"""
    explog.write(d, report, {
        "title": "관계/맥락 프로브", "date": datetime.date.today().isoformat(),
        "headline": f"best={best[0]} top1 {best[1]}±{best[2]}% (local {base_t1}±{base_std}, Δ{delta}) → {verdict}",
        "variants": {v: {"top1": m1, "top1_std": s1, "top5": m5} for v, m1, s1, m5, s5 in rows_out}})
    print(f"\n{verdict}  ->  {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
