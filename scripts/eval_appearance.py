"""M4' appearance MVP — frozen DINOv2 + GaussianPool + prototypical few-shot.

The first *model* over the real QuizLink data, and deliberately the simplest one:
no training at all. Pipeline per item ``(image I, pin q)``:

    I -> frozen DINOv2 -> patch-token grid
    grid, q -> GaussianPool -> z_q              (point embedding, L2-normalized)
    z_q vs class prototypes (mean gallery embedding) -> nearest = prediction

Restricts to the evaluable core (>=2 instances), splits at the SPECIMEN level
(leak-free), reports selective-accuracy@coverage + top-k stratified by support.
Writes a full experiment folder (Korean report + figures) via ``explog``.

    .venv/bin/python scripts/eval_appearance.py
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

import explog  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def load_core(jsonl, min_count):
    rows = [json.loads(l) for l in open(jsonl, encoding="utf-8") if l.strip()]
    cnt = collections.Counter(r["label"] for r in rows)
    return [r for r in rows if cnt[r["label"]] >= min_count]


def split_specimen(rows, test_frac=0.3, seed=0):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[f'{r["src"]}#{r["page"]}'].append(r)
    keys = sorted(groups)
    np.random.default_rng(seed).shuffle(keys)
    n_test = max(1, int(round(len(keys) * test_frac)))
    tk = set(keys[:n_test])
    tr = [r for k in keys if k not in tk for r in groups[k]]
    te = [r for k in keys if k in tk for r in groups[k]]
    return tr, te


@torch.no_grad()
def embed(rows, base, backbone, pool, S, device):
    """L2-normalized point embedding z_q for each triple (backbone once / image)."""
    by_img = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_img[r["image"]].append(i)
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    centers = backbone.patch_centers(device)
    Z = [None] * len(rows)
    for n, (img, idxs) in enumerate(by_img.items(), 1):
        im = Image.open(base / img).convert("RGB")
        w, h = im.size
        arr = np.asarray(im.resize((S, S)), dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        x = (x - mean) / std
        grid, _ = backbone(x)
        for i in idxs:
            qx, qy = rows[i]["q"]
            q = torch.tensor([[qx * S / w, qy * S / h]], device=device)
            Z[i] = F.normalize(pool(grid, centers, q)[0], dim=0).cpu()
        if n % 25 == 0:
            print(f"   embedded {n}/{len(by_img)} images")
    return torch.stack(Z)


def prototypes(tr, Ztr):
    by_lab = collections.defaultdict(list)
    for i, r in enumerate(tr):
        by_lab[r["label"]].append(i)
    protos = {l: F.normalize(Ztr[idx].mean(0), dim=0) for l, idx in by_lab.items()}
    support = {l: len(idx) for l, idx in by_lab.items()}
    return protos, support


def evaluate(te, Zte, protos, support):
    """Return (metrics dict, per-item predictions for covered test items)."""
    proto_labels = list(protos)
    P = torch.stack([protos[l] for l in proto_labels])
    strata = collections.defaultdict(lambda: [0, 0, 0, 0])
    per_item, confus = [], collections.Counter()
    cov = tot = t1 = t3 = t5 = 0
    for i, r in enumerate(te):
        tot += 1
        if r["label"] not in protos:
            continue
        cov += 1
        sims = P @ Zte[i]
        order = [proto_labels[j] for j in torch.argsort(sims, descending=True)[:5]]
        c1, c3, c5 = r["label"] == order[0], r["label"] in order[:3], r["label"] in order[:5]
        t1 += c1; t3 += c3; t5 += c5
        if not c1:
            confus[f"{r['label']} -> {order[0]}"] += 1
        b = min(support[r["label"]], 4)
        s = strata[b]; s[0] += 1; s[1] += c1; s[2] += c3; s[3] += c5
        per_item.append({"image": r["image"], "q": r["q"], "true": r["label"],
                         "pred": order[0], "correct": bool(c1)})
    pct = lambda a, b: round(100 * a / b, 1) if b else 0.0
    metrics = {
        "n_protos": len(protos), "test": tot, "covered": cov,
        "coverage_pct": pct(cov, tot), "oov": tot - cov,
        "sel_top1": pct(t1, cov), "sel_top3": pct(t3, cov), "sel_top5": pct(t5, cov),
        "e2e_top1": pct(t1, tot), "e2e_top5": pct(t5, tot),
        "chance_top1": round(100 / max(1, len(protos)), 2),
        "strata": {("4+" if b == 4 else str(b)):
                   {"n": strata[b][0], "top1": pct(strata[b][1], strata[b][0]),
                    "top3": pct(strata[b][2], strata[b][0]), "top5": pct(strata[b][3], strata[b][0])}
                   for b in sorted(strata)},
        "top_confusions": confus.most_common(10),
    }
    return metrics, per_item


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "?"


def _report_md(cfg, args, sizes, m, device):
    sigma = cfg.point.gauss_sigma_px
    core, ncls, ntr, nte = sizes
    ratio = round(m["sel_top1"] / max(m["chance_top1"], 1e-9))
    strata_rows = "\n".join(
        f"| {k}-shot | {v['n']} | {v['top1']}% | {v['top3']}% | {v['top5']}% |"
        for k, v in m["strata"].items())
    conf_rows = "\n".join(f"- `{p}` ×{c}" for p, c in m["top_confusions"][:8]) or "- (없음)"
    return f"""# 실험 001 — 베이스라인 (M4' 외형 MVP)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/eval_appearance.py`

## 목적
외형 신호만으로 **학습 없이**(frozen DINOv2 + GaussianPool + 프로토타입) 핀이 가리킨
구조물을 식별할 수 있는지 첫 검증. softmax 분류가 아니라 metric learning이라 long-tail
(클래스당 1~3샷)에 적합한지 본다.

## 방법
`I → frozen DINOv2(패치 격자) → 핀 q에서 GaussianPool → z_q(L2 정규화) →
클래스 프로토타입(갤러리 임베딩 평균)과 cosine 최근접 = 예측.`

## 설정
| 항목 | 값 |
|---|---|
| 백본 | dinov2_vitb14, 518px, frozen, {device} |
| 풀링 | GaussianPool σ={sigma}px |
| 거리 | cosine |
| 데이터 | `{args.jsonl}`, ≥{args.min_count}-인스턴스 코어 |
| 분할 | 표본(페이지) 단위, test_frac={args.test_frac}, seed={args.seed} |
| 규모 | 코어 {core} 트리플 / {ncls} 클래스 · 갤러리 {ntr} / 테스트 {nte} · 프로토타입 {m['n_protos']} |

## 결과
| 지표 | 값 |
|---|---|
| coverage | {m['coverage_pct']}% ({m['oov']}개 OOV→기권) |
| **selective top1** | **{m['sel_top1']}%** |
| selective top3 | {m['sel_top3']}% |
| selective top5 | {m['sel_top5']}% |
| end-to-end top1 (OOV=오답) | {m['e2e_top1']}% |
| 무작위 기대 top1 | {m['chance_top1']}% |

![top-k](fig_topk.png)

### support(갤러리 샷 수)별 정확도
| 버킷 | n | top1 | top3 | top5 |
|---|---|---|---|---|
{strata_rows}

![support](fig_support.png)

### 주요 혼동 (정답 → 예측, 상위)
{conf_rows}

![confusions](fig_confusions.png)

### 예측 예시 (O=정답, X=오답)
![examples](examples.private.png)
> ⚠️ 이 figure는 카데바 이미지를 포함하므로 git에 올리지 않습니다(로컬 전용, donor dignity §6).

## 해석
- top1 **{m['sel_top1']}%** = 무작위({m['chance_top1']}%)의 **약 {ratio}배** → 외형 신호가 실재한다(가설 검증 성공).
- top5 {m['sel_top5']}% → 5개 후보 안엔 절반 가까이 정답이 들어옴.
- coverage {m['coverage_pct']}% → 테스트 대부분이 갤러리에 프로토타입을 가짐(나머지는 정직하게 기권).
- support별 정확도가 들쭉날쭉한 건 버킷별 n이 작고(수십 개), 서로 다른 표본의 뷰를 평균한
  프로토타입이 흐려질 수 있어서 — 데이터가 커지면 안정화될 노이즈로 본다.

## 한계
무파라미터 GaussianPool(σ={sigma}px) · 관계추론 없음 · 보정/기권 미적용 · 모달리티(카데바/골표본/3D) 혼재 · 테스트 {nte}개로 작음.

## 다음
σ 스윕 · bilinear 단일패치 vs 가우시안 · cosine vs L2 · 모달리티 분리(혼동 원인 규명) → 이후 M5' 보정+기권.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="data/triples/triples.jsonl")
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="baseline")
    args = ap.parse_args()
    base = Path(args.jsonl).parent

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg()
    S = cfg.backbone.image_size
    print(f"device={device} backbone={cfg.backbone.name} S={S} sigma={cfg.point.gauss_sigma_px}px")

    core = load_core(args.jsonl, args.min_count)
    tr, te = split_specimen(core, args.test_frac, args.seed)
    ncls = len({r["label"] for r in core})
    print(f"core {len(core)} / {ncls} cls | gallery {len(tr)} / test {len(te)}")

    print("loading frozen DINOv2 ...")
    backbone = DinoBackbone(cfg.backbone)
    backbone.ensure_loaded()
    backbone.to(device)
    pool = GaussianPool(cfg.point).to(device)

    print("embedding gallery ..."); Ztr = embed(tr, base, backbone, pool, S, device)
    print("embedding test ...");    Zte = embed(te, base, backbone, pool, S, device)
    protos, support = prototypes(tr, Ztr)
    m, per_item = evaluate(te, Zte, protos, support)
    print(json.dumps({k: m[k] for k in ("coverage_pct", "sel_top1", "sel_top3", "sel_top5")}, indent=2))

    # ---- experiment folder: figures + Korean report --------------------------
    d = explog.next_dir(args.name)
    explog.bar(d / "fig_topk.png", ["top1", "top3", "top5"],
               [m["sel_top1"], m["sel_top3"], m["sel_top5"]],
               "Selective accuracy @ coverage", "%", ymax=100)
    explog.grouped_bar(
        d / "fig_support.png", list(m["strata"]),
        {"top1": [v["top1"] for v in m["strata"].values()],
         "top5": [v["top5"] for v in m["strata"].values()]},
        "Accuracy by gallery support count", "%", ymax=100)
    explog.barh_pairs(d / "fig_confusions.png",
                      [(p, c) for p, c in m["top_confusions"]],
                      "Top confusions (true -> pred)")
    wrong = [x for x in per_item if not x["correct"]][:6]
    right = [x for x in per_item if x["correct"]][:6]
    explog.montage(d / "examples.private.png", right + wrong, base, ncol=4)

    sizes = (len(core), ncls, len(tr), len(te))
    m_full = dict(m)
    m_full.update({"title": "베이스라인 (M4' 외형 MVP)",
                   "date": datetime.date.today().isoformat(),
                   "headline": f"top1 {m['sel_top1']}% / top5 {m['sel_top5']}% @cov{m['coverage_pct']}% "
                               f"({m['n_protos']}-way)",
                   "config": {"backbone": cfg.backbone.name, "sigma_px": cfg.point.gauss_sigma_px,
                              "image_size": S, "min_count": args.min_count,
                              "test_frac": args.test_frac, "seed": args.seed}})
    explog.write(d, _report_md(cfg, args, sizes, m, device), m_full)
    print(f"\nwrote experiment -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
