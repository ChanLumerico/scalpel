"""M4' appearance MVP — frozen DINOv2 + GaussianPool + prototypical few-shot.

The first *model* over the real QuizLink data, and deliberately the simplest one:
no training at all. Pipeline per item ``(image I, pin q)``:

    I -> frozen DINOv2 -> patch-token grid
    grid, q -> GaussianPool -> z_q              (point embedding, L2-normalized)
    z_q vs class prototypes (mean gallery embedding) -> nearest = prediction

Restricts to the evaluable core (>=2 instances), splits at the SPECIMEN level
(leak-free). The test set is small (~180), so a single split is noisy (+/-3-4%p);
by default we embed ONCE and evaluate over many seeds, reporting **mean +/- std**.
Writes a full experiment folder (Korean report + figures) via ``explog``.

    .venv/bin/python scripts/eval_appearance.py            # 10-seed baseline
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import statistics as st
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


def split_indices(core, test_frac, seed):
    """Specimen-level (per page) split -> (train_idx, test_idx) into ``core``."""
    g = collections.defaultdict(list)
    for i, r in enumerate(core):
        g[f'{r["src"]}#{r["page"]}'].append(i)
    keys = sorted(g)
    np.random.default_rng(seed).shuffle(keys)
    nt = max(1, int(round(len(keys) * test_frac)))
    tk = set(keys[:nt])
    tr = [i for k in keys if k not in tk for i in g[k]]
    te = [i for k in keys if k in tk for i in g[k]]
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
    """One split -> (metrics dict, per-item predictions for covered test items)."""
    proto_labels = list(protos)
    P = torch.stack([protos[l] for l in proto_labels])
    strata = collections.defaultdict(lambda: [0, 0, 0])
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
        s = strata[b]; s[0] += 1; s[1] += c1; s[2] += c5
        per_item.append({"image": r["image"], "q": r["q"], "true": r["label"],
                         "pred": order[0], "correct": bool(c1)})
    pct = lambda a, b: round(100 * a / b, 1) if b else 0.0
    metrics = {
        "n_protos": len(protos), "coverage_pct": pct(cov, tot),
        "sel_top1": pct(t1, cov), "sel_top3": pct(t3, cov), "sel_top5": pct(t5, cov),
        "strata": {("4+" if b == 4 else str(b)):
                   {"n": strata[b][0], "top1": pct(strata[b][1], strata[b][0]),
                    "top5": pct(strata[b][2], strata[b][0])} for b in sorted(strata)},
        "confusions": confus,
    }
    return metrics, per_item


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "?"


def _ms(vals):
    return {"mean": round(st.mean(vals), 1), "std": round(st.pstdev(vals), 1)}


def _report_md(cfg, args, sizes, agg, device):
    sigma = cfg.point.gauss_sigma_px
    core, ncls = sizes
    t1, t5 = agg["sel_top1"], agg["sel_top5"]
    chance = round(100 / max(1, agg["n_protos"]), 2)
    ratio = round(t1["mean"] / max(chance, 1e-9))
    strata_rows = "\n".join(
        f"| {k}-shot | {v['n']} | {v['top1']}% | {v['top5']}% |"
        for k, v in agg["strata"].items())
    conf_rows = "\n".join(f"- `{p}` x{c}" for p, c in agg["confusions"][:8]) or "- (없음)"
    return f"""# 베이스라인 ({args.name}) — M4' 외형 MVP

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/eval_appearance.py`  (multi-seed 기본)

## 목적
외형 신호만으로 **학습 없이**(frozen DINOv2 + GaussianPool + 프로토타입) 핀이 가리킨
구조물을 식별할 수 있는지 검증. 테스트셋이 작아(~180) 단일 split은 ±3-4%p로 흔들리므로,
임베딩을 한 번만 하고 **{args.n_seeds}개 seed**로 분할만 바꿔 **mean±std**를 보고한다.

## 방법
`I → frozen DINOv2(패치 격자) → 핀 q에서 GaussianPool → z_q(L2 정규화) →
클래스 프로토타입(갤러리 평균)과 cosine 최근접 = 예측.` (metric learning, long-tail용)

## 설정
| 항목 | 값 |
|---|---|
| 백본 | dinov2_vitb14, 518px, frozen, {device} |
| 풀링 | GaussianPool σ={sigma}px |
| 데이터 | `{args.jsonl}`, ≥{args.min_count} 코어: {core} 트리플 / {ncls} 클래스 |
| 분할 | 표본(페이지) 단위, test_frac={args.test_frac}, seeds=0..{args.n_seeds - 1} |
| 프로토타입 | 평균 {agg['n_protos']}개 |

## 결과 ({args.n_seeds}-seed, mean±std)
| 지표 | 값 |
|---|---|
| coverage | {agg['coverage_pct']['mean']}% |
| **selective top1** | **{t1['mean']} ± {t1['std']}%** |
| selective top3 | {agg['sel_top3']['mean']} ± {agg['sel_top3']['std']}% |
| selective top5 | {t5['mean']} ± {t5['std']}% |
| 무작위 기대 top1 | {chance}% |

per-seed top1: {agg['per_seed_top1']}

![top-k](fig_topk.png)

### support(갤러리 샷 수)별 정확도 (seed 평균)
| 버킷 | n(평균) | top1 | top5 |
|---|---|---|---|
{strata_rows}

![support](fig_support.png)

### 주요 혼동 (정답 → 예측, seed 합산 상위)
{conf_rows}

![confusions](fig_confusions.png)

### 예측 예시 (seed 0, O=정답 X=오답)
![examples](examples.private.png)
> ⚠️ 카데바 이미지 포함 → git 제외(로컬 전용, §6).

## 해석
- top1 **{t1['mean']}±{t1['std']}%** = 무작위({chance}%)의 약 **{ratio}배** → 외형 신호 실재.
- ±{t1['std']}%p가 분할 노이즈 폭. 단일 seed 비교로 작은 차이를 논하면 안 됨.
- 혼동은 대부분 해부학적 인접/유사 구조물 → 관계추론(M6')이 메울 지점.

## 한계
무파라미터 GaussianPool · 관계추론 없음 · 보정/기권 미적용 · 모달리티 혼재 · 코어 {core} 트리플.

## 다음
풀링(σ) · 모달리티 분리 · M5' 보정+기권. 비교는 항상 multi-seed mean±std로.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="data/triples/triples.jsonl")
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--name", default="baseline")
    args = ap.parse_args()
    base = Path(args.jsonl).parent

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg()
    S = cfg.backbone.image_size
    print(f"device={device} backbone={cfg.backbone.name} sigma={cfg.point.gauss_sigma_px}px seeds={args.n_seeds}")

    core = load_core(args.jsonl, args.min_count)
    ncls = len({r["label"] for r in core})
    print(f"core {len(core)} / {ncls} cls")

    print("loading frozen DINOv2 + embedding core once ...")
    backbone = DinoBackbone(cfg.backbone)
    backbone.ensure_loaded()
    backbone.to(device)
    pool = GaussianPool(cfg.point).to(device)
    Z = embed(core, base, backbone, pool, S, device)

    # ---- evaluate over seeds (embedding is split-independent) ----------------
    ms, per_item0 = [], None
    for seed in range(args.n_seeds):
        tr, te = split_indices(core, args.test_frac, seed)
        trr, ter = [core[i] for i in tr], [core[i] for i in te]
        protos, support = prototypes(trr, Z[tr])
        m, per_item = evaluate(ter, Z[te], protos, support)
        ms.append(m)
        if seed == 0:
            per_item0 = per_item
        print(f"  seed {seed}: top1 {m['sel_top1']:.1f} top5 {m['sel_top5']:.1f} cov {m['coverage_pct']:.0f}")

    # ---- aggregate -----------------------------------------------------------
    agg = {
        "n_protos": round(st.mean([m["n_protos"] for m in ms])),
        "coverage_pct": _ms([m["coverage_pct"] for m in ms]),
        "sel_top1": _ms([m["sel_top1"] for m in ms]),
        "sel_top3": _ms([m["sel_top3"] for m in ms]),
        "sel_top5": _ms([m["sel_top5"] for m in ms]),
        "per_seed_top1": [m["sel_top1"] for m in ms],
    }
    sa, sn, s5 = (collections.defaultdict(list) for _ in range(3))
    for m in ms:
        for b, v in m["strata"].items():
            sa[b].append(v["top1"]); sn[b].append(v["n"]); s5[b].append(v["top5"])
    agg["strata"] = {b: {"n": round(st.mean(sn[b])), "top1": round(st.mean(sa[b]), 1),
                         "top5": round(st.mean(s5[b]), 1)} for b in sorted(sa)}
    conf = collections.Counter()
    for m in ms:
        conf.update(m["confusions"])
    agg["confusions"] = conf.most_common(10)
    print(f"\n== {args.n_seeds}-seed: top1 {agg['sel_top1']['mean']}±{agg['sel_top1']['std']}  "
          f"top5 {agg['sel_top5']['mean']}±{agg['sel_top5']['std']}  cov {agg['coverage_pct']['mean']}% ==")

    # ---- experiment folder ---------------------------------------------------
    d = explog.next_dir(args.name)
    explog.bar(d / "fig_topk.png", ["top1", "top3", "top5"],
               [agg["sel_top1"]["mean"], agg["sel_top3"]["mean"], agg["sel_top5"]["mean"]],
               f"Selective accuracy ({args.n_seeds}-seed mean±std)", "%", ymax=100,
               errors=[agg["sel_top1"]["std"], agg["sel_top3"]["std"], agg["sel_top5"]["std"]])
    explog.grouped_bar(
        d / "fig_support.png", list(agg["strata"]),
        {"top1": [v["top1"] for v in agg["strata"].values()],
         "top5": [v["top5"] for v in agg["strata"].values()]},
        "Accuracy by gallery support count (seed mean)", "%", ymax=100)
    explog.barh_pairs(d / "fig_confusions.png", agg["confusions"],
                      "Top confusions (true -> pred, seeds summed)")
    right = [x for x in per_item0 if x["correct"]][:6]
    wrong = [x for x in per_item0 if not x["correct"]][:6]
    explog.montage(d / "examples.private.png", right + wrong, base, ncol=4)

    m_full = dict(agg)
    m_full.update({"title": f"베이스라인 ({args.name})",
                   "date": datetime.date.today().isoformat(),
                   "headline": f"top1 {agg['sel_top1']['mean']}±{agg['sel_top1']['std']}% / "
                               f"top5 {agg['sel_top5']['mean']}±{agg['sel_top5']['std']}% "
                               f"@cov{agg['coverage_pct']['mean']}% ({agg['n_protos']}-way, {args.n_seeds} seeds)",
                   "config": {"backbone": cfg.backbone.name, "sigma_px": cfg.point.gauss_sigma_px,
                              "min_count": args.min_count, "test_frac": args.test_frac,
                              "n_seeds": args.n_seeds}})
    explog.write(d, _report_md(cfg, args, (len(core), ncls), agg, device), m_full)
    print(f"wrote experiment -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
