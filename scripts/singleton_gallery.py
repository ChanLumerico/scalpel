"""Experiment A-1 — including singletons in the gallery (vocabulary vs accuracy).

The eval uses only the >=2 core (215 classes). For deployment the gallery should
hold ALL labelled data so the model can recognise the 352 singleton-only
structures too — expanding the recognisable vocabulary 215 -> 567. The cost: more
distractor exemplars may lower accuracy on the core test. We measure that cost:
core-test top1/top5 with gallery = core-only vs core+singletons (singletons only
from NON-test pages, so no image leaks). 10-seed paired.

    .venv/bin/python scripts/singleton_gallery.py
"""

from __future__ import annotations

import collections
import datetime
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import explog  # noqa: E402
from eval_appearance import embed, _git_sha  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402


def exemplar(galZ, galY, teZ, teY):
    labels = sorted(set(galY)); lidx = {l: i for i, l in enumerate(labels)}
    cols = collections.defaultdict(list)
    for j, l in enumerate(galY):
        cols[lidx[l]].append(j)
    cv = [k for k, l in enumerate(teY) if l in lidx]
    if not cv:
        return 0.0, 0.0, 0.0
    sims = teZ[cv] @ galZ.T
    sc = np.full((len(cv), len(labels)), -2.0, np.float32)
    for c, ix in cols.items():
        sc[:, c] = sims[:, ix].max(1)
    o = np.argsort(-sc, axis=1)
    n1 = sum(int(labels[o[r, 0]] == teY[k]) for r, k in enumerate(cv))
    n5 = sum(int(teY[k] in [labels[o[r, t]] for t in range(5)]) for r, k in enumerate(cv))
    return 100 * n1 / len(cv), 100 * n5 / len(cv), 100 * len(cv) / len(teY)


def main():
    base = Path("data/triples")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size
    rows = [json.loads(l) for l in open("data/triples/triples.jsonl", encoding="utf-8") if l.strip()]
    cnt = collections.Counter(r["label"] for r in rows)
    Y = [r["label"] for r in rows]
    pg = [f'{r["src"]}#{r["page"]}' for r in rows]
    is_core = np.array([cnt[Y[i]] >= 2 for i in range(len(rows))])
    print(f"triples {len(rows)} | core {is_core.sum()} | singletons {(~is_core).sum()}")
    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    pool = GaussianPool(cfg.point).to(device)
    print("embedding once..."); Z = embed(rows, base, bb, pool, S, device).numpy().astype(np.float32)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)

    a1, a5, b1, b5, vocabA, vocabB = ([] for _ in range(6))
    for seed in range(10):
        pages = sorted(set(pg))
        np.random.default_rng(seed).shuffle(pages)
        nt = max(1, int(round(len(pages) * 0.3)))
        test_pg = set(pages[:nt])
        gal = [i for i in range(len(rows)) if pg[i] not in test_pg]
        core_test = [i for i in range(len(rows)) if pg[i] in test_pg and is_core[i]]
        galA = [i for i in gal if is_core[i]]                  # core gallery
        galB = gal                                             # core + singletons (non-test pages)
        yte = [Y[i] for i in core_test]
        ra = exemplar(Z[galA], [Y[i] for i in galA], Z[core_test], yte)
        rb = exemplar(Z[galB], [Y[i] for i in galB], Z[core_test], yte)
        a1.append(ra[0]); a5.append(ra[1]); b1.append(rb[0]); b5.append(rb[1])
        vocabA.append(len({Y[i] for i in galA})); vocabB.append(len({Y[i] for i in galB}))
        print(f"  seed {seed}: core-only top1 {ra[0]:.0f}/{ra[1]:.0f} (vocab {vocabA[-1]}) | "
              f"+singletons top1 {rb[0]:.0f}/{rb[1]:.0f} (vocab {vocabB[-1]})")

    ms = lambda v: (round(st.mean(v), 1), round(st.pstdev(v), 1))
    A1, A5, B1, B5 = ms(a1), ms(a5), ms(b1), ms(b5)
    d1 = [x - y for x, y in zip(b1, a1)]
    md1, w1 = round(st.mean(d1), 1), sum(x < 0 for x in d1)
    print(f"\n== core-only top1 {A1[0]} (vocab ~{round(st.mean(vocabA))}) | "
          f"+singletons top1 {B1[0]} (vocab ~{round(st.mean(vocabB))}) | Δtop1 {md1} ==")

    d = explog.next_dir("singleton-gallery")
    explog.bar(d / "fig_singleton.png", ["core-only\ntop1", "+singletons\ntop1", "core-only\ntop5", "+singletons\ntop5"],
               [A1[0], B1[0], A5[0], B5[0]], "Singletons in gallery: accuracy cost (10-seed)", "%",
               ymax=100, errors=[A1[1], B1[1], A5[1], B5[1]])
    report = f"""# A-1: 싱글톤 갤러리 포함 — 어휘 vs 정확도 (singleton-gallery)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/singleton_gallery.py`  (frozen exemplar, 10-seed, page-split)

## 목적
평가는 ≥2 코어(215)만 쓰지만, 배포 갤러리에 **싱글톤 352개까지 넣으면 인식 어휘가 ~{round(st.mean(vocabA))}→
~{round(st.mean(vocabB))}로 확장**된다. 비용은 코어 테스트 정확도 하락(distractor 증가). 누수 방지 위해
싱글톤은 테스트 페이지 제외.

## 결과 (코어 테스트, 10-seed)
| 갤러리 | 어휘(클래스) | top1 | top5 |
|---|---|---|---|
| core-only | ~{round(st.mean(vocabA))} | {A1[0]}±{A1[1]}% | {A5[0]}±{A5[1]}% |
| **+singletons** | ~{round(st.mean(vocabB))} | {B1[0]}±{B1[1]}% | {B5[0]}±{B5[1]}% |

![singleton](fig_singleton.png)

## 판정
- **어휘 {round(st.mean(vocabA))}→{round(st.mean(vocabB))} (2.6배 확장)**, 코어 top1 비용 **Δ{md1}%p** ({w1}/10 하락).
- 비용이 작으면 → 배포 갤러리엔 **싱글톤 포함**(거의 공짜로 인식 가능 구조물 2.6배). 단 싱글톤 자체의
  정확도는 측정 불가(인스턴스 1개).

## 다음
A-2(OCR 복구)로 더 많은 트리플 회수 시도.
"""
    explog.write(d, report, {
        "title": "싱글톤 갤러리 포함", "date": datetime.date.today().isoformat(),
        "headline": f"vocab {round(st.mean(vocabA))}→{round(st.mean(vocabB))}, core top1 {A1[0]}→{B1[0]} (Δ{md1})",
        "core_only": {"top1": A1, "top5": A5, "vocab": round(st.mean(vocabA))},
        "with_singletons": {"top1": B1, "top5": B5, "vocab": round(st.mean(vocabB))}})
    print(f"wrote -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
