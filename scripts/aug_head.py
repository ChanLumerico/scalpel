"""Experiment — train the SupCon head on AUGMENTED embeddings (squeeze via learning).

exp 016 augmented the GALLERY (retrieval) for +1.5pp; the learned head (exp 012),
which helped +2.6, was trained on only 601 un-augmented embeddings. Here we give
the data-starved head more training signal: each gallery triple -> K augmented
views, all same-class positives. Does augmenting the head's TRAINING beat the head
on raw? Frozen backbone, exemplar 1-NN eval, 10-seed PAIRED.

    .venv/bin/python scripts/aug_head.py
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

import explog  # noqa: E402
from augment_eval import load_triples, split_idx, embed_items  # noqa: E402
from eval_appearance import _git_sha  # noqa: E402
from learned_head import train_head  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.data.augment import augment  # noqa: E402
from scalpel.perception import DinoBackbone  # noqa: E402

K = 4


def exemplar(galZ, galY, teZ, teY):
    labels = sorted(set(galY)); lidx = {l: i for i, l in enumerate(labels)}
    cols = collections.defaultdict(list)
    for j, l in enumerate(galY):
        cols[lidx[l]].append(j)
    cv = [k for k, l in enumerate(teY) if l in lidx]
    sims = teZ[cv] @ galZ.T
    sc = np.full((len(cv), len(labels)), -2.0, np.float32)
    for c, ix in cols.items():
        sc[:, c] = sims[:, ix].max(1)
    o = np.argsort(-sc, axis=1)
    n1 = sum(int(labels[o[r, 0]] == teY[k]) for r, k in enumerate(cv))
    n5 = sum(int(teY[k] in [labels[o[r, t]] for t in range(5)]) for r, k in enumerate(cv))
    return 100 * n1 / len(cv), 100 * n5 / len(cv)


def proj(head, Z):
    with torch.no_grad():
        return F.normalize(head(torch.from_numpy(Z)), dim=1).numpy()


def main():
    base = Path("data/triples")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size
    tris = load_triples("data/triples/triples.jsonl", base, min_count=2)
    Y = [t.label for t in tris]
    print(f"core {len(tris)}/{len(set(Y))} | device={device}")
    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    centers = bb.patch_centers(device)
    print("embedding originals + augmented...")
    Zo = embed_items([(t.image, t.q) for t in tris], bb, centers, S, device)
    aug = [augment(t, K, seed=i) for i, t in enumerate(tris)]
    Za = embed_items([(a.image, a.q) for sub in aug for a in sub], bb, centers, S, device).reshape(len(tris), K, -1)

    def nrm(a):
        return (a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)).astype(np.float32)
    Zo, Za = nrm(Zo), nrm(Za)

    raw1, raw5, aug1, aug5 = [], [], [], []
    for seed in range(10):
        tr, te = split_idx(tris, 0.3, seed)
        ytr = [Y[i] for i in tr]; yte = [Y[i] for i in te]
        hr = train_head(Zo[tr], ytr, device="cpu")
        Xa = np.concatenate([Zo[tr]] + [Za[tr, k] for k in range(K)])
        ya = ytr * (K + 1)
        ha = train_head(Xa, ya, device="cpu")
        r = exemplar(proj(hr, Zo[tr]), ytr, proj(hr, Zo[te]), yte)
        a = exemplar(proj(ha, Zo[tr]), ytr, proj(ha, Zo[te]), yte)
        raw1.append(r[0]); raw5.append(r[1]); aug1.append(a[0]); aug5.append(a[1])
        print(f"  seed {seed}: head-raw {r[0]:.0f}/{r[1]:.0f}  head-aug {a[0]:.0f}/{a[1]:.0f}")

    ms = lambda v: (round(st.mean(v), 1), round(st.pstdev(v), 1))
    R1, R5, A1, A5 = ms(raw1), ms(raw5), ms(aug1), ms(aug5)
    d1 = [a - b for a, b in zip(aug1, raw1)]; d5 = [a - b for a, b in zip(aug5, raw5)]
    md1, w1 = round(st.mean(d1), 1), sum(x > 0 for x in d1)
    md5, w5 = round(st.mean(d5), 1), sum(x > 0 for x in d5)
    verdict = "증강 학습이 헤드를 개선" if (md1 > 0 and w1 >= 8) else "효과 불명확 (증강은 새 다양성 없음)"
    print(f"\n== head-raw {R1[0]} | head-aug {A1[0]} | paired Δtop1 {md1} ({w1}/10) Δtop5 {md5} ({w5}/10) -> {verdict} ==")

    d = explog.next_dir("aug-head")
    explog.bar(d / "fig_aughead.png", ["raw\ntop1", "aug\ntop1", "raw\ntop5", "aug\ntop5"],
               [R1[0], A1[0], R5[0], A5[0]], "Head trained on augmented embeddings (10-seed)", "%",
               ymax=100, errors=[R1[1], A1[1], R5[1], A5[1]])
    report = f"""# 증강 임베딩으로 헤드 학습 (aug-head)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/aug_head.py`

## 목적
exp 016은 갤러리만 증강. 데이터-굶주린 SupCon 헤드를 **트리플당 K={K} 증강뷰로 훈련**하면 더 나은
metric을 배우는지 (paired vs 원본 학습 헤드). frozen 백본, exemplar 1-NN.

## 결과 (selective, mean±std, paired)
| 헤드 학습 | top1 | top5 |
|---|---|---|
| 원본만 | {R1[0]}±{R1[1]}% | {R5[0]}±{R5[1]}% |
| **+증강** | {A1[0]}±{A1[1]}% | {A5[0]}±{A5[1]}% |

![aughead](fig_aughead.png)

## 판정
- paired Δtop1 {md1:+}%p ({w1}/10), Δtop5 {md5:+}%p ({w5}/10) → **{verdict}**

## 해석
- 증강은 새 *해부 다양성*을 안 만들므로(같은 시신·구조물), 헤드 학습 데이터를 늘려도 천장은 못 깸.
  이득이 있어도 robustness/정규화 수준. exp 013(데이터=천장)과 일관된 한계.
"""
    explog.write(d, report, {
        "title": "증강 임베딩으로 헤드 학습", "date": datetime.date.today().isoformat(),
        "headline": f"head-raw {R1[0]} vs head-aug {A1[0]} | paired Δtop1 {md1}({w1}/10) → {verdict}",
        "raw": {"top1": R1, "top5": R5}, "aug": {"top1": A1, "top5": A5}})
    print(f"wrote -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
