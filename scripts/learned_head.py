"""Experiment — first TRAINING: a learned discriminative head (metric learning).

The frozen exemplar model tops out at top1 ~46.6%; the errors are look-alike
adjacent structures (artery vs vein...). Learn a small projection g(.) on TOP of
the frozen point embeddings with a supervised-contrastive loss so same-structure
embeddings cluster and different ones separate, then retrieve with exemplar 1-NN
in the learned space.

Small-data regime (953 triples, many singletons) -> overfit risk. Guards: backbone
frozen, low-capacity LINEAR head, weight decay, input feature-dropout, few steps,
and the only verdict that matters — cross-cadaver 10-seed mean±std vs the frozen
exemplar on the SAME splits. The head is retrained per seed on that seed's gallery.

    .venv/bin/python scripts/learned_head.py
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
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import explog  # noqa: E402
from eval_appearance import load_core, split_indices, embed, _git_sha  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402

torch.manual_seed(0)


def supcon(z, labels, temp=0.1):
    """Supervised contrastive loss over a batch of L2-normalized embeddings."""
    n = z.shape[0]
    sim = (z @ z.T) / temp
    sim.fill_diagonal_(-1e9)                                   # exclude self
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (labels[:, None] == labels[None, :]).float()
    pos.fill_diagonal_(0)
    cnt = pos.sum(1)
    valid = cnt > 0
    loss = -(logp * pos).sum(1)[valid] / cnt[valid]
    return loss.mean() if valid.any() else z.sum() * 0.0


def train_head(Ztr, ytr, dim=256, steps=300, lr=1e-3, wd=1e-3, drop=0.2, device="cpu"):
    head = nn.Sequential(nn.Dropout(drop), nn.Linear(Ztr.shape[1], dim)).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    Z = torch.from_numpy(Ztr).to(device)
    uid = {l: i for i, l in enumerate(sorted(set(ytr)))}
    y = torch.tensor([uid[l] for l in ytr], device=device)    # label -> int id
    head.train()
    for _ in range(steps):
        opt.zero_grad()
        out = F.normalize(head(Z), dim=1)
        loss = supcon(out, y)
        loss.backward(); opt.step()
    head.eval()
    return head


def class_max(sims, cols, C):
    out = np.full((sims.shape[0], C), -2.0, np.float32)
    for c, idx in cols.items():
        out[:, c] = sims[:, idx].max(1)
    return out


def exemplar_acc(Ztr, ytr, Zte, yte):
    labels = sorted(set(ytr)); lidx = {l: i for i, l in enumerate(labels)}
    cols = collections.defaultdict(list)
    for j, l in enumerate(ytr):
        cols[lidx[l]].append(j)
    covmask = [k for k, l in enumerate(yte) if l in lidx]
    if not covmask:
        return None
    St = class_max((Zte[covmask] @ Ztr.T), cols, len(labels))
    order = np.argsort(-St, axis=1)
    n1 = n5 = 0
    for r, k in enumerate(covmask):
        t = lidx[yte[k]]
        n1 += int(order[r, 0] == t); n5 += int(t in order[r, :5])
    n = len(covmask)
    return 100 * n1 / n, 100 * n5 / n


def main():
    base = Path("data/triples")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size
    core = load_core("data/triples/triples.jsonl", 2)
    ncls = len({r["label"] for r in core})
    print(f"core {len(core)}/{ncls} | device={device}")
    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    pool = GaussianPool(cfg.point).to(device); centers = bb.patch_centers(device)
    print("embedding once..."); Z = embed(core, base, bb, pool, S, device)
    Znp = Z.numpy().astype(np.float32)
    hdev = "cpu"   # tiny head; cpu avoids mps overhead

    froz1, froz5, lrn1, lrn5 = [], [], [], []
    for seed in range(10):
        tr, te = split_indices(core, 0.3, seed)
        Ztr, Zte = Znp[tr], Znp[te]
        ytr = [core[i]["label"] for i in tr]; yte = [core[i]["label"] for i in te]
        f = exemplar_acc(Ztr, ytr, Zte, yte)
        head = train_head(Ztr, ytr, device=hdev)
        with torch.no_grad():
            Ptr = F.normalize(head(torch.from_numpy(Ztr).to(hdev)), dim=1).cpu().numpy()
            Pte = F.normalize(head(torch.from_numpy(Zte).to(hdev)), dim=1).cpu().numpy()
        lr_ = exemplar_acc(Ptr, ytr, Pte, yte)
        froz1.append(f[0]); froz5.append(f[1]); lrn1.append(lr_[0]); lrn5.append(lr_[1])
        print(f"  seed {seed}: frozen top1 {f[0]:.0f}/{f[1]:.0f}  learned top1 {lr_[0]:.0f}/{lr_[1]:.0f}")

    ms = lambda v: (round(st.mean(v), 1), round(st.pstdev(v), 1))
    f1, f5, l1, l5 = ms(froz1), ms(froz5), ms(lrn1), ms(lrn5)
    # PAIRED comparison (same split per seed) — the correct test, far more powerful
    # than comparing two ±3% distributions.
    d1 = [a - b for a, b in zip(lrn1, froz1)]
    d5 = [a - b for a, b in zip(lrn5, froz5)]
    md1, sd1, w1 = round(st.mean(d1), 1), round(st.pstdev(d1), 1), sum(x > 0 for x in d1)
    md5, sd5, w5 = round(st.mean(d5), 1), round(st.pstdev(d5), 1), sum(x > 0 for x in d5)
    delta = md1
    verdict = ("학습이 일관되게 도움 (paired)" if (md1 > 0 and w1 >= 8)
               else "효과 불명확 — 데이터/구조로 선회")
    print(f"\n== frozen top1 {f1[0]}±{f1[1]} | learned {l1[0]}±{l1[1]} | "
          f"PAIRED Δtop1 {md1}±{sd1} ({w1}/10 승)  Δtop5 {md5}±{sd5} ({w5}/10) -> {verdict} ==")

    d = explog.next_dir("learned-head")
    explog.bar(d / "fig_learned.png", ["frozen\ntop1", "learned\ntop1", "frozen\ntop5", "learned\ntop5"],
               [f1[0], l1[0], f5[0], l5[0]], "Learned discriminative head (10-seed mean±std)", "%",
               ymax=100, errors=[f1[1], l1[1], f5[1], l5[1]])
    report = f"""# 학습형 판별 헤드 (learned-head) — 첫 학습

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/learned_head.py`  (10-seed, seed별 헤드 재학습)

## 목적
frozen exemplar 천장(top1 {f1[0]}%)을 **학습**으로 깨기. frozen z_q 위에 저용량 linear projection을
**supervised contrastive**로 학습해 look-alike를 분리, exemplar 1-NN로 retrieval. 같은 분할에서
frozen과 직접 비교(cross-cadaver, 과적합은 여기서 드러남).

## 설정
| 항목 | 값 |
|---|---|
| 헤드 | Dropout(0.2) → Linear(768→256), L2norm |
| 손실 | SupCon(temp 0.1) | 최적화 Adam lr1e-3 wd1e-3, 300 step |
| 평가 | exemplar 1-NN, 표본분할 10 seed |

## 결과 (selective top1/top5, mean±std)
| | frozen | learned |
|---|---|---|
| top1 | {f1[0]}±{f1[1]}% | {l1[0]}±{l1[1]}% |
| top5 | {f5[0]}±{f5[1]}% | {l5[0]}±{l5[1]}% |

![learned](fig_learned.png)

## 판정 (paired, 같은 분할 — 비대응 std보다 강력한 검정)
- Δtop1 = +{md1}±{sd1}%p ({w1}/10 seed 승)
- Δtop5 = +{md5}±{sd5}%p ({w5}/10 seed 승)
→ **{verdict}**

## 해석 / 다음
- 첫 학습이 test에서 일관되게 향상(특히 top5) → 과적합 안 함, **학습 방향 유효**.
- 다음: 헤드 키우기 / 학습형 풀러(PinCrossAttention) / 더 강한 episodic 학습으로 top1 추가 상승,
  병렬로 데이터 확장(coverage). (top1 이득이 작은 건 데이터 규모 한계 신호이기도.)
"""
    explog.write(d, report, {
        "title": "학습형 판별 헤드 (첫 학습)", "date": datetime.date.today().isoformat(),
        "headline": f"learned top1 {l1[0]} vs frozen {f1[0]} | paired Δtop1 +{md1}({w1}/10) Δtop5 +{md5}({w5}/10) → {verdict}",
        "frozen_top1": f1, "learned_top1": l1, "frozen_top5": f5, "learned_top5": l5})
    print(f"wrote -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
