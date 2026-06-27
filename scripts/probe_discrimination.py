"""Experiment — cheap discrimination diagnostic (training-free).

The top1 ceiling (~39%) comes from confusing adjacent look-alike structures. Before
investing in a learned metric head, ask cheaply: can a *training-free* change to
the retrieval rule already separate them? Three levers, all on the SAME frozen
embeddings, 10-seed mean±std:

    proto      : nearest MEAN prototype (current MVP)
    exemplar   : nearest single gallery exemplar (max sim) — does averaging wash
                 out fine detail?
    knn-vote   : majority vote of k nearest exemplars
    region     : proto score + lambda * region(CLS) similarity — does excluding
                 wrong-REGION look-alikes help? (lambda swept)

If any clears the baseline noise band on top1, a learned discriminator is likely
to pay off; if none does, the ceiling is feature-fundamental -> data / structured
relation.

    .venv/bin/python scripts/probe_discrimination.py
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
from eval_appearance import load_core, split_indices, _git_sha, _MEAN, _STD  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402

LAMBDAS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


@torch.no_grad()
def embed(core, base, backbone, pool, centers, S, device):
    """Per triple: pin embedding Zq (gauss40) and region descriptor CLS."""
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    by_img = collections.defaultdict(list)
    for i, r in enumerate(core):
        by_img[r["image"]].append(i)
    Zq = [None] * len(core)
    Cl = [None] * len(core)
    for n, (img, idxs) in enumerate(by_img.items(), 1):
        im = Image.open(base / img).convert("RGB")
        w, h = im.size
        arr = np.asarray(im.resize((S, S)), dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        grid, cls = backbone((x - mean) / std)
        c = F.normalize(cls[0], dim=0).cpu()
        for i in idxs:
            qx, qy = core[i]["q"]
            q = torch.tensor([[qx * S / w, qy * S / h]], device=device)
            Zq[i] = F.normalize(pool(grid, centers, q)[0], dim=0).cpu()
            Cl[i] = c
        if n % 25 == 0:
            print(f"   embedded {n}/{len(by_img)}")
    return torch.stack(Zq), torch.stack(Cl)


def topk_acc(scores, labels_order, true, k=(1, 5)):
    order = np.argsort(-scores)
    ranked = [labels_order[i] for i in order[:max(k)]]
    return {kk: (true in ranked[:kk]) for kk in k}


def run_seed(core, Zq, Cl, seed):
    tr, te = split_indices(core, 0.3, seed)
    Ztr, Ctr = Zq[tr].numpy(), Cl[tr].numpy()
    ytr = [core[i]["label"] for i in tr]
    # class prototypes + region signatures
    by = collections.defaultdict(list)
    for j, l in enumerate(ytr):
        by[l].append(j)
    labels = list(by)
    P = np.stack([_norm(Ztr[idx].mean(0)) for idx in by.values()])
    R = np.stack([_norm(Ctr[idx].mean(0)) for idx in by.values()])
    lidx = {l: i for i, l in enumerate(labels)}

    res = collections.defaultdict(lambda: [0, 0, 0])  # method -> [n, t1, t5]
    lam_t1 = {lam: [0, 0] for lam in LAMBDAS}          # region sweep: [n, t1]
    for j in te:
        true = core[j]["label"]
        if true not in lidx:
            continue
        zq, cl = Zq[j].numpy(), Cl[j].numpy()
        app = P @ zq                                  # (C,) appearance sim to prototypes
        # proto
        _acc(res["proto"], app, labels, true)
        # exemplar (max over gallery exemplars per class)
        sims = Ztr @ zq                               # (Ntr,)
        ex = np.full(len(labels), -2.0)
        np.maximum.at(ex, [lidx[y] for y in ytr], sims)
        _acc(res["exemplar"], ex, labels, true)
        # knn-vote (k=5): score = votes + tiny maxsim tiebreak
        nn = np.argsort(-sims)[:5]
        vote = np.zeros(len(labels))
        for rank, e in enumerate(nn):
            vote[lidx[ytr[e]]] += 1.0
        knn = vote + 1e-3 * ex                        # tiebreak by exemplar sim
        _acc(res["knn5"], knn, labels, true)
        # region-gated sweep
        reg = R @ cl
        for lam in LAMBDAS:
            s = app + lam * reg
            lam_t1[lam][0] += 1
            lam_t1[lam][1] += int(labels[int(np.argmax(s))] == true)
    best_lam = max(LAMBDAS, key=lambda L: lam_t1[L][1] / max(1, lam_t1[L][0]))
    out = {m: (100 * v[1] / v[0], 100 * v[2] / v[0]) for m, v in res.items()}
    out["region"] = (100 * lam_t1[best_lam][1] / max(1, lam_t1[best_lam][0]),
                     out["proto"][1])                  # top5 same family
    out["_best_lam"] = best_lam
    return out


def _norm(v):
    return v / (np.linalg.norm(v) + 1e-9)


def _acc(slot, scores, labels, true):
    order = np.argsort(-scores)
    ranked = [labels[i] for i in order[:5]]
    slot[0] += 1
    slot[1] += int(ranked[0] == true)
    slot[2] += int(true in ranked)


def main():
    base = Path("data/triples")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size
    core = load_core("data/triples/triples.jsonl", 2)
    ncls = len({r["label"] for r in core})
    print(f"core {len(core)}/{ncls} | device={device}")
    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    pool = GaussianPool(cfg.point).to(device); centers = bb.patch_centers(device)
    print("embedding (Zq + CLS) once...")
    Zq, Cl = embed(core, base, bb, pool, centers, S, device)

    methods = ["proto", "exemplar", "knn5", "region"]
    acc = {m: ([], []) for m in methods}
    lams = []
    for seed in range(10):
        o = run_seed(core, Zq, Cl, seed)
        lams.append(o["_best_lam"])
        for m in methods:
            acc[m][0].append(o[m][0]); acc[m][1].append(o[m][1])
        print(f"  seed {seed}: " + " ".join(f"{m} {o[m][0]:.0f}" for m in methods)
              + f"  (region λ={o['_best_lam']})")

    ms = lambda v: (round(st.mean(v), 1), round(st.pstdev(v), 1))
    rows = [(m, *ms(acc[m][0]), *ms(acc[m][1])) for m in methods]
    base_t1, base_sd = rows[0][1], rows[0][2]
    best = max(rows, key=lambda r: r[1])
    gain = round(best[1] - base_t1, 1)
    verdict = ("학습/구조 보상 신호 있음" if gain > base_sd
               else "노이즈 안 — training-free 재순위로는 top1 못 올림")

    d = explog.next_dir("discrimination-probe")
    explog.bar(d / "fig_methods.png", [r[0] for r in rows], [r[1] for r in rows],
               "Discrimination diagnostic: top1 (10-seed mean±std)", "%", ymax=100,
               errors=[r[2] for r in rows])
    table = "\n".join(f"| {m} | {t1}±{t1s}% | {t5}±{t5s}% |" for m, t1, t1s, t5, t5s in rows)
    report = f"""# 판별 진단 (discrimination-probe) — multi-seed

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/probe_discrimination.py`

## 목적
top1 천장(~{base_t1}%)이 *특징 한계*인지, *재순위/판별 규칙*으로 풀리는지 **학습 없이** 진단.
같은 frozen 임베딩에 retrieval 규칙만 바꿔본다: 평균 프로토타입 vs 최근접 exemplar vs k-NN
투표 vs region(CLS)-gated. 어느 하나라도 노이즈 밖으로 top1을 올리면 학습형 판별 헤드가 유망.

## 설정
| 항목 | 값 |
|---|---|
| 백본/풀링 | dinov2_vitb14 518px frozen · GaussianPool σ40 |
| 데이터 | ≥2 코어 {len(core)} 트리플 / {ncls} 클래스, 표본분할 10 seed |
| region λ | {LAMBDAS} 중 seed별 최적 (평균 λ≈{round(st.mean(lams),1)}) |

## 결과 (selective accuracy, mean±std)
| 방법 | top1 | top5 |
|---|---|---|
{table}

![methods](fig_methods.png)

## 판정
- 베스트: **{best[0]}** top1 {best[1]}±{best[2]}% (proto {base_t1}±{base_sd}, Δ{'+' if gain>=0 else ''}{gain}%p)
- → **{verdict}**

## 해석 / 다음
- training-free 재순위가 천장을 못 깨면 → **학습형 판별 헤드(metric learning / PinCrossAttention)**
  로 look-alike를 임베딩 공간에서 분리하는 게 다음 (frozen 백본 + 작은 헤드 + episodic + 증강 +
  cross-cadaver multi-seed 정직 평가).
- region이 도움되면 → 그 신호를 학습 목표/입력에 포함.
"""
    explog.write(d, report, {
        "title": "판별 진단", "date": datetime.date.today().isoformat(),
        "headline": f"best={best[0]} top1 {best[1]}±{best[2]}% (proto {base_t1}) → {verdict}",
        "methods": {m: {"top1": t1, "top1_std": t1s, "top5": t5} for m, t1, t1s, t5, t5s in rows}})
    print(f"\n{verdict}  ->  {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
