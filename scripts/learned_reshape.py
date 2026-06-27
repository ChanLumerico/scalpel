"""Experiment 049 — M-rep1: learned representation reshape (the last untested representation lever).

045 (resolution) delivered; 046 (tissue gate) / 047 (relational) / 048 (resolution refine) were
negatives. The one lever never tried: *learn* a reshape of the space so the nearest exemplar stops
crossing regions/tissues. 046 warns the exemplar ALREADY uses DINO's implicit tissue axis (AUC 0.76),
so a learned head must ADD beyond that — and the class-level SupCon trap (502 classes, ~1.7/fold-class)
is real. Mitigations (handout §0.2): train at TISSUE level (367/tissue) and a HIERARCHICAL tissue+class
objective; keep the frozen region structure by also evaluating [frozen ; head] concat (region for top5,
head for within-region tissue id).

Supervised-contrastive head (MLP on global+L256, frozen backbone), trained PER FOLD on the gallery only
(leak-safe). Objectives: tissue / class / hierarchical. Eval: exemplar 1-NN on (head) and (frozen⊕head).
Protocol §1.7: dev 10-seed CV select, sealed test once. Baseline = frozen global+L256 (045: dev 33.5).

    .venv/bin/python scripts/learned_reshape.py
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
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import explog  # noqa: E402
from eval_appearance import _git_sha  # noqa: E402
from eval_merged import load, exemplar_eval, block_split, BASE  # noqa: E402
from split_devtest import get_split  # noqa: E402

SEEDS = 10
EPOCHS = 120
TAU = 0.1
TIS = {"artery": "artery", "arteries": "artery", "vein": "vein", "veins": "vein",
       "nerve": "nerve", "nerves": "nerve", "cn": "nerve", "muscle": "muscle",
       "muscles": "muscle", "bone": "bone", "joint": "bone"}


def tissue(lab):
    toks = lab.split()
    if "cn" in toks:
        return "nerve"
    for t in reversed(toks):
        if t in TIS:
            return TIS[t]
    return "other"


def unit(Z):
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)


def ms(v):
    v = [x for x in v if x == x]
    return (round(st.mean(v), 1), round(st.pstdev(v), 1)) if v else (float("nan"), 0.0)


class Head(nn.Module):
    def __init__(self, din, dout=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, 512), nn.GELU(), nn.Linear(512, dout))

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


def supcon(z, lab_t, lab_c, mode, device):
    """Supervised contrastive loss. mode: 'tissue' | 'class' | 'hier' (tissue+0.5class)."""
    sim = z @ z.T / TAU
    n = z.shape[0]
    self_mask = torch.eye(n, device=device, dtype=torch.bool)
    sim = sim.masked_fill(self_mask, -1e9)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    def term(lab):
        L = torch.tensor([hash(x) for x in lab], device=device)
        pos = (L[:, None] == L[None, :]) & ~self_mask
        cnt = pos.sum(1)
        ok = cnt > 0
        if ok.sum() == 0:
            return torch.tensor(0.0, device=device)
        return -((logp * pos).sum(1)[ok] / cnt[ok]).mean()

    if mode == "tissue":
        return term(lab_t)
    if mode == "class":
        return term(lab_c)
    return term(lab_t) + 0.5 * term(lab_c)


def train_head(Xtr, lab_t, lab_c, mode, din, device):
    head = Head(din).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    X = torch.tensor(Xtr, device=device)
    for _ in range(EPOCHS):
        opt.zero_grad()
        z = head(X)
        loss = supcon(z, lab_t, lab_c, mode, device)
        loss.backward(); opt.step()
    head.eval()
    with torch.no_grad():
        return head(torch.tensor(Xtr, device=device)).cpu().numpy(), head


def embed_all(head, X, device):
    with torch.no_grad():
        return head(torch.tensor(X, device=device)).cpu().numpy()


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = load(); Y = [r["label"] for r in rows]
    Ty = [tissue(y) for y in Y]
    split = get_split()
    cnt = collections.Counter(Y)
    core = [i for i in range(len(rows)) if cnt[Y[i]] >= 2]
    dev = [i for i in core if split[rows[i]["image"]] == "dev"]
    test = [i for i in core if split[rows[i]["image"]] == "test"]
    block = [json.loads((BASE / "_blocks.json").read_text())[r["image"]] for r in rows]
    zg = unit(np.load(BASE / "_dino_cache.npy").astype(np.float32))
    zl = unit(np.load(BASE / "_local256_cache.npy").astype(np.float32))
    Z = unit(np.concatenate([zg, zl], 1)).astype(np.float32)  # frozen global+L256 (045 best)
    din = Z.shape[1]
    print(f"core {len(core)} | dev {len(dev)} / test {len(test)} | din {din} | {device}")

    splits = [block_split(dev, block, s) for s in range(SEEDS)]

    def eval_space(Zspace, tr, te):
        return exemplar_eval(Zspace, Y, tr, te)[0]

    # baseline: frozen global+L256
    base = [eval_space(Z, tr, te) for tr, te in splits]
    print(f"baseline frozen global+L256 dev-CV {ms(base)[0]}±{ms(base)[1]}")

    modes = ["tissue", "class", "hier"]
    # variants: head-only and frozen⊕head, per objective
    res = {f"{m}:{sp}": [] for m in modes for sp in ("head", "frozen+head")}
    for si, (tr, te) in enumerate(splits):
        lab_t = [Ty[i] for i in tr]; lab_c = [Y[i] for i in tr]
        Xtr = Z[tr]
        for m in modes:
            _, head = train_head(Xtr, lab_t, lab_c, m, din, device)
            H = unit(embed_all(head, Z, device))
            res[f"{m}:head"].append(eval_space(H, tr, te))
            res[f"{m}:frozen+head"].append(eval_space(unit(np.concatenate([Z, H], 1)), tr, te))
        print(f"  seed {si}: " + " ".join(f"{k.split(':')[0][:3]}/{k.split(':')[1][:1]} {res[k][-1]:.0f}" for k in res))

    def paired(a):
        d = [x - y for x, y in zip(a, base)]
        return round(st.mean(d), 2), int(sum(x > 0 for x in d))

    print(f"\n== dev-CV 10-seed (baseline {ms(base)[0]}) ==")
    table = {}
    for k in res:
        dlt = paired(res[k]); table[k] = (ms(res[k]), dlt)
        print(f"  {k:22} top1 {ms(res[k])[0]}±{ms(res[k])[1]}  Δ {dlt[0]:+} ({dlt[1]}/10)")

    best = max(res, key=lambda k: st.mean(res[k]))
    d_best = table[best][1]
    adopt = d_best[0] > 0 and d_best[1] >= 7

    # sealed test: retrain best objective on full dev, eval test
    m_best, sp_best = best.split(":")
    lab_t = [Ty[i] for i in dev]; lab_c = [Y[i] for i in dev]
    _, head = train_head(Z[dev], lab_t, lab_c, m_best, din, device)
    H = unit(embed_all(head, Z, device))
    Zbest = H if sp_best == "head" else unit(np.concatenate([Z, H], 1))
    bt1, bt5, bcov = exemplar_eval(Zbest, Y, dev, test)
    base_test = exemplar_eval(Z, Y, dev, test)[0]
    labset = set(Y[i] for i in dev); cov = [q for q in test if Y[q] in labset]
    sims = Zbest[cov] @ Zbest[dev].T
    cols = collections.defaultdict(list); labs = sorted(labset); li = {l: j for j, l in enumerate(labs)}
    for j, i in enumerate(dev):
        cols[li[Y[i]]].append(j)
    sc = np.full((len(cov), len(labs)), -2.0, np.float32)
    for c, ix in cols.items():
        sc[:, c] = sims[:, ix].max(1)
    corr = np.array([labs[sc[r].argmax()] == Y[cov[r]] for r in range(len(cov))])
    rng = np.random.default_rng(0)
    boot = sorted(100 * corr[rng.integers(0, len(corr), len(corr))].mean() for _ in range(2000))
    ci = (round(boot[50], 1), round(boot[1950], 1))
    print(f"\n  ★ SEALED TEST: frozen global+L256 {round(base_test,1)} → best({best}) {round(bt1,1)} "
          f"(CI {ci[0]}–{ci[1]}) | dev Δ {d_best[0]:+} ({d_best[1]}/10) → "
          f"{'🟢 ADOPT (학습형 표현이 이김)' if adopt else '🔴 학습형 표현도 frozen 못 넘음 (SupCon 함정 재확인)'}")

    d = explog.EXP / "049-learned-reshape"; d.mkdir(parents=True, exist_ok=True)
    ks = list(res)
    explog.bar(d / "fig1_objectives.png", [k.replace(":", "\n") for k in ks],
               [table[k][0][0] for k in ks], "049 learned reshape (SupCon) vs frozen global+L256",
               "dev-CV top1 %", ymax=42, errors=[table[k][0][1] for k in ks])
    explog.bar(d / "fig2_delta.png", [k.replace(":", "\n") for k in ks],
               [table[k][1][0] for k in ks], "049 paired Δtop1 vs frozen global+L256 (33.5)", "Δ top1 pp")

    rowmd = "\n".join(f"| {k} | {table[k][0][0]}±{table[k][0][1]} | {table[k][1][0]:+} | {table[k][1][1]}/10 |" for k in ks)
    verdict = (f"🟢 **{best}** 가 frozen global+L256를 이김 (dev Δ{d_best[0]:+}, {d_best[1]}/10, 봉인 {round(bt1,1)}) → 채택."
               if adopt else
               "🔴 **학습형 표현도 frozen exemplar를 못 넘는다.** tissue/class/hier 어느 SupCon 목적도, head 단독도 "
               "frozen⊕head도 가산 없음 — 046의 'exemplar가 이미 암묵 조직축을 쓴다'가 학습 차원에서 재확인. "
               "표현 축 전체(해상도 제외) 소진 → 데이터가 유일하게 남은 검증된 레버.")
    report = f"""# 049 — M-rep1: 학습형 표현 reshape (SupCon head, frozen global+L256 위)

- 날짜: {datetime.date.today().isoformat()} · 커밋 `main @ {_git_sha()}` · `scripts/learned_reshape.py`
- clean 502 (dev {len(dev)}/test {len(test)} 봉인), dev 10-seed CV 선택 + 봉인 test 1회 (§1.7).
- SupCon head(MLP 1536→512→128), **train fold만** 학습(누수안전). 목적: tissue/class/hier. 평가: head 단독 + frozen⊕head.
- 046 경고 반영: exemplar가 이미 DINO 암묵 조직축(0.76)을 쓰므로 head는 *그 이상*을 더해야 채택.

## 결과 (paired Δ vs frozen global+L256)
| variant (목적:공간) | dev-CV top1 | Δ | wins |
|---|---|---|---|
{rowmd}

- **봉인 TEST: frozen {round(base_test,1)} → best({best}) {round(bt1,1)}** (CI {ci[0]}–{ci[1]}).
- 판정: {verdict}

![objectives](fig1_objectives.png)
![delta](fig2_delta.png)

## 핵심
- tissue-level은 SupCon 함정(class 1.7/fold) 회피하나, {'그래도 ' if not adopt else ''}region 보존(frozen⊕head)으로도 {('가산' if adopt else '무가산')}.
- {'학습형 표현 채택 — 표현 축 추가 전진.' if adopt else 'frozen exemplar가 학습형 reshape를 이김 — 표현 축은 해상도(045) 외 소진. 다음 = 데이터.'}
"""
    explog.write(d, report, {
        "title": "학습형 표현 reshape (SupCon head)", "date": datetime.date.today().isoformat(),
        "headline": f"M-rep1 best={best} dev {ms(res[best])[0]} (Δ vs frozen {d_best[0]:+}, {d_best[1]}/10) | "
                    f"봉인 frozen {round(base_test,1)}→best {round(bt1,1)} (CI {ci[0]}-{ci[1]}) {'🟢채택' if adopt else '🔴무가산'}",
        "baseline_devcv": ms(base), "best": best, "adopt": bool(adopt),
        "devcv": {k: {"top1": table[k][0], "delta": table[k][1]} for k in ks},
        "sealed": {"frozen": round(base_test, 1), "best": round(bt1, 1), "ci": list(ci)}})
    print(f"\nwrote -> {d}  (2 figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
