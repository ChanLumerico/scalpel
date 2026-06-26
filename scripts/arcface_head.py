"""Experiment — angular-margin head (ArcFace-style) to sharpen look-alike boundaries.

The SupCon head helped (+2.6 top1) by clustering same-structure embeddings. Plain
SupCon has NO margin — positives only need to be closer than negatives, not closer
by a gap. For fine same-region look-alikes (artery vs vein) a hard ANGULAR MARGIN
should carve a cleaner boundary. We add an additive angular margin m to positive
pairs inside the supervised-contrastive objective (ArcFace's cos(θ+m), applied
contrastively so it stays few-shot compatible — no fixed classifier weights).

m=0 reproduces plain SupCon, so this is a clean paired ablation. exemplar 1-NN in
the learned space, 10-seed, paired vs frozen AND vs m=0.

    .venv/bin/python scripts/arcface_head.py
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
from learned_head import class_max, exemplar_acc  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402

torch.manual_seed(0)
MARGINS = [0.0, 0.1, 0.2, 0.3]


def margin_supcon(z, labels, temp=0.1, m=0.2):
    """Supervised contrastive loss with an additive angular margin on positives."""
    cos = (z @ z.T).clamp(-1 + 1e-6, 1 - 1e-6)
    pos = (labels[:, None] == labels[None, :]).clone()
    pos.fill_diagonal_(False)
    if m > 0:
        theta = torch.acos(cos)
        cos_m = torch.cos(theta + m)                 # harder target: cos(θ+m) ≤ cosθ
        sim = torch.where(pos, cos_m, cos) / temp
    else:
        sim = cos / temp
    sim.fill_diagonal_(-1e9)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    posf = pos.float()
    cnt = posf.sum(1); valid = cnt > 0
    loss = -(logp * posf).sum(1)[valid] / cnt[valid]
    return loss.mean() if valid.any() else z.sum() * 0.0


def train_head(Ztr, ytr, m, dim=256, steps=300, lr=1e-3, wd=1e-3, drop=0.2, device="cpu"):
    torch.manual_seed(0)
    head = nn.Sequential(nn.Dropout(drop), nn.Linear(Ztr.shape[1], dim)).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    Z = torch.from_numpy(Ztr).to(device)
    uid = {l: i for i, l in enumerate(sorted(set(ytr)))}
    y = torch.tensor([uid[l] for l in ytr], device=device)
    head.train()
    for _ in range(steps):
        opt.zero_grad()
        out = F.normalize(head(Z), dim=1)
        loss = margin_supcon(out, y, m=m)
        loss.backward(); opt.step()
    head.eval()
    return head


def main():
    base = Path("data/triples")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size
    core = load_core("data/triples/triples.jsonl", 2)
    Y = [r["label"] for r in core]
    print(f"core {len(core)}/{len(set(Y))} | device={device}")
    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    pool = GaussianPool(cfg.point).to(device)
    print("embedding once..."); Znp = embed(core, base, bb, pool, S, device).numpy().astype(np.float32)
    hdev = "cpu"

    froz = ([], [])
    learned = {m: ([], []) for m in MARGINS}
    for seed in range(10):
        tr, te = split_indices(core, 0.3, seed)
        Ztr, Zte = Znp[tr], Znp[te]
        ytr = [Y[i] for i in tr]; yte = [Y[i] for i in te]
        f = exemplar_acc(Ztr, ytr, Zte, yte)
        froz[0].append(f[0]); froz[1].append(f[1])
        line = f"  seed {seed}: frozen {f[0]:.0f}"
        for m in MARGINS:
            head = train_head(Ztr, ytr, m, device=hdev)
            with torch.no_grad():
                Ptr = F.normalize(head(torch.from_numpy(Ztr).to(hdev)), dim=1).cpu().numpy()
                Pte = F.normalize(head(torch.from_numpy(Zte).to(hdev)), dim=1).cpu().numpy()
            lr_ = exemplar_acc(Ptr, ytr, Pte, yte)
            learned[m][0].append(lr_[0]); learned[m][1].append(lr_[1])
            line += f" | m{m} {lr_[0]:.0f}"
        print(line)

    ms = lambda v: (round(st.mean(v), 1), round(st.pstdev(v), 1))
    f1, f5 = ms(froz[0]), ms(froz[1])
    rows = []
    for m in MARGINS:
        t1, t5 = learned[m]
        dvf = [a - b for a, b in zip(t1, froz[0])]
        dv0 = [a - b for a, b in zip(t1, learned[0.0][0])]
        rows.append((m, ms(t1), ms(t5), round(st.mean(dvf), 1), sum(x > 0 for x in dvf),
                     round(st.mean(dv0), 1), sum(x > 0 for x in dv0)))
    best = max(rows[1:], key=lambda r: r[5])          # best margin vs m=0
    verdict = ("각마진이 SupCon보다 향상 (paired)" if (best[5] > 0 and best[6] >= 8)
               else "각마진 추가 이득 없음 (SupCon로 충분)")
    print(f"\n== frozen {f1[0]} | " + " ".join(f"m{m} {ms(learned[m][0])[0]}" for m in MARGINS) +
          f" | best m{best[0]} Δvs-m0 {best[5]:+}({best[6]}/10) -> {verdict} ==")

    d = explog.next_dir("arcface-head")
    explog.bar(d / "fig_margin.png", ["frozen"] + [f"m={m}" for m in MARGINS],
               [f1[0]] + [ms(learned[m][0])[0] for m in MARGINS],
               "Angular-margin head: top1 (10-seed)", "%", ymax=100,
               errors=[f1[1]] + [ms(learned[m][0])[1] for m in MARGINS])
    tab = "\n".join(
        f"| {m if m else '0 (SupCon)'} | {t1[0]}±{t1[1]}% | {t5[0]}% | {df:+} ({wf}/10) | {d0:+} ({w0}/10) |"
        for m, t1, t5, df, wf, d0, w0 in rows)
    report = f"""# 각마진 헤드 (arcface-head)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/arcface_head.py`

## 목적
SupCon 헤드(마진 없음) 위에 **additive angular margin** `cos(θ+m)` 를 positive 쌍에 부과해 same-region
look-alike 경계를 더 날카롭게. m=0 = 기존 SupCon → 깨끗한 paired ablation. exemplar 1-NN, 10-seed.

## 결과 (mean±std, paired)
| margin m | top1 | top5 | Δ vs frozen | Δ vs m=0 |
|---|---|---|---|---|
{tab}

(frozen top1 {f1[0]}±{f1[1]}%)

![margin](fig_margin.png)

## 판정
- 베스트 m={best[0]}: vs SupCon(m=0) Δtop1 {best[5]:+}%p ({best[6]}/10) → **{verdict}**

## 해석
- 마진이 도우면 → 경계가 데이터로 underdetermined가 아니라 *손실 형태*가 레버. 안 도우면 → 천장은
  여전히 데이터(같은-부위 판별 정보가 외형에 없음), 손실 trick 무효.
"""
    explog.write(d, report, {
        "title": "각마진 헤드", "date": datetime.date.today().isoformat(),
        "headline": f"best m={best[0]} vs-SupCon Δtop1 {best[5]:+}({best[6]}/10) → {verdict}",
        "frozen_top1": f1, "margins": {str(m): {"top1": t1, "top5": t5, "dvs_frozen": df, "dvs_m0": d0}
                                       for m, t1, t5, df, wf, d0, w0 in rows}})
    print(f"wrote -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
