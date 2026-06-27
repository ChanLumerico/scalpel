"""Experiment — backbone scaling: does a bigger frozen DINOv2 lift the ceiling?

We have only ever used dinov2_vitb14 (the small end). Swap in vitl14 / vitg14
(stronger self-supervised features, frozen) and re-run the canonical exemplar
1-NN eval. No training. 10-seed mean±std, same splits, paired vs vitb14.

    .venv/bin/python scripts/backbone_scale.py
"""

from __future__ import annotations

import collections
import datetime
import statistics as st
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import explog  # noqa: E402
from eval_appearance import load_core, split_indices, embed, _git_sha  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402

BACKBONES = [("dinov2_vitb14", 768), ("dinov2_vitl14", 1024), ("dinov2_vitg14", 1536)]


def exemplar(Ztr, ytr, Zte, yte):
    labels = sorted(set(ytr)); lidx = {l: i for i, l in enumerate(labels)}
    cols = collections.defaultdict(list)
    for j, l in enumerate(ytr):
        cols[lidx[l]].append(j)
    cv = [k for k, l in enumerate(yte) if l in lidx]
    sims = Zte[cv] @ Ztr.T
    sc = np.full((len(cv), len(labels)), -2.0, np.float32)
    for c, ix in cols.items():
        sc[:, c] = sims[:, ix].max(1)
    o = np.argsort(-sc, axis=1)
    n1 = sum(int(labels[o[r, 0]] == yte[k]) for r, k in enumerate(cv))
    n5 = sum(int(yte[k] in [labels[o[r, t]] for t in range(5)]) for r, k in enumerate(cv))
    cov = 100 * len(cv) / len(yte)
    return 100 * n1 / len(cv), 100 * n5 / len(cv), cov


def main():
    base = Path("data/triples")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size
    core = load_core("data/triples/triples.jsonl", 2)
    Y = [r["label"] for r in core]
    print(f"core {len(core)}/{len(set(Y))} | device={device}")

    results = {}
    for name, dim in BACKBONES:
        print(f"\n=== {name} ===")
        try:
            bcfg = replace(cfg.backbone, name=name, embed_dim=dim)
            bb = DinoBackbone(bcfg); bb.ensure_loaded(); bb.to(device)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {name}: {type(e).__name__}: {e}")
            continue
        pool = GaussianPool(cfg.point).to(device)
        Z = embed(core, base, bb, pool, S, device).numpy().astype(np.float32)
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        del bb
        if device == "mps":
            torch.mps.empty_cache()
        t1, t5 = [], []
        for seed in range(10):
            tr, te = split_indices(core, 0.3, seed)
            a, b, _ = exemplar(Z[tr], [Y[i] for i in tr], Z[te], [Y[i] for i in te])
            t1.append(a); t5.append(b)
        results[name] = (t1, t5)
        print(f"  top1 {st.mean(t1):.1f}±{st.pstdev(t1):.1f}  top5 {st.mean(t5):.1f}±{st.pstdev(t5):.1f}")

    ms = lambda v: (round(st.mean(v), 1), round(st.pstdev(v), 1))
    names = [n for n, _ in BACKBONES if n in results]
    base1 = ms(results["dinov2_vitb14"][0])[0] if "dinov2_vitb14" in results else None
    rows = []
    for n in names:
        t1, t5 = results[n]
        m1, m5 = ms(t1), ms(t5)
        paired = ""
        if base1 is not None and n != "dinov2_vitb14":
            d = [a - b for a, b in zip(t1, results["dinov2_vitb14"][0])]
            paired = f"+{round(st.mean(d),1)} ({sum(x>0 for x in d)}/10)"
        rows.append((n.replace("dinov2_", ""), m1, m5, paired))
    print("\n== " + " | ".join(f"{n} {r1[0]}" for n, r1, _, _ in rows) + " ==")

    d = explog.next_dir("backbone-scale")
    explog.bar(d / "fig_backbone.png", [r[0] for r in rows], [r[1][0] for r in rows],
               "Backbone scaling: exemplar top1 (10-seed)", "%", ymax=100, errors=[r[1][1] for r in rows])
    tab = "\n".join(f"| {n} | {r1[0]}±{r1[1]}% | {r5[0]}±{r5[1]}% | {p or '—'} |" for n, r1, r5, p in rows)
    report = f"""# 백본 스케일링 (backbone-scale)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/backbone_scale.py`  (frozen exemplar 1-NN, 10-seed)

## 목적
지금껏 가장 작은 `vitb14`만 사용 → **더 큰 frozen DINOv2(vitl14/vitg14)** 로 특징을 강화하면
천장이 오르는지. 학습 없음, 같은 분할.

## 결과 (exemplar 1-NN, mean±std)
| 백본 | top1 | top5 | paired Δtop1 (vs vitb14) |
|---|---|---|---|
{tab}

![backbone](fig_backbone.png)

## 해석 / 다음
- 큰 백본이 노이즈 밖으로 올리면 → **정식 백본 교체**(가장 싼 큰 레버). 그 위에 SupCon 헤드/관계추론.
- 미미하면 → 특징은 포화, **구조(M6')·데이터**가 진짜 레버.
"""
    explog.write(d, report, {
        "title": "백본 스케일링", "date": datetime.date.today().isoformat(),
        "headline": " | ".join(f"{n} top1 {r1[0]}%" for n, r1, _, _ in rows),
        "results": {n: {"top1": r1, "top5": r5} for n, r1, r5, _ in rows}})
    print(f"wrote -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
