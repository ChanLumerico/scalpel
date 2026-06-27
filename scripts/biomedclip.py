"""Experiment — BiomedCLIP: bring anatomical KNOWLEDGE (vision-language) to the task.

We diagnosed that the missing info for same-region look-alikes is anatomical
knowledge, not local appearance. BiomedCLIP (a medical CLIP, image+text) carries
that knowledge in its text encoder. Four ways, on a crop around the pin:

  dino           : frozen DINOv2 exemplar 1-NN (our baseline)
  bmc-img        : BiomedCLIP image features, exemplar 1-NN (medical-pretrained)
  bmc-text (0sh) : zero-shot — crop vs text embedding of each class name (knowledge,
                   no gallery images)
  dino+textλ     : DINO appearance score + λ · BiomedCLIP text similarity (knowledge
                   prior re-ranks appearance; λ picked per seed = upper-bound signal)

10-seed, paired vs DINO.

    .venv/bin/python scripts/biomedclip.py
"""

from __future__ import annotations

import collections
import datetime
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import open_clip  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import explog  # noqa: E402
from eval_appearance import load_core, split_indices, embed, _git_sha  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402

BMC = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
LAM = [0.3, 0.6, 1.0]


def crop(im, q, frac=0.4):
    w, h = im.size
    s = int(frac * min(w, h)); x, y = q
    return im.crop((max(0, x - s // 2), max(0, y - s // 2), min(w, x + s // 2), min(h, y + s // 2)))


@torch.no_grad()
def bmc_images(core, base, model, prep, device, chunk=64):
    Z, buf = [], []
    for r in core:
        im = Image.open(base / r["image"]).convert("RGB")
        buf.append(prep(crop(im, r["q"])))
        if len(buf) == chunk:
            Z.append(model.encode_image(torch.stack(buf).to(device)).cpu()); buf = []
    if buf:
        Z.append(model.encode_image(torch.stack(buf).to(device)).cpu())
    Z = torch.cat(Z).numpy()
    return (Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)).astype(np.float32)


@torch.no_grad()
def bmc_texts(labels, model, tok, device):
    emb, names = {}, list(labels)
    for s in range(0, len(names), 128):
        ch = names[s:s + 128]
        e = model.encode_text(tok([f"a dissection photo of a {n}" for n in ch]).to(device)).cpu().numpy()
        for n, v in zip(ch, e):
            emb[n] = (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)
    return emb


def ex_scores(Ztr, ytr, Zte, labels, lidx, cols):
    sims = Zte @ Ztr.T
    sc = np.full((len(Zte), len(labels)), -2.0, np.float32)
    for c, ix in cols.items():
        sc[:, c] = sims[:, ix].max(1)
    return sc


def topk(sc, labels, Y, cv):
    o = np.argsort(-sc, axis=1)
    n1 = sum(int(labels[o[r, 0]] == Y[cv[r]]) for r in range(len(cv)))
    n5 = sum(int(Y[cv[r]] in [labels[o[r, t]] for t in range(5)]) for r in range(len(cv)))
    return 100 * n1 / len(cv), 100 * n5 / len(cv)


def main():
    base = Path("data/triples")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size
    core = load_core("data/triples/triples.jsonl", 2)
    Y = [r["label"] for r in core]
    print(f"core {len(core)}/{len(set(Y))} | device={device}")

    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    pool = GaussianPool(cfg.point).to(device)
    print("DINO embed..."); Zd = embed(core, base, bb, pool, S, device).numpy().astype(np.float32)
    Zd = Zd / (np.linalg.norm(Zd, axis=1, keepdims=True) + 1e-9)

    print("BiomedCLIP load + embed..."); model, prep = open_clip.create_model_from_pretrained(BMC)
    model = model.to(device).eval(); tok = open_clip.get_tokenizer(BMC)
    Zb = bmc_images(core, base, model, prep, device)
    txt = bmc_texts(set(Y), model, tok, device)

    methods = ["dino", "bmc-img", "bmc-text", "dino+textλ"]
    acc = {m: ([], []) for m in methods}
    for seed in range(10):
        tr, te = split_indices(core, 0.3, seed)
        ytr = [Y[i] for i in tr]
        labels = sorted(set(ytr)); lidx = {l: j for j, l in enumerate(labels)}
        cols = collections.defaultdict(list)
        for j, l in enumerate(ytr):
            cols[lidx[l]].append(j)
        cv = [k for k in te if Y[k] in lidx]
        T = np.stack([txt[l] for l in labels])
        scd = ex_scores(Zd[tr], ytr, Zd[cv], labels, lidx, cols)
        scb = ex_scores(Zb[tr], ytr, Zb[cv], labels, lidx, cols)
        sct = Zb[cv] @ T.T
        for m, sc in [("dino", scd), ("bmc-img", scb), ("bmc-text", sct)]:
            a = topk(sc, labels, Y, cv); acc[m][0].append(a[0]); acc[m][1].append(a[1])
        best = max(((topk(scd + lam * sct, labels, Y, cv)) for lam in LAM), key=lambda x: x[0])
        acc["dino+textλ"][0].append(best[0]); acc["dino+textλ"][1].append(best[1])
        print(f"  seed {seed}: " + " ".join(f"{m} {acc[m][0][-1]:.0f}" for m in methods))

    ms = lambda v: (round(st.mean(v), 1), round(st.pstdev(v), 1))
    base1 = acc["dino"][0]
    rows = []
    for m in methods:
        t1, t5 = acc[m]
        dd = [a - b for a, b in zip(t1, base1)]
        rows.append((m, ms(t1), ms(t5), round(st.mean(dd), 1), sum(x > 0 for x in dd)))
    print("\n== " + " | ".join(f"{m} {ms(acc[m][0])[0]}" for m in methods) + " ==")

    d = explog.next_dir("biomedclip")
    explog.bar(d / "fig_bmc.png", [r[0] for r in rows], [r[1][0] for r in rows],
               "BiomedCLIP (knowledge): top1 (10-seed)", "%", ymax=100, errors=[r[1][1] for r in rows])
    tab = "\n".join(f"| {m} | {t1[0]}±{t1[1]}% | {t5[0]}% | {dd:+} ({w}/10) |" for m, t1, t5, dd, w in rows)
    report = f"""# BiomedCLIP — vision-language 지식 (biomedclip)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/biomedclip.py`

## 목적
같은-부위 look-alike 구분에 필요한 *해부 지식*을 의료 CLIP 텍스트 인코더로 끌어옴. 핀 주변 크롭으로
DINO / BiomedCLIP-이미지 / zero-shot 텍스트 / DINO+텍스트사전 비교. (dino+text는 seed별 λ=상한 신호.)

## 결과 (10-seed, paired vs dino)
| 방법 | top1 | top5 | Δtop1 |
|---|---|---|---|
{tab}

![bmc](fig_bmc.png)

## 해석
- bmc-text(0shot)/dino+text가 dino를 넘으면 → **지식이 실재 보완 신호**. 둘 다 낮으면 → BiomedCLIP도
  박리 사진엔 OOD(논문 figure 학습).
"""
    explog.write(d, report, {
        "title": "BiomedCLIP 지식", "date": datetime.date.today().isoformat(),
        "headline": " | ".join(f"{m} {ms(acc[m][0])[0]}" for m in methods),
        "methods": {m: {"top1": t1, "top5": t5, "dtop1": dd} for m, t1, t5, dd, w in rows}})
    print(f"wrote -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
