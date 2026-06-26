"""M4' appearance MVP — frozen DINOv2 + GaussianPool + prototypical few-shot.

The first *model* over the real QuizLink data, and deliberately the simplest one:
no training at all. Pipeline per item ``(image I, pin q)``:

    I -> frozen DINOv2 -> patch-token grid
    grid, q -> GaussianPool -> z_q                  (point embedding, L2-normalized)
    z_q vs class prototypes (mean gallery embedding) -> nearest = prediction

We restrict to the evaluable core (classes with >=2 instances), split at the
SPECIMEN level (per PDF page, leak-free), build prototypes from the gallery, and
report selective-accuracy@coverage + top-k, stratified by support count. This is
metric learning, not softmax classification, so it suits the long tail
(handout v2 §2.6, §5.4).

    .venv/bin/python scripts/eval_appearance.py
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from scalpel.config import PipelineCfg
from scalpel.perception import DinoBackbone, GaussianPool

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def load_core(jsonl: str, min_count: int):
    rows = [json.loads(l) for l in open(jsonl, encoding="utf-8") if l.strip()]
    cnt = collections.Counter(r["label"] for r in rows)
    core = [r for r in rows if cnt[r["label"]] >= min_count]
    return core, cnt


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
    """L2-normalized point embedding z_q for every triple (backbone once/image)."""
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="data/triples/triples.jsonl")
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    base = Path(a.jsonl).parent

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg()
    S = cfg.backbone.image_size
    print(f"device={device}  backbone={cfg.backbone.name}  image_size={S}  sigma={cfg.point.gauss_sigma_px}px")

    core, _ = load_core(a.jsonl, a.min_count)
    tr, te = split_specimen(core, a.test_frac, a.seed)
    labels = sorted({r["label"] for r in core})
    print(f"core: {len(core)} triples, {len(labels)} classes (>= {a.min_count} inst) | "
          f"gallery {len(tr)} / test {len(te)} (specimen-split)")

    print("loading frozen DINOv2 (first run downloads weights)...")
    backbone = DinoBackbone(cfg.backbone)
    backbone.ensure_loaded()
    backbone.to(device)
    pool = GaussianPool(cfg.point).to(device)

    print("embedding gallery...")
    Ztr = embed(tr, base, backbone, pool, S, device)
    print("embedding test...")
    Zte = embed(te, base, backbone, pool, S, device)

    # ---- prototypes: mean gallery embedding per class -----------------------
    protos, support = {}, {}
    by_lab = collections.defaultdict(list)
    for i, r in enumerate(tr):
        by_lab[r["label"]].append(i)
    for lab, idxs in by_lab.items():
        protos[lab] = F.normalize(Ztr[idxs].mean(0), dim=0)
        support[lab] = len(idxs)
    proto_labels = list(protos)
    P = torch.stack([protos[l] for l in proto_labels])         # (C, D)

    # ---- evaluate nearest prototype ----------------------------------------
    strata = collections.defaultdict(lambda: [0, 0, 0, 0])     # bucket -> [n,t1,t3,t5]
    cov = tot = t1 = t3 = t5 = 0
    for i, r in enumerate(te):
        tot += 1
        if r["label"] not in protos:
            continue                                           # OOV: no prototype -> abstain
        cov += 1
        sims = P @ Zte[i]                                       # cosine (all normalized)
        order = [proto_labels[j] for j in torch.argsort(sims, descending=True)[:5]]
        c1, c3, c5 = r["label"] == order[0], r["label"] in order[:3], r["label"] in order[:5]
        t1 += c1; t3 += c3; t5 += c5
        b = min(support[r["label"]], 4)                        # support bucket (1,2,3,4+)
        s = strata[b]; s[0] += 1; s[1] += c1; s[2] += c3; s[3] += c5

    print("\n================ M4' appearance MVP ================")
    print(f"classes with a prototype: {len(protos)}")
    print(f"test items: {tot} | covered (have prototype): {cov} = {100*cov/tot:.0f}% coverage"
          f"  ({tot-cov} OOV -> abstain)")
    if cov:
        print(f"selective-accuracy@coverage:  top1 {100*t1/cov:.1f}%  "
              f"top3 {100*t3/cov:.1f}%  top5 {100*t5/cov:.1f}%")
        print(f"end-to-end (OOV=wrong):       top1 {100*t1/tot:.1f}%  top5 {100*t5/tot:.1f}%")
        rand = 100 / len(protos)
        print(f"(random-chance top1 over {len(protos)} classes ~= {rand:.2f}%)")
        print("\nby gallery support count:")
        for b in sorted(strata):
            n, a1, a3, a5 = strata[b]
            tag = f"{b}+" if b == 4 else str(b)
            print(f"  {tag}-shot: n={n:3d}  top1 {100*a1/n:4.1f}%  top3 {100*a3/n:4.1f}%  top5 {100*a5/n:4.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
