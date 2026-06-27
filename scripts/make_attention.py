"""Pin-conditioned saliency maps for correctly-classified samples (4x4).

For each correct test pin we show WHERE the model's point embedding "looks": the
per-patch cosine similarity between every DINOv2 patch token and the pooled pin
embedding z_q (the exact signal the prototypical head compares). High = this
region looks like the pinned structure. Upsampled, JET-colormapped, blended onto
the original photo, with the pin marked.

    .venv/bin/python scripts/make_attention.py
Output: a *.private.png montage (cadaver imagery -> gitignored).
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from eval_appearance import (load_core, split_indices, embed, prototypes,  # noqa: E402
                             evaluate, _MEAN, _STD)
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402

OUT = "/private/tmp/claude-501/-Users-chanlee-Desktop-Programming-scalpel/320890f6-79e3-48e4-86bc-86ffdb842a81/scratchpad/attention_4x4.png"


@torch.no_grad()
def saliency(img_path, q_xy, backbone, pool, centers, S, device, sigma=40.0):
    """(blended BGR image, pin in original px). Heat = patch cosine to z_q."""
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    arr = np.asarray(im.resize((S, S)), dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    grid, _ = backbone((x - mean) / std)               # (1,g,g,D)
    g = grid.shape[1]
    qx, qy = q_xy[0] * S / w, q_xy[1] * S / h
    q = torch.tensor([[qx, qy]], device=device)
    z = F.normalize(pool(grid, centers, q)[0], dim=0)  # (D,)
    tok = F.normalize(grid[0].reshape(g * g, -1), dim=1)   # (g*g, D)
    heat = (tok @ z).reshape(g, g).float().cpu().numpy()   # (g,g) cosine
    heat = np.clip((heat - heat.min()) / (np.ptp(heat) + 1e-9), 0, 1)
    orig = np.asarray(im)[:, :, ::-1].copy()           # BGR
    hm = cv2.resize((heat * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    blend = cv2.addWeighted(orig, 0.70, hm, 0.40, 0)
    cv2.circle(blend, (int(q_xy[0]), int(q_xy[1])), max(8, w // 60), (255, 255, 255), -1)
    cv2.circle(blend, (int(q_xy[0]), int(q_xy[1])), max(8, w // 60), (0, 0, 0), 3)
    return blend


def main():
    base = Path("data/triples")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg()
    S = cfg.backbone.image_size
    core = load_core("data/triples/triples.jsonl", 2)
    backbone = DinoBackbone(cfg.backbone); backbone.ensure_loaded(); backbone.to(device)
    pool = GaussianPool(cfg.point).to(device)
    centers = backbone.patch_centers(device)

    print("embedding to find correct samples...")
    Z = embed(core, base, backbone, pool, S, device)
    tr, te = split_indices(core, 0.3, 0)
    protos, sup = prototypes([core[i] for i in tr], Z[tr])
    _, per_item = evaluate([core[i] for i in te], Z[te], protos, sup)

    # 16 correct, diverse (distinct labels, then distinct images)
    seen_lab, seen_img, picks = set(), set(), []
    for r in per_item:
        if not r["correct"] or r["true"] in seen_lab or r["image"] in seen_img:
            continue
        seen_lab.add(r["true"]); seen_img.add(r["image"]); picks.append(r)
        if len(picks) >= 16:
            break
    print(f"correct picks: {len(picks)}")

    tiles = []
    for r in picks:
        b = saliency(base / r["image"], r["q"], backbone, pool, centers, S, device)
        s = 360
        b = cv2.resize(b, (s, int(s * b.shape[0] / b.shape[1])))
        b = np.vstack([b, np.full((max(0, 392 - b.shape[0]), s, 3), 18, np.uint8)])[:392]
        cv2.putText(b, r["true"][:26], (6, 384), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        tiles.append(b)
    while len(tiles) < 16:
        tiles.append(np.full_like(tiles[0], 18))
    grid = np.vstack([np.hstack(tiles[i * 4:(i + 1) * 4]) for i in range(4)])
    cv2.imwrite(OUT, grid)
    print("saved", OUT, "| labels:", [r["true"] for r in picks])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
