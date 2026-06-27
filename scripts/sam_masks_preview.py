"""Extract SAM masks + pooling weights — VISUAL inspection only, NO recognition metric.

Refined design after inspection. SAM can only find a useful local region where the
structure is a DISTINCT object with boundaries against its surroundings:
  - thin (artery/vein/nerve/duct): a real tubular object → SAM masks it well.
    Pooling weight = feather(mask) × pin-Gaussian  (follows the vessel, no bleed).
  - bulk (muscle/bone/brain/gland): uniform tissue with NO internal boundary →
    SAM gives either the whole structure (too big, dilutes the pin signal) or, if
    force-clipped, an arbitrary circle (= just a Gaussian). So we DON'T mask bulk:
    pooling weight = plain pin-Gaussian.

This script only RENDERS, per example:
  col0 original + pin | col1 SAM region (raw) | col2 pooling weight USED

Output: one montage saved as *.private.png (cadaver imagery → gitignored, §3). No
JSONL, no accuracy — inspect first, decide later.

    .venv/bin/python scripts/sam_masks_preview.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from segment_anything import SamPredictor, sam_model_registry  # noqa: E402

from eval_appearance import load_core  # noqa: E402

SAM_CKPT = ".cache/sam/sam_vit_b.pth"
THIN_KW = ("artery", "arteria", "vein", "vena", "nerve", "nervus", "vessel", "duct", "ductus")
WANT = [("thin", "artery", 2), ("thin", "nerve", 2), ("thin", "vein", 2),
        ("bulk", "muscle", 3), ("bulk", "bone", 1), ("bulk", "gland", 1), ("bulk", "lobe", 1)]


def coarse(label):
    s = label.lower()
    return "thin" if any(k in s for k in THIN_KW) else "bulk"


SEED = 3   # change to surface a DIFFERENT sample of specimens


def pick(core):
    order = np.random.default_rng(SEED).permutation(len(core))     # shuffle → different photos
    chosen, used_img = [], set()
    for cat, kw, k in WANT:
        c = 0
        for i in order:
            if c >= k:
                break
            r = core[int(i)]
            if kw in r["label"].lower() and coarse(r["label"]) == cat and r["image"] not in used_img:
                chosen.append((cat, int(i))); used_img.add(r["image"]); c += 1
    return chosen


def thin_mask(masks, H, W):
    """Tightest SAM mask (the vessel/nerve object) — no disk clip."""
    area = masks.reshape(len(masks), -1).mean(1)
    order = np.argsort(area)
    cands = [o for o in order if area[o] < 0.15] or [order[0]]
    return masks[cands[0]]


def bulk_region(masks, H, W):
    """The largest coherent SAM mask — shown for reference only (NOT used to pool)."""
    area = masks.reshape(len(masks), -1).mean(1)
    order = np.argsort(area)
    cands = [o for o in order[::-1] if area[o] < 0.85] or [order[-1]]
    return masks[cands[0]]


def feather(mask, W):
    sig = max(2.0, 0.010 * W)
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=sig)
    return soft / (soft.max() + 1e-9)


def pin_gauss(H, W, qx, qy, sig):
    yy, xx = np.mgrid[0:H, 0:W]
    return np.exp(-((xx - qx) ** 2 + (yy - qy) ** 2) / (2 * sig ** 2)).astype(np.float32)


def crop_box(qx, qy, mask, H, W):
    ys, xs = np.where(mask)
    side = 220 if len(xs) < 5 else int(np.clip(2.2 * max(xs.max() - xs.min(), ys.max() - ys.min()), 200, min(H, W)))
    half = side // 2
    x0 = int(np.clip(qx - half, 0, max(0, W - side))); y0 = int(np.clip(qy - half, 0, max(0, H - side)))
    return x0, y0, min(side, W - x0), min(side, H - y0)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    base = Path("data/triples")
    core = load_core("data/triples/triples.jsonl", 2)
    ex = pick(core)
    print(f"selected {len(ex)} examples")
    try:
        sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT).to(device)
    except Exception as e:  # noqa: BLE001
        print(f"  SAM {device} failed ({type(e).__name__}); cpu"); sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT).to("cpu")
    predictor = SamPredictor(sam)

    n = len(ex)
    fig, axes = plt.subplots(n, 3, figsize=(11.5, 3.4 * n))
    if n == 1:
        axes = axes[None, :]
    titles = ["original + pin", "SAM region (raw)", "pooling weight USED"]
    for row, (cat, i) in enumerate(ex):
        r = core[i]
        im = Image.open(base / r["image"]).convert("RGB"); W, H = im.size
        rgb = np.asarray(im); qx, qy = int(r["q"][0]), int(r["q"][1])
        predictor.set_image(rgb)
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[qx, qy]], np.float32),
            point_labels=np.array([1], np.int32), multimask_output=True)
        pg = pin_gauss(H, W, qx, qy, 0.11 * min(H, W))
        if cat == "thin":
            m = thin_mask(masks, H, W)
            soft = feather(m, W)
            weight = soft * pg; weight /= (weight.max() + 1e-9)
            rule = "mask-gated"
        else:
            m = bulk_region(masks, H, W)              # display only
            weight = pg / (pg.max() + 1e-9)           # plain Gaussian, NO mask
            rule = "plain Gaussian (no mask)"

        x0, y0, sw, sh = crop_box(qx, qy, m, H, W)
        sl = (slice(y0, y0 + sh), slice(x0, x0 + sw))
        crop = rgb[sl]; gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY); px, py = qx - x0, qy - y0

        axes[row, 0].imshow(crop); axes[row, 0].plot(px, py, "o", mfc="red", mec="white", ms=9, mew=1.5)
        axes[row, 0].set_ylabel(f"{cat}\n{r['label'][:22]}\n[{rule}]", fontsize=8)

        axes[row, 1].imshow(crop)
        mc = m[sl].astype(np.uint8)
        ov = np.zeros((*mc.shape, 4)); ov[mc > 0] = [0, 1, 1, 0.30]      # cyan for all
        axes[row, 1].imshow(ov)
        for cnt in cv2.findContours(mc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            axes[row, 1].plot(cnt[:, 0, 0], cnt[:, 0, 1], "-", color="cyan", lw=1.3)
        axes[row, 1].plot(px, py, "o", mfc="red", mec="white", ms=7, mew=1.2)
        axes[row, 1].set_title(f"area {100*m.mean():.1f}%" + ("" if cat == "thin" else "  (not used)"), fontsize=8)

        axes[row, 2].imshow(gray, cmap="gray")
        axes[row, 2].imshow(weight[sl], cmap="jet", alpha=0.55)
        axes[row, 2].plot(px, py, "o", mfc="white", mec="black", ms=6, mew=1)

        for c in range(3):
            axes[row, c].set_xticks([]); axes[row, c].set_yticks([])
            if row == 0:
                axes[row, c].set_title(titles[c], fontsize=10)
        print(f"  row {row} {cat:4s} {r['label'][:20]:20s} rawarea {100*m.mean():4.1f}%  {rule}")

    fig.suptitle("thin → SAM mask-gated Gaussian   ·   bulk → plain Gaussian (uniform tissue, SAM not used)", fontsize=12, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = Path("experiments/033-sam-thingate"); out.mkdir(parents=True, exist_ok=True)
    p = out / "masks_preview_set2.private.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"wrote -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
