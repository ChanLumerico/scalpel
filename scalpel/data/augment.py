"""Photometric + geometric augmentation (HANDOUT v2 §5.3).

Boosts *quantity* (the diversity ceiling is still the number of real photos -- no
new anatomy is created, §8.8). The pin ``q`` follows every geometric transform.
"""

from __future__ import annotations

import numpy as np

from .parse import Triple


def _affine(img, q, M):
    """Apply a 2x3 affine to image + point q."""
    import cv2

    H, W = img.shape[:2]
    out = cv2.warpAffine(
        img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
    x, y = q
    qx, qy = M @ np.array([x, y, 1.0])
    return out, (int(np.clip(qx, 0, W - 1)), int(np.clip(qy, 0, H - 1)))


def _photometric(img, rng):
    img = img.astype(np.float32)
    img *= rng.uniform(0.8, 1.2)  # brightness
    img = (img - 128) * rng.uniform(0.85, 1.15) + 128  # contrast
    img *= rng.uniform(0.92, 1.08, size=3)[None, None, :]  # colour jitter
    if rng.random() < 0.3:
        import cv2

        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    img += rng.normal(0, rng.uniform(0, 6), img.shape)  # sensor noise
    return np.clip(img, 0, 255).astype(np.uint8)


def augment(triple: Triple, n: int, seed: int = 0) -> list[Triple]:
    """Return ``n`` augmented copies of ``triple`` (q follows geometry)."""
    import cv2

    rng = np.random.default_rng(seed)
    H, W = triple.image.shape[:2]
    cx, cy = W / 2.0, H / 2.0
    out: list[Triple] = []
    for _ in range(n):
        img, q = triple.image, triple.q
        if rng.random() < 0.5:  # horizontal flip
            img = img[:, ::-1].copy()
            q = (W - 1 - q[0], q[1])
        deg = float(rng.uniform(-15, 15))
        scale = float(rng.uniform(0.9, 1.1))
        M = cv2.getRotationMatrix2D((cx, cy), deg, scale)
        M[:, 2] += rng.uniform(-0.04, 0.04, 2) * [W, H]  # small translate
        img, q = _affine(img, q, M)
        img = _photometric(img, rng)
        out.append(Triple(img, q, triple.label, triple.page, triple.src))
    return out
