"""QuizLink PDF -> (clean_I, q, y) triples (HANDOUT v2 §5.2, §4 — core).

Verified PDF structure (§4): each page is one baked JPEG2000 image ``Im0``
(3000×2250 RGB) holding the photo + blue leader lines + label boxes + the
**answer text**. Click-to-reveal is an AcroForm button overlay that *masks* the
answer, so the raw ``Im0`` already shows it — no click simulation needed.

For each label box (a button-widget rect):
  * OCR the box crop      -> answer ``y``     (normalize via :class:`Vocab`)
  * trace the blue leader -> pin ``q``        (solid=endpoint, dashed=region centroid)
  * inpaint box + leaders -> ``clean_I``      (★ remove label leak, §8.1)

Traps: PDF-rect ↔ Im0-pixel transform (§4.4); solid vs dashed leader (§4.5); OCR
only the clean box, not the photo (§8.5); pull q inward off the boundary (§8.1).

fitz / cv2 / pytesseract are imported lazily. The CV thresholds here are tuned
against a real page at the M2' acceptance (visual check, label-leak == 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# blue leader-line HSV band (OpenCV H in 0..179); tune at M2' on a real page
_BLUE_LO = (95, 60, 40)
_BLUE_HI = (135, 255, 255)


@dataclass
class Triple:
    image: np.ndarray  # clean_I (H, W, 3) uint8 -- answer/leader removed
    q: tuple[int, int]  # pin (x, y) in Im0 pixels
    label: str  # normalized structure name
    page: int
    src: str


# --------------------------------------------------------------------------- #
# PDF extraction                                                              #
# --------------------------------------------------------------------------- #
def _page_image(page) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Return the page's main baked image (Im0) as RGB uint8 + its placement rect.

    The rect (fitz top-left points) is where Im0 is drawn on the page; it maps
    PDF coordinates to Im0 pixels (§4.4).
    """
    import cv2

    imgs = page.get_images(full=True)
    if not imgs:
        raise ValueError(f"page {page.number}: no image")
    # the dissection photo is by far the largest image on the page
    xref = max(imgs, key=lambda im: im[2] * im[3])[0]  # im[2],im[3] = w,h
    raw = page.parent.extract_image(xref)["image"]
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    rects = page.get_image_rects(xref)
    r = rects[0]
    return arr, (r.x0, r.y0, r.width, r.height)


def _button_rects(page) -> list[tuple[float, float, float, float]]:
    """Mask/label button-widget rects (PDF points); near-duplicates merged (§4.2)."""
    rects: list[tuple[float, float, float, float]] = []
    for w in page.widgets() or []:
        if w.field_type_string and "Button" in w.field_type_string:
            r = w.rect
            rects.append((r.x0, r.y0, r.width, r.height))
    # the mask + hide-trigger come in near-identical pairs -> dedup
    merged: list[tuple[float, float, float, float]] = []
    for r in rects:
        if not any(abs(r[0] - m[0]) < 3 and abs(r[1] - m[1]) < 3 for m in merged):
            merged.append(r)
    return merged


def _pdf_rect_to_px(rect, place, wh) -> tuple[int, int, int, int]:
    """Transform a PDF-point rect to Im0 pixel box (x0, y0, x1, y1) (§4.4)."""
    px0, py0, pw, ph = place
    W, H = wh
    x0 = (rect[0] - px0) / pw * W
    y0 = (rect[1] - py0) / ph * H
    x1 = (rect[0] + rect[2] - px0) / pw * W
    y1 = (rect[1] + rect[3] - py0) / ph * H
    return int(x0), int(y0), int(x1), int(y1)


# --------------------------------------------------------------------------- #
# leader line -> pin                                                          #
# --------------------------------------------------------------------------- #
def _blue_mask(im0: np.ndarray) -> np.ndarray:
    import cv2

    hsv = cv2.cvtColor(im0, cv2.COLOR_RGB2HSV)
    return cv2.inRange(hsv, np.array(_BLUE_LO), np.array(_BLUE_HI))


def _leader_pin(im0: np.ndarray, leader_blue: np.ndarray, box) -> tuple[int, int]:
    """Pin from the leader attached to ``box``: tissue endpoint (solid) or region
    centroid (dashed). ``leader_blue`` must already have the label boxes removed
    (so box borders/text don't pollute the trace).

    Solid leaders are thin lines -> take the point farthest from the box (the
    tissue end). Dashed leaders outline a region (their convex hull encloses real
    area while staying thin) -> take the hull centroid (§4.5).
    """
    import cv2
    from scipy import ndimage as ndi

    x0, y0, x1, y1 = box
    bx, by = (x0 + x1) / 2, (y0 + y1) / 2
    lbl, n = ndi.label(leader_blue > 0)
    if n == 0:
        return int(bx), int(by)
    best, bestd = None, 1e18  # component nearest the box
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        d = ((xs - bx) ** 2 + (ys - by) ** 2).min()
        if d < bestd:
            bestd, best = d, (xs, ys)
    xs, ys = best
    pts = np.stack([xs, ys], 1).astype(np.int32)
    hull_area = cv2.contourArea(cv2.convexHull(pts)) if xs.size >= 3 else 0.0
    enclosed = hull_area > 40 * 40 and xs.size / (hull_area + 1.0) < 0.08
    if enclosed:  # dashed outline -> region centroid
        return int(xs.mean()), int(ys.mean())
    far = int(np.argmax((xs - bx) ** 2 + (ys - by) ** 2))  # solid -> tissue endpoint
    return int(xs[far]), int(ys[far])


def _pull_inward(im0: np.ndarray, q, k: int = 9) -> tuple[int, int]:
    """Nudge q off any boundary toward the local region interior (§8.1)."""
    import cv2
    from scipy import ndimage as ndi

    x, y = q
    H, W = im0.shape[:2]
    patch = im0[max(0, y - 40) : y + 40, max(0, x - 40) : x + 40]
    if patch.size == 0:
        return int(x), int(y)
    # local region = pixels similar to q's colour; erode and take nearest interior
    ref = im0[min(H - 1, y), min(W - 1, x)].astype(int)
    sim = np.abs(patch.astype(int) - ref).sum(-1) < 60
    er = ndi.binary_erosion(sim, iterations=k)
    if er.any():
        yy, xx = np.where(er)
        oy, ox = max(0, y - 40), max(0, x - 40)
        j = np.argmin((xx + ox - x) ** 2 + (yy + oy - y) ** 2)
        return int(xx[j] + ox), int(yy[j] + oy)
    return int(x), int(y)


def _snap_to_tissue(crop: np.ndarray, q, radius: int = 70, thr: int = 35):
    """Ensure the pin sits on tissue, not a black gap/background (§8.1).

    Many leaders end in a dark gap between structures; pooling DINO features there
    is meaningless. If q is on (near-)black, snap to the nearest tissue pixel
    within ``radius``; if there's no tissue nearby, return ``None`` so the caller
    drops the triple (a pin we can't trust to a structure is worse than no pin).
    """
    import cv2

    h, w = crop.shape[:2]
    x, y = int(q[0]), int(q[1])
    val = crop.max(2).astype(np.uint8)               # brightness ~ max channel
    tissue = cv2.blur(val, (7, 7)) > thr             # smoothed -> ignore specks
    if 0 <= y < h and 0 <= x < w and tissue[y, x]:
        return x, y
    y0, y1 = max(0, y - radius), min(h, y + radius)
    x0, x1 = max(0, x - radius), min(w, x + radius)
    ys, xs = np.where(tissue[y0:y1, x0:x1])
    if ys.size == 0:
        return None                                  # no tissue nearby -> drop
    j = int(np.argmin((xs + x0 - x) ** 2 + (ys + y0 - y) ** 2))
    return int(xs[j] + x0), int(ys[j] + y0)


# --------------------------------------------------------------------------- #
# clean image                                                                 #
# --------------------------------------------------------------------------- #
def _clean(im0: np.ndarray, boxes, blue: np.ndarray) -> np.ndarray:
    """Inpaint label boxes + leader lines so no answer text/leader leaks (§8.1)."""
    import cv2

    mask = np.zeros(im0.shape[:2], np.uint8)
    for x0, y0, x1, y1 in boxes:
        mask[max(0, y0) : y1, max(0, x0) : x1] = 255
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    mask = cv2.bitwise_or(mask, cv2.dilate(blue, np.ones((5, 5), np.uint8)))
    bgr = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(bgr, mask, 5, cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _ocr(crop: np.ndarray) -> str:
    import pytesseract

    return pytesseract.image_to_string(crop, config="--psm 7").strip()


def _photo_bbox(img: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box of the cadaver photo: the large saturated tissue region.

    Crops out the page borders, the region caption ("Thorax, anterior") and the
    attribution text -- all gray/white, low-saturation -- so the model sees ONLY
    the dissection photo (and no caption-region leak).
    """
    import cv2
    from scipy import ndimage as ndi

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    sat, val = hsv[..., 1], hsv[..., 2]
    mask = ((sat > 30) & (val > 35) & (val < 248)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    lbl, n = ndi.label(mask > 0)
    if n == 0:
        return 0, 0, img.shape[1], img.shape[0]
    big = int(np.argmax(np.bincount(lbl.ravel())[1:])) + 1  # largest tissue blob
    ys, xs = np.where(lbl == big)
    H, W = img.shape[:2]
    m = int(0.02 * max(np.ptp(xs), np.ptp(ys)))  # small margin
    return (
        max(0, int(xs.min()) - m),
        max(0, int(ys.min()) - m),
        min(W, int(xs.max()) + 1 + m),
        min(H, int(ys.max()) + 1 + m),
    )


# --------------------------------------------------------------------------- #
# entry point                                                                 #
# --------------------------------------------------------------------------- #
def parse_quizlink(pdf_path, vocab) -> list[Triple]:
    """Parse a QuizLink PDF into ``(clean_I, q, y)`` triples (one per label box)."""
    import fitz

    triples: list[Triple] = []
    doc = fitz.open(pdf_path)
    src = str(pdf_path)
    for page in doc:
        try:
            im0, place = _page_image(page)
        except ValueError:
            continue
        H, W = im0.shape[:2]
        boxes = [_pdf_rect_to_px(r, place, (W, H)) for r in _button_rects(page)]
        boxes = [b for b in boxes if 0 <= b[0] < W and 0 <= b[1] < H and b[2] > b[0]]
        if not boxes:
            continue
        blue = _blue_mask(im0)
        clean = _clean(im0, boxes, blue)
        leader_blue = blue.copy()  # boxes removed: box borders
        for x0, y0, x1, y1 in boxes:  # must not pollute the trace
            leader_blue[max(0, y0 - 6) : y1 + 6, max(0, x0 - 6) : x1 + 6] = 0
        cx0, cy0, cx1, cy1 = _photo_bbox(clean)  # crop to the cadaver photo
        crop = clean[cy0:cy1, cx0:cx1]
        ch, cw = crop.shape[:2]
        for box in boxes:
            x0, y0, x1, y1 = box
            label = vocab.normalize(_ocr(im0[max(0, y0) : y1, max(0, x0) : x1]))
            if not label or "http" in label or "adobe" in label:  # skip intro/junk
                continue
            q = _pull_inward(clean, _leader_pin(im0, leader_blue, box))
            qx, qy = q[0] - cx0, q[1] - cy0  # to crop coordinates
            if not (0 <= qx < cw and 0 <= qy < ch):  # pin outside the photo -> skip
                continue
            snapped = _snap_to_tissue(crop, (qx, qy))  # keep the pin on tissue
            if snapped is None:  # pin on background, no tissue nearby -> drop
                continue
            triples.append(Triple(crop, snapped, label, page.number, src))
    doc.close()
    return triples
