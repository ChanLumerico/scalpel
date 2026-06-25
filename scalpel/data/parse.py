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
    image: np.ndarray            # clean_I (H, W, 3) uint8 -- answer/leader removed
    q: tuple[int, int]           # pin (x, y) in Im0 pixels
    label: str                   # normalized structure name
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
    xref = max(imgs, key=lambda im: im[2] * im[3])[0]     # im[2],im[3] = w,h
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


def _leader_pin(im0: np.ndarray, blue: np.ndarray, box) -> tuple[int, int]:
    """Pin from the leader attached to ``box``: endpoint (solid) or centroid (dashed).

    Dashed leaders outline a region; solid leaders point at one spot (§4.5). We
    take the blue component nearest the box; if it is fragmented/spread (dashed
    outline) we use the centroid of its convex hull, else the far endpoint.
    """
    import cv2
    from scipy import ndimage as ndi

    x0, y0, x1, y1 = box
    bx, by = (x0 + x1) / 2, (y0 + y1) / 2
    lbl, n = ndi.label(blue > 0)
    if n == 0:
        return int(bx), int(by)
    # component whose pixels come nearest the box
    best, bestd = None, 1e18
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        d = ((xs - bx) ** 2 + (ys - by) ** 2).min()
        if d < bestd:
            bestd, best = d, (xs, ys)
    xs, ys = best
    fill = xs.size / max(1.0, (xs.ptp() + 1) * (ys.ptp() + 1))
    if fill < 0.15:                                       # spread/dashed -> region
        return int(xs.mean()), int(ys.mean())
    far = np.argmax((xs - bx) ** 2 + (ys - by) ** 2)      # solid -> far endpoint
    return int(xs[far]), int(ys[far])


def _pull_inward(im0: np.ndarray, q, k: int = 9) -> tuple[int, int]:
    """Nudge q off any boundary toward the local region interior (§8.1)."""
    import cv2
    from scipy import ndimage as ndi

    x, y = q
    H, W = im0.shape[:2]
    patch = im0[max(0, y - 40):y + 40, max(0, x - 40):x + 40]
    if patch.size == 0:
        return int(x), int(y)
    # local region = pixels similar to q's colour; erode and take nearest interior
    ref = im0[min(H - 1, y), min(W - 1, x)].astype(int)
    sim = (np.abs(patch.astype(int) - ref).sum(-1) < 60)
    er = ndi.binary_erosion(sim, iterations=k)
    if er.any():
        yy, xx = np.where(er)
        oy, ox = max(0, y - 40), max(0, x - 40)
        j = np.argmin((xx + ox - x) ** 2 + (yy + oy - y) ** 2)
        return int(xx[j] + ox), int(yy[j] + oy)
    return int(x), int(y)


# --------------------------------------------------------------------------- #
# clean image                                                                 #
# --------------------------------------------------------------------------- #
def _clean(im0: np.ndarray, boxes, blue: np.ndarray) -> np.ndarray:
    """Inpaint label boxes + leader lines so no answer text/leader leaks (§8.1)."""
    import cv2
    mask = np.zeros(im0.shape[:2], np.uint8)
    for x0, y0, x1, y1 in boxes:
        mask[max(0, y0):y1, max(0, x0):x1] = 255
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    mask = cv2.bitwise_or(mask, cv2.dilate(blue, np.ones((5, 5), np.uint8)))
    bgr = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(bgr, mask, 5, cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _ocr(crop: np.ndarray) -> str:
    import pytesseract
    return pytesseract.image_to_string(crop, config="--psm 7").strip()


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
        for box in boxes:
            x0, y0, x1, y1 = box
            label = vocab.normalize(_ocr(im0[max(0, y0):y1, max(0, x0):x1]))
            if not label:
                continue
            q = _pull_inward(clean, _leader_pin(im0, blue, box))
            triples.append(Triple(clean, q, label, page.number, src))
    doc.close()
    return triples
