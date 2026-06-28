"""Experiment 061 / M-rohen0 STEP 1 — semi-automatic Rohen (I,q,y) extractor (human q-verification).

STEP 0 (060) passed: Rohen is in-domain. Now extract (I,q,y): the machine proposes candidate pins q by
tracing the black leader line from each margin number to its tissue endpoint; the HUMAN verifies (marks
the wrong numbers). Black leader lines are the crux (naive Hough got 6/31) — so this is best-effort +
review, not fully automatic.

Per page: main photo, legend (number→name via extract_text), margin-number positions (OCR), and a
number-anchored dark-line march → candidate q. Outputs an overlay (numbered candidate pins) for review
and a candidates JSON. Run on clean single-photo pages first.

    .venv/bin/python scripts/rohen_extract.py --pages 250,73,251,254,258
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytesseract  # noqa: E402
from pypdf import PdfReader  # noqa: E402

PDF = "data/color_atlas_of_anatomy.pdf"
OUT = Path("/private/tmp/claude-501/-Users-chanlee-Desktop-Programming-scalpel/320890f6-79e3-48e4-86bc-86ffdb842a81/scratchpad/rohen_pilot")


def parse_legend(text):
    """number → structure name from the page text (e.g. '1 Semispinalis capitis muscle 2 External ...')."""
    text = re.sub(r"\s+", " ", text)
    # split on ' <num> ' boundaries; keep pairs
    parts = re.split(r"(?<!\d)(\d{1,2})\s", " " + text + " ")
    leg = {}
    for i in range(1, len(parts) - 1, 2):
        try:
            num = int(parts[i]); name = parts[i + 1].strip()
        except ValueError:
            continue
        name = re.split(r"\s\d{1,2}\s", name)[0].strip()
        name = re.sub(r"[^A-Za-z .,()-]", "", name).strip(" .,-")
        if 1 <= num <= 60 and 3 <= len(name) <= 50 and num not in leg:
            leg[num] = name.lower()
    return leg


def ocr_margins(gray, W):
    out = []
    for x0, x1, side in [(0, int(W * 0.14), "L"), (int(W * 0.86), W, "R")]:
        strip = gray[:, x0:x1]
        d = pytesseract.image_to_data(strip, config="--psm 11 -c tessedit_char_whitelist=0123456789",
                                       output_type=pytesseract.Output.DICT)
        for i, t in enumerate(d["text"]):
            t = t.strip()
            if t.isdigit() and int(d["conf"][i]) > 25:
                out.append((int(t), x0 + d["left"][i] + d["width"][i] // 2,
                            d["top"][i] + d["height"][i] // 2, side))
    return out


def trace(mask, mx, my, side, W, band=5, maxgap=6):
    """March inward from a margin number along the leader line (dual-polarity mask handles black↔white
    flips); stop at the tissue endpoint. Reject if the line strays too far vertically (jumped to a crosser)."""
    H, Wm = mask.shape
    dirx = 1 if side == "L" else -1
    x = mx + dirx * 14; y = my
    last = None; gap = 0; steps = 0
    while 0 <= x < Wm and steps < W:
        steps += 1; x += dirx
        if not (0 <= x < Wm):
            break
        lo, hi = max(0, y - band), min(H, y + band + 1)
        ys = np.where(mask[lo:hi, x] > 0)[0]
        if len(ys):
            ny = lo + int(ys[np.argmin(np.abs(lo + ys - y))])   # nearest line px to current y (no jumping)
            if abs(ny - my) > 0.13 * H:                          # strayed too far vertically → a crosser
                break
            y = ny; last = (x, y); gap = 0
        else:
            gap += 1
            if gap > maxgap:
                break
    return last


def process(pg, reader):
    page = reader.pages[pg]
    imgs = sorted(page.images, key=lambda im: len(im.data), reverse=True)
    if not imgs or len(imgs[0].data) < 40000:
        return None
    pil = imgs[0].image.convert("RGB")
    img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    legend = parse_legend(page.extract_text() or "")
    # leader lines are drawn to stay visible: DARK on light tissue, LIGHT on dark tissue. So a thin
    # horizontal line is one that CONTRASTS (either polarity) with its vertical neighbours. Detect BOTH
    # dark lines (vertical black-hat) AND light lines (vertical top-hat) → follow through black↔white flips.
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, vk)      # thin DARK horizontal (black-on-light)
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, vk)        # thin LIGHT horizontal (white-on-dark)
    _, mb = cv2.threshold(bh, 12, 255, cv2.THRESH_BINARY)
    _, mt = cv2.threshold(th, 12, 255, cv2.THRESH_BINARY)
    m0 = cv2.bitwise_or(mb, mt)
    # keep only horizontally-long (straight leader lines), bridge small gaps at contrast flips
    mask = cv2.morphologyEx(m0, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (41, 1)))
    nums = ocr_margins(gray, W)
    cands = []
    seen = set()
    for n, mx, my, side in nums:
        if (n, side, my // 20) in seen or n not in legend:
            continue
        seen.add((n, side, my // 20))
        q = trace(mask, mx, my, side, W)
        travel = abs(q[0] - mx) if q else 0
        # accept only if: deep enough inside the photo, travelled inward (rejects margin-stuck),
        # and not past mid-image (rejects cross-image jumps to the wrong side).
        if q and W * 0.13 < q[0] < W * 0.87 and 0.08 * W < travel < 0.55 * W:
            cands.append({"num": n, "name": legend[n], "q": [int(q[0]), int(q[1])], "side": side, "travel": travel})
    # drop convergent endpoints: pins that bunch within 22 px are unreliable (traces lost in a dense
    # crossing region and stopped at its near edge, not at distinct structures).
    drop = set()
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            if abs(cands[i]["q"][0] - cands[j]["q"][0]) < 22 and abs(cands[i]["q"][1] - cands[j]["q"][1]) < 22:
                drop.add(i); drop.add(j)
    cands = [{k: c[k] for k in ("num", "name", "q", "side")} for i, c in enumerate(cands) if i not in drop]
    # overlay
    vis = img.copy()
    for c in cands:
        x, y = c["q"]
        cv2.circle(vis, (x, y), 9, (0, 0, 255), 2); cv2.circle(vis, (x, y), 2, (0, 255, 255), -1)
        cv2.putText(vis, str(c["num"]), (x + 11, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return {"page": pg, "size": [W, H], "legend": legend, "candidates": cands, "img": img, "vis": vis}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="250,73,251,254,258,262")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(PDF)
    pages = [int(p) for p in args.pages.split(",")]
    summary = []
    allc = []
    for pg in pages:
        r = process(pg, reader)
        if not r:
            print(f"  p{pg}: no main photo"); continue
        cv2.imwrite(str(OUT / f"p{pg}_overlay.png"), r["vis"])
        cv2.imwrite(str(OUT / f"p{pg}_clean.png"), r["img"])
        print(f"  p{pg}: legend {len(r['legend'])} | candidate q's {len(r['candidates'])} "
              f"({sorted(c['num'] for c in r['candidates'])})")
        summary.append({"page": pg, "n_legend": len(r["legend"]), "n_cand": len(r["candidates"])})
        for c in r["candidates"]:
            allc.append({"page": pg, **{k: c[k] for k in ("num", "name", "q", "side")}})
    (OUT / "candidates.json").write_text(json.dumps(allc, ensure_ascii=False, indent=2))
    print(f"\n{len(allc)} candidate (q,y) across {len(summary)} pages → {OUT}/candidates.json")
    print(f"overlays: {OUT}/p<page>_overlay.png  (review: which numbers' pins are wrong?)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
