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


def trace(mask, mx, my, side, W, band=9, maxgap=34):
    """March inward from a margin number along the dark leader line; return the tissue endpoint q.
    (ORIGINAL version — the one the user judged best, only ~2-3 wrong/page.)"""
    H, Wm = mask.shape
    dirx = 1 if side == "L" else -1
    x = mx + dirx * 14; y = my
    last = None; gap = 0; steps = 0
    while 0 <= x < Wm and steps < W:
        steps += 1; x += dirx
        if not (0 <= x < Wm):
            break
        lo, hi = max(0, y - band), min(H, y + band + 1)
        col = mask[lo:hi, x]
        ys = np.where(col > 0)[0]
        if len(ys):
            y = lo + int(np.median(ys)); last = (x, y); gap = 0
        else:
            gap += 1
            if gap > maxgap:
                break
    return last


EXCLUDE = ("section", "mri", "radiograph", "schematic", "diagram", "ct ", "tomograph")


def process(pg, reader):
    page = reader.pages[pg]
    txt = (page.extract_text() or "")
    if any(k in txt.lower() for k in EXCLUDE):          # skip imaging/section/illustration pages
        return None
    imgs = sorted(page.images, key=lambda im: len(im.data), reverse=True)
    if not imgs or len(imgs[0].data) < 40000:
        return None
    pil = imgs[0].image.convert("RGB")
    img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    legend = parse_legend(page.extract_text() or "")
    # ORIGINAL leader-line mask (user judged this best): thin DARK horizontal (vertical black-hat) kept
    # only where horizontally LONG (open) → isolates straight leader lines from curvy tissue texture.
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9)))
    _, m0 = cv2.threshold(bh, 12, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(m0, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (41, 1)))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    nums = ocr_margins(gray, W)
    cands = []
    seen = set()
    for n, mx, my, side in nums:
        if (n, side, my // 20) in seen or n not in legend:
            continue
        seen.add((n, side, my // 20))
        q = trace(mask, mx, my, side, W)
        if q and W * 0.12 < q[0] < W * 0.88:        # endpoint is inside the photo, not margin
            cands.append({"num": n, "name": legend[n], "q": [int(q[0]), int(q[1])], "side": side})
    # overlay
    vis = img.copy()
    for c in cands:
        x, y = c["q"]
        cv2.circle(vis, (x, y), 9, (0, 0, 255), 2); cv2.circle(vis, (x, y), 2, (0, 255, 255), -1)
        cv2.putText(vis, str(c["num"]), (x + 11, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return {"page": pg, "size": [W, H], "legend": legend, "candidates": cands, "img": img, "vis": vis}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="")
    ap.add_argument("--range", default="20-540", help="full-book page range start-end")
    ap.add_argument("--min-cand", type=int, default=3, help="save overlay only if >= this many candidates")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(PDF)
    if args.pages:
        pages = [int(p) for p in args.pages.split(",")]
    else:
        a, b = args.range.split("-"); pages = list(range(int(a), min(int(b), len(reader.pages))))
    summary = []; allc = []; n_photo = 0
    for pg in pages:
        try:
            r = process(pg, reader)
        except Exception as e:
            print(f"  p{pg}: err {str(e)[:40]}"); continue
        if not r:
            continue
        n_photo += 1
        nc = len(r["candidates"])
        if nc >= args.min_cand:
            cv2.imwrite(str(OUT / f"p{pg}_overlay.png"), r["vis"])
            cv2.imwrite(str(OUT / f"p{pg}_clean.png"), r["img"])
        if nc:
            print(f"  p{pg}: legend {len(r['legend'])} | q {nc} ({sorted(c['num'] for c in r['candidates'])})")
        summary.append({"page": pg, "n_legend": len(r["legend"]), "n_cand": nc})
        for c in r["candidates"]:
            allc.append({"page": pg, **{k: c[k] for k in ("num", "name", "q", "side")}})
    (OUT / "candidates.json").write_text(json.dumps(allc, ensure_ascii=False, indent=2))
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    pages_with_cand = sum(1 for s in summary if s["n_cand"] > 0)
    print(f"\n==== FULL BOOK ====")
    print(f"  pages scanned {len(pages)} | with main cadaver photo {n_photo} | with >=1 candidate {pages_with_cand}")
    print(f"  total candidate (I,q,y) {len(allc)} | overlays saved for pages with >={args.min_cand} candidates")
    print(f"  → {OUT}/candidates.json ({len(allc)}), overlays {OUT}/p<page>_overlay.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
