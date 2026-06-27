"""BlueLink labeled-slide extractor — (I, q, y) from baked-label cadaver images.

The new data source (data/bluelink_images/, 462 slides) has labels *baked into the
image*: a blue leader line connects a blue-bordered label box (structure name) to a
point on the tissue. We recover, per slide:
  - y  = OCR of the label box text (normalized via scalpel.data.vocab.Vocab)
  - q  = the tissue end of that label's leader line (the pin)
  - region = the bottom-left title ("Pancreas, anterior") — free context metadata
  - I  = the slide with ALL annotations inpainted out (leak-free, CLAUDE.md §2)
Each slide yields MULTIPLE (q, y) on ONE image → multi-pin pages (dissolves exp 040
crack #0). Cadaver images stay in gitignored data/ (§3).

Grounding (measured on 5 slides / 3 regions, scripts probe):
  blue annotation: HSV H∈[115,128] S>120 V>120 ≈ RGB(12,10,207); 0.4–1.4% of px,
  perfectly separable from warm tissue (R>B). OCR conf 91–96. Box polarity varies
  (white-on-black ↔ black-on-white) → Otsu both ways. Label vs title/copyright =
  presence of an enclosing blue frame.

    .venv/bin/python scripts/bluelink_extract.py --viz --limit 6   # visual check
    .venv/bin/python scripts/bluelink_extract.py --out data/bluelink_triples
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytesseract  # noqa: E402
from pytesseract import Output  # noqa: E402

from scalpel.data.vocab import Vocab  # noqa: E402

COPYRIGHT_RE = re.compile(r"kathleen|alsup|glenn|fox|bluelink|©|copyright", re.I)


# ----------------------------------------------------------------- blue mask
def blue_mask(rgb: np.ndarray) -> np.ndarray:
    R, G, B = (rgb[..., i].astype(int) for i in range(3))
    m = (B > 110) & (B - R > 60) & (B - G > 60)
    return (m.astype(np.uint8)) * 255


# ----------------------------------------------------------------- OCR words
def ocr_words(gray: np.ndarray, min_conf=55):
    """Confident alpha words from BOTH polarities (boxes vary in fill)."""
    out = {}
    for im in (gray, 255 - gray):
        d = pytesseract.image_to_data(im, config="--psm 11", output_type=Output.DICT)
        for i in range(len(d["text"])):
            t = d["text"][i].strip()
            c = int(d["conf"][i])
            if len(t) >= 2 and c >= min_conf and any(ch.isalpha() for ch in t):
                x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
                key = (round(x / 30), round(y / 30))
                if key not in out or c > out[key][1]:
                    out[key] = (t, c, x, y, w, h)
    return list(out.values())


def cluster_words(words, gap_x=120, gap_y=70):
    """Union-find words into label-sized clusters (same box: same line + stacked lines)."""
    n = len(words)
    par = list(range(n))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a

    def near(A, B):
        (_, _, ax, ay, aw, ah) = A
        (_, _, bx, by, bw, bh) = B
        # horizontal gap on (near) same line, or vertical stack with x-overlap
        same_line = abs(ay - by) < max(ah, bh) and (bx - (ax + aw) < gap_x) and (ax - (bx + bw) < gap_x)
        x_ovl = min(ax + aw, bx + bw) - max(ax, bx) > -40
        stacked = x_ovl and (0 <= by - (ay + ah) < gap_y or 0 <= ay - (by + bh) < gap_y)
        return same_line or stacked

    for i in range(n):
        for j in range(i + 1, n):
            if near(words[i], words[j]):
                par[find(i)] = find(j)
    groups = collections.defaultdict(list)
    for i in range(n):
        groups[find(i)].append(words[i])
    clusters = []
    for g in groups.values():
        g.sort(key=lambda w: (w[3], w[2]))  # reading order: top, then left
        xs = [w[2] for w in g]; ys = [w[3] for w in g]
        xe = [w[2] + w[4] for w in g]; ye = [w[3] + w[5] for w in g]
        bbox = (min(xs), min(ys), max(xe) - min(xs), max(ye) - min(ys))
        text = " ".join(w[0] for w in g)
        conf = sum(w[1] for w in g) / len(g)
        clusters.append({"text": text, "bbox": bbox, "conf": conf, "n": len(g)})
    return clusters


# ----------------------------------------------------------------- blue box (scan outward)
def find_box(bmask: np.ndarray, bbox, max_pad=150, min_run=0.45, need_sides=3):
    """Find the blue rectangle enclosing a text bbox by scanning OUTWARD on each side
    until a blue border line is hit (padding varies per box, so a fixed band fails).
    Returns (x0,y0,x1,y1,n_sides) or None. The border spans (most of) the text extent."""
    H, W = bmask.shape
    tx, ty, tw, th = bbox

    def edge_h(y_start, step):  # horizontal border above/below text
        for d in range(4, max_pad):
            yy = y_start + step * d
            if not (0 <= yy < H):
                break
            if bmask[yy, tx:tx + tw].sum() / 255.0 > min_run * tw:
                return yy
        return None

    def edge_v(x_start, step):  # vertical border left/right of text
        for d in range(4, max_pad):
            xx = x_start + step * d
            if not (0 <= xx < W):
                break
            if bmask[ty:ty + th, xx].sum() / 255.0 > min_run * th:
                return xx
        return None

    top, bot = edge_h(ty, -1), edge_h(ty + th, +1)
    lft, rgt = edge_v(tx, -1), edge_v(tx + tw, +1)
    n = sum(e is not None for e in (top, bot, lft, rgt))
    if n < need_sides:
        return None
    x0 = lft if lft is not None else tx - 12
    x1 = rgt if rgt is not None else tx + tw + 12
    y0 = top if top is not None else ty - 10
    y1 = bot if bot is not None else ty + th + 10
    return (max(0, x0), max(0, y0), min(W, x1), min(H, y1), n)


def ocr_box(gray, rect):
    """Re-OCR a detected label box (psm 6, both polarities) for a clean full string."""
    x0, y0, x1, y1 = rect[:4]
    pad = 6
    crop = gray[y0 + pad:y1 - pad, x0 + pad:x1 - pad]
    if crop.size == 0:
        return "", 0.0
    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    best = ("", 0.0)
    for im in (crop, 255 - crop):
        _, th = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        d = pytesseract.image_to_data(th, config="--psm 6", output_type=Output.DICT)
        ws = [(d["text"][i], int(d["conf"][i])) for i in range(len(d["text"]))
              if d["text"][i].strip() and int(d["conf"][i]) > 0]
        if ws:
            txt = " ".join(w for w, _ in ws); mc = sum(c for _, c in ws) / len(ws)
            if mc > best[1]:
                best = (txt, mc)
    return best


# ----------------------------------------------------------------- leader endpoint
def _attachment(line_blue, box_rect, band=46):
    """Blue line pixel closest to the box rectangle = where the leader leaves the box."""
    H, W = line_blue.shape
    x0, y0, x1, y1 = box_rect
    ys, xs = np.where(line_blue > 0)
    if len(xs) == 0:
        return None, None
    dx = np.maximum.reduce([x0 - xs, xs - x1, np.zeros_like(xs)])
    dy = np.maximum.reduce([y0 - ys, ys - y1, np.zeros_like(ys)])
    d = dx * dx + dy * dy
    sel = d < band * band
    if not sel.any():
        return None, None
    sx, sy, sd = xs[sel], ys[sel], d[sel]
    j = sd.argmin()
    return (int(sx[j]), int(sy[j])), (sx, sy)


def leader_endpoint(line_blue, box_rect, tissue, step=4.0, win=8, max_steps=800):
    """Trace the leader line from its box attachment to the tissue tip = q.
    Direction is fixed from a local PCA fit at the attachment, then we march along
    that straight ray snapping to blue — so CROSSING leader lines are passed through,
    not followed (robust + precise). box_rect=(x0,y0,x1,y1)."""
    import math
    H, W = line_blue.shape
    x0, y0, x1, y1 = box_rect
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    A, _ = _attachment(line_blue, box_rect)
    if A is None:
        return None
    ax, ay = A
    # local PCA direction from blue pixels within R of A (this line's segment near box)
    R = 110
    yy, xx = np.where(line_blue[max(0, ay - R):ay + R, max(0, ax - R):ax + R] > 0)
    xx = xx + max(0, ax - R); yy = yy + max(0, ay - R)
    if len(xx) < 8:
        return A
    pts = np.stack([xx - ax, yy - ay], 1).astype(float)
    # restrict to the local cluster around A (drop a stray crossing line far in window)
    near = (pts[:, 0] ** 2 + pts[:, 1] ** 2) < R * R
    pts = pts[near]
    cov = pts.T @ pts
    w, V = np.linalg.eigh(cov)
    d = V[:, -1]                       # principal axis
    if d[0] * (ax - cx) + d[1] * (ay - cy) < 0:  # orient AWAY from the box
        d = -d
    dx, dy = float(d[0]), float(d[1])
    # march along fixed direction, snapping to nearest blue in a perpendicular window
    px, py = float(ax), float(ay)
    last = (ax, ay)
    miss = 0
    for _ in range(max_steps):
        px += dx * step; py += dy * step
        ix, iy = int(round(px)), int(round(py))
        if not (0 <= ix < W and 0 <= iy < H):
            break
        # nearest blue within window, preferring on-ray pixels
        best = None; bestd = 1e9
        for oy in range(-win, win + 1):
            for ox in range(-win, win + 1):
                qx, qy = ix + ox, iy + oy
                if 0 <= qx < W and 0 <= qy < H and line_blue[qy, qx]:
                    dd = ox * ox + oy * oy
                    if dd < bestd:
                        bestd = dd; best = (qx, qy)
        if best is None:
            miss += 1
            if miss > 6:               # ~24px gap with no blue → real tip reached
                break
            continue
        miss = 0
        px, py = best                  # re-center on the line (correct drift)
        last = best
    return (int(last[0]), int(last[1]))


# ----------------------------------------------------------------- per-image
def extract(path: Path, vocab: Vocab):
    bgr = cv2.imread(str(path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    R, G, B = (rgb[..., i].astype(int) for i in range(3))
    bmask = blue_mask(rgb)
    tissue = (R > 60) & (R >= B - 10) & ~((R < 25) & (G < 25) & (B < 25))  # warm, not black

    words = ocr_words(gray)
    clusters = cluster_words(words)

    # detect blue label boxes (scan outward); separate title/copyright (no blue frame)
    labels, title_cands = [], []
    box_rects = []
    for c in clusters:
        if COPYRIGHT_RE.search(c["text"]):
            continue
        box = find_box(bmask, c["bbox"])
        x, y, w, h = c["bbox"]
        if box is not None:
            rect = box[:4]
            txt, mc = ocr_box(gray, rect)            # clean re-OCR of the box
            if not txt or any(ch.isalpha() for ch in txt) is False:
                txt, mc = c["text"], c["conf"]
            labels.append({"text": txt, "conf": mc, "rect": rect})
            box_rects.append(rect)
        elif y > H * 0.66 and x < W * 0.33 and h > 35:  # bottom-left big text = title
            title_cands.append(c)

    # line_blue = blue minus the box rectangles (so components are pure leader lines)
    line_blue = bmask.copy()
    for (x0, y0, x1, y1) in box_rects:
        b = 16
        line_blue[max(0, y0 - b):y1 + b, max(0, x0 - b):x1 + b] = 0

    triples = []
    for lb in labels:
        q = leader_endpoint(line_blue, lb["rect"], tissue)
        y_norm = vocab.normalize(lb["text"])
        triples.append({"q": q, "label_raw": lb["text"], "label": y_norm,
                        "box": lb["rect"], "conf": round(lb["conf"], 1)})
    region = " ".join(c["text"] for c in sorted(title_cands, key=lambda c: c["bbox"][0]))
    # pieces for leak-free inpainting
    copyright_boxes = [c["bbox"] for c in clusters if COPYRIGHT_RE.search(c["text"])]
    title_bbox = None
    if title_cands:
        xs = [c["bbox"][0] for c in title_cands]; ys = [c["bbox"][1] for c in title_cands]
        xe = [c["bbox"][0] + c["bbox"][2] for c in title_cands]
        ye = [c["bbox"][1] + c["bbox"][3] for c in title_cands]
        title_bbox = (min(xs), min(ys), max(xe) - min(xs), max(ye) - min(ys))
    return {"image": path.name, "W": W, "H": H, "region": region, "triples": triples,
            "label_rects": [lb["rect"] for lb in labels], "title_bbox": title_bbox,
            "copyright_boxes": copyright_boxes}, rgb, bmask


def _logo_mask(rgb):
    """Mask bright logo/© marks that sit in the BLACK MARGIN (ring around them is black),
    e.g. the BlueLink logo — WITHOUT eating pale tissue (which is surrounded by tissue,
    not black). Restricted to the bottom strip where logo/copyright live."""
    H, W = rgb.shape[:2]
    R, G, B = (rgb[..., i].astype(int) for i in range(3))
    mn = np.minimum.reduce([R, G, B]); mx = np.maximum.reduce([R, G, B])
    bright = (mn > 170) | ((R > 175) & (G > 140) & (B < 115))   # near-white OR logo-yellow
    black = mx < 32
    strip = np.zeros((H, W), bool); strip[int(0.84 * H):, :] = True
    cand = (bright & strip).astype(np.uint8)
    out = np.zeros((H, W), np.uint8)
    if cand.sum() == 0:
        return out
    num, lab, stats, _ = cv2.connectedComponentsWithStats(cand)
    for i in range(1, num):
        x, y, w, h, a = stats[i]
        if a < 50:
            continue
        pad = 20
        rx0, ry0, rx1, ry1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
        ring = black[ry0:ry1, rx0:rx1].sum() - black[y:y + h, x:x + w].sum()
        ring_n = (rx1 - rx0) * (ry1 - ry0) - w * h
        if ring_n > 0 and ring / ring_n > 0.55:        # surrounded by black margin → annotation
            cv2.rectangle(out, (rx0, ry0), (rx1, ry1), 255, -1)
    return out


def make_clean(bgr, rgb, bmask, result):
    """Inpaint ALL annotations (blue lines+boxes, box text, title, copyright, logo) →
    leak-free I (CLAUDE.md §2). Leader lines crossing tissue are reconstructed."""
    H, W = bgr.shape[:2]
    mask = cv2.dilate(bmask, np.ones((9, 9), np.uint8))  # blue lines + box borders (+halo)

    def fill_xywh(b, pad=8):
        x, y, w, h = b
        cv2.rectangle(mask, (max(0, x - pad), max(0, y - pad)),
                      (min(W, x + w + pad), min(H, y + h + pad)), 255, -1)

    for r in result["label_rects"]:  # (x0,y0,x1,y1) box interiors (text)
        cv2.rectangle(mask, (max(0, r[0] - 8), max(0, r[1] - 8)),
                      (min(W, r[2] + 8), min(H, r[3] + 8)), 255, -1)
    if result["title_bbox"]:
        fill_xywh(result["title_bbox"])
    for b in result["copyright_boxes"]:
        fill_xywh(b)
    mask = np.maximum(mask, _logo_mask(rgb))     # BlueLink logo / © in the black margin
    return cv2.inpaint(bgr, mask, 4, cv2.INPAINT_TELEA)


def q_on_tissue(rgb, q, r=13):
    """QC: is q on the dissection photo (not the black margin)?"""
    if q is None:
        return False
    x, y = q; H, W = rgb.shape[:2]
    patch = rgb[max(0, y - r):y + r, max(0, x - r):x + r]
    if patch.size == 0:
        return False
    return float((patch.max(2) < 28).mean()) < 0.55


def photo_bbox(clean_bgr, thr=24, pad=2):
    """Tight bbox of the dissection photo (largest non-black region) in the CLEAN image
    — its margins are pure black after inpaint, so the photo is the dominant non-black blob.
    Used to crop away the black margins; q is then offset by the crop origin."""
    H, W = clean_bgr.shape[:2]
    nb = (clean_bgr.max(2) > thr).astype(np.uint8)
    nb = cv2.morphologyEx(nb, cv2.MORPH_CLOSE, np.ones((35, 35), np.uint8))
    num, lab, stats, _ = cv2.connectedComponentsWithStats(nb)
    if num <= 1:
        return (0, 0, W, H)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
    w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
    return (max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad))


def visualize(rgb, result, out_path):
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for t in result["triples"]:
        x0, y0, x1, y1 = t["box"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 3)
        if t["q"]:
            qx, qy = t["q"]
            cv2.circle(vis, (qx, qy), 22, (0, 0, 255), -1)
            cv2.circle(vis, (qx, qy), 24, (255, 255, 255), 3)
            cv2.line(vis, ((x0 + x1) // 2, (y0 + y1) // 2), (qx, qy), (0, 255, 0), 2)
            cv2.putText(vis, t["label"][:18], (qx + 26, qy), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, (0, 255, 255), 3)
    cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 80])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/bluelink_images")
    ap.add_argument("--out", default=None, help="write triples.jsonl + clean images here")
    ap.add_argument("--viz", action="store_true", help="write overlay images for visual QC")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--glob", default="*.[Jj][Pp][Gg]")
    args = ap.parse_args()

    src = Path(args.src)
    files = sorted(p for p in src.glob(args.glob))
    if args.limit:
        files = files[: args.limit]
    vocab = Vocab()
    vizdir = src / "_viz"
    if args.viz:
        vizdir.mkdir(exist_ok=True)

    out = Path(args.out) if args.out else None
    if out:
        (out / "clean").mkdir(parents=True, exist_ok=True)
    fout = (out / "triples.jsonl").open("w", encoding="utf-8") if out else None

    n_trip = n_q = n_offtissue = 0
    labels_per = []
    classes = collections.Counter()
    flags = []  # OCR-suspect labels for §2 hand-QC
    SUSPECT = re.compile(r"\b[bcdfghjklmnpqrstvwxz]{4,}\b|brachil|\blac\b")
    for i, p in enumerate(files, 1):
        res, rgb, bmask = extract(p, vocab)
        theme = re.sub(r"_(Copy_of_)?Slide.*", "", p.stem)
        clean = make_clean(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), rgb, bmask, res) if out else None
        ox = oy = 0; W, H = res["W"], res["H"]
        if clean is not None:
            bx0, by0, bx1, by1 = photo_bbox(clean)   # crop away black margins
            clean = clean[by0:by1, bx0:bx1]
            ox, oy = bx0, by0; H, W = clean.shape[:2]
            cv2.imwrite(str(out / "clean" / (p.stem + ".jpg")), clean, [cv2.IMWRITE_JPEG_QUALITY, 92])
        nq = 0
        for t in res["triples"]:
            n_trip += 1
            if not t["q"]:
                continue
            ok_tissue = q_on_tissue(rgb, t["q"])     # checked on original coords
            qx, qy = t["q"][0] - ox, t["q"][1] - oy  # ★ offset q into the cropped frame
            inside = 0 <= qx < W and 0 <= qy < H
            ok = ok_tissue and inside
            n_offtissue += (not ok)
            nq += 1; classes[t["label"]] += 1
            if t["conf"] < 80 or SUSPECT.search(t["label"]):
                flags.append((p.stem, t["label_raw"], t["label"], round(t["conf"], 0)))
            if fout:
                fout.write(json.dumps({
                    "image": f"clean/{p.stem}.jpg", "q": [qx, qy], "label": t["label"],
                    "label_raw": t["label_raw"], "region": res["region"], "conf": t["conf"],
                    "src": theme, "page": p.stem, "w": W, "h": H,
                    "q_on_tissue": ok, "source": "bluelink_images"}, ensure_ascii=False) + "\n")
        n_q += nq; labels_per.append(len(res["triples"]))
        if args.viz:
            visualize(rgb, res, vizdir / (p.stem + ".viz.jpg"))
        if i % 50 == 0:
            print(f"  ...{i}/{len(files)} slides, {n_q} triples so far")
    if fout:
        fout.close()

    nsl = len(files)
    print(f"\n=== {nsl} slides | {n_trip} labels | {n_q} q ({100*n_q/max(1,n_trip):.0f}%) | "
          f"{n_trip/max(1,nsl):.1f} labels/slide ===")
    print(f"distinct classes: {len(classes)} | q off-tissue (QC flag): {n_offtissue} "
          f"({100*n_offtissue/max(1,n_q):.1f}%)")
    print(f"OCR-suspect labels for hand-QC: {len(flags)}")
    for f in flags[:12]:
        print(f"   {f[0][:30]:32} '{f[1]}' -> '{f[2]}' (conf {f[3]:.0f})")
    if out:
        qc = {"slides": nsl, "labels": n_trip, "q_ok": n_q, "labels_per_slide": round(n_trip / max(1, nsl), 2),
              "distinct_classes": len(classes), "q_off_tissue": n_offtissue,
              "top_classes": classes.most_common(25), "suspect_labels": flags}
        (out / "qc_report.json").write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\ntriples -> {out/'triples.jsonl'} | clean imgs -> {out/'clean'} | QC -> {out/'qc_report.json'}")
    if args.viz:
        print(f"overlays -> {vizdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
