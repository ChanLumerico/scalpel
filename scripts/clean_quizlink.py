"""Clean QuizLink images in the merged set (parity with BlueLink cleaning).

The existing QuizLink images were NOT put through the BlueLink cleaning, so many still
carry the BlueLink logo, ©, the title text, and black margins (the answer labels are
already absent — quiz version). This removes them and crops to tissue, offsetting q.
Reuses bluelink_extract helpers. Operates in-place on data/merged_clean (.png = quizlink).

    .venv/bin/python scripts/clean_quizlink.py --test 00372   # write /tmp previews
    .venv/bin/python scripts/clean_quizlink.py --apply        # clean all png in-place
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from bluelink_extract import blue_mask, _logo_mask, ocr_words, COPYRIGHT_RE, q_on_tissue  # noqa: E402

BASE = Path("data/merged_clean")


def margin_trim(bgr, black_thr=24, edge_black=0.90):
    """Trim near-uniform black margins from each edge (works even when a little tissue
    protrudes into a corner — unlike a bbox of the largest blob). Stops at the first
    row/col that is < edge_black fraction black."""
    nb = (bgr.max(2) > black_thr)
    H, W = nb.shape
    rowf, colf = nb.mean(1), nb.mean(0)
    keep = 1.0 - edge_black
    y0 = 0
    while y0 < H and rowf[y0] < keep:
        y0 += 1
    y1 = H
    while y1 > y0 and rowf[y1 - 1] < keep:
        y1 -= 1
    x0 = 0
    while x0 < W and colf[x0] < keep:
        x0 += 1
    x1 = W
    while x1 > x0 and colf[x1 - 1] < keep:
        x1 -= 1
    if x1 - x0 < W * 0.3 or y1 - y0 < H * 0.3:    # safety: never over-trim
        return 0, 0, W, H
    return x0, y0, x1, y1


def clean_quizlink(bgr, qs=(), pad=28):
    """Remove logo/©/title/blue-remnant (bottom region) + trim black margins. The crop is
    PIN-AWARE: it is expanded so EVERY pin in ``qs`` stays inside (never lose a labelled
    structure — e.g. a coccyx pin at the very bottom). Returns (clean_cropped, (ox, oy))."""
    H, W = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = cv2.dilate(blue_mask(rgb), np.ones((9, 9), np.uint8))   # any blue remnant
    mask = np.maximum(mask, _logo_mask(rgb))                       # BlueLink logo
    for (t, c, x, y, w, h) in ocr_words(gray, min_conf=45):        # title / © / logo text
        if y > 0.80 * H or COPYRIGHT_RE.search(t):
            cv2.rectangle(mask, (max(0, x - 10), max(0, y - 10)),
                          (min(W, x + w + 10), min(H, y + h + 10)), 255, -1)
    clean = cv2.inpaint(bgr, mask, 4, cv2.INPAINT_TELEA)
    bx0, by0, bx1, by1 = margin_trim(clean)
    for (qx, qy) in qs:                                            # ★ never crop out a pin
        bx0 = min(bx0, max(0, qx - pad)); by0 = min(by0, max(0, qy - pad))
        bx1 = max(bx1, min(W, qx + pad)); by1 = max(by1, min(H, qy + pad))
    return clean[by0:by1, bx0:bx1], (bx0, by0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", nargs="*", help="image numbers (e.g. 00372) → /tmp previews")
    ap.add_argument("--apply", action="store_true", help="clean all .png in-place + update triples")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(BASE / "triples.jsonl", encoding="utf-8")]
    by_img = collections.defaultdict(list)
    for r in rows:
        by_img[r["image"]].append(r)

    if args.test is not None:
        for num in args.test:
            img = f"images/{num}.png"
            if img not in by_img:
                print(f"  {img} not in dataset"); continue
            bgr = cv2.imread(str(BASE / img))
            clean, (ox, oy) = clean_quizlink(bgr, [tuple(r["q"]) for r in by_img[img]])
            ch, cw = clean.shape[:2]
            vis = clean.copy()
            for r in by_img[img]:
                qx, qy = r["q"][0] - ox, r["q"][1] - oy
                cv2.circle(vis, (qx, qy), 14, (0, 0, 255), -1)
                cv2.circle(vis, (qx, qy), 16, (255, 255, 255), 2)
                cv2.putText(vis, r["label"][:16], (qx + 18, qy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imwrite(f"/tmp/qlclean_{num}.jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
            print(f"  {img} {bgr.shape[1]}x{bgr.shape[0]} -> {cw}x{ch} origin=({ox},{oy}) | -> /tmp/qlclean_{num}.jpg")
        return 0

    if args.apply:
        OUT = Path("data/merged_final")     # NON-DESTRUCTIVE: merged_clean stays as backup
        if OUT.exists():
            shutil.rmtree(OUT)
        (OUT / "images").mkdir(parents=True)
        changed = 0
        for k, (img, rs) in enumerate(by_img.items(), 1):
            if img.endswith(".png"):        # quizlink → clean
                bgr = cv2.imread(str(BASE / img))
                clean, (ox, oy) = clean_quizlink(bgr, [tuple(r["q"]) for r in rs])
                ch, cw = clean.shape[:2]
                if (ox, oy) != (0, 0) or cw != bgr.shape[1] or ch != bgr.shape[0]:
                    changed += 1
                cv2.imwrite(str(OUT / img), clean)
                for r in rs:
                    r["q"] = [r["q"][0] - ox, r["q"][1] - oy]; r["w"], r["h"] = cw, ch
            else:                            # bluelink → already clean, copy as-is
                shutil.copy2(BASE / img, OUT / img)
            if k % 100 == 0:
                print(f"   ...{k}/{len(by_img)}")
        # VALUE-PRESERVING: drop ONLY out-of-bounds q (unusable). off-tissue → FLAG not drop.
        kept, oob, flags = [], 0, []
        for img, rs in by_img.items():
            rgb = cv2.cvtColor(cv2.imread(str(OUT / img)), cv2.COLOR_BGR2RGB)
            for r in rs:
                if not (0 <= r["q"][0] < r["w"] and 0 <= r["q"][1] < r["h"]):
                    oob += 1; continue
                if not q_on_tissue(rgb, r["q"]):
                    flags.append({"image": img, "q": r["q"], "label": r["label"], "source": r["source"]})
                kept.append(r)
        with (OUT / "triples.jsonl").open("w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (OUT / "qc_offtissue_flags.json").write_text(json.dumps(flags, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote -> {OUT}/ (NON-destructive; merged_clean kept as backup)")
        print(f"  cleaned {sum(1 for i in by_img if i.endswith('.png'))} quizlink png ({changed} trimmed) | "
              f"triples {len(kept)} (dropped {oob} OOB only) | off-tissue FLAGGED: {len(flags)}")
        return 0

    ap.error("pass --test or --apply")


if __name__ == "__main__":
    raise SystemExit(main())
