"""Dedup IDENTICAL photos (strict) + union pins → leak-safe clean merged dataset.

"Same photo" = exact byte hash ∪ 64x64 grayscale corr >= 0.99 (STRICT — similar-but-
different 0.90-0.97 photos are NOT merged, per 2026-06-27 decision). For each identity
group: one canonical image (BlueLink-clean preferred = annotations already inpainted, no
pin-marker leakage; then largest area), all pins unioned onto it (q scaled to canonical
size — valid because corr>=0.99 ⇒ same framing), duplicate pins (same label + near q)
removed, and ONE specimen id per photo (so a split never separates copies = no leak).

    .venv/bin/python scripts/dedup_union.py [--apply]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

SRC = "data/merged/triples.jsonl"
BASE = Path("data/merged")
OUT = Path("data/merged_clean")
CORR = 0.99
DEDUP_PX = 35


def dhash(p):
    a = np.asarray(Image.open(p).convert("L").resize((9, 8)), np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write data/merged_clean/ (else report only)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    files = sorted(set(r["image"] for r in rows))
    img_rows = collections.defaultdict(list)
    for r in rows:
        img_rows[r["image"]].append(r)

    # --- cluster images into photo-identity groups (exact hash ∪ corr>=0.99) ---
    H = {f: dhash(BASE / f) for f in files}
    sha = {f: hashlib.sha256((BASE / f).read_bytes()).hexdigest() for f in files}
    g64 = {f: np.asarray(Image.open(BASE / f).convert("L").resize((64, 64)), float).flatten() for f in files}
    pc = lambda x: bin(int(x)).count("1")
    par = {f: f for f in files}

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a

    fl = list(files)
    by_sha = collections.defaultdict(list)
    for f in fl:
        by_sha[sha[f]].append(f)
    for g in by_sha.values():
        for x in g[1:]:
            par[find(x)] = find(g[0])
    for i in range(len(fl)):
        for j in range(i + 1, len(fl)):
            a, b = fl[i], fl[j]
            if find(a) == find(b):
                continue
            if pc(H[a] ^ H[b]) <= 10 and float(np.corrcoef(g64[a], g64[b])[0, 1]) >= CORR:
                par[find(a)] = find(b)

    groups = collections.defaultdict(list)
    for f in files:
        groups[find(f)].append(f)
    print(f"images {len(files)} -> {len(groups)} photo-identity groups (corr>={CORR})")
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"  groups with >1 copy: {len(multi)} (merging {sum(len(v) for v in multi.values())} imgs)")

    # --- per group: canonical + union pins (scaled) + dedup ---
    def area(f):
        r0 = img_rows[f][0]; return r0["w"] * r0["h"]

    out_rows = []
    img_map = {}        # canonical image -> new numeric name
    n_union = n_dedup = 0
    gi = 0
    for gid, members in groups.items():
        # canonical: prefer bluelink, then largest area
        canon = sorted(members, key=lambda f: (img_rows[f][0]["source"] != "bluelink", -area(f)))[0]
        cw, ch = img_rows[canon][0]["w"], img_rows[canon][0]["h"]
        spec = f"photo_{gi:04d}"; gi += 1
        kept = []   # (label, qx, qy, source, conf, region, label_raw)
        for f in members:
            sw, shh = img_rows[f][0]["w"], img_rows[f][0]["h"]
            sx, sy = cw / sw, ch / shh
            for r in img_rows[f]:
                qx, qy = int(round(r["q"][0] * sx)), int(round(r["q"][1] * sy))
                if f != canon:
                    n_union += 1
                # dedup: same label + near q already kept?
                dup = any(k[0] == r["label"] and abs(k[1] - qx) < DEDUP_PX and abs(k[2] - qy) < DEDUP_PX
                          for k in kept)
                if dup:
                    n_dedup += 1
                    continue
                kept.append((r["label"], qx, qy, r["source"], r.get("conf", 100),
                             r.get("region", ""), r.get("label_raw", "")))
        img_map[canon] = None
        for (lab, qx, qy, source, conf, region, raw) in kept:
            out_rows.append({"_canon": canon, "q": [qx, qy], "label": lab, "src": spec,
                             "source": source, "w": cw, "h": ch, "conf": conf,
                             "region": region, "label_raw": raw})

    # stats
    cnt = collections.Counter(r["label"] for r in out_rows)
    bysrc = collections.Counter(r["source"] for r in out_rows)
    core = sum(1 for v in cnt.values() if v >= 2)
    print(f"\n== CLEAN dataset ==")
    print(f"  unique photos {len(img_map)} | triples {len(out_rows)} ({dict(bysrc)})")
    print(f"  pins unioned from dup copies: {n_union} | duplicate pins removed: {n_dedup}")
    print(f"  distinct classes {len(cnt)} | core(>=2) {core} | singletons {sum(1 for v in cnt.values() if v==1)}")
    print(f"  specimens {len({r['src'] for r in out_rows})}")

    if not args.apply:
        print("\n(report only — pass --apply to write data/merged_clean/)")
        return 0

    # write
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)
    for idx, canon in enumerate(sorted(img_map)):
        ext = Path(canon).suffix.lower()
        name = f"{idx:05d}{ext}"
        shutil.copy2(BASE / canon, OUT / "images" / name)
        img_map[canon] = f"images/{name}"
    with (OUT / "triples.jsonl").open("w", encoding="utf-8") as f:
        for r in out_rows:
            r2 = {"image": img_map[r["_canon"]], "q": r["q"], "label": r["label"], "src": r["src"],
                  "source": r["source"], "w": r["w"], "h": r["h"], "conf": r["conf"],
                  "region": r["region"], "label_raw": r["label_raw"]}
            f.write(json.dumps(r2, ensure_ascii=False) + "\n")
    print(f"\nwrote -> {OUT}/  ({len(img_map)} images, {len(out_rows)} triples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
