"""Joint QC over BOTH datasets (existing QuizLink + new BlueLink Images) before merge.

Merging two OCR-derived label sets risks class fragmentation: the same structure spelled
two ways (OCR variant or synonym) becomes two classes, inflating the singleton tail and
corrupting evaluation. This holds both sources to one bar and surfaces fixes:

  1. label garble flags   — consonant clusters / known OCR errors (both sources)
  2. near-duplicate labels — fuzzy pairs (token-blocked) = OCR-variant/synonym to unify
  3. q-on-tissue           — pins that land in black margin / off the photo (both sources)

    .venv/bin/python scripts/qc_datasets.py            # report only
"""

from __future__ import annotations

import collections
import difflib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

NEW = "data/bluelink_triples/triples.jsonl"
NEW_BASE = Path("data/bluelink_triples")
OLD = "data/triples/triples.jsonl"
OLD_BASE = Path("data/triples")

TISSUE_WORDS = {"artery", "arteries", "vein", "veins", "nerve", "nerves", "muscle", "muscles",
                "ligament", "tendon", "duct", "gland", "node", "nodes", "bone", "vessel", "vessels",
                "joint", "membrane", "cartilage", "process", "ramus", "rami"}
GENERIC = {"l", "r", "left", "right", "superior", "inferior", "anterior", "posterior", "deep",
           "superficial", "lateral", "medial", "internal", "external", "common", "cn"}
SUSPECT = re.compile(r"[bcdfghjklmnpqrstvwxz]{4,}|brachil|\blac\b|\bm\b$|nerv$|arter$", re.I)


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def name_tokens(lab):
    return [t for t in lab.split() if t not in TISSUE_WORDS and t not in GENERIC and len(t) > 1]


def garble_flags(counter, label_to_ex):
    out = []
    for lab, n in counter.items():
        toks = lab.split()
        bad = SUSPECT.search(lab) or (len(lab) < 3) or any(len(t) >= 4 and not re.search("[aeiou]", t) for t in toks)
        if bad:
            out.append((lab, n, label_to_ex.get(lab, "")))
    return sorted(out, key=lambda x: -x[1])


def near_dups(labels_with_meta, thr=0.86):
    """Token-blocked fuzzy matching. labels_with_meta: {label: (count, sources)}.
    Returns candidate pairs (a, b, ratio) likely the same structure spelled differently."""
    labels = list(labels_with_meta)
    block = collections.defaultdict(list)
    for lab in labels:
        for t in set(name_tokens(lab)):
            block[t].append(lab)
    seen = set(); pairs = []
    for t, group in block.items():
        if len(group) > 60:        # skip ultra-common tokens (weak block)
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a == b:
                    continue
                key = tuple(sorted((a, b)))
                if key in seen:
                    continue
                seen.add(key)
                r = difflib.SequenceMatcher(None, a, b).ratio()
                if r >= thr:
                    pairs.append((a, b, round(r, 3)))
    pairs.sort(key=lambda x: -x[2])
    return pairs


def q_off_tissue(records, base, cache):
    """Flag q's whose neighborhood is mostly black (off the photo / in margin)."""
    flags = []
    by_img = collections.defaultdict(list)
    for i, r in enumerate(records):
        by_img[r["image"]].append(i)
    for img, idxs in by_img.items():
        p = base / img
        im = cache.get(p)
        if im is None:
            im = cv2.imread(str(p))
            cache[p] = im
        if im is None:
            continue
        H, W = im.shape[:2]
        for i in idxs:
            x, y = records[i]["q"]
            x, y = int(x), int(y)
            patch = im[max(0, y - 13):y + 13, max(0, x - 13):x + 13]
            if patch.size == 0 or float((patch.max(2) < 28).mean()) >= 0.55:
                flags.append((img, records[i]["label"], [x, y]))
    return flags


def main():
    new, old = load(NEW), load(OLD)
    nc = collections.Counter(r["label"] for r in new)
    oc = collections.Counter(r["label"] for r in old)
    ex_new = {r["label"]: r.get("label_raw", "") for r in new}
    ex_old = {r["label"]: r.get("label_raw", "") for r in old}

    print(f"NEW {len(new)} triples / {len(nc)} classes | OLD {len(old)} / {len(oc)} classes\n")

    # 1. garble
    gn = garble_flags(nc, ex_new); go = garble_flags(oc, ex_old)
    print(f"== 1. label garble flags ==  NEW {len(gn)} | OLD {len(go)}")
    for lab, n, raw in gn[:10]:
        print(f"   NEW x{n:<3} '{lab}'  (raw '{raw}')")
    for lab, n, raw in go[:8]:
        print(f"   OLD x{n:<3} '{lab}'  (raw '{raw}')")

    # 2. near-duplicates across the merged vocabulary
    meta = {}
    for lab in set(nc) | set(oc):
        src = ("N" if lab in nc else "") + ("Q" if lab in oc else "")
        meta[lab] = (nc.get(lab, 0) + oc.get(lab, 0), src)
    pairs = near_dups(meta)
    cross = [(a, b, r) for a, b, r in pairs if meta[a][1] != meta[b][1] or "NQ" in (meta[a][1] + meta[b][1])]
    print(f"\n== 2. near-duplicate label pairs (ratio>=0.86): {len(pairs)} (cross-source-relevant {len(cross)}) ==")
    for a, b, r in pairs[:22]:
        print(f"   {r}  '{a}'[{meta[a][1]}x{meta[a][0]}]  <->  '{b}'[{meta[b][1]}x{meta[b][0]}]")

    # 3. q-on-tissue (both)
    cache = {}
    qn = q_off_tissue(new, NEW_BASE, cache)
    qo = q_off_tissue(old, OLD_BASE, cache)
    print(f"\n== 3. q off-tissue (mostly-black neighborhood) ==  NEW {len(qn)} ({100*len(qn)/len(new):.1f}%) | "
          f"OLD {len(qo)} ({100*len(qo)/len(old):.1f}%)")
    for img, lab, q in qn[:5]:
        print(f"   NEW {lab:28} q={q}  {img[:40]}")
    for img, lab, q in qo[:5]:
        print(f"   OLD {lab:28} q={q}  {img[:40]}")

    report = {
        "new_triples": len(new), "old_triples": len(old),
        "garble_new": gn, "garble_old": go,
        "near_dup_pairs": [[a, b, r, meta[a][1], meta[b][1]] for a, b, r in pairs],
        "q_off_tissue_new": qn, "q_off_tissue_old": qo,
    }
    Path("data/bluelink_triples/qc_joint.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote -> data/bluelink_triples/qc_joint.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
