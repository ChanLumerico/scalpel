"""Clean BOTH datasets to one bar, then merge (QC policy agreed 2026-06-27).

SAFE deterministic canonicalization only — NOT fuzzy auto-merge (which would wrongly fuse
genuinely different structures: minor/major, flexor/extensor, medius/maximus, artery/nerve,
molars/premolars, v2/v3, l/r). We only:
  1. strip trailing STATE descriptors (cut/reflected/opened/...) — same structure, dissection state
  2. depluralize a trailing tissue word IFF the singular is attested (m. vs mm. OCR variance)
  3. apply a CURATED OCR-fix map (cn viid→viii, lac→iliac, vell→veli, ...) — hand-verified
  4. drop garbage labels (single char / no alpha)
  5. drop q out-of-bounds (negative / ≥ w,h). Dark-but-in-bounds q (foramina, lumens) kept.
Then merge into one jsonl with source-prefixed specimen keys (leak-safe split unit).

    .venv/bin/python scripts/clean_merge.py
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

NEW = ("data/bluelink_triples/triples.jsonl", "data/bluelink_triples", "bluelink")
OLD = ("data/triples/triples.jsonl", "data/triples", "quizlink")
OUT = Path("data/merged")

STATE = {"cut", "reflected", "opened", "retracted", "removed", "divided", "refld"}
PLURAL = {"muscles": "muscle", "veins": "vein", "nerves": "nerve", "arteries": "artery",
          "nodes": "node", "vessels": "vessel", "ligaments": "ligament", "tendons": "tendon",
          "canals": "canal", "sinuses": "sinus", "foramina": "foramen", "branches": "branch",
          "tubercles": "tubercle", "processes": "process"}
OCR_FIX = {
    "common lac artery": "common iliac artery", "common lac vein": "common iliac vein",
    "external lac artery": "external iliac artery", "external lac vein": "external iliac vein",
    "internal lac artery": "internal iliac artery", "internal lac vein": "internal iliac vein",
    "vestibulocochlear nerve cn viid": "vestibulocochlear nerve cn viii",
    "hypoglossal nerve cn xil": "hypoglossal nerve cn xii",
    "oculomotor nerve cn iid": "oculomotor nerve cn iii",
    "optic nerve cn il": "optic nerve cn ii",
    "levator vell palatini muscle": "levator veli palatini muscle",
    "ateral cord": "lateral cord",
    "acromioclavicular ac joint": "acromioclavicular joint",
    "posterior inferior cerebellar artery pica": "posterior inferior cerebellar artery",
    "perpendicular plate ethmoid": "perpendicular plate of ethmoid",
}


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def is_garbage(lab):
    return len(lab.strip()) < 2 or not any(c.isalpha() for c in lab)


def strip_state(lab):
    toks = lab.split()
    while toks and toks[-1] in STATE:
        toks.pop()
    return " ".join(toks)


def canon_pass1(lab):
    """OCR fix + state strip (no depluralize yet — needs attested set)."""
    lab = OCR_FIX.get(lab, lab)
    lab = strip_state(lab)
    return OCR_FIX.get(lab, lab)


def depluralize(lab, attested):
    toks = lab.split()
    if toks and toks[-1] in PLURAL:
        sing = toks[:-1] + [PLURAL[toks[-1]]]
        if " ".join(sing) in attested:
            return " ".join(sing)
    return lab


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for path, base, source in (OLD, NEW):
        for r in load(path):
            rows.append({**r, "_base": base, "_source": source})

    # pass 1: OCR + state strip; drop garbage; compute attested singular set
    changes = collections.Counter()
    kept = []
    for r in rows:
        lab0 = r["label"]
        lab1 = canon_pass1(lab0)
        if is_garbage(lab1):
            changes["dropped_garbage"] += 1
            continue
        r["_lab1"] = lab1
        if lab1 != lab0:
            changes["ocr_or_state"] += 1
        kept.append(r)
    attested = {r["_lab1"] for r in kept}

    # pass 2: depluralize (only if singular attested) + OOB q drop
    merged = []
    qdrop = 0
    relabel = collections.Counter()
    for r in kept:
        lab = depluralize(r["_lab1"], attested)
        if lab != r["_lab1"]:
            changes["depluralized"] += 1
        if lab != r["label"]:
            relabel[(r["label"], lab)] += 1
        q = r["q"]
        w, h = r["w"], r["h"]
        if not (0 <= q[0] < w and 0 <= q[1] < h):
            qdrop += 1
            continue
        img = f'{r["_base"]}/{r["image"]}'
        spec = f'{r["_source"]}:{r["src"]}#{r["page"]}'
        merged.append({"image": img, "q": [int(q[0]), int(q[1])], "label": lab,
                       "src": spec, "source": r["_source"], "w": w, "h": h,
                       "region": r.get("region", ""), "label_raw": r.get("label_raw", "")})

    with (OUT / "triples.jsonl").open("w", encoding="utf-8") as f:
        for m in merged:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # stats
    def core(c):
        return sum(1 for v in c.values() if v >= 2)
    cnt = collections.Counter(m["label"] for m in merged)
    by_src = collections.Counter(m["source"] for m in merged)
    print(f"== canonicalization changes ==")
    for k, v in changes.most_common():
        print(f"   {k}: {v}")
    print(f"   q out-of-bounds dropped: {qdrop}")
    print(f"\n== top relabels (old -> canonical) ==")
    for (a, b), n in relabel.most_common(15):
        print(f"   x{n:<3} '{a}' -> '{b}'")
    print(f"\n== MERGED dataset ==")
    print(f"   triples {len(merged)} ({dict(by_src)})")
    print(f"   distinct classes {len(cnt)} | core(>=2) {core(cnt)} | singletons {sum(1 for v in cnt.values() if v==1)}")
    print(f"   specimens {len({m['src'] for m in merged})}")
    print(f"\nwrote -> {OUT/'triples.jsonl'}")

    (OUT / "clean_report.json").write_text(json.dumps({
        "changes": dict(changes), "q_oob_dropped": qdrop,
        "relabels": [[a, b, n] for (a, b), n in relabel.most_common()],
        "merged_triples": len(merged), "by_source": dict(by_src),
        "distinct": len(cnt), "core": core(cnt)}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
