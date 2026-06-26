"""Label cleaning + canonicalization for QuizLink triples (HANDOUT v2 §5.5).

OCR over the baked label boxes is noisy: it yields garbage strings ("ee", "ma",
"inferior opens"), OCR misspellings ("deltold m."), and many surface variants of
the same structure ("biceps brachii" / "biceps brachii muscle"). Left alone this
*inflates* the class count and starves every class of instances. This module:

1. **drops** strings that don't look anatomical (junk filter);
2. **canonicalizes** the rest by greedy, frequency-priority fuzzy clustering -
   a rare/misspelt label snaps onto a more frequent near-identical one, so
   variants collapse and the evaluable (>=2-instance) core grows. Precision is
   favoured (high threshold) - a wrong merge corrupts a label, so we print every
   merge for inspection;
3. tags each image's **modality** (cadaver / bone / model) with a coarse colour-
   texture heuristic, so different appearance domains can be filtered later.

    python -m scalpel.data.clean --in data/triples/triples.jsonl \
        --out data/triples/triples.clean.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from .vocab import Vocab

# tokens that signal a label is a real gross-anatomy structure
_ANAT = re.compile(
    r"\b(muscle|nerve|artery|vein|bone|process|fossa|ligament|tendon|gland|joint|"
    r"sinus|canal|foramen|tubercle|spine|notch|condyle|fissure|duct|node|trunk|"
    r"plexus|ramus|crest|head|neck|body|membrane|septum|sulcus|gyrus|ventricle|"
    r"cortex|lobe|vessel|cartilage|capsule|fascia|aponeurosis|symphysis|disc|"
    r"meatus|antrum|cavity|recess|hiatus|aorta|atrium|valve|cord|root|branch|"
    r"fold|arch|sheath|bursa|groove|ala|horn|limb|wall|margin|angle|border|"
    r"surface| duct|tract|nucleus|bundle|fiber|ligamentum|process|tuberosity|"
    r"malleolus|epicondyle|trochanter|ramus|septum|raphe|commissure|peduncle|"
    r"colliculus|hippocampus|cerebellum|thalamus|pons|medulla|tonsil|ossicle)\b"
)
_VOWEL = re.compile(r"[aeiou]")

# single-word gross-anatomy structures: valid even though they have no category
# suffix and aren't multi-token (so "tibia", "uvula" are kept, "peel", "made" not)
_STRUCTURES = {
    # bones
    "tibia", "fibula", "femur", "humerus", "radius", "ulna", "mandible",
    "maxilla", "scapula", "clavicle", "sternum", "patella", "talus", "calcaneus",
    "navicular", "cuboid", "sacrum", "coccyx", "atlas", "axis", "vomer",
    "ethmoid", "sphenoid", "occiput", "zygomatic", "hyoid", "manubrium",
    "ilium", "ischium", "pubis", "acetabulum", "rib", "ribs", "vertebra",
    "phalanx", "phalanges", "metacarpal", "metatarsal", "carpals", "tarsals",
    # viscera / organs
    "liver", "spleen", "pancreas", "stomach", "kidney", "bladder", "uterus",
    "ovary", "prostate", "testis", "thymus", "thyroid", "tonsil", "tonsils",
    "appendix", "cecum", "duodenum", "jejunum", "ileum", "colon", "rectum",
    "esophagus", "trachea", "larynx", "pharynx", "epiglottis", "diaphragm",
    "lung", "lungs", "heart", "brain", "gallbladder", "ureter", "urethra",
    "spleen", "tongue",
    # brain / neuro
    "cerebellum", "pons", "medulla", "thalamus", "hypothalamus", "hippocampus",
    "amygdala", "cortex", "cerebrum", "midbrain", "infundibulum", "pituitary",
    "hypophysis", "uvula",
    # eye / ear
    "cornea", "retina", "iris", "lens", "sclera", "pupil", "cochlea", "malleus",
    "incus", "stapes",
    # serous membranes / misc
    "septum", "mediastinum", "peritoneum", "pleura", "pericardium", "omentum",
    "mesentery", "falx", "tentorium", "aorta", "esophagus",
}

# tissue-type suffixes: a difference in these does NOT make a different structure
_TISSUE = {
    "muscle", "artery", "vein", "nerve", "bone", "bones", "ligament", "tendon",
    "gland", "joint", "duct", "sinus", "vessel", "node", "branch", "process",
}
# contrastive modifiers: a difference in these DOES define a different structure
_CONTRAST = {
    "internal", "external", "superior", "inferior", "anterior", "posterior",
    "medial", "lateral", "deep", "superficial", "major", "minor", "greater",
    "lesser", "proximal", "distal", "ascending", "descending", "transverse",
    "left", "right", "l", "r", "common", "middle", "accessory", "first",
    "second", "third", "fourth", "fifth", "1st", "2nd", "3rd", "4th", "5th",
    # Latin size/depth/length modifiers - each names a distinct structure
    "maximus", "minimus", "medius", "longus", "brevis", "magnus", "profundus",
    "superficialis", "externus", "internus",
}
# curated anatomy terms that are real but often appear only once (so the df>=2
# self-lexicon misses them) and are NOT in the English system dictionary
_ANATOMY_LEX = {
    "omohyoid", "mylohyoid", "stylohyoid", "geniohyoid", "thyrohyoid",
    "sternohyoid", "sternothyroid", "thyroarytenoid", "cricothyroid",
    "cricoarytenoid", "digastric", "buccinator", "mentalis", "masseter",
    "platysma", "digitorum", "hallucis", "pollicis", "indicis", "iliac",
    "epicondyle", "condyle", "malleolus", "trochanter", "tuberosity",
    "infraspinatus", "supraspinatus", "subscapularis", "teres", "deltoid",
    "trapezius", "rhomboid", "scalene", "splenius", "semispinalis",
}
_LEVEL = re.compile(r"^(?:c|t|l|s|cn)?\d+$")      # c5, t1, l4, s3, 12
_ROMAN = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"}
# bare category words: a label that is ONLY one of these (no specifier) is a
# truncated OCR read ("...artery" with the name lost), not an identifiable class
_GENERIC_SOLO = _TISSUE | {
    "body", "head", "neck", "trunk", "root", "horn", "lobe", "wall", "cavity",
    "membrane", "cord", "fold", "arch", "crest", "notch", "spine", "angle",
    "border", "margin", "surface", "ramus", "branch", "sheath", "fossa",
    "process", "tubercle", "tuberosity", "canal", "foramen", "groove", "sulcus",
}


def _safe_merge(a: str, b: str, real: set, pair_thresh: int = 80) -> bool:
    """True if a->b is an OCR/variant fix, not a merge of two distinct structures.

    Differing tokens are paired across the two labels by character similarity
    (OCR variants pair up); any *unpaired* leftover that is a contrastive
    modifier, a level (c5/s3/cn ii) or a real word (seen elsewhere, not a tissue
    suffix) means the labels denote different structures -> refuse.
    """
    from rapidfuzz import fuzz

    sa, sb = set(a.split()), set(b.split())
    a_only, b_only = sa - sb, sb - sa
    avail = list(b_only)
    leftover = []
    for x in a_only:                                   # greedily OCR-pair a->b
        hit = None
        for i, y in enumerate(avail):
            if x in real and y in real:                # two distinct real words ->
                continue                               # a genuine difference, not a typo
            if max(fuzz.ratio(x, y), fuzz.token_sort_ratio(x, y)) >= pair_thresh:
                hit = i
                break
        if hit is None:
            leftover.append(x)
        else:
            avail.pop(hit)
    leftover += avail
    for t in leftover:
        if t in _CONTRAST or t in _ROMAN or _LEVEL.match(t):
            return False
        if t in real and t not in _TISSUE:
            return False
    return True


def is_valid(label: str) -> bool:
    """True if ``label`` plausibly names an anatomical structure (not OCR junk)."""
    label = label.strip()
    toks = label.split()
    if len(label) < 4 or not _VOWEL.search(label):
        return False
    if all(len(t) < 3 for t in toks):                 # only tiny tokens -> junk
        return False
    if len(toks) == 1 and toks[0] in _GENERIC_SOLO:   # bare "artery"/"body" -> truncated
        return False
    if _ANAT.search(label):                           # has an anatomy keyword
        return True
    if len(toks) == 1:                                # single word -> must be a known structure
        return toks[0] in _STRUCTURES or toks[0] in _ANATOMY_LEX
    # otherwise require >=2 alphabetic tokens, each with a vowel (a proper name)
    alpha = [t for t in toks if re.fullmatch(r"[a-z]+", t) and _VOWEL.search(t)]
    return len(alpha) >= 2


def canonicalize(counts: dict, thresh: int = 90, seed=()):
    """Greedy frequency-priority fuzzy clustering -> ``(mapping, merges)``.

    Frequent valid labels become cluster centroids; a later label snaps onto an
    existing centroid when its similarity (max of token-sort and plain ratio)
    reaches ``thresh``, else it starts its own centroid. ``mapping[label]`` is the
    canonical form ("" = dropped as junk).
    """
    from rapidfuzz import fuzz, process

    # lexicon of "real" tokens: df>=2 in our data + English dict + anatomy terms
    tok_df = collections.Counter()
    for lab in counts:
        if is_valid(lab):
            tok_df.update(set(lab.split()))
    real = {t for t, df in tok_df.items() if df >= 2} | _CONTRAST | _ANATOMY_LEX | _STRUCTURES
    try:                                            # macOS/Linux system word list
        words = Path("/usr/share/dict/words").read_text().lower().split()
        real |= {w for w in words if len(w) >= 4}   # skip 1-3 char noise words
    except OSError:
        pass

    order = sorted(counts, key=lambda l: (-counts[l], l))
    canon: list[str] = list(seed)
    mapping: dict[str, str] = {}
    merges: list[tuple] = []
    for lab in order:
        if not is_valid(lab):
            mapping[lab] = ""
            continue
        if canon:
            # best match under token-sort and plain edit ratio (take the higher)
            cands = [m for m in (
                process.extractOne(lab, canon, scorer=fuzz.token_sort_ratio),
                process.extractOne(lab, canon, scorer=fuzz.ratio),
            ) if m]
            m = max(cands, key=lambda x: x[1], default=None)
            if m and m[1] >= thresh and _safe_merge(lab, m[0], real):
                mapping[lab] = m[0]
                if m[0] != lab:
                    merges.append((lab, m[0], round(m[1])))
                continue
        canon.append(lab)
        mapping[lab] = lab
    return mapping, merges


def modality(img_path) -> str:
    """Coarse appearance-domain tag: cadaver | bone | model (heuristic, approximate)."""
    import cv2
    import numpy as np

    im = cv2.imread(str(img_path))
    if im is None:
        return "cadaver"
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sat, val = float(s.mean()), float(v.mean())
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    tex = float(cv2.Laplacian(gray, cv2.CV_64F).var())   # texture energy (smoothness)
    # pink/salmon 3D-render hue (OpenCV H 0-179: magenta~150, red wraps 0/179)
    pink = float((((h > 145) | (h < 8)) & (s > 60) & (v > 120)).mean())
    if val > 190 and sat < 35:
        return "bone"                                    # bright, desaturated specimen
    if pink > 0.4 and tex < 250:                         # vivid pink + smooth -> render
        return "model"
    return "cadaver"


def clean_dataset(in_jsonl: str, out_jsonl: str, thresh: int = 90, tag: bool = True,
                  prune_images: bool = False):
    rows = [json.loads(l) for l in open(in_jsonl, encoding="utf-8") if l.strip()]
    for r in rows:                                     # apply the latest vocab
        r["label_raw"] = r["label"]                    # (nn->nerves, m->muscle, ...)
        r["label"] = Vocab.normalize(r["label"])
    counts = collections.Counter(r["label"] for r in rows)
    mapping, merges = canonicalize(dict(counts), thresh)

    base = Path(in_jsonl).parent
    mod_cache: dict[str, str] = {}
    vocab = Vocab()
    kept = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            canon = mapping.get(r["label"], "")
            if not canon:                                # dropped junk
                continue
            r = dict(r)
            r["label"] = canon
            r["label_id"] = vocab.index(canon)
            if tag:
                img = r["image"]
                if img not in mod_cache:
                    mod_cache[img] = modality(base / img)
                r["modality"] = mod_cache[img]
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            kept += 1
    vocab.save(Path(out_jsonl).with_name("vocab.json"))

    # ---- delete orphan images (pages whose every label was junk) ----------
    pruned = 0
    if prune_images:
        base = Path(out_jsonl).parent
        keep = {Path(r["image"]).name
                for r in (json.loads(l) for l in open(out_jsonl, encoding="utf-8"))}
        imgdir = base / "images"
        if imgdir.is_dir():
            for f in imgdir.glob("*.png"):
                if f.name not in keep:
                    f.unlink()
                    pruned += 1

    # ---- report ----------------------------------------------------------
    def core(rs):
        c = collections.Counter(x["label"] for x in rs)
        ge2 = [k for k, v in c.items() if v >= 2]
        return len(c), len(ge2), sum(c[k] for k in ge2)

    raw_cls = len(counts)
    new_rows = [json.loads(l) for l in open(out_jsonl, encoding="utf-8")]
    nc, nge2, ncov = core(new_rows)
    print(f"raw:     {len(rows)} triples, {raw_cls} classes")
    print(f"dropped: {len(rows) - kept} junk triples")
    print(f"merges:  {len(merges)} variant->canonical")
    print(f"clean:   {kept} triples, {nc} classes")
    if prune_images:
        print(f"pruned:  {pruned} orphan images (page had only junk labels)")
    print(f"  evaluable core (>=2 inst): {nge2} classes / {ncov} triples "
          f"(was {sum(1 for v in counts.values() if v >= 2)} classes)")
    if tag:
        md = collections.Counter(r.get("modality") for r in new_rows)
        print(f"  modality: {dict(md)}")
    print("\nsample merges (variant -> canonical, score):")
    for a, b, s in merges[:25]:
        print(f"  {a!r:42s} -> {b!r:32s} {s}")
    return {"kept": kept, "classes": nc, "core_classes": nge2, "merges": len(merges)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Clean + canonicalize QuizLink labels")
    ap.add_argument("--in", dest="inp", default="data/triples/triples.jsonl")
    ap.add_argument("--out", default="data/triples/triples.clean.jsonl")
    ap.add_argument("--thresh", type=int, default=90)
    ap.add_argument("--no-tag", action="store_true")
    ap.add_argument("--prune", action="store_true",
                    help="delete orphan images (pages with no surviving triple)")
    a = ap.parse_args()
    clean_dataset(a.inp, a.out, a.thresh, tag=not a.no_tag, prune_images=a.prune)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
