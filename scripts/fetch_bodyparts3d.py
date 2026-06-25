#!/usr/bin/env python3
"""Fetch a curated region of BodyParts3D meshes for SCALPEL's synthetic data (M1).

Reproducible mesh acquisition (handout §5.1). Pulls individual STL files from the
BodyParts3D GitHub mirror by raw URL — *not* the 908 MB full clone — selecting a
region by anatomical name via the project's own `parts_list_e.txt` (FMA id ->
English name), with readable output filenames.

Source (CC BY-SA 2.1 JP — attribution required):
  Kevin-Mattheus-Moerman/BodyParts3D (clone of BodyParts3D / Anatomography)
  https://github.com/Kevin-Mattheus-Moerman/BodyParts3D

Note: this mirror's STL set is **bones + muscles** (no nerves/arteries/veins).
For nerve/artery/vein structures (handout §1.5's full tissue mix) a different
source such as Z-Anatomy is needed.

Usage::

    python scripts/fetch_bodyparts3d.py --out .cache/bodyparts3d/meshes
    python scripts/fetch_bodyparts3d.py --region upper-limb --side right --validate

Only stdlib is needed to download; `--validate` additionally needs Open3D and
checks every mesh loads with geometry (run it in the render venv).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import urllib.request

API_TREE = "https://api.github.com/repos/Kevin-Mattheus-Moerman/BodyParts3D/git/trees/main?recursive=1"
RAW = "https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/main/assets/BodyParts3D_data"

# region -> (include regex, exclude regex). Exclusions kill look-alikes from other
# regions (e.g. flexor digitorum *brevis* is a FOOT muscle, not forearm).
REGIONS = {
    "upper-limb": (
        r"humer|radius|ulna|clavicle|scapula|subscapularis|deltoid|levator scapulae|"
        r"interosseous membrane of (right|left) forearm|extensor carpi|flexor carpi|"
        r"biceps brachii|triceps brachii|brachialis|brachii|coracobrachialis|"
        r"pronator|supinator|flexor digitorum profundus|flexor digitorum superficialis|"
        r"flexor pollicis|extensor digitorum|extensor pollicis|extensor indicis|"
        r"proximal phalanx of (right|left) .*finger|metacarp",
        r"hallucis|digitorum longus|brevis|hallux|foot|toe|tibia|femur|patella|fibula|"
        r"tarsal|metatars|thigh|knee|ankle|\bleg\b|plantar|sole|lumbar|cervical",
    ),
}


def _ssl_context() -> ssl.SSLContext:
    """Prefer certifi's CA bundle (python.org framework builds lack a system one)."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


_CTX = _ssl_context()


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "scalpel-fetch"})
    with urllib.request.urlopen(req, context=_CTX) as r:  # noqa: S310 (trusted public source)
        return r.read()


def available_stems() -> set[str]:
    tree = json.loads(_get(API_TREE))
    stems = set()
    for node in tree.get("tree", []):
        m = re.search(r"BodyParts3D_data/stl/(.+)\.stl$", node["path"])
        if m:
            stems.add(m.group(1))
    if tree.get("truncated"):
        print("  WARNING: file tree truncated by GitHub API; some meshes may be missed")
    return stems


def name_map() -> dict[str, str]:
    text = _get(f"{RAW}/parts_list_e.txt").decode("utf-8", "replace")
    out = {}
    for line in text.splitlines()[1:]:          # skip header
        parts = line.rstrip("\r").split("\t")
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return out


def sanitize(name: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", re.sub(r"[ /\-]+", "_", name.lower()))


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch a curated BodyParts3D region")
    ap.add_argument("--out", default=".cache/bodyparts3d/meshes")
    ap.add_argument("--region", default="upper-limb", choices=sorted(REGIONS))
    ap.add_argument("--side", default="right", choices=["right", "left", "both"])
    ap.add_argument("--max", type=int, default=40)
    ap.add_argument("--validate", action="store_true", help="check each mesh loads (needs open3d)")
    args = ap.parse_args()

    inc, exc = REGIONS[args.region]
    inc_re, exc_re = re.compile(inc, re.I), re.compile(exc, re.I)
    side_re = None if args.side == "both" else re.compile(rf"\b{args.side}\b", re.I)

    print("[1/3] enumerating available STL files + names ...")
    stems, names = available_stems(), name_map()

    selected = []
    for stem in sorted(stems):
        nm = names.get(stem)
        if not nm or not inc_re.search(nm) or exc_re.search(nm):
            continue
        if side_re and not side_re.search(nm):
            continue
        selected.append((stem, nm))
    selected = selected[: args.max]
    print(f"      selected {len(selected)} '{args.region}' ({args.side}) structures")

    os.makedirs(args.out, exist_ok=True)
    print(f"[2/3] downloading STLs -> {args.out}")
    manifest = []
    for stem, nm in selected:
        fn = sanitize(nm) + ".stl"
        dst = os.path.join(args.out, fn)
        try:
            data = _get(f"{RAW}/stl/{stem}.stl")
            with open(dst, "wb") as f:
                f.write(data)
            manifest.append((fn, stem, nm))
        except Exception as e:
            print(f"      MISS {stem} ({nm}): {e}")
    with open(os.path.join(args.out, "manifest.tsv"), "w") as f:
        f.writelines(f"{fn}\t{stem}\t{nm}\n" for fn, stem, nm in manifest)
    print(f"      downloaded {len(manifest)} meshes (+ manifest.tsv)")

    if args.validate:
        print("[3/3] validating meshes load (open3d) ...")
        import open3d as o3d
        ok = 0
        for fn, _stem, _nm in manifest:
            m = o3d.io.read_triangle_mesh(os.path.join(args.out, fn))
            if len(m.vertices) and len(m.triangles):
                ok += 1
            else:
                print(f"      EMPTY {fn}")
        print(f"      {ok}/{len(manifest)} meshes load with geometry")
    else:
        print("[3/3] skipped load validation (pass --validate in the render venv)")

    print(f"\nDONE. Render in your GUI Terminal:\n"
          f"  python scripts/demo_render.py --mesh-dir {args.out} --out outputs/demo.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
