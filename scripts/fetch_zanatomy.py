#!/usr/bin/env python3
"""Fetch Z-Anatomy meshes (incl. nerves + vessels) for a region (handout §5.1).

Z-Anatomy (CC BY-SA, github.com/LluisV/Z-Anatomy) ships per-SYSTEM FBX files,
each holding many *named* structures. Unlike BodyParts3D, it includes peripheral
nerves (brachial plexus, median/ulnar/radial) and vessels (axillary/brachial
arteries, basilic/cephalic veins) — the structures a real spot exam pins.

Pipeline: download the system FBX -> convert FBX->glb with the assimp CLI
(`brew install assimp`) -> read named geometries with trimesh -> filter to a
region by name -> export one STL per structure (all in one shared coordinate
frame, so they align in a scene).

Requires: assimp CLI on PATH, `pip install trimesh`.

Usage:
    python scripts/fetch_zanatomy.py --out .cache/zanatomy/upperlimb --region upper-limb
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import ssl
import subprocess
import urllib.request

RAW = "https://raw.githubusercontent.com/LluisV/Z-Anatomy/PC-Version/Resources/Models/FBX"
SYSTEMS = {  # FBX stem on the repo
    "nervous": "NervousSystem100",
    "cardio": "CardioVascular41",
    "muscular": "MuscularSystem100",
    "skeletal": "SkeletalSystem100",
}

REGIONS = {
    "upper-limb": (
        r"humer|radius|ulna|clavicle|scapula|carpal|metacarp|phalan|deltoid|biceps|"
        r"triceps|brachial|brachii|brachialis|coracobrach|pectoral|latissimus|teres|"
        r"supraspinatus|infraspinatus|subscapularis|serratus|rhomboid|trapezius|subclavius|"
        r"pronator|supinator|flexor|extensor|interosseous|anconeus|palmaris|brachioradialis|"
        r"median nerve|ulnar nerve|radial nerve|musculocut|axillary nerve|brachial plexus|"
        r"antebrachial cutaneous|interosseous nerve|palmar digital|cord of brachial|"
        r"trunk of brachial|suprascapular|subscapular nerve|thoracodorsal|long thoracic|"
        r"axillary artery|brachial artery|radial artery|ulnar artery|interosseous artery|"
        r"subclavian|collateral artery|recurrent artery|circumflex humeral|thoracoacromial|"
        r"palmar|digital arter|axillary vein|brachial vein|basilic|cephalic|"
        r"interosseous vein|digital vein",
        r"foot|toe|hallu|thigh|femor|tibia|peroneal|fibular|sural|plantar|saphen|poplit|"
        r"gluteal|head|brain|cerebr|facial|maxillar|cranial|optic|ophthalm|nasal|lingual|"
        r"abdomin|pelvi|iliac|lumbar|sacr|intercostal|lung|heart|spleen|liver|kidney|"
        r"carotid|jugular|intestin|gastric|hepatic|renal|mesenteric|coronary|pulmonary|"
        r"\baorta|vena cava|vertebr|knee|ankle|psoas|phrenic|vagus|recurrent laryngeal|"
        r"sympathetic|esophag|trache|digitorum longus|fibularis|soleus|gastrocnem|"
        r"popliteus|tibialis|peroneus|calcane",
    ),
}


def _ctx() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _download(stem: str, cache: str) -> str:
    fbx = os.path.join(cache, stem + ".fbx")
    if not os.path.exists(fbx):
        print(f"  downloading {stem}.fbx ...")
        req = urllib.request.Request(f"{RAW}/{stem}.fbx", headers={"User-Agent": "scalpel"})
        with urllib.request.urlopen(req, context=_ctx()) as r, open(fbx, "wb") as f:
            shutil.copyfileobj(r, f)
    return fbx


def _to_glb(fbx: str, cache: str) -> str:
    glb = os.path.splitext(fbx)[0] + ".glb"
    if not os.path.exists(glb):
        assimp = shutil.which("assimp")
        if not assimp:
            raise SystemExit("assimp CLI not found — run `brew install assimp`")
        subprocess.run([assimp, "export", fbx, glb], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return glb


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch a Z-Anatomy region (with nerves+vessels)")
    ap.add_argument("--out", default=".cache/zanatomy/upperlimb")
    ap.add_argument("--region", default="upper-limb", choices=sorted(REGIONS))
    ap.add_argument("--cache", default=".cache/zanatomy/_fbx")
    ap.add_argument("--both-sides", action="store_true", help="keep left+right (default: one side)")
    args = ap.parse_args()

    import trimesh

    inc, exc = (re.compile(p, re.I) for p in REGIONS[args.region])
    nums = re.compile(r"\.\d+$")
    os.makedirs(args.cache, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    for f in os.listdir(args.out):
        if f.endswith(".stl"):
            os.remove(os.path.join(args.out, f))

    seen, per = set(), {}
    for key, stem in SYSTEMS.items():
        glb = _to_glb(_download(stem, args.cache), args.cache)
        scene = trimesh.load(glb)
        if not isinstance(scene, trimesh.Scene):
            continue
        k = 0
        for name, geom in scene.geometry.items():
            if ".j." in name or (not args.both_sides and nums.search(name)):
                continue
            if not inc.search(name) or exc.search(name) or len(geom.vertices) < 40:
                continue
            fn = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            if not fn or fn in seen:
                continue
            seen.add(fn)
            geom.export(os.path.join(args.out, fn + ".stl"))
            k += 1
        per[key] = k
        print(f"  {key}: {k} structures")
    print(f"\nDONE: {sum(per.values())} structures -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
