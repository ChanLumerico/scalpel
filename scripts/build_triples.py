#!/usr/bin/env python3
"""Build a synthetic (I, q, y) training dataset from 3D meshes (handout §2.3, M1->M2).

For each rendered view it saves a **clean** RGB image (no burned-in marker), the
ground-truth label map, and a list of pin coordinates with their labels. The pin
is stored as a *coordinate* q = (x, y) -- the form both experts consume -- not as
a drawn arrow (see the marking rationale: arrows are for the frozen-VLM path /
parsing real exams, and are added only as optional DR augmentation, never baked
into the training signal).

Run in the render venv (a window flashes per scene)::

    python scripts/build_triples.py --mesh-dir .cache/bodyparts3d/meshes \
        --out data/synth_upperlimb --n-scenes 6 --preview      # small test first
    python scripts/build_triples.py --mesh-dir .cache/bodyparts3d/meshes \
        --out data/synth_upperlimb --n-scenes 400               # then scale up

Layout:
    out/vocab.json                {label_int: structure_name}
    out/manifest.json             [{id, rgb, idmap, pins:[{x,y,label,name}]}, ...]
    out/scenes/0000_rgb.png       clean render
    out/scenes/0000_idmap.png     8-bit label map (lossless)
    out/previews/0000_marked.png  (with --preview) arrow + answer, for eyeballing
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scalpel.config import SynthCfg  # noqa: E402
from scalpel.llm_reasoner import mark_pin  # noqa: E402
from scalpel.synth import DomainRandomizer, SyntheticRenderer, sample_triples  # noqa: E402
from demo_render import find_meshes, stage_meshes  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a synthetic (I, q, y) dataset")
    ap.add_argument("--mesh-dir", required=True)
    ap.add_argument("--out", default="data/synth_upperlimb")
    ap.add_argument("--n-scenes", type=int, default=6)
    ap.add_argument("--pins", type=int, default=6, help="pin samples per scene")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--max-meshes", type=int, default=24)
    ap.add_argument("--cadaveric-prob", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", action="store_true", help="save arrow-marked previews")
    ap.add_argument("--no-window", action="store_true", help="try offscreen (no window flashing)")
    args = ap.parse_args()

    paths = find_meshes(args.mesh_dir)
    if not paths:
        raise SystemExit(f"no meshes under {args.mesh_dir}")
    staging = os.path.join(args.out, "_staged")
    label_map = stage_meshes(paths, staging, args.max_meshes)
    # readable text labels: underscores -> spaces, strip trailing mesh-id digit cruft
    names = {v: re.sub(r"(\s\d+)+$", "", k.replace("_", " ")).strip()
             for k, v in label_map.items()}
    print(f"[vocab] {len(label_map)} structures")

    cfg = SynthCfg(image_size=args.image_size, cadaveric_prob=args.cadaveric_prob,
                   erosion_iters=4, min_pin_area=200, window_visible=not args.no_window)
    rng = np.random.default_rng(args.seed)
    dr = DomainRandomizer(cfg, rng)
    renderer = SyntheticRenderer(staging, label_map, cfg)

    scenes = os.path.join(args.out, "scenes")
    os.makedirs(scenes, exist_ok=True)
    if args.preview:
        os.makedirs(os.path.join(args.out, "previews"), exist_ok=True)

    manifest, total_pins = [], 0
    for i in range(args.n_scenes):
        rgb, idmap = renderer.render_pair(dr, rng)
        triples = sample_triples(rgb, idmap, cfg, rng, n_per_image=args.pins)
        if not triples:
            print(f"  scene {i}: no pinnable structure in view, skipped")
            continue
        rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        rgb_name, id_name = f"scenes/{i:04d}_rgb.png", f"scenes/{i:04d}_idmap.png"
        Image.fromarray(rgb_u8).save(os.path.join(args.out, rgb_name))
        Image.fromarray(idmap.astype(np.uint8), mode="L").save(os.path.join(args.out, id_name))

        pins = [{"x": int(x), "y": int(y), "label": int(lab), "name": names.get(int(lab), "?")}
                for (_rgb, (x, y), lab) in triples]
        manifest.append({"id": i, "rgb": rgb_name, "idmap": id_name, "pins": pins})
        total_pins += len(pins)

        if args.preview and i < 6:
            x, y, lab = pins[0]["x"], pins[0]["y"], pins[0]["label"]
            pv = mark_pin(Image.fromarray(rgb_u8), (x, y), r=16)
            pv.save(os.path.join(args.out, f"previews/{i:04d}_marked_{names.get(lab,'?')}.png"))
        print(f"  scene {i}: {len(pins)} pins")

    json.dump(names, open(os.path.join(args.out, "vocab.json"), "w"), indent=2)
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=2)
    print(f"\nDONE: {len(manifest)} scenes, {total_pins} (I,q,y) triples, "
          f"vocab {len(label_map)} -> {args.out}")
    if args.preview:
        print(f"eyeball: {args.out}/previews/  (arrow-marked, for inspection only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
