#!/usr/bin/env python3
"""SCALPEL M1 pre-validation: render one synthetic ``(I, q, y)`` triple and look at it.

This is the demo that de-risks milestone M1 *before* handing it to the agent: it
drives the **real** ``scalpel.synth`` code path end-to-end (the same renderer,
codec and pin sampler the pipeline will use at M1 - no demo-only rendering
logic) and answers two questions by eye and by assertion:

1. Does a labelled mesh actually render into an ``(rgb, idmap)`` pair, and how
   "rough"/clean does the synthetic image look?  -> the 3-panel PNG.
2. Did the ID pass decode cleanly (no anti-aliased garbage labels, §8.2) and is
   the pin a valid in-structure triple (§8.1)?  -> the decode-sanity report.

Output: a 3-panel PNG ``RGB | idmap(colourised) | pin overlay`` plus a stdout
sanity report. Exit code is non-zero if the ID pass produced foreign labels
(the canonical §8.2 failure: MSAA left on).

Requirements (handout §9 + project memory): run this in a **Python 3.11 venv**
with Open3D installed - Open3D's Apple-Silicon wheels lag Python releases, so
the 3.14 main env won't have it::

    python3.11 -m venv .venv-render && source .venv-render/bin/activate
    pip install open3d numpy scipy pillow
    python scripts/demo_render.py --mesh-dir path/to/meshes --out outputs/demo.png
    # or let it fetch BodyParts3D itself:
    python scripts/demo_render.py --download --max-meshes 24 --out outputs/demo.png

Mesh sources (CC BY-SA, attribution required):
* BodyParts3D GitHub mirror - https://github.com/Kevin-Mattheus-Moerman/BodyParts3D
* DBCLS OBJ archive (isa_BP3D_4.0_obj_99.zip, FMA-named OBJs) -
  ftp://ftp.biosciencedbc.jp/archive/bodyparts3d/LATEST/isa_BP3D_4.0_obj_99.zip
"""

from __future__ import annotations

import argparse
import colorsys
import glob
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile

import numpy as np
from PIL import Image, ImageDraw

# make `scalpel` importable when run from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalpel.config import SynthCfg  # noqa: E402
from scalpel.llm_reasoner import mark_pin  # noqa: E402
from scalpel.synth import (  # noqa: E402  (reuse the real code path)
    DomainRandomizer,
    SyntheticRenderer,
    sample_triples,
)

MESH_EXTS = (".obj", ".ply", ".stl", ".off", ".gltf", ".glb")
# Default download source: the BodyParts3D GitHub mirror (reliable HTTPS). The
# DBCLS OBJ archive below is the canonical FMA-named alternative (large, FTP).
GITHUB_MIRROR_URL = "https://github.com/Kevin-Mattheus-Moerman/BodyParts3D/archive/refs/heads/master.zip"
DBCLS_OBJ_URL = "ftp://ftp.biosciencedbc.jp/archive/bodyparts3d/LATEST/isa_BP3D_4.0_obj_99.zip"


# --------------------------------------------------------------------------- #
# Mesh acquisition                                                            #
# --------------------------------------------------------------------------- #
def download_and_extract(url: str, cache_dir: str) -> str:
    """Download an archive and extract it; return the extracted root dir."""
    os.makedirs(cache_dir, exist_ok=True)
    archive = os.path.join(cache_dir, os.path.basename(url) or "meshes.zip")
    if not os.path.exists(archive):
        print(f"[download] {url}\n           -> {archive}  (may be large; one-time)")
        urllib.request.urlretrieve(url, archive)  # noqa: S310 (trusted public source)
    extract_dir = os.path.join(cache_dir, "extracted")
    if not os.path.isdir(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as z:
                z.extractall(extract_dir)
        elif tarfile.is_tarfile(archive):
            with tarfile.open(archive) as t:
                try:
                    t.extractall(extract_dir, filter="data")  # py3.12+ safe extract
                except TypeError:
                    t.extractall(extract_dir)                  # py3.11 fallback
        else:
            raise RuntimeError(f"unknown archive format: {archive}")
    return extract_dir


def find_meshes(root: str) -> list[str]:
    """Recursively collect mesh files under ``root`` (layout-agnostic)."""
    found: list[str] = []
    for ext in MESH_EXTS:
        found += glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True)
    return sorted(set(found))


def stage_meshes(paths: list[str], staging_dir: str, max_meshes: int) -> dict[str, int]:
    """Flatten chosen meshes into one dir and return ``{stem: label}`` (1..N).

    ``SyntheticRenderer._load_meshes`` does a *flat* ``os.listdir`` and matches
    file stems against ``label_map``, so we stage symlinks (or copies) into a
    single directory and assign small sequential labels - which the 24-bit codec
    encodes/decodes cleanly.
    """
    if os.path.isdir(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)

    label_map: dict[str, int] = {}
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem in label_map:
            continue
        dst = os.path.join(staging_dir, os.path.basename(p))
        try:
            os.symlink(os.path.abspath(p), dst)
        except OSError:
            shutil.copy(p, dst)
        label_map[stem] = len(label_map) + 1
        if len(label_map) >= max_meshes:
            break
    return label_map


def resolve_meshes(args) -> tuple[str, dict[str, int]]:
    if args.mesh_dir:
        source_root = args.mesh_dir
        if not os.path.isdir(source_root):
            raise SystemExit(f"--mesh-dir not found: {source_root}")
    elif args.download:
        source_root = download_and_extract(args.url, args.cache)
    else:
        raise SystemExit(
            "no meshes: pass --mesh-dir PATH, or --download to fetch BodyParts3D."
        )
    paths = find_meshes(source_root)
    if not paths:
        raise SystemExit(f"no mesh files ({', '.join(MESH_EXTS)}) under {source_root!r}")
    label_map = stage_meshes(paths, os.path.join(args.cache, "_staged"), args.max_meshes)
    print(f"[meshes] {len(paths)} found, using {len(label_map)} (cap={args.max_meshes})")
    return os.path.join(args.cache, "_staged"), label_map


# --------------------------------------------------------------------------- #
# Visualisation                                                               #
# --------------------------------------------------------------------------- #
def colorize_labelmap(idmap: np.ndarray) -> np.ndarray:
    """Map an integer label map to a distinct-colour RGB image (bg=black)."""
    out = np.zeros((*idmap.shape, 3), dtype=np.uint8)
    labels = [int(v) for v in np.unique(idmap) if v != 0]
    for k, lab in enumerate(sorted(labels)):
        h = (k * 0.61803398875) % 1.0           # golden-ratio hue spacing
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 1.0)
        out[idmap == lab] = (int(r * 255), int(g * 255), int(b * 255))
    return out


def labeled_panel(arr_u8: np.ndarray, caption: str, bar: int = 22) -> Image.Image:
    img = Image.fromarray(arr_u8)
    canvas = Image.new("RGB", (img.width, img.height + bar), (20, 20, 20))
    canvas.paste(img, (0, bar))
    ImageDraw.Draw(canvas).text((6, 5), caption, fill=(240, 240, 240))
    return canvas


def hconcat(panels: list[Image.Image], gap: int = 8) -> Image.Image:
    h = max(p.height for p in panels)
    w = sum(p.width for p in panels) + gap * (len(panels) - 1)
    canvas = Image.new("RGB", (w, h), (20, 20, 20))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + gap
    return canvas


# --------------------------------------------------------------------------- #
# Decode sanity (handout §8.2, §8.1)                                          #
# --------------------------------------------------------------------------- #
def decode_sanity(idmap: np.ndarray, label_map: dict[str, int],
                  pin, pin_label, names: dict[int, str]) -> bool:
    decoded = {int(v) for v in np.unique(idmap)} - {0}
    valid = set(label_map.values())
    foreign = sorted(decoded - valid)     # ids in image but not in label_map -> DECODE BUG
    missing = sorted(valid - decoded)     # labels not visible -> occlusion (benign)

    print("\n=== decode sanity (ID pass, §8.2) ===")
    print(f"  structures in label_map : {len(valid)}")
    print(f"  distinct ids decoded    : {len(decoded)}  "
          f"(coverage {len(decoded)}/{len(valid)}; missing = occluded, benign)")
    ok = True
    if foreign:
        ok = False
        print(f"  !! FOREIGN ids (not in label_map): {foreign[:10]}"
              f"{' ...' if len(foreign) > 10 else ''}")
        print("     -> ID pass is not clean. Likely MSAA/post-processing left ON (§8.2).")
    else:
        print("  no foreign ids  -> ID pass decodes cleanly (MSAA off, §8.2) OK")

    print("\n=== pin triple (eroded interior, §8.1) ===")
    if pin is None:
        ok = False
        print("  !! no triple sampled (no structure large enough to pin)")
    else:
        x, y = pin
        decoded_at_pin = int(idmap[y, x])
        match = decoded_at_pin == int(pin_label)
        ok = ok and match
        print(f"  pin (x,y)            : ({x}, {y})")
        print(f"  decoded label at pin : {decoded_at_pin} = {names.get(decoded_at_pin, '?')}")
        print(f"  sampler's label      : {int(pin_label)} = {names.get(int(pin_label), '?')}")
        print(f"  match                : {'OK' if match else '!! MISMATCH'}")
    return ok


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="SCALPEL M1 pre-validation render demo")
    src = ap.add_argument_group("mesh source")
    src.add_argument("--mesh-dir", help="directory of mesh files (used directly if given)")
    src.add_argument("--download", action="store_true", help="fetch BodyParts3D if no --mesh-dir")
    src.add_argument("--url", default=GITHUB_MIRROR_URL,
                     help=f"download URL (default: GitHub mirror; DBCLS OBJ: {DBCLS_OBJ_URL})")
    src.add_argument("--cache", default=".cache/bodyparts3d", help="download/staging cache dir")
    src.add_argument("--max-meshes", type=int, default=24, help="cap structures for a fast demo")

    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/demo_render.png")
    args = ap.parse_args()

    staging_dir, label_map = resolve_meshes(args)
    names = {v: k for k, v in label_map.items()}

    cfg = SynthCfg(image_size=args.image_size)
    rng = np.random.default_rng(args.seed)
    dr = DomainRandomizer(cfg, rng)
    renderer = SyntheticRenderer(staging_dir, label_map, cfg)

    print(f"[render] {args.image_size}x{args.image_size}, seed={args.seed} ...")
    try:
        rgb, idmap = renderer.render_pair(dr, rng)            # the real M1 code path
    except RuntimeError as e:
        print(f"\n[render FAILED] {e}", file=sys.stderr)
        print("If this is an Open3D import error, you're likely not in the 3.11 "
              "render venv (see this file's header / project memory).", file=sys.stderr)
        return 1

    # sample exactly one in-structure pin triple (§8.1, via the real sampler)
    triples = sample_triples(rgb, idmap, cfg, rng, n_per_image=1)
    pin, pin_label = (triples[0][1], triples[0][2]) if triples else (None, None)

    # 3-panel figure
    rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    idviz = colorize_labelmap(idmap)
    pin_src = mark_pin(Image.fromarray(rgb_u8), pin) if pin else Image.fromarray(rgb_u8)
    panels = [
        labeled_panel(rgb_u8, "RGB  (pass 1: lit + domain-rand)"),
        labeled_panel(idviz, "idmap  (pass 2: unlit ID, colourised)"),
        labeled_panel(np.asarray(pin_src), "pin @ eroded interior  (I, q, y)"),
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    hconcat(panels).save(args.out)
    print(f"[saved] {args.out}")

    ok = decode_sanity(idmap, label_map, pin, pin_label, names)
    print(f"\nRESULT: {'OK - M1 render path validated' if ok else 'PROBLEMS (see above)'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
