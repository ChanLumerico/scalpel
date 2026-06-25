#!/usr/bin/env python3
"""Probe whether Open3D's *legacy* Visualizer can render on this macOS.

The Filament OffscreenRenderer fails here with "EGL Headless is not supported".
The legacy Visualizer uses a native GL window + screen capture, which works on
macOS. This opens a small window briefly, captures it, and saves outputs/probe.png.

Run inside the render venv:  python scripts/_probe_render.py
"""
import glob
import os
import sys

import numpy as np
import open3d as o3d
from PIL import Image

meshdir = sys.argv[1] if len(sys.argv) > 1 else ".cache/bodyparts3d/meshes"
files = sorted(glob.glob(os.path.join(meshdir, "*.stl")))[:6]
if not files:
    raise SystemExit(f"no STL meshes found in {meshdir}")

meshes = []
for f in files:
    m = o3d.io.read_triangle_mesh(f)
    m.compute_vertex_normals()
    m.paint_uniform_color([0.82, 0.56, 0.50])   # fleshy
    meshes.append(m)

os.makedirs("outputs", exist_ok=True)
vis = o3d.visualization.Visualizer()
vis.create_window(width=640, height=640, visible=True)   # real window (macOS-friendly)
for m in meshes:
    vis.add_geometry(m)
opt = vis.get_render_option()
opt.background_color = np.array([0.04, 0.05, 0.08])
opt.light_on = True
vis.poll_events()
vis.update_renderer()
buf = vis.capture_screen_float_buffer(do_render=True)
vis.destroy_window()

img = (np.clip(np.asarray(buf), 0, 1) * 255).astype(np.uint8)
Image.fromarray(img).save("outputs/probe.png")
nonblack = int((img.sum(-1) > 25).sum())
print(f"OK: saved outputs/probe.png  shape={img.shape}  non-background px={nonblack}")
print("=> RENDER WORKS" if nonblack > 5000 else "=> WARNING: image is blank (capture failed)")
