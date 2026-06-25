"""Synthetic data generation (handout §2.3, §4.2).

Render labelled 3-D meshes into ``(I, q, y)`` triples: the RGB pass gives the
image, the ID pass gives a per-pixel label map for free, and pins are sampled
inside structures. The sim-to-real gap is closed by *domain randomization*, not
photorealism (handout §2.3, §8.5).

Only :class:`SyntheticRenderer` needs Open3D + meshes; it is imported lazily so
the rest of this module (label codec, :class:`DomainRandomizer`,
:func:`sample_triples`) is pure numpy and runs in the smoke path (handout §8.8).

Two hard-won conventions:
* ID pass is **unlit flat with MSAA off** so label colours are never
  interpolated and decode cleanly (handout §8.2).
* Pins are sampled from the **eroded interior** of a mask, never its
  anti-aliased boundary, where the label is ambiguous (handout §8.1).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .config import SynthCfg


# --------------------------------------------------------------------------- #
# Label <-> RGB codec (24-bit)                                                #
# --------------------------------------------------------------------------- #
def id_to_rgb(i: int) -> tuple[float, float, float]:
    """Integer label -> 24-bit RGB in [0, 1] (little-endian over R, G, B)."""
    i = int(i)
    r = (i & 0xFF) / 255.0
    g = ((i >> 8) & 0xFF) / 255.0
    b = ((i >> 16) & 0xFF) / 255.0
    return r, g, b


def rgb_to_id(arr: np.ndarray) -> np.ndarray:
    """``(H, W, 3) uint8`` -> ``(H, W) int`` label map (inverse of :func:`id_to_rgb`)."""
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = (
            np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
            if arr.max() <= 1.0
            else arr.astype(np.uint8)
        )
    r = arr[..., 0].astype(np.int64)
    g = arr[..., 1].astype(np.int64)
    b = arr[..., 2].astype(np.int64)
    return r + (g << 8) + (b << 16)


TISSUE_RGB = {
    "nerve": (0.94, 0.90, 0.62),   # pale yellow-cream
    "artery": (0.80, 0.12, 0.11),  # red
    "vein": (0.28, 0.32, 0.58),    # bluish
    "bone": (0.92, 0.89, 0.80),    # pale cream
    "muscle": (0.66, 0.30, 0.28),  # red-brown
}


def tissue_of(name: str) -> str:
    """Classify a structure by name -> nerve / artery / vein / bone / muscle."""
    n = name.lower()
    if any(k in n for k in ("nerve", "plexus", "cord", "trunk", "cutaneous", "ganglion")):
        return "nerve"
    if any(k in n for k in ("arter", "aorta", "circumflex", "collateral", "recurrent")):
        return "artery"
    if any(k in n for k in ("vein", "basilic", "cephalic", "venous", "vena")):
        return "vein"
    if any(k in n for k in ("humer", "radius", "ulna", "clavicle", "scapula", "carpal",
                            "metacarp", "phalan", "sternum", "rib", "vertebra", "bone")):
        return "bone"
    return "muscle"


def tissue_color(name: str) -> tuple[float, float, float]:
    """Default dissection-readable RGB base colour for a structure (by tissue)."""
    return TISSUE_RGB[tissue_of(name)]


# --------------------------------------------------------------------------- #
# Domain randomization (xi)                                                   #
# --------------------------------------------------------------------------- #
class DomainRandomizer:
    """Samples and applies the randomization parameters xi (handout §2.3)."""

    def __init__(self, cfg: SynthCfg, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()

    def sample_light(self) -> float:
        lo, hi = self.cfg.light_range
        return float(self.rng.uniform(lo, hi))

    def perturb_color(self, base_rgb) -> np.ndarray:
        base = np.asarray(base_rgb, dtype=np.float64)
        j = self.cfg.color_jitter
        factor = self.rng.uniform(1.0 - j, 1.0 + j, size=base.shape)
        return np.clip(base * factor, 0.0, 1.0)

    def cadaverize(self, img: np.ndarray) -> np.ndarray:
        """Shift colours toward a pale, desaturated formalin-cadaver look (§2.3, §8.5).

        The evaluation target (formalin-fixed cadaver) is pale tan and very
        low-saturation — unlike bright mesh renders. This DR op spans that gap so
        synthetic training covers the real exam domain rather than chasing
        photorealism.
        """
        out = np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0)
        lum = (out * np.array([0.299, 0.587, 0.114])).sum(-1, keepdims=True)
        s = float(self.rng.uniform(0.12, 0.32))      # low residual saturation
        desat = lum * (1 - s) + out * s
        tint = np.array([0.95, 0.83, 0.66])          # warm tan / beige
        pale = (0.40 + 0.55 * lum) * tint            # pale, low-contrast base
        return np.clip(0.5 * desat + 0.5 * pale, 0.0, 1.0)

    def corrupt(self, img: np.ndarray) -> np.ndarray:
        """Apply noise / blur / occlusion to an RGB float image in [0, 1]."""
        out = np.asarray(img, dtype=np.float64).copy()
        H, W = out.shape[:2]

        # gaussian sensor noise
        if self.cfg.gauss_noise_std > 0:
            out = out + self.rng.normal(0.0, self.cfg.gauss_noise_std, size=out.shape)

        # defocus blur
        if self.cfg.blur_sigma_max > 0:
            sigma = float(self.rng.uniform(0.0, self.cfg.blur_sigma_max))
            if sigma > 1e-3:
                for c in range(out.shape[2]):
                    out[..., c] = ndi.gaussian_filter(out[..., c], sigma=sigma)

        # rectangular occluder
        if self.rng.random() < self.cfg.occlusion_prob:
            frac = float(self.rng.uniform(0.02, self.cfg.occlusion_max_frac))
            ow = max(1, int(W * np.sqrt(frac)))
            oh = max(1, int(H * np.sqrt(frac)))
            ox = int(self.rng.integers(0, max(1, W - ow)))
            oy = int(self.rng.integers(0, max(1, H - oh)))
            # darken the patch (a soft shadow/occluder) rather than paint a solid block
            out[oy : oy + oh, ox : ox + ow, :] *= float(self.rng.uniform(0.1, 0.45))

        return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Pin sampling                                                                #
# --------------------------------------------------------------------------- #
def sample_triples(
    rgb: np.ndarray,
    idmap: np.ndarray,
    cfg: SynthCfg,
    rng: np.random.Generator | None = None,
    n_per_image: int = 8,
) -> list[tuple[np.ndarray, tuple[int, int], int]]:
    """Sample ``(rgb, (x, y), label)`` triples with pins in eroded interiors.

    Each pin lands strictly inside a structure (mask eroded by
    ``cfg.erosion_iters``) so the boundary anti-aliasing never makes its label
    ambiguous (handout §8.1).
    """
    rng = rng or np.random.default_rng()
    idmap = np.asarray(idmap)
    struct = ndi.generate_binary_structure(2, 2)

    # collect, per label, the set of safe interior pixels
    interiors: dict[int, np.ndarray] = {}
    for lab in np.unique(idmap):
        if lab == 0:
            continue
        mask = idmap == lab
        if mask.sum() < cfg.min_pin_area:
            continue
        eroded = ndi.binary_erosion(mask, struct, iterations=cfg.erosion_iters)
        ys, xs = np.nonzero(eroded)
        if xs.size == 0:
            continue
        interiors[int(lab)] = np.stack([xs, ys], axis=1)  # (n, 2) as (x, y)

    triples: list[tuple[np.ndarray, tuple[int, int], int]] = []
    if not interiors:
        return triples

    labels = list(interiors.keys())
    for _ in range(n_per_image):
        lab = labels[int(rng.integers(0, len(labels)))]
        pts = interiors[lab]
        x, y = pts[int(rng.integers(0, pts.shape[0]))]
        triples.append((rgb, (int(x), int(y)), int(lab)))
    return triples


# --------------------------------------------------------------------------- #
# Renderer (Open3D - lazy; validated at M1)                                   #
# --------------------------------------------------------------------------- #
class SyntheticRenderer:
    """Two-pass offscreen renderer producing ``(rgb, idmap)`` (handout §4.2).

    Open3D is imported lazily inside :meth:`_ensure_open3d` so that importing
    :mod:`scalpel.synth` never drags Open3D in; this class is *not* touched on
    the smoke path. Full validation happens at milestone M1, once meshes are
    available.
    """

    def __init__(self, mesh_dir: str, label_map: dict[str, int], cfg: SynthCfg):
        self.mesh_dir = mesh_dir
        self.label_map = label_map  # mesh filename stem -> integer label
        self.cfg = cfg
        self._o3d = None
        self._meshes: list[tuple[int, object]] = []  # (label, geometry)

    def _ensure_open3d(self):
        if self._o3d is None:
            try:
                import open3d as o3d
            except Exception as e:  # pragma: no cover - exercised only at M1
                raise RuntimeError(
                    "SyntheticRenderer requires Open3D. Install it with "
                    "`pip install open3d` (handout §9). This dependency is "
                    "intentionally excluded from the smoke path (§8.8)."
                ) from e
            self._o3d = o3d
        return self._o3d

    def _load_meshes(self):
        import os

        o3d = self._ensure_open3d()
        if self._meshes:
            return self._meshes
        self._names = {}
        for fname in sorted(os.listdir(self.mesh_dir)):
            stem = os.path.splitext(fname)[0]
            if stem not in self.label_map:
                continue
            mesh = o3d.io.read_triangle_mesh(os.path.join(self.mesh_dir, fname))
            mesh.compute_vertex_normals()
            lab = self.label_map[stem]
            self._meshes.append((lab, mesh))
            self._names[lab] = stem
        if not self._meshes:
            raise RuntimeError(f"no labelled meshes found under {self.mesh_dir!r}")
        return self._meshes

    @staticmethod
    def _distinct_colors(n: int) -> np.ndarray:
        """``n`` bright, well-separated RGB colours for the ID pass (decode-robust)."""
        import colorsys
        return np.array(
            [colorsys.hsv_to_rgb((i / max(1, n)) % 1.0, 0.92, 0.95) for i in range(n)]
        )

    @staticmethod
    def _fit(arr: np.ndarray, S: int, nearest: bool) -> np.ndarray:
        """Resize a capture to ``(S, S)`` (macOS Retina backing buffers are 2x)."""
        if arr.shape[0] == S and arr.shape[1] == S:
            return arr
        from PIL import Image
        if nearest:
            im = Image.fromarray(arr.astype(np.uint8)).resize((S, S), Image.NEAREST)
            return np.asarray(im)
        im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
        return np.asarray(im.resize((S, S), Image.BILINEAR)).astype(np.float64) / 255.0

    def render_pair(self, dr: DomainRandomizer, rng: np.random.Generator | None = None):
        """Render one ``(rgb float[0,1] HxWx3, idmap int HxW)`` pair.

        Uses Open3D's legacy ``Visualizer`` (native GL window + screen capture),
        not the Filament ``OffscreenRenderer`` -- the latter fails on macOS with
        "EGL Headless is not supported". A window briefly appears per render
        (``SynthCfg.window_visible``).

        The ID pass paints each structure a distinct bright colour and decodes by
        nearest-colour match, which is robust to the GL pipeline's gamma/sRGB and
        edge anti-aliasing (the §8.2 concern) without depending on an exact
        24-bit round-trip.
        """
        o3d = self._ensure_open3d()
        rng = rng or np.random.default_rng()
        meshes = self._load_meshes()
        S = self.cfg.image_size

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=S, height=S, visible=self.cfg.window_visible)
        try:
            # dissection depth: drop a random fraction of muscle layers so the deep
            # nerves/vessels are exposed (and pinnable) in some scenes, like a real
            # layered dissection. Nerves/vessels/bone are never dropped.
            drop_p = float(rng.uniform(0.0, self.cfg.muscle_drop_max))
            geoms = [(label, o3d.geometry.TriangleMesh(mesh)) for label, mesh in meshes
                     if not (tissue_of(self._names.get(label, "")) == "muscle"
                             and rng.random() < drop_p)]
            if not geoms:                                  # safety: never render empty
                geoms = [(label, o3d.geometry.TriangleMesh(mesh)) for label, mesh in meshes]
            # normalized copies (origin-centred, unit-scaled) so framing is reliable
            # regardless of the meshes' world coordinates.
            allpts = np.vstack([np.asarray(g.vertices) for _, g in geoms])
            center = allpts.mean(axis=0)
            scale = 1.0 / (float(np.linalg.norm(allpts.max(0) - allpts.min(0))) + 1e-9)
            for _label, g in geoms:
                g.translate(-center)
                g.scale(scale, center=(0.0, 0.0, 0.0))
                g.compute_vertex_normals()
                vis.add_geometry(g)
            opt = vis.get_render_option()
            opt.mesh_show_back_face = True

            # framing: centred on the body, zoom so the region + surrounding body
            # fill the frame (limited elevation -> natural front/side views, not
            # top-down). Real spot-exam photos are close-ups embedded in context.
            ctr = vis.get_view_control()
            ctr.set_zoom(float(rng.uniform(self.cfg.zoom_min, self.cfg.zoom_max)))
            ctr.rotate(float(rng.uniform(-180, 180)), float(rng.uniform(-35, 35)))

            # ----- pass 1: lit RGB, fleshy jittered colour per structure
            opt.light_on = True
            opt.background_color = np.array([0.05, 0.05, 0.07])
            for label, g in geoms:
                base = tissue_color(self._names.get(label, ""))
                g.paint_uniform_color(np.asarray(dr.perturb_color(base)))
                vis.update_geometry(g)
            vis.poll_events()
            vis.update_renderer()
            rgb = np.asarray(vis.capture_screen_float_buffer(do_render=True))

            # ----- pass 2: unlit flat distinct ID colours (light OFF -> flat)
            palette = self._distinct_colors(len(geoms))
            labels = [lab for lab, _ in geoms]
            opt.light_on = False
            opt.background_color = np.array([0.0, 0.0, 0.0])
            for (_label, g), col in zip(geoms, palette):
                g.paint_uniform_color(col)
                vis.update_geometry(g)
            vis.poll_events()
            vis.update_renderer()
            id_buf = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        finally:
            vis.destroy_window()

        rgb = self._fit(rgb, S, nearest=False)
        id_buf = self._fit(id_buf, S, nearest=False)

        # decode by nearest palette colour; far-from-any / near-black -> background
        flat = id_buf.reshape(-1, 3)
        idmap = np.zeros(flat.shape[0], dtype=np.int64)
        best = np.full(flat.shape[0], np.inf)
        for col, lab in zip(palette, labels):
            d = ((flat - col) ** 2).sum(axis=1)
            upd = d < best
            best[upd] = d[upd]
            idmap[upd] = lab
        idmap[(best > 0.15) | (flat.sum(axis=1) < 0.08)] = 0
        idmap = idmap.reshape(S, S)

        rgb = np.clip(rgb, 0.0, 1.0) ** 0.85               # mild brightness lift
        rgb = dr.corrupt(rgb)
        if rng.random() < self.cfg.cadaveric_prob:         # formalin look, TISSUE only
            fg = (idmap > 0)[..., None]                    # keep the background dark
            rgb = np.where(fg, dr.cadaverize(rgb), rgb)
        return rgb, idmap
