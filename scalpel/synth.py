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

FOV_DEG = 60.0  # vertical field of view (degrees) for the offscreen camera


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
            shade = self.rng.uniform(0.0, 1.0, size=(1, 1, out.shape[2]))
            out[oy : oy + oh, ox : ox + ow, :] = shade

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
        for fname in sorted(os.listdir(self.mesh_dir)):
            stem = os.path.splitext(fname)[0]
            if stem not in self.label_map:
                continue
            mesh = o3d.io.read_triangle_mesh(os.path.join(self.mesh_dir, fname))
            mesh.compute_vertex_normals()
            self._meshes.append((self.label_map[stem], mesh))
        if not self._meshes:
            raise RuntimeError(f"no labelled meshes found under {self.mesh_dir!r}")
        return self._meshes

    def _random_camera(self, renderer, rng):
        """Set a randomized camera; return ``(center, eye, up)`` to reuse for pass 2."""
        bounds = renderer.scene.bounding_box
        center = bounds.get_center()
        extent = float(np.linalg.norm(bounds.get_extent()))
        az = float(rng.uniform(0, 2 * np.pi))
        el = float(rng.uniform(-np.pi / 4, np.pi / 4))
        radius = extent * float(rng.uniform(1.2, 2.0))
        eye = center + radius * np.array(
            [np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)]
        )
        # camera roll via an up-vector tilt (handout's xi includes camera roll)
        roll = np.deg2rad(
            float(rng.uniform(-self.cfg.camera_roll_deg, self.cfg.camera_roll_deg))
        )
        up = np.array([np.sin(roll), np.cos(roll), 0.0])
        # setup_camera sets BOTH projection (fov) and view; look_at alone leaves
        # the projection uninitialised and renders an empty image.
        renderer.setup_camera(FOV_DEG, center, eye, up)
        return center, eye, up

    @staticmethod
    def _disable_aa(renderer) -> None:
        """Disable post-processing + FXAA + MSAA so ID colours decode cleanly (§8.2).

        Post-processing also applies tone-mapping that would shift the flat ID
        colours; turning it off is what keeps the label codec reversible.
        Guarded because the exact View API varies across Open3D versions.
        """
        try:
            view = renderer.scene.view
            view.set_post_processing(False)
            view.set_antialiasing(False)
            view.set_sample_count(1)
        except Exception:
            pass

    def render_pair(self, dr: DomainRandomizer, rng: np.random.Generator | None = None):
        """Render one ``(rgb float[0,1] HxWx3, idmap int HxW)`` pair."""
        o3d = self._ensure_open3d()
        rng = rng or np.random.default_rng()
        meshes = self._load_meshes()
        S = self.cfg.image_size
        rendering = o3d.visualization.rendering
        renderer = rendering.OffscreenRenderer(S, S)

        # ----- pass 1: lit RGB with randomized lighting + per-structure colour
        renderer.scene.set_background([0, 0, 0, 1])
        intensity = dr.sample_light()
        renderer.scene.scene.set_sun_light(
            [0.3, -0.7, -0.6], [1.0, 1.0, 1.0], 75000 * intensity
        )
        renderer.scene.scene.enable_sun_light(True)
        for label, mesh in meshes:
            mat = rendering.MaterialRecord()
            mat.shader = "defaultLit"
            base = dr.perturb_color([0.8, 0.55, 0.5])  # fleshy base, jittered
            mat.base_color = [*base, 1.0]
            renderer.scene.add_geometry(f"rgb_{label}", mesh, mat)
        cam = self._random_camera(renderer, rng)        # sample the view ONCE
        rgb = np.asarray(renderer.render_to_image(), dtype=np.float64) / 255.0
        rgb = dr.corrupt(rgb)
        if rng.random() < self.cfg.cadaveric_prob:      # span model -> formalin look
            rgb = dr.cadaverize(rgb)

        # ----- pass 2: unlit flat ID colours, AA + post-processing OFF (§8.2)
        renderer.scene.clear_geometry()
        self._disable_aa(renderer)
        for label, mesh in meshes:
            mat = rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            mat.base_color = [*id_to_rgb(label), 1.0]
            renderer.scene.add_geometry(f"id_{label}", mesh, mat)
        renderer.setup_camera(FOV_DEG, *cam)            # identical view to the RGB pass
        id_rgb = np.asarray(renderer.render_to_image(), dtype=np.uint8)
        idmap = rgb_to_id(id_rgb)
        return rgb, idmap
