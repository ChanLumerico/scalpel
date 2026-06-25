"""Scene graph over a dense label map (handout §2.4, §4.4).

Turns a per-pixel label map (from the segmenter, or a synthetic ID pass) into a
typed graph of anatomical regions. Nodes are regions; directed, typed edges
encode their spatial relations. This is pure numpy/scipy - no torch, no
external assets - so it runs in the smoke path unchanged.

Relation vocabulary (``n_relations = 6``). Every edge ``(i -> j, rel)`` reads
**"i is [rel] j"** and that direction is kept *consistent* everywhere
(handout §4.4, §8.3 - flipping a sign makes central structures serialize as
"lateral", which is the canonical bug)::

    0 adjacent       i and j touch (overlap after dilation)
    1 surrounds      i encloses j
    2 surrounded_by  i is enclosed by j
    3 medial         i is closer to the midline than j
    4 lateral        i is farther from the midline than j
    5 deep           i is deeper than j (synthetic depth, else centrality)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from .config import GraphCfg

# relation ids ------------------------------------------------------------- #
REL_ADJACENT = 0
REL_SURROUNDS = 1
REL_SURROUNDED_BY = 2
REL_MEDIAL = 3
REL_LATERAL = 4
REL_DEEP = 5
RELATION_NAMES = [
    "adjacent",
    "surrounds",
    "surrounded_by",
    "medial",
    "lateral",
    "deep",
]


# --------------------------------------------------------------------------- #
# Data structures                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class Region:
    label: int
    mask: np.ndarray  # bool (H, W)
    area: int
    centroid: tuple[float, float]  # (x, y)
    eccentricity: float
    solidity: float
    depth: float = 0.0  # mean depth if a depth map was supplied


@dataclass
class SceneGraph:
    regions: list[Region]
    edges: list[list[tuple[int, int]]] = field(
        default_factory=list
    )  # edges[i] = [(j, rel), ...]

    def neighbors(self, i: int) -> list[tuple[int, int]]:
        return self.edges[i]

    def __len__(self) -> int:
        return len(self.regions)

    @property
    def labels(self) -> list[int]:
        return [r.label for r in self.regions]


# --------------------------------------------------------------------------- #
# Region property extraction                                                  #
# --------------------------------------------------------------------------- #
def _region_props(mask: np.ndarray) -> tuple[int, tuple[float, float], float, float]:
    """Return ``(area, (cx, cy), eccentricity, solidity)`` for a binary mask."""
    ys, xs = np.nonzero(mask)
    area = int(xs.size)
    cx, cy = float(xs.mean()), float(ys.mean())

    # eccentricity from the second central moments (covariance eigenvalues)
    if area >= 2:
        mxx = float(((xs - cx) ** 2).mean())
        myy = float(((ys - cy) ** 2).mean())
        mxy = float(((xs - cx) * (ys - cy)).mean())
        cov = np.array([[mxx, mxy], [mxy, myy]])
        evals = np.linalg.eigvalsh(cov)  # ascending: [l2, l1]
        l2, l1 = float(evals[0]), float(evals[1])
        ecc = float(np.sqrt(max(0.0, 1.0 - l2 / l1))) if l1 > 1e-8 else 0.0
    else:
        ecc = 0.0

    # solidity = area / convex-hull area (1.0 for degenerate / convex shapes).
    # Collinear specks make Qhull raise QhullError, and other degeneracies can
    # surface as runtime errors; treat any of them as "convex enough" (1.0).
    solidity = 1.0
    if area >= 4:
        try:
            from scipy.spatial import ConvexHull

            pts = np.stack([xs, ys], axis=1).astype(np.float64)
            hull_area = float(ConvexHull(pts).volume)  # 2-D: .volume == area
            if hull_area > 1e-8:
                solidity = float(min(1.0, area / hull_area))
        except Exception:
            solidity = 1.0
    return area, (cx, cy), ecc, solidity


def _extract_regions(
    label_map: np.ndarray, cfg: GraphCfg, depth_map: np.ndarray | None
) -> list[Region]:
    regions: list[Region] = []
    for lab in np.unique(label_map):
        if lab == cfg.background_label:
            continue
        mask = label_map == lab
        area = int(mask.sum())
        if area < cfg.min_region_area:
            continue
        a, c, ecc, sol = _region_props(mask)
        depth = float(depth_map[mask].mean()) if depth_map is not None else 0.0
        regions.append(Region(int(lab), mask, a, c, ecc, sol, depth))
    return regions


# --------------------------------------------------------------------------- #
# Pairwise relations                                                          #
# --------------------------------------------------------------------------- #
def _surrounds(mask_i: np.ndarray, mask_j: np.ndarray, frac: float = 0.5) -> bool:
    """True if region ``i`` encloses region ``j`` (j sits in i's holes)."""
    filled_i = ndi.binary_fill_holes(mask_i)
    holes_i = filled_i & ~mask_i
    if not holes_i.any():
        return False
    inside = float((mask_j & holes_i).sum())
    return inside / max(1.0, float(mask_j.sum())) > frac


def build_scene_graph(
    label_map: np.ndarray, cfg: GraphCfg, depth_map: np.ndarray | None = None
) -> SceneGraph:
    """Build a typed scene graph from a dense integer ``label_map`` (H, W).

    Edges are emitted only between *neighbouring* regions (adjacent after
    dilation, or in a surround relationship). For each such pair every
    applicable typed edge is added, always in the "i is [rel] j" direction.
    """
    label_map = np.asarray(label_map)
    regions = _extract_regions(label_map, cfg, depth_map)
    n = len(regions)
    edges: list[list[tuple[int, int]]] = [[] for _ in range(n)]

    if n == 0:
        return SceneGraph(regions, edges)

    H, W = label_map.shape
    axis_x = cfg.medial_axis_x if cfg.medial_axis_x is not None else W / 2.0
    center = np.array([W / 2.0, H / 2.0])

    # precompute dilated masks + bounding boxes once per region
    struct = ndi.generate_binary_structure(2, 2)
    dil = [
        ndi.binary_dilation(r.mask, struct, iterations=cfg.dilate_iters)
        for r in regions
    ]
    bbox = [_bbox(r.mask) for r in regions]

    def add(i: int, j: int, rel: int) -> None:
        if (j, rel) not in edges[i]:
            edges[i].append((j, rel))

    for i in range(n):
        for j in range(i + 1, n):
            if not _bbox_overlap(bbox[i], bbox[j], pad=cfg.dilate_iters + 1):
                continue
            surr_ij = _surrounds(regions[i].mask, regions[j].mask)
            surr_ji = _surrounds(regions[j].mask, regions[i].mask)
            touch = bool(
                (dil[i] & regions[j].mask).any() or (dil[j] & regions[i].mask).any()
            )
            if not (touch or surr_ij or surr_ji):
                continue

            # 0 adjacent (symmetric)
            add(i, j, REL_ADJACENT)
            add(j, i, REL_ADJACENT)

            # 1/2 surrounds / surrounded_by
            if surr_ij:
                add(i, j, REL_SURROUNDS)
                add(j, i, REL_SURROUNDED_BY)
            if surr_ji:
                add(j, i, REL_SURROUNDS)
                add(i, j, REL_SURROUNDED_BY)

            # 3/4 medial / lateral (by distance of centroid-x to the midline)
            di = abs(regions[i].centroid[0] - axis_x)
            dj = abs(regions[j].centroid[0] - axis_x)
            if di < dj:
                add(i, j, REL_MEDIAL)
                add(j, i, REL_LATERAL)
            elif dj < di:
                add(j, i, REL_MEDIAL)
                add(i, j, REL_LATERAL)

            # 5 deep (depth if available, else centrality: nearer image center = deeper)
            if depth_map is not None:
                deeper = i if regions[i].depth > regions[j].depth else j
                shallower = j if deeper == i else i
                if regions[i].depth != regions[j].depth:
                    add(deeper, shallower, REL_DEEP)
            else:
                ci = float(np.linalg.norm(np.array(regions[i].centroid) - center))
                cj = float(np.linalg.norm(np.array(regions[j].centroid) - center))
                if ci < cj:
                    add(i, j, REL_DEEP)
                elif cj < ci:
                    add(j, i, REL_DEEP)

    return SceneGraph(regions, edges)


def pin_region_index(graph: SceneGraph, q: tuple[float, float]) -> int | None:
    """Index of the region whose mask contains pixel ``q = (x, y)`` (or None)."""
    x, y = int(round(q[0])), int(round(q[1]))
    for i, r in enumerate(graph.regions):
        H, W = r.mask.shape
        if 0 <= y < H and 0 <= x < W and r.mask[y, x]:
            return i
    return None


# --------------------------------------------------------------------------- #
# small geometry helpers                                                       #
# --------------------------------------------------------------------------- #
def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _bbox_overlap(a, b, pad: int = 0) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (
        ax1 + pad < bx0 or bx1 + pad < ax0 or ay1 + pad < by0 or by1 + pad < ay0
    )
