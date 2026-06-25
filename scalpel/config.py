"""Configuration objects for SCALPEL.

Every magic number in the system lives here as a dataclass field (handout §4.1).
A single :class:`PipelineCfg` bundles the per-component configs and, in
``__post_init__``, propagates the cross-cutting quantities (``n_classes``,
``image_size``) down into the sub-configs so they can never drift apart.

``default_device()`` follows the handout's mps > cuda > cpu preference.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace


def default_device() -> str:
    """Return the preferred torch device string: ``mps`` > ``cuda`` > ``cpu``.

    Imported lazily so that merely importing :mod:`scalpel.config` does not pull
    in torch (keeps the lightweight import path cheap; handout §4.10 / §8.7).
    """
    try:
        import torch
    except Exception:  # pragma: no cover - torch always present in practice
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# --------------------------------------------------------------------------- #
# Per-component configs                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class BackboneCfg:
    """Frozen DINOv2 backbone (handout §2.2, §4.3)."""

    name: str = "dinov2_vitb14"  # torch.hub entry point
    repo: str = "facebookresearch/dinov2"
    patch_size: int = 14
    image_size: int = 518  # 518 / 14 == 37 patch grid
    embed_dim: int = 768  # ViT-B
    frozen: bool = True

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size


@dataclass
class PointCondCfg:
    """Point-conditioned pooler (handout §2.1, §4.3)."""

    mode: str = "attn"  # "attn" | "gauss"
    gauss_sigma_px: float = 40.0  # GaussianPool kernel width (pixels)
    attn_dim: int = 256  # PinCrossAttention internal width
    n_heads: int = 4
    n_pos_freqs: int = 6  # Fourier positional-encoding bands gamma(q)
    dropout: float = 0.0


@dataclass
class SegmenterCfg:
    """Dense patch segmenter probe (handout §4.3)."""

    hidden: int = 256
    n_classes: int = 50  # synced from PipelineCfg
    dropout: float = 0.1


@dataclass
class GraphCfg:
    """Scene-graph construction (handout §4.4)."""

    n_relations: int = 6
    dilate_iters: int = 2  # adjacency = overlap after dilation
    min_region_area: int = 16  # drop specks
    background_label: int = 0  # excluded from nodes
    medial_axis_x: float | None = None  # None -> use image mid-x at build time


@dataclass
class GNNCfg:
    """Relational expert / R-GCN (handout §2.9, §4.5)."""

    hidden: int = 128
    n_layers: int = 2
    geom_dim: int = 5  # [sqrt(area), cx, cy, ecc, solidity]
    n_relations: int = 6  # must match GraphCfg.n_relations
    n_classes: int = 50  # synced from PipelineCfg
    dropout: float = 0.1


@dataclass
class HeadCfg:
    """Prototypical head, calibration, PoE fusion + abstention (handout §2.5-2.7)."""

    n_classes: int = 50  # synced from PipelineCfg
    proto_metric: str = "sqeuclidean"  # "sqeuclidean" | "cosine"
    proto_tau: float = 1.0  # distance temperature in proto softmax
    poe_alpha: float = 1.0  # appearance expert weight
    poe_beta: float = 1.0  # relational expert weight
    init_T: float = 1.0  # initial temperature for calibration
    min_top1_prob: float = 0.40  # abstain below this top-1 probability
    max_entropy_bits: float = 2.5  # abstain above this predictive entropy


@dataclass
class SynthCfg:
    """Synthetic renderer + domain randomization (handout §2.3, §4.2)."""

    image_size: int = 518
    erosion_iters: int = 3  # pin sampled inside eroded mask (§8.1)
    n_pins_per_image: int = 8
    min_pin_area: int = 32  # skip regions too small to pin safely
    # domain randomization (xi) ranges
    light_range: tuple[float, float] = (0.4, 1.6)
    color_jitter: float = 0.15  # per-channel multiplicative jitter
    gauss_noise_std: float = 0.04
    blur_sigma_max: float = 1.5
    occlusion_prob: float = 0.25
    occlusion_max_frac: float = 0.07  # max occluder area as frac of image
    cadaveric_prob: float = 0.5  # P(shift render toward pale formalin-cadaver look)
    camera_roll_deg: float = 20.0
    mesh_dir: str | None = None  # populated by the user for M1+
    unlit_id_pass: bool = True  # ID pass: unlit flat, MSAA off (§8.2)
    window_visible: bool = True  # legacy Visualizer needs a real window on macOS
    zoom_min: float = 0.30  # close framing so region+context fill the frame
    zoom_max: float = 0.42
    muscle_drop_max: float = 0.7  # max fraction of muscle layers dropped (dissection depth)
    tissue_fill: bool = True  # fill background+gaps with fat/fascia (continuous tissue field)


# --------------------------------------------------------------------------- #
# Bundle                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class PipelineCfg:
    """Top-level config bundling every component config.

    ``n_classes`` and ``image_size`` are owned here and pushed down into the
    sub-configs by ``__post_init__`` so the closed vocabulary size stays
    consistent across the segmenter, the GNN and the heads.
    """

    n_classes: int = 50  # |V| - closed structure vocabulary (§1.5)
    image_size: int = 518

    backbone: BackboneCfg = field(default_factory=BackboneCfg)
    point: PointCondCfg = field(default_factory=PointCondCfg)
    segmenter: SegmenterCfg = field(default_factory=SegmenterCfg)
    graph: GraphCfg = field(default_factory=GraphCfg)
    gnn: GNNCfg = field(default_factory=GNNCfg)
    head: HeadCfg = field(default_factory=HeadCfg)
    synth: SynthCfg = field(default_factory=SynthCfg)

    device: str = field(default_factory=default_device)

    def __post_init__(self) -> None:
        # Propagate cross-cutting quantities so sub-configs cannot drift.
        self.backbone = replace(self.backbone, image_size=self.image_size)
        self.segmenter = replace(self.segmenter, n_classes=self.n_classes)
        self.gnn = replace(
            self.gnn,
            n_classes=self.n_classes,
            n_relations=self.graph.n_relations,
        )
        self.head = replace(self.head, n_classes=self.n_classes)
        self.synth = replace(self.synth, image_size=self.image_size)

    # convenience -----------------------------------------------------------
    @property
    def embed_dim(self) -> int:
        return self.backbone.embed_dim

    @property
    def grid_size(self) -> int:
        return self.backbone.grid_size

    def summary(self) -> dict:
        """Flat dict of the headline numbers, handy for logging."""
        return {
            "n_classes": self.n_classes,
            "image_size": self.image_size,
            "grid_size": self.grid_size,
            "embed_dim": self.embed_dim,
            "point_mode": self.point.mode,
            "n_relations": self.graph.n_relations,
            "device": self.device,
        }


def field_names(cfg) -> list[str]:
    """Utility: names of a dataclass config's fields (used in tests)."""
    return [f.name for f in fields(cfg)]
