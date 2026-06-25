"""End-to-end SCALPEL pipeline (handout §3, §4.8).

``predict`` flow::

    preprocess (ImageNet norm + coord scale)
        -> backbone -> grid, cls
        -> point conditioner -> z_q -> prototypical head -> log p_app
        -> segmenter -> dense label map -> scene graph -> R-GCN -> log p_rel
        -> Product of Experts fuse -> temperature calibrate -> decide

The appearance and relational experts are conditionally independent given the
label, so fusing them in log space (PoE) sharpens the distribution when they
agree (handout §2.5). Calibration happens *before* the abstention decision
(handout §8.6).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PipelineCfg
from .heads import PrototypicalHead, ProductOfExperts, TemperatureScaler
from .perception import DinoBackbone, PatchSegmenter, PointConditioner
from .relational_gnn import RelationalGNN, to_tensors
from .scene_graph import build_scene_graph, pin_region_index

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class ScalpelPipeline(nn.Module):
    """The full point-conditioned Product-of-Experts model."""

    def __init__(self, cfg: PipelineCfg, point_mode: str = "attn"):
        super().__init__()
        self.cfg = cfg
        self.image_size = cfg.image_size
        embed_dim = cfg.embed_dim

        self.backbone = DinoBackbone(cfg.backbone)
        self.point_conditioner = PointConditioner(
            cfg.point, embed_dim, cfg.image_size, mode=point_mode
        )
        self.segmenter = PatchSegmenter(cfg.segmenter, embed_dim)
        self.gnn = RelationalGNN(cfg.gnn, embed_dim)
        self.proto_head = PrototypicalHead(cfg.head, embed_dim)
        self.temp_scaler = TemperatureScaler(cfg.head)
        self.poe = ProductOfExperts(cfg.head)

        self.label_names: list[str] = [str(i) for i in range(cfg.n_classes)]

        self.register_buffer("_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    # -- utilities ----------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self._mean.device

    def set_label_names(self, names: list[str]) -> None:
        assert (
            len(names) == self.cfg.n_classes
        ), "label_names length must match n_classes"
        self.label_names = list(names)

    # -- preprocessing ------------------------------------------------------
    def preprocess(self, img, q_xy):
        """PIL image + pin (original px) -> normalised tensor + scaled pin.

        Returns ``(x (1,3,S,S), q_px (1,2))`` with the pin remapped into the
        resized pixel frame so it lines up with the patch centers.
        """
        S = self.image_size
        w, h = img.size
        arr = np.asarray(img.convert("RGB").resize((S, S)), dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        x = (x - self._mean) / self._std
        qx = q_xy[0] * (S / w)
        qy = q_xy[1] * (S / h)
        q_px = torch.tensor([[qx, qy]], dtype=torch.float32, device=self.device)
        return x, q_px

    # -- region pooling -----------------------------------------------------
    def _region_embeds(self, grid: torch.Tensor, regions) -> torch.Tensor:
        """Mean-pool DINO tokens over each region mask -> ``(N, D)`` (handout §4.5)."""
        _, g, _, d = grid.shape
        tokens = grid[0]  # (g, g, D)
        S = self.image_size
        p = S // g
        out = torch.zeros((len(regions), d), device=grid.device)
        for i, r in enumerate(regions):
            mask = r.mask
            if mask.shape == (g * p, g * p):
                cov = mask.reshape(g, p, g, p).mean(axis=(1, 3))  # (g, g) coverage
            else:  # fallback: resize coverage
                cov = _resize_coverage(mask, g)
            sel = torch.from_numpy(cov > 0.5).to(grid.device)
            if not bool(sel.any()):
                sel = torch.from_numpy(cov >= cov.max()).to(grid.device)
            out[i] = tokens[sel].mean(dim=0)
        return out

    # -- forward / predict --------------------------------------------------
    def predict(self, img, q_xy) -> dict:
        """Identify the structure under pin ``q_xy`` in ``img`` (handout §4.8)."""
        self.eval()
        C = self.cfg.n_classes
        with torch.no_grad():
            x, q_px = self.preprocess(img, q_xy)
            grid, _cls = self.backbone(x)  # (1,g,g,D)
            centers = self.backbone.patch_centers(self.device)  # (g*g,2)

            # ---- appearance expert -----------------------------------------
            z_q = self.point_conditioner(grid, centers, q_px)  # (1,D)
            if self.proto_head.filled == 0:
                logp_app = _uniform_logp(C, self.device)
            else:
                logp_app = F.log_softmax(self.proto_head(z_q)[0], dim=-1)

            # ---- relational expert -----------------------------------------
            logp_rel, pin_label = self._relational_logp(grid, q_px, C)

            # ---- fuse, calibrate, decide -----------------------------------
            fused = self.poe.fuse(logp_app, logp_rel)  # (C,)
            calibrated = F.log_softmax(self.temp_scaler(fused), dim=-1)
            decision = self.poe.decide(calibrated, label_names=self.label_names)

        decision["pin_region_label"] = pin_label
        decision["temperature"] = self.temp_scaler.T
        return decision

    def _relational_logp(self, grid: torch.Tensor, q_px: torch.Tensor, C: int):
        """Relational log-distribution for the pin's region (uniform if none)."""
        label_grid = self.segmenter.dense_map(grid, out_size=self.image_size)
        lg = label_grid[0].cpu().numpy().astype(np.int64)
        graph = build_scene_graph(lg, self.cfg.graph)
        q = (float(q_px[0, 0]), float(q_px[0, 1]))
        pin_idx = pin_region_index(graph, q)
        if len(graph) == 0 or pin_idx is None:
            return _uniform_logp(C, grid.device), None

        region_embeds = self._region_embeds(grid, graph.regions)
        node_features, adj = to_tensors(
            graph, region_embeds, self.cfg.graph.n_relations, grid.device
        )
        node_logits = self.gnn(node_features, adj)  # (N, C)
        logp_rel = F.log_softmax(node_logits[pin_idx], dim=-1)
        return logp_rel, int(graph.regions[pin_idx].label)

    # -- persistence --------------------------------------------------------
    def state_dict(self, *args, **kwargs):
        """State dict of the *trained* heads only.

        The frozen DINO backbone is deliberately excluded (it is reproduced from
        torch.hub, not a learned artefact of this project).
        """
        sd = super().state_dict(*args, **kwargs)
        return {k: v for k, v in sd.items() if not k.startswith("backbone.")}

    def load_state_dict(self, state_dict, strict: bool = False):
        return super().load_state_dict(state_dict, strict=strict)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _uniform_logp(C: int, device) -> torch.Tensor:
    return F.log_softmax(torch.zeros(C, device=device), dim=-1)


def _resize_coverage(mask: np.ndarray, g: int) -> np.ndarray:
    """Average-pool a boolean mask down to a ``(g, g)`` coverage grid."""
    H, W = mask.shape
    ys = (np.arange(g + 1) * H / g).astype(int)
    xs = (np.arange(g + 1) * W / g).astype(int)
    cov = np.zeros((g, g), dtype=np.float64)
    m = mask.astype(np.float64)
    for r in range(g):
        for c in range(g):
            block = m[ys[r] : ys[r + 1], xs[c] : xs[c + 1]]
            cov[r, c] = block.mean() if block.size else 0.0
    return cov
