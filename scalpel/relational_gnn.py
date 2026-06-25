"""Relational expert: reason about identity from neighbourhood (handout §2.9, §4.5).

An R-GCN (Schlichtkrull et al. 2018) over the scene graph. Each relation type
gets its *own* weight matrix, so "adjacent", "surrounds", "deep", ... contribute
differently::

    h_i^{(l+1)} = sigma( W0 h_i^{(l)} + sum_r sum_{j in N_r(i)} (1/c_{i,r}) W_r h_j^{(l)} )

The expert is trained on synthetic graphs (labels are free) and transfers to
real data because *topology is more domain-invariant than appearance*
(handout §2.9).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GNNCfg
from .scene_graph import SceneGraph


class RGCNLayer(nn.Module):
    """One relational graph-convolution layer.

    Because aggregation is linear, ``W_r (A_r h)`` equals ``A_r (h W_r^T)``; we
    aggregate first (``A_r @ h``) then apply the per-relation linear map, which
    is cheaper and numerically identical.
    """

    def __init__(self, in_dim: int, out_dim: int, n_relations: int, bias: bool = True):
        super().__init__()
        self.n_relations = n_relations
        self.self_lin = nn.Linear(in_dim, out_dim, bias=bias)
        self.rel_lins = nn.ModuleList(
            [nn.Linear(in_dim, out_dim, bias=False) for _ in range(n_relations)]
        )

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """``h (N, in)``, ``adj (n_rel, N, N)`` (row-normalised) -> ``(N, out)``."""
        out = self.self_lin(h)
        for r in range(self.n_relations):
            out = out + self.rel_lins[r](adj[r] @ h)
        return out


class RelationalGNN(nn.Module):
    """Stack of :class:`RGCNLayer` producing per-node class logits."""

    def __init__(self, cfg: GNNCfg, embed_dim: int):
        super().__init__()
        self.cfg = cfg
        in_dim = embed_dim + cfg.geom_dim
        dims = [in_dim] + [cfg.hidden] * (cfg.n_layers - 1) + [cfg.n_classes]
        self.layers = nn.ModuleList(
            [
                RGCNLayer(dims[k], dims[k + 1], cfg.n_relations)
                for k in range(cfg.n_layers)
            ]
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """``h (N, in)``, ``adj (n_rel, N, N)`` -> per-node logits ``(N, C)``."""
        for k, layer in enumerate(self.layers):
            h = layer(h, adj)
            if k < len(self.layers) - 1:  # no activation on the final logits
                h = self.dropout(F.relu(h))
        return h

    def predict_node(
        self, h: torch.Tensor, adj: torch.Tensor, node: int
    ) -> torch.Tensor:
        """Logits ``(C,)`` for a single node (e.g. the pin's region)."""
        return self.forward(h, adj)[node]


# --------------------------------------------------------------------------- #
# Graph -> tensors                                                            #
# --------------------------------------------------------------------------- #
def to_tensors(graph: SceneGraph, region_embeds: torch.Tensor, n_rel: int, device=None):
    """Pack a scene graph into ``(node_features, adj)``.

    * ``node_features (N, D + geom_dim)`` - region appearance embedding
      concatenated with geometry ``[sqrt(area), cx, cy, ecc, solidity]``,
      normalised by the image extent so the values are scale-free.
    * ``adj (n_rel, N, N)`` - per-relation adjacency, *row-normalised* by
      ``c_{i,r}`` (the ``1/c_{i,r}`` term in the R-GCN update). ``adj[r, i, j]``
      is the message weight from ``j`` into ``i`` under relation ``r``.
    """
    n = len(graph.regions)
    embed_dim = region_embeds.shape[1]
    device = device or region_embeds.device

    if n == 0:
        return (
            torch.zeros((0, embed_dim + 5), device=device),
            torch.zeros((n_rel, 0, 0), device=device),
        )

    H, W = graph.regions[0].mask.shape
    diag = float(np.sqrt(H * W))
    geom = torch.zeros((n, 5), device=device)
    for i, r in enumerate(graph.regions):
        cx, cy = r.centroid
        geom[i, 0] = (r.area**0.5) / diag
        geom[i, 1] = cx / W
        geom[i, 2] = cy / H
        geom[i, 3] = r.eccentricity
        geom[i, 4] = r.solidity
    node_features = torch.cat([region_embeds.to(device), geom], dim=1)

    adj = torch.zeros((n_rel, n, n), device=device)
    for i in range(n):
        for j, rel in graph.edges[i]:
            if 0 <= rel < n_rel:
                adj[rel, i, j] = 1.0
    # row-normalise: divide each (relation, source-row) by its neighbour count c_{i,r}
    deg = adj.sum(dim=2, keepdim=True).clamp_min(1.0)
    adj = adj / deg
    return node_features, adj
