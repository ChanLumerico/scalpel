"""Perception: "what is where" (handout §2.4, §4.3).

Components
----------
* :class:`DinoBackbone`     - frozen DINOv2 ViT, returns a patch-token grid + CLS.
                              The heavy ``torch.hub`` load is deferred until the
                              first forward so constructing it is asset-free
                              (smoke path injects a mock; handout §8.8).
* :class:`GaussianPool`     - parameter-free point pooler (baseline).
* :class:`PinCrossAttention`- learned point pooler (seed = bilinear sample at q,
                              cross-attends over the token grid).
* :class:`PointConditioner` - dispatches to one of the two poolers by ``mode``.
* :class:`PatchSegmenter`   - lightweight MLP probe over the frozen tokens.

Token-grid flattening is **row-major** (index = r*g + c) everywhere, and
:meth:`DinoBackbone.patch_centers` returns centers in the same order so the
pooler's spatial weights line up with the tokens (handout §4.3).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BackboneCfg, PointCondCfg, SegmenterCfg


# --------------------------------------------------------------------------- #
# Backbone                                                                     #
# --------------------------------------------------------------------------- #
class DinoBackbone(nn.Module):
    """Frozen DINOv2 backbone.

    The actual network is loaded lazily (``ensure_loaded``) the first time a
    forward pass is requested, so merely constructing a pipeline never touches
    the network or the torch.hub cache.
    """

    def __init__(self, cfg: BackboneCfg):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.embed_dim
        self.patch_size = cfg.patch_size
        self.image_size = cfg.image_size
        self.grid_size = cfg.grid_size
        self._model: nn.Module | None = None

    # -- lazy loading -------------------------------------------------------
    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        model = torch.hub.load(self.cfg.repo, self.cfg.name)
        if self.cfg.frozen:
            for p in model.parameters():
                p.requires_grad_(False)
            model.eval()
        self._model = model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # -- forward ------------------------------------------------------------
    def forward(self, imgs: torch.Tensor):
        """``imgs (B,3,H,W)`` -> ``(grid (B,g,g,D), cls (B,D))``."""
        self.ensure_loaded()
        ctx = torch.no_grad() if self.cfg.frozen else torch.enable_grad()
        with ctx:
            out = self._model.forward_features(imgs)
        patch = out["x_norm_patchtokens"]  # (B, g*g, D)
        cls = out["x_norm_clstoken"]  # (B, D)
        b, n, d = patch.shape
        g = self.grid_size
        if n != g * g:
            # input wasn't the configured size; infer a square grid
            g = int(round(n**0.5))
        grid = patch.reshape(b, g, g, d)
        return grid, cls

    # -- geometry -----------------------------------------------------------
    def patch_centers(self, device=None) -> torch.Tensor:
        """Pixel center ``(x, y)`` of each patch, row-major: ``(g*g, 2)``."""
        g, p = self.grid_size, self.patch_size
        rows = torch.arange(g, dtype=torch.float32, device=device)
        cols = torch.arange(g, dtype=torch.float32, device=device)
        cy = (rows + 0.5) * p
        cx = (cols + 0.5) * p
        # row-major: row index outer, col index inner -> index = r*g + c
        yy = cy.view(g, 1).expand(g, g).reshape(-1)
        xx = cx.view(1, g).expand(g, g).reshape(-1)
        return torch.stack([xx, yy], dim=-1)  # (g*g, 2) as (x, y)


# --------------------------------------------------------------------------- #
# Point poolers                                                                #
# --------------------------------------------------------------------------- #
def _fourier_encode(coords_norm: torch.Tensor, n_freqs: int) -> torch.Tensor:
    """Fourier positional encoding gamma(q).

    ``coords_norm (B, 2)`` in [-1, 1] -> ``(B, 2 * 2 * n_freqs)``.
    """
    freqs = 2.0 ** torch.arange(
        n_freqs, device=coords_norm.device, dtype=coords_norm.dtype
    )
    ang = coords_norm[:, :, None] * freqs[None, None, :] * torch.pi  # (B,2,F)
    enc = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (B,2,2F)
    return enc.reshape(coords_norm.shape[0], -1)  # (B,4F)


class GaussianPool(nn.Module):
    """Parameter-free point pooler (handout §4.3 baseline).

    ``z_q = sum_i softmax_i(-||p_i - q||^2 / 2 sigma^2) z_i``.
    """

    def __init__(self, cfg: PointCondCfg):
        super().__init__()
        self.sigma = cfg.gauss_sigma_px

    def forward(
        self, grid: torch.Tensor, centers: torch.Tensor, q_px: torch.Tensor
    ) -> torch.Tensor:
        b, g, _, d = grid.shape
        tokens = grid.reshape(b, g * g, d)  # (B,M,D)
        centers = centers.to(grid.device)  # (M,2)
        # squared distance from each patch center to the pin, per batch element
        diff = centers[None, :, :] - q_px[:, None, :]  # (B,M,2)
        d2 = (diff**2).sum(-1)  # (B,M)
        w = torch.softmax(-d2 / (2.0 * self.sigma**2), dim=1)  # (B,M)
        z_q = torch.einsum("bm,bmd->bd", w, tokens)  # (B,D)
        return z_q


class PinCrossAttention(nn.Module):
    """Learned point pooler (handout §2.1, §4.3).

    ``seed = bilinear_sample(grid, q)``; ``Q = Wq[seed; gamma(q)]``;
    ``z_q = softmax(QK^T/sqrt(d)) V`` over the token grid, residually added to
    the seed.
    """

    def __init__(self, cfg: PointCondCfg, embed_dim: int, image_size: int):
        super().__init__()
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.n_heads = cfg.n_heads
        self.dim = cfg.attn_dim
        assert self.dim % self.n_heads == 0, "attn_dim must be divisible by n_heads"
        self.head_dim = self.dim // self.n_heads
        fourier_dim = 2 * 2 * cfg.n_pos_freqs
        self.n_pos_freqs = cfg.n_pos_freqs

        self.q_proj = nn.Linear(embed_dim + fourier_dim, self.dim)
        self.k_proj = nn.Linear(embed_dim, self.dim)
        self.v_proj = nn.Linear(embed_dim, self.dim)
        self.out_proj = nn.Linear(self.dim, embed_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def _seed(self, grid: torch.Tensor, q_px: torch.Tensor) -> torch.Tensor:
        """Bilinearly sample the token grid at the pin location -> ``(B,D)``."""
        b, g, _, d = grid.shape
        feat = grid.permute(0, 3, 1, 2)  # (B,D,g,g)
        q_norm = 2.0 * (q_px / self.image_size) - 1.0  # (B,2) in [-1,1]
        samp = q_norm.view(b, 1, 1, 2)  # (B,1,1,2) -> (x,y)
        seed = F.grid_sample(
            feat, samp, mode="bilinear", align_corners=False, padding_mode="border"
        )
        return seed.view(b, d)  # (B,D)

    def forward(
        self, grid: torch.Tensor, centers: torch.Tensor, q_px: torch.Tensor
    ) -> torch.Tensor:
        b, g, _, d = grid.shape
        tokens = grid.reshape(b, g * g, d)  # (B,M,D)
        seed = self._seed(grid, q_px)  # (B,D)
        q_norm = 2.0 * (q_px / self.image_size) - 1.0
        gamma = _fourier_encode(q_norm, self.n_pos_freqs)  # (B,4F)

        q = self.q_proj(torch.cat([seed, gamma], dim=-1))  # (B,dim)
        k = self.k_proj(tokens)  # (B,M,dim)
        v = self.v_proj(tokens)  # (B,M,dim)

        nh, hd = self.n_heads, self.head_dim
        q = q.view(b, nh, 1, hd)
        k = k.view(b, -1, nh, hd).transpose(1, 2)  # (B,nh,M,hd)
        v = v.view(b, -1, nh, hd).transpose(1, 2)  # (B,nh,M,hd)
        attn = torch.softmax((q @ k.transpose(-2, -1)) / hd**0.5, dim=-1)
        attn = self.dropout(attn)
        ctx = (attn @ v).reshape(b, self.dim)  # (B,dim)
        z_q = self.out_proj(ctx) + seed  # residual
        return z_q


class PointConditioner(nn.Module):
    """Dispatch to the configured point pooler (handout §4.3).

    ``mode`` is ``"attn"`` (learned :class:`PinCrossAttention`) or ``"gauss"``
    (parameter-free :class:`GaussianPool`). Output is always ``(B, embed_dim)``.
    """

    def __init__(
        self,
        cfg: PointCondCfg,
        embed_dim: int,
        image_size: int,
        mode: str | None = None,
    ):
        super().__init__()
        self.mode = mode or cfg.mode
        self.embed_dim = embed_dim
        if self.mode == "attn":
            self.pool = PinCrossAttention(cfg, embed_dim, image_size)
        elif self.mode == "gauss":
            self.pool = GaussianPool(cfg)
        else:
            raise ValueError(f"unknown point-conditioner mode: {self.mode!r}")

    def forward(
        self, grid: torch.Tensor, centers: torch.Tensor, q_px: torch.Tensor
    ) -> torch.Tensor:
        return self.pool(grid, centers, q_px)


# --------------------------------------------------------------------------- #
# Segmenter                                                                    #
# --------------------------------------------------------------------------- #
class PatchSegmenter(nn.Module):
    """Per-token MLP probe over frozen DINO features (handout §4.3).

    Cheap dense labelling used to seed the scene graph; the backbone stays
    frozen, only this probe is trained.
    """

    def __init__(self, cfg: SegmenterCfg, embed_dim: int):
        super().__init__()
        self.n_classes = cfg.n_classes
        self.net = nn.Sequential(
            nn.Linear(embed_dim, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.n_classes),
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """``grid (B,g,g,D)`` -> per-patch logits ``(B,g,g,C)``."""
        return self.net(grid)

    @torch.no_grad()
    def dense_map(self, grid: torch.Tensor, out_size: int) -> torch.Tensor:
        """Argmax label map upsampled to ``(B, out_size, out_size)`` (nearest)."""
        logits = self.forward(grid)  # (B,g,g,C)
        labels = logits.argmax(dim=-1).float().unsqueeze(1)  # (B,1,g,g)
        up = F.interpolate(labels, size=(out_size, out_size), mode="nearest")
        return up.squeeze(1).long()  # (B,out,out)
