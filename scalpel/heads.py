"""Adaptation, calibration and fusion (handout §2.5-2.7, §4.6).

* :class:`PrototypicalHead` - few-shot classifier. Adding a class is just
  averaging a few embeddings: *no gradient, no overfitting* (handout §2.6). Real
  photos are consumed as a gallery, not a training set.
* :class:`TemperatureScaler` - post-hoc calibration with a single scalar T
  (Guo et al. 2017). Calibrated probabilities are what make the abstention
  threshold meaningful (handout §2.7, §8.6 - calibrate *before* abstaining).
* :class:`ProductOfExperts` - fuses the appearance and relational experts in log
  space (``alpha * log p_app + beta * log p_rel``) and turns the fused
  distribution into an answer-or-abstain decision (handout §2.5).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HeadCfg

_NEG_INF = float("-inf")


# --------------------------------------------------------------------------- #
# Calibration error                                                           #
# --------------------------------------------------------------------------- #
def expected_calibration_error(
    probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15
) -> float:
    """15-bin ECE over a batch of probability vectors (handout §7)."""
    probs, labels = probs.detach(), labels.detach()
    conf, pred = probs.max(dim=1)
    acc = pred.eq(labels).float()
    edges = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = torch.zeros((), device=probs.device)
    n = probs.shape[0]
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        in_bin = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        m = in_bin.float().sum()
        if m > 0:
            ece = ece + (m / n) * (acc[in_bin].mean() - conf[in_bin].mean()).abs()
    return float(ece)


# --------------------------------------------------------------------------- #
# Prototypical head                                                           #
# --------------------------------------------------------------------------- #
class PrototypicalHead(nn.Module):
    """Distance-to-prototype classifier (handout §2.6, §4.6)."""

    def __init__(self, cfg: HeadCfg, embed_dim: int):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = embed_dim
        self.metric = cfg.proto_metric
        self.tau = cfg.proto_tau
        self.register_buffer("prototypes", torch.zeros(cfg.n_classes, embed_dim))
        self.register_buffer(
            "filled_mask", torch.zeros(cfg.n_classes, dtype=torch.bool)
        )

    @torch.no_grad()
    def fit(self, embeds: torch.Tensor, labels: torch.Tensor) -> None:
        """Build prototypes from a gallery (no gradient; handout §2.6)."""
        embeds = embeds.to(self.prototypes.device)
        labels = labels.to(self.prototypes.device)
        for k in labels.unique():
            ki = int(k)
            if not (0 <= ki < self.prototypes.shape[0]):
                continue
            self.prototypes[ki] = embeds[labels == k].mean(dim=0)
            self.filled_mask[ki] = True

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``z (B, D)`` -> class logits ``(B, C)`` (unfilled classes -> -inf)."""
        if self.metric == "cosine":
            zc = F.normalize(z, dim=-1)
            pc = F.normalize(self.prototypes, dim=-1)
            logits = (zc @ pc.t()) / self.tau
        else:  # squared euclidean
            d2 = torch.cdist(z, self.prototypes) ** 2
            logits = -d2 / self.tau
        mask = self.filled_mask.unsqueeze(0).expand_as(logits)
        return logits.masked_fill(~mask, _NEG_INF)

    @property
    def filled(self) -> int:
        """Number of classes with a prototype (handout M5 acceptance)."""
        return int(self.filled_mask.sum())


# --------------------------------------------------------------------------- #
# Temperature scaling                                                         #
# --------------------------------------------------------------------------- #
class TemperatureScaler(nn.Module):
    """Single-scalar temperature calibration (handout §2.7, §4.6).

    ``log_temperature`` is the free parameter (``T = exp(log_temperature)``) so
    the temperature stays strictly positive during the LBFGS fit.
    """

    def __init__(self, cfg: HeadCfg | None = None, init_T: float | None = None):
        super().__init__()
        t0 = init_T if init_T is not None else (cfg.init_T if cfg else 1.0)
        self.log_temperature = nn.Parameter(torch.tensor(float(t0)).log())

    @property
    def T(self) -> float:
        return float(self.log_temperature.detach().exp())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_temperature.exp()

    def fit(
        self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100
    ) -> dict:
        """Fit T by minimising NLL via LBFGS; report ECE before/after."""
        logits = logits.detach()
        labels = labels.detach()
        ece_before = expected_calibration_error(logits.softmax(dim=1), labels)

        opt = torch.optim.LBFGS([self.log_temperature], lr=0.1, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss

        opt.step(closure)
        ece_after = expected_calibration_error(
            self.forward(logits).softmax(dim=1), labels
        )
        return {"T": self.T, "ece_before": ece_before, "ece_after": ece_after}


# --------------------------------------------------------------------------- #
# Product of Experts + decision                                               #
# --------------------------------------------------------------------------- #
class ProductOfExperts(nn.Module):
    """Log-space fusion of the two experts + answer/abstain decision (§2.5)."""

    def __init__(self, cfg: HeadCfg):
        super().__init__()
        self.alpha = cfg.poe_alpha
        self.beta = cfg.poe_beta
        self.min_top1_prob = cfg.min_top1_prob
        self.max_entropy_bits = cfg.max_entropy_bits

    def fuse(self, logp_app: torch.Tensor, logp_rel: torch.Tensor) -> torch.Tensor:
        """``alpha * log p_app + beta * log p_rel`` then renormalise in log space.

        Inputs are per-class *log probabilities*; output is a normalised log
        distribution over the vocabulary.
        """
        combined = self.alpha * logp_app + self.beta * logp_rel
        return F.log_softmax(combined, dim=-1)

    def decide(
        self, logp: torch.Tensor, topk: int = 5, label_names: list[str] | None = None
    ) -> dict:
        """Turn one fused log-distribution ``(C,)`` into a decision dict.

        Abstains when the calibrated top-1 probability is too low *or* the
        predictive entropy is too high (handout §4.6).
        """
        logp = logp.detach()
        if logp.dim() == 2:  # accept (1, C)
            assert logp.shape[0] == 1, "decide() handles a single distribution"
            logp = logp[0]
        p = logp.exp()
        # entropy in bits, ignoring the -inf/0 entries
        nz = p > 0
        entropy_bits = float(-(p[nz] * logp[nz]).sum()) / math.log(2.0)
        k = min(topk, int(p.numel()))
        top_p, top_i = p.topk(k)
        pred = int(top_i[0])
        prob = float(top_p[0])
        abstain = bool(
            prob < self.min_top1_prob or entropy_bits > self.max_entropy_bits
        )

        def name(idx: int):
            return label_names[idx] if label_names and idx < len(label_names) else idx

        return {
            "pred": pred,
            "pred_name": name(pred),
            "prob": prob,
            "entropy_bits": entropy_bits,
            "abstain": abstain,
            "topk": [int(i) for i in top_i.tolist()],
            "topk_names": [name(int(i)) for i in top_i.tolist()],
            "topk_prob": [float(x) for x in top_p.tolist()],
        }
