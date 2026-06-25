"""Training, adaptation and evaluation loops (handout §4.9, §6, §7).

* :func:`synthetic_pretrain` - train the point conditioner + segmenter on
  synthetic triples (labels are free). The backbone stays frozen. A throwaway
  linear classifier provides the appearance training signal; at adaptation time
  it is discarded in favour of prototypes.
* :func:`pretrain_gnn` - train the relational expert on synthetic graphs.
* :func:`fewshot_adapt` - build prototypes from a real gallery (no gradient;
  handout §2.6).
* :func:`evaluate` - selective-accuracy / coverage / ECE on a held-out set,
  the honest sim-to-real benchmark (handout §7). Splits must be specimen-level
  (handout §5.2) - that is the caller's responsibility.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import expected_calibration_error


# --------------------------------------------------------------------------- #
# Synthetic pretraining (appearance)                                          #
# --------------------------------------------------------------------------- #
def synthetic_pretrain(pipe, triple_loader, epochs: int = 1, lr: float = 1e-3) -> dict:
    """Train the point conditioner + segmenter on synthetic triples.

    ``triple_loader`` yields batch dicts with keys:
        ``images`` (B,3,S,S) normalised, ``pins`` (B,2) in resized px,
        ``labels`` (B,), and optionally ``seg_maps`` (B,g,g) per-patch labels.
    """
    device = pipe.device
    embed_dim = pipe.cfg.embed_dim
    n_classes = pipe.cfg.n_classes
    centers = pipe.backbone.patch_centers(device)

    # throwaway appearance classifier (discarded at adapt time)
    clf = nn.Linear(embed_dim, n_classes).to(device)
    params = (
        list(pipe.point_conditioner.parameters())
        + list(pipe.segmenter.parameters())
        + list(clf.parameters())
    )
    opt = torch.optim.Adam(params, lr=lr)

    pipe.point_conditioner.train()
    pipe.segmenter.train()
    history = {"loss": [], "app_acc": [], "seg_loss": []}

    for _ in range(epochs):
        ep_loss, ep_correct, ep_n, ep_seg = 0.0, 0, 0, 0.0
        for batch in triple_loader:
            images = batch["images"].to(device)
            pins = batch["pins"].to(device).float()
            labels = batch["labels"].to(device).long()

            with torch.no_grad():
                grid, _ = pipe.backbone(images)  # backbone frozen

            z_q = pipe.point_conditioner(grid, centers, pins)
            app_logits = clf(z_q)
            loss = F.cross_entropy(app_logits, labels)

            seg_loss_val = 0.0
            if batch.get("seg_maps") is not None:
                seg_target = batch["seg_maps"].to(device).long()  # (B,g,g)
                seg_logits = pipe.segmenter(grid)  # (B,g,g,C)
                seg_loss = F.cross_entropy(seg_logits.permute(0, 3, 1, 2), seg_target)
                loss = loss + seg_loss
                seg_loss_val = float(seg_loss.detach())

            opt.zero_grad()
            loss.backward()
            opt.step()

            ep_loss += float(loss.detach()) * labels.size(0)
            ep_correct += int((app_logits.argmax(-1) == labels).sum())
            ep_n += labels.size(0)
            ep_seg += seg_loss_val * labels.size(0)

        history["loss"].append(ep_loss / max(1, ep_n))
        history["app_acc"].append(ep_correct / max(1, ep_n))
        history["seg_loss"].append(ep_seg / max(1, ep_n))
    return history


# --------------------------------------------------------------------------- #
# Synthetic pretraining (relational)                                          #
# --------------------------------------------------------------------------- #
def pretrain_gnn(pipe, graph_samples, epochs: int = 1, lr: float = 1e-3) -> dict:
    """Train the relational expert on synthetic graphs.

    ``graph_samples`` yields dicts with keys ``node_features`` (N, D+geom),
    ``adj`` (n_rel, N, N) and ``labels`` (N,) - the per-node ground-truth
    structure ids.
    """
    device = pipe.device
    opt = torch.optim.Adam(pipe.gnn.parameters(), lr=lr)
    pipe.gnn.train()
    history = {"loss": [], "node_acc": []}

    for _ in range(epochs):
        ep_loss, ep_correct, ep_n = 0.0, 0, 0
        for sample in graph_samples:
            h = sample["node_features"].to(device)
            adj = sample["adj"].to(device)
            labels = sample["labels"].to(device).long()
            if h.shape[0] == 0:
                continue
            logits = pipe.gnn(h, adj)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach()) * labels.size(0)
            ep_correct += int((logits.argmax(-1) == labels).sum())
            ep_n += labels.size(0)
        history["loss"].append(ep_loss / max(1, ep_n))
        history["node_acc"].append(ep_correct / max(1, ep_n))
    return history


# --------------------------------------------------------------------------- #
# Few-shot adaptation (real gallery -> prototypes, no grad)                    #
# --------------------------------------------------------------------------- #
def _gallery_embeddings(pipe, gallery):
    """Return ``(embeds (M,D), labels (M,))`` from a variety of gallery forms."""
    if isinstance(gallery, dict) and "embeds" in gallery:
        e = torch.as_tensor(gallery["embeds"], dtype=torch.float32)
        y = torch.as_tensor(gallery["labels"], dtype=torch.long)
        return e, y

    embeds, labels = [], []
    centers = pipe.backbone.patch_centers(pipe.device)
    for item in gallery:
        if len(item) == 3:  # (PIL image, q, label)
            img, q, lab = item
            x, q_px = pipe.preprocess(img, q)
            with torch.no_grad():
                grid, _ = pipe.backbone(x)
                z = pipe.point_conditioner(grid, centers, q_px)[0]
            embeds.append(z)
            labels.append(int(lab))
        else:  # (embedding, label)
            emb, lab = item
            embeds.append(torch.as_tensor(emb, dtype=torch.float32))
            labels.append(int(lab))
    return torch.stack(embeds), torch.tensor(labels, dtype=torch.long)


def fewshot_adapt(pipe, gallery) -> dict:
    """Build prototypes from a real-image gallery (no gradient; handout §2.6)."""
    embeds, labels = _gallery_embeddings(pipe, gallery)
    pipe.proto_head.fit(embeds, labels)
    return {
        "filled": pipe.proto_head.filled,
        "n_support": int(labels.numel()),
        "n_classes_seen": int(labels.unique().numel()),
    }


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #
def _selective_ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    n = conf.size
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() > 0:
            ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def evaluate(pipe, test_set) -> dict:
    """Evaluate on ``(image, q, label)`` triples (handout §7).

    Reports coverage, selective accuracy at the model's own abstention
    threshold, ECE, top-k accuracy, and a coverage/selective-accuracy curve
    obtained by sweeping the confidence threshold.
    """
    confs, correct, answered, topk_hit = [], [], [], []
    for img, q, label in test_set:
        d = pipe.predict(img, q)
        confs.append(d["prob"])
        correct.append(int(d["pred"] == int(label)))
        answered.append(not d["abstain"])
        topk_hit.append(int(int(label) in d["topk"]))

    confs = np.array(confs, dtype=np.float64)
    correct = np.array(correct, dtype=np.float64)
    answered = np.array(answered, dtype=bool)
    topk_hit = np.array(topk_hit, dtype=np.float64)
    n = int(confs.size)

    cov = float(answered.mean()) if n else 0.0
    sel_acc = float(correct[answered].mean()) if answered.any() else 0.0

    # coverage / selective-accuracy curve by sweeping the confidence threshold
    curve = []
    for t in np.linspace(0.0, 1.0, 21):
        keep = confs >= t
        c = float(keep.mean()) if n else 0.0
        a = float(correct[keep].mean()) if keep.any() else float("nan")
        curve.append({"threshold": float(t), "coverage": c, "selective_accuracy": a})

    return {
        "n": n,
        "coverage": cov,
        "selective_accuracy": sel_acc,
        "overall_accuracy": float(correct.mean()) if n else 0.0,
        "topk_accuracy": float(topk_hit.mean()) if n else 0.0,
        "ece": _selective_ece(confs, correct) if n else 0.0,
        "curve": curve,
    }
