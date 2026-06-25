#!/usr/bin/env python3
"""M3 pre-validation: does the relational expert actually *learn* from topology?

Milestone M3's acceptance is "node-classification accuracy rises on synthetic
graphs" (handout §6). This is fully testable today - no Open3D, no meshes - by
driving the *real* relational code path end-to-end:

    procedural label map -> build_scene_graph -> to_tensors -> RelationalGNN
                         -> pretrain_gnn (the real training loop)

The task is deliberately **purely relational**: each region carries a random
marker, and its class is a function of its *neighbours'* markers, not its own
features. So:

* a features-only MLP baseline cannot beat the majority-class rate, but
* the R-GCN, which aggregates over typed edges, can.

If the R-GCN clears the MLP/majority baseline, the relational machinery (typed
adjacency -> message passing -> per-node logits) works. (The real anatomical
relational-reasoning claim still needs mesh-derived graphs at M3 proper.)

Run (any env with torch + scipy; e.g. the 3.14 main env)::

    python scripts/validate_m3_relational.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalpel.config import BackboneCfg, PipelineCfg  # noqa: E402
from scalpel.loops import pretrain_gnn  # noqa: E402
from scalpel.pipeline import ScalpelPipeline  # noqa: E402
from scalpel.relational_gnn import to_tensors  # noqa: E402
from scalpel.scene_graph import build_scene_graph  # noqa: E402

N_CLASSES = 4
EMBED_DIM = 8


def make_layout(rng: np.random.Generator, g: int = 5, cell: int = 36,
                drop_prob: float = 0.15) -> np.ndarray:
    """Procedural grid label map: each kept cell is one region (touching = adjacent)."""
    H = W = g * cell
    lab = np.zeros((H, W), dtype=np.int64)
    nxt = 1
    for r in range(g):
        for c in range(g):
            if rng.random() < drop_prob:
                continue
            y0, x0 = r * cell, c * cell
            lab[y0:y0 + cell, x0:x0 + cell] = nxt  # cells touch -> adjacency detected
            nxt += 1
    return lab


def make_sample(rng: np.random.Generator, cfg) -> dict | None:
    """One relational graph: class_i = bucket(mean of neighbours' markers)."""
    lab = make_layout(rng)
    graph = build_scene_graph(lab, cfg.graph)
    n = len(graph)
    if n < 4:
        return None

    # random marker per node; node embedding carries its OWN marker + noise
    markers = rng.uniform(0.0, 1.0, size=n)
    region_embeds = torch.zeros((n, EMBED_DIM))
    region_embeds[:, 0] = torch.tensor(markers, dtype=torch.float32)
    region_embeds[:, 1:] = torch.randn(n, EMBED_DIM - 1) * 0.1

    # label depends ONLY on neighbours' markers (not the node's own) -> relational
    labels = np.zeros(n, dtype=np.int64)
    for i in range(n):
        nbrs = {j for (j, _rel) in graph.neighbors(i)}
        if nbrs:
            m = float(np.mean([markers[j] for j in nbrs]))
            labels[i] = min(N_CLASSES - 1, int(m * N_CLASSES))
    node_features, adj = to_tensors(graph, region_embeds, cfg.graph.n_relations, "cpu")
    return {"node_features": node_features, "adj": adj,
            "labels": torch.tensor(labels), "markers": markers, "graph": graph}


@torch.no_grad()
def node_accuracy(gnn, samples) -> float:
    correct = total = 0
    for s in samples:
        if s["node_features"].shape[0] == 0:
            continue
        pred = gnn(s["node_features"], s["adj"]).argmax(-1)
        correct += int((pred == s["labels"]).sum())
        total += s["labels"].numel()
    return correct / max(1, total)


def majority_baseline(train, val) -> float:
    counts = np.zeros(N_CLASSES, dtype=np.int64)
    for s in train:
        for y in s["labels"].tolist():
            counts[y] += 1
    maj = int(counts.argmax())
    correct = total = 0
    for s in val:
        correct += int((s["labels"] == maj).sum())
        total += s["labels"].numel()
    return correct / max(1, total)


def mlp_baseline(train, val, epochs=60) -> float:
    """Features-only MLP (no graph). Should not beat majority on a relational task."""
    Xtr = torch.cat([s["node_features"] for s in train])
    ytr = torch.cat([s["labels"] for s in train])
    in_dim = Xtr.shape[1]
    net = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, N_CLASSES))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(net(Xtr), ytr)
        loss.backward()
        opt.step()
    Xval = torch.cat([s["node_features"] for s in val])
    yval = torch.cat([s["labels"] for s in val])
    with torch.no_grad():
        acc = float((net(Xval).argmax(-1) == yval).float().mean())
    return acc


def main() -> int:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    cfg = PipelineCfg(n_classes=N_CLASSES, image_size=70, backbone=BackboneCfg(embed_dim=EMBED_DIM))

    train = [s for s in (make_sample(rng, cfg) for _ in range(160)) if s]
    val = [s for s in (make_sample(rng, cfg) for _ in range(40)) if s]
    n_nodes = sum(s["labels"].numel() for s in train)
    print(f"[data] {len(train)} train / {len(val)} val graphs, {n_nodes} train nodes, "
          f"avg degree { _avg_degree(train):.2f}")

    pipe = ScalpelPipeline(cfg, point_mode="gauss")  # gauss => no unused attn params
    hist = pretrain_gnn(pipe, train, epochs=40, lr=1e-2)

    gnn_val = node_accuracy(pipe.gnn, val)
    maj = majority_baseline(train, val)
    mlp = mlp_baseline(train, val)

    print("\n=== M3 relational learning ===")
    print(f"  train node-acc: {hist['node_acc'][0]:.3f} (epoch 1) -> "
          f"{hist['node_acc'][-1]:.3f} (epoch {len(hist['node_acc'])})")
    print(f"  R-GCN  val acc : {gnn_val:.3f}")
    print(f"  MLP    val acc : {mlp:.3f}   (features only, no graph)")
    print(f"  majority base  : {maj:.3f}")

    rose = hist["node_acc"][-1] > hist["node_acc"][0] + 0.10
    beats = gnn_val > max(maj, mlp) + 0.10
    ok = rose and beats
    print(f"\n  accuracy rose over epochs : {'YES' if rose else 'NO'}")
    print(f"  R-GCN beats MLP/majority  : {'YES' if beats else 'NO'}  "
          f"(=> uses relations, not just node features)")
    print(f"\nRESULT: {'M3 relational machinery VALIDATED' if ok else 'NOT validated (see above)'}")
    return 0 if ok else 2


def _avg_degree(samples) -> float:
    degs = []
    for s in samples:
        g = s["graph"]
        for i in range(len(g)):
            degs.append(len({j for (j, _r) in g.neighbors(i)}))
    return float(np.mean(degs)) if degs else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
