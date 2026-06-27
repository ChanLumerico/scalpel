"""SCALPEL smoke test - milestone M0 acceptance (handout §6, §8).

Threads synthetic tensors through *every* module with the DINO backbone mocked
and exercises the §8 traps. Runs with **zero external assets**: no torch.hub
download, no Open3D, no LLM backend. Run directly::

    python tests/smoke_test.py

Exits non-zero if any check fails.
"""

from __future__ import annotations

import os
import sys
import traceback

# --- lazy-import guard: importing `scalpel` must NOT pull in heavy entry points #
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scalpel  # noqa: E402

assert "scalpel.pipeline" not in sys.modules, "pipeline must be lazily imported (§4.10)"
assert (
    "open3d" not in sys.modules
), "open3d must never be imported on the light path (§8.8)"
_ = scalpel.PipelineCfg  # config is the one eager export
_ = scalpel.default_device()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from PIL import Image  # noqa: E402

from scalpel.config import BackboneCfg, PipelineCfg, field_names  # noqa: E402
from scalpel import (
    heads,
    llm_reasoner,
    loops,
    perception,
    scene_graph,
)  # noqa: E402
from scalpel.relational_gnn import RelationalGNN, to_tensors  # noqa: E402
from scalpel.pipeline import ScalpelPipeline  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)


# --------------------------------------------------------------------------- #
# Mock backbone (stands in for frozen DINOv2; handout §8.8)                    #
# --------------------------------------------------------------------------- #
class MockBackbone(nn.Module):
    """Returns random tensors with the real backbone's shapes/interface."""

    def __init__(self, cfg: BackboneCfg):
        super().__init__()
        self.embed_dim = cfg.embed_dim
        self.patch_size = cfg.patch_size
        self.image_size = cfg.image_size
        self.grid_size = cfg.grid_size
        self.is_loaded = True

    def ensure_loaded(self):
        pass

    def forward(self, imgs):
        b = imgs.shape[0]
        g, d = self.grid_size, self.embed_dim
        return torch.randn(b, g, g, d), torch.randn(b, d)

    def patch_centers(self, device=None):
        g, p = self.grid_size, self.patch_size
        rows = torch.arange(g, dtype=torch.float32, device=device)
        cy = (rows + 0.5) * p
        cx = (torch.arange(g, dtype=torch.float32, device=device) + 0.5) * p
        yy = cy.view(g, 1).expand(g, g).reshape(-1)
        xx = cx.view(1, g).expand(g, g).reshape(-1)
        return torch.stack([xx, yy], dim=-1)


# small config: grid 5x5 (70/14), embed 32, 8 classes -> tiny + fast
def tiny_cfg() -> PipelineCfg:
    return PipelineCfg(n_classes=8, image_size=70, backbone=BackboneCfg(embed_dim=32))


def make_pipe() -> ScalpelPipeline:
    cfg = tiny_cfg()
    pipe = ScalpelPipeline(cfg, point_mode="attn")
    assert (
        not pipe.backbone.is_loaded
    ), "real backbone must not load on construction (§8.8)"
    pipe.backbone = MockBackbone(cfg.backbone)  # mock it
    return pipe


def rand_image(h=100, w=120) -> Image.Image:
    return Image.fromarray((np.random.rand(h, w, 3) * 255).astype(np.uint8))


# --------------------------------------------------------------------------- #
# Checks                                                                       #
# --------------------------------------------------------------------------- #
def check_config():
    cfg = tiny_cfg()
    assert cfg.grid_size == 5 and cfg.embed_dim == 32
    # cross-cutting quantities propagated into sub-configs (§4.1)
    assert cfg.segmenter.n_classes == 8
    assert cfg.gnn.n_classes == 8 and cfg.gnn.n_relations == cfg.graph.n_relations
    assert cfg.head.n_classes == 8
    assert cfg.backbone.image_size == 70
    assert "n_classes" in field_names(cfg)
    assert scalpel.default_device() in {"mps", "cuda", "cpu"}


def check_poolers():
    cfg = tiny_cfg()
    b, g, d = 2, cfg.grid_size, cfg.embed_dim
    grid = torch.randn(b, g, g, d)
    bb = MockBackbone(cfg.backbone)
    centers = bb.patch_centers()
    assert centers.shape == (g * g, 2)
    # row-major: first center is the top-left patch (~ p/2, p/2)
    assert torch.allclose(centers[0], torch.tensor([7.0, 7.0]))
    q = torch.tensor([[35.0, 35.0], [10.0, 60.0]])

    gp = perception.GaussianPool(cfg.point)
    assert gp(grid, centers, q).shape == (b, d)
    pca = perception.PinCrossAttention(cfg.point, d, cfg.image_size)
    assert pca(grid, centers, q).shape == (b, d)
    for mode in ("attn", "gauss"):
        pc = perception.PointConditioner(cfg.point, d, cfg.image_size, mode=mode)
        out = pc(grid, centers, q)
        assert out.shape == (b, d) and torch.isfinite(out).all()


def check_segmenter():
    cfg = tiny_cfg()
    b, g, d, c = 2, cfg.grid_size, cfg.embed_dim, cfg.n_classes
    seg = perception.PatchSegmenter(cfg.segmenter, d)
    grid = torch.randn(b, g, g, d)
    assert seg(grid).shape == (b, g, g, c)
    dm = seg.dense_map(grid, out_size=cfg.image_size)
    assert dm.shape == (b, cfg.image_size, cfg.image_size)
    assert dm.dtype == torch.long and int(dm.min()) >= 0 and int(dm.max()) < c


def _medial_lateral_map():
    """Two adjacent regions: a central column (label 1) and a lateral block (2)."""
    lab = np.zeros((60, 60), dtype=np.int64)
    lab[10:50, 24:36] = 1  # central, centroid x ~ 29.5 (axis = 30)
    lab[20:40, 36:54] = 2  # lateral, centroid x ~ 44.5
    return lab


def check_scene_graph_relation_direction():
    """§8.3: the central structure must serialise as medial/deep, never lateral."""
    cfg = tiny_cfg()
    g = scene_graph.build_scene_graph(_medial_lateral_map(), cfg.graph)
    assert len(g) == 2
    labels = g.labels
    a = labels.index(1)  # central
    b = labels.index(2)  # lateral
    edges_a = dict_from_edges(g.edges[a])
    edges_b = dict_from_edges(g.edges[b])
    # central -> lateral reads "central is medial to lateral" (and deep)
    assert scene_graph.REL_MEDIAL in edges_a.get(
        b, set()
    ), "central must be medial to lateral"
    assert scene_graph.REL_LATERAL not in edges_a.get(
        b, set()
    ), "central must NOT be lateral (§8.3)"
    assert scene_graph.REL_DEEP in edges_a.get(
        b, set()
    ), "central (image centre) must be deep"
    assert scene_graph.REL_LATERAL in edges_b.get(
        a, set()
    ), "lateral must be lateral to central"
    # serialisation reads consistently
    names = ["bg", "biceps", "humerus", *[f"s{i}" for i in range(6)]]
    text = llm_reasoner.serialise_neighbourhood(g, a, names)
    assert "medial" in text and "lateral" not in text, f"bad serialisation:\n{text}"


def check_scene_graph_surrounds():
    cfg = tiny_cfg()
    lab = np.zeros((60, 60), dtype=np.int64)
    lab[15:45, 15:45] = 1  # ring (frame)
    lab[22:38, 22:38] = 3  # centre filling the ring's hole
    lab[20:40, 20:42] = np.where(lab[20:40, 20:42] == 3, 3, lab[20:40, 20:42])
    # carve the hole so 1 is a frame around 3
    lab[20:40, 20:40][(lab[20:40, 20:40] != 3)] = 1
    lab[22:38, 22:38] = 3
    g = scene_graph.build_scene_graph(lab, cfg.graph)
    ring = g.labels.index(1)
    centre = g.labels.index(3)
    er = dict_from_edges(g.edges[ring])
    ec = dict_from_edges(g.edges[centre])
    assert scene_graph.REL_SURROUNDS in er.get(centre, set()), "ring surrounds centre"
    assert scene_graph.REL_SURROUNDED_BY in ec.get(
        ring, set()
    ), "centre surrounded_by ring"
    # pin lookup
    assert scene_graph.pin_region_index(g, (30, 30)) == centre
    assert scene_graph.pin_region_index(g, (16, 16)) == ring
    assert scene_graph.pin_region_index(g, (0, 0)) is None


def check_relational_gnn():
    cfg = tiny_cfg()
    g = scene_graph.build_scene_graph(_medial_lateral_map(), cfg.graph)
    n, d = len(g), cfg.embed_dim
    region_embeds = torch.randn(n, d)
    node_features, adj = to_tensors(g, region_embeds, cfg.graph.n_relations, "cpu")
    assert node_features.shape == (n, d + cfg.gnn.geom_dim)
    assert adj.shape == (cfg.graph.n_relations, n, n)
    # row-normalised: each (relation, row) sums to 0 or 1
    rsum = adj.sum(dim=2)
    assert torch.all((rsum == 0) | torch.isclose(rsum, torch.ones_like(rsum)))

    gnn = RelationalGNN(cfg.gnn, cfg.embed_dim)
    logits = gnn(node_features, adj)
    assert logits.shape == (n, cfg.n_classes)
    assert gnn.predict_node(node_features, adj, 0).shape == (cfg.n_classes,)


def check_heads():
    cfg = tiny_cfg()
    d, c = cfg.embed_dim, cfg.n_classes
    # prototypical head: filled count == unique labels; unfilled -> -inf
    ph = heads.PrototypicalHead(cfg.head, d)
    assert ph.filled == 0
    embeds = torch.randn(20, d)
    labels = torch.randint(0, 5, (20,))  # only classes 0..4 present
    ph.fit(embeds, labels)
    assert ph.filled == int(labels.unique().numel())
    logits = ph(torch.randn(3, d))
    assert logits.shape == (3, c)
    assert torch.isinf(logits[:, 6]).all() and (logits[:, 6] < 0).all()  # class 6 empty

    # temperature scaling: ECE reported, T finite (§2.7)
    ts = heads.TemperatureScaler(cfg.head)
    lg = torch.randn(200, c) * 3.0
    yy = torch.randint(0, c, (200,))
    info = ts.fit(lg, yy)
    assert {"T", "ece_before", "ece_after"} <= set(info)
    assert np.isfinite(info["T"]) and info["T"] > 0
    assert torch.allclose(ts(lg), lg / ts.T)

    # PoE fuse normalised; decide returns the contract dict
    poe = heads.ProductOfExperts(cfg.head)
    la = torch.log_softmax(torch.randn(c), dim=-1)
    lr = torch.log_softmax(torch.randn(c), dim=-1)
    fused = poe.fuse(la, lr)
    assert abs(float(fused.exp().sum()) - 1.0) < 1e-5
    dec = poe.decide(fused)
    for k in ("pred", "prob", "entropy_bits", "abstain", "topk", "topk_prob"):
        assert k in dec, f"decide() missing {k}"
    assert isinstance(dec["abstain"], bool) and len(dec["topk"]) == 5


def check_llm_reasoner():
    cfg = tiny_cfg()
    g = scene_graph.build_scene_graph(_medial_lateral_map(), cfg.graph)
    names = ["bg", "biceps", "humerus", *[f"s{i}" for i in range(6)]]
    pin_idx = 0
    # serialisation + prompt
    text = llm_reasoner.serialise_neighbourhood(g, pin_idx, names)
    assert isinstance(text, str) and "pin is on" in text.lower()
    # §8.9: reason(shortlist=) must reach build_prompt(candidate_shortlist=)
    prompt = llm_reasoner.build_prompt(
        g, pin_idx, names, candidate_shortlist=["biceps", "humerus"]
    )
    assert "biceps" in prompt and "humerus" in prompt
    # mark_pin returns an image
    marked = llm_reasoner.mark_pin(rand_image(), (10, 10))
    assert marked.size == (120, 100)
    # full reason() with stub backend; shortlist forwarded
    reasoner = llm_reasoner.LLMReasoner(label_names=names)
    out = reasoner.reason(rand_image(), (30, 30), g, pin_idx, shortlist=["biceps"])
    assert {"raw", "answer", "confidence"} <= set(out)
    assert out["answer"] == "unknown" and out["confidence"] == 0.0


def check_loops_and_pipeline():
    pipe = make_pipe()
    cfg = pipe.cfg
    S, g, d, c = cfg.image_size, cfg.grid_size, cfg.embed_dim, cfg.n_classes

    # synthetic_pretrain over a tiny fake loader
    def loader():
        for _ in range(3):
            yield {
                "images": torch.randn(4, 3, S, S),
                "pins": torch.rand(4, 2) * S,
                "labels": torch.randint(0, c, (4,)),
                "seg_maps": torch.randint(0, c, (4, g, g)),
            }

    hist = loops.synthetic_pretrain(pipe, loader(), epochs=2, lr=1e-3)
    assert len(hist["app_acc"]) == 2 and len(hist["seg_loss"]) == 2

    # pretrain_gnn over fake graphs
    def graphs():
        for n in (3, 4):
            yield {
                "node_features": torch.randn(n, d + cfg.gnn.geom_dim),
                "adj": torch.rand(cfg.graph.n_relations, n, n),
                "labels": torch.randint(0, c, (n,)),
            }

    ghist = loops.pretrain_gnn(pipe, graphs(), epochs=2, lr=1e-3)
    assert len(ghist["node_acc"]) == 2

    # fewshot_adapt from a gallery dict (no grad)
    gal = {"embeds": torch.randn(30, d), "labels": torch.randint(0, c, (30,))}
    info = loops.fewshot_adapt(pipe, gal)
    assert info["filled"] == int(gal["labels"].unique().numel())
    assert pipe.proto_head.filled == info["filled"]

    # end-to-end predict
    dec = pipe.predict(rand_image(), (40, 50))
    for k in ("pred", "prob", "entropy_bits", "abstain", "topk", "topk_prob"):
        assert k in dec
    assert 0.0 <= dec["prob"] <= 1.0 and isinstance(dec["abstain"], bool)

    # evaluate over a few triples -> metrics + curve
    test_set = [
        (
            rand_image(),
            (np.random.randint(0, 120), np.random.randint(0, 100)),
            np.random.randint(0, c),
        )
        for _ in range(6)
    ]
    metrics = loops.evaluate(pipe, test_set)
    for k in ("n", "coverage", "selective_accuracy", "ece", "topk_accuracy", "curve"):
        assert k in metrics
    assert metrics["n"] == 6 and 0.0 <= metrics["coverage"] <= 1.0
    assert len(metrics["curve"]) == 21

    # state_dict excludes the frozen backbone and round-trips
    sd = pipe.state_dict()
    assert not any(
        k.startswith("backbone.") for k in sd
    ), "backbone must be excluded from state_dict"
    pipe2 = make_pipe()
    pipe2.load_state_dict(sd)
    assert pipe2.proto_head.filled == pipe.proto_head.filled


def check_lazy_imports_final():
    # after exercising everything, the heavy synthetic dep must still be absent
    assert "open3d" not in sys.modules, "open3d leaked into the smoke path (§8.8)"


# --------------------------------------------------------------------------- #
# helpers + runner                                                            #
# --------------------------------------------------------------------------- #
def dict_from_edges(edge_list):
    d: dict[int, set] = {}
    for j, rel in edge_list:
        d.setdefault(j, set()).add(rel)
    return d


CHECKS = [
    check_config,
    check_poolers,
    check_segmenter,
    check_scene_graph_relation_direction,
    check_scene_graph_surrounds,
    check_relational_gnn,
    check_heads,
    check_llm_reasoner,
    check_loops_and_pipeline,
    check_lazy_imports_final,
]


def main() -> int:
    passed, failed = 0, 0
    for chk in CHECKS:
        try:
            chk()
            print(f"  PASS  {chk.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {chk.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{'=' * 48}\nM0 smoke: {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
