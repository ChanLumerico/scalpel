# SCALPEL 🔪

**S**ynthetic-to-real **C**adaveric **A**natomy **L**ocalization via
**P**oint-conditioned **E**xpert **L**earning.

A point-conditioned, scene-graph reasoning model for gross-anatomy spot exams
(*땡시*). SCALPEL estimates `p(y | I, q)` — the structure `y` under a pin at pixel
`q` in image `I` — as a **Product of Experts** over:

- an **appearance expert** — frozen DINOv2 → point-conditioned pooling →
  prototypical few-shot head, and
- a **relational expert** — patch segmentation → typed scene graph → R-GCN,

then calibrates the fused distribution with temperature scaling and **abstains**
when unsure. The full design rationale lives in [`HANDOUT.md`](HANDOUT.md).

## Status — milestone M0 ✅

The complete package scaffold and interface contracts (handout §4) are in place,
and the smoke test passes with **zero external assets**:

```bash
python tests/smoke_test.py        # M0 acceptance: backbone mocked, all modules threaded
```

Milestone map (handout §6): **M0 scaffold ✅** → M1 synthetic render → M2
appearance pretrain → M3 relational expert → M4 PoE + calibration → M5 few-shot
adapt → M6 evaluation → M7 (optional) MLX + LLM.

## Package layout

| Module | Responsibility |
|---|---|
| `scalpel/config.py` | All hyperparameters as dataclasses; `default_device()` (mps>cuda>cpu) |
| `scalpel/perception.py` | Frozen `DinoBackbone`, point poolers (`GaussianPool`, `PinCrossAttention`), `PatchSegmenter` |
| `scalpel/scene_graph.py` | Typed scene graph (`build_scene_graph`, `pin_region_index`) |
| `scalpel/relational_gnn.py` | R-GCN relational expert + `to_tensors` |
| `scalpel/heads.py` | `PrototypicalHead`, `TemperatureScaler`, `ProductOfExperts` |
| `scalpel/llm_reasoner.py` | Optional frozen-LM reasoning layer (Set-of-Mark / graph prompt) |
| `scalpel/synth.py` | Label codec, domain randomization, `SyntheticRenderer` (Open3D, lazy) |
| `scalpel/pipeline.py` | `ScalpelPipeline.predict` — end-to-end |
| `scalpel/loops.py` | `synthetic_pretrain`, `pretrain_gnn`, `fewshot_adapt`, `evaluate` |

Heavy entry points are exposed via PEP 562 lazy imports, so importing `scalpel`
or a lightweight submodule never pulls in torch.hub / Open3D (handout §4.10, §8.7).

## Install

```bash
pip install -r requirements.txt          # core: torch, torchvision, numpy, scipy, pillow
# pip install open3d                      # for M1+ synthetic rendering (Apple-Silicon friendly)
```

Target hardware: Apple Silicon (`mps`); the bottleneck is setup, not compute
(handout §9).

## Data sources, licenses & attribution

SCALPEL trains on synthetic renders, adapts on a small real gallery, and is
evaluated on a held-out real benchmark (handout §5). **Respect every license and
treat donor imagery with dignity.**

| Source | Use | License / terms |
|---|---|---|
| **BodyParts3D / Z-Anatomy** | Synthetic meshes (training) | CC BY-SA — **attribution required**; share-alike |
| **Visible Human Project** (NLM) | Real cadaver cross-sections | Public domain |
| **BlueLink** (Univ. of Michigan) | Real photos + QuizLink (eval only) | Not CC — educational/non-commercial use, attribution, author notification |

**Not used** (copyright risk): commercial atlases (Netter / Sobotta /
Photographic Atlas) and Internet Archive scans. Most permitted sources are
non-commercial; re-verify terms before any commercial distribution.

**Evaluation discipline:** BlueLink **QuizLink** is the closest match to the real
exam format and is used as the **test set only** — never for training (handout
§2.10). Train/val/test splits are made at the **specimen** level, not the image
level, because triples from one photo are strongly correlated (handout §5.2).

## License

Code: TBD by the author. Note the CC BY-SA share-alike obligation that attaches
to any redistributed BodyParts3D/Z-Anatomy-derived assets.
