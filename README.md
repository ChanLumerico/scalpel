# SCALPEL 🔪 (v2 — real-data pivot)

**S**ynthetic-to-real **C**adaveric **A**natomy **L**ocalization via
**P**oint-conditioned **E**xpert **L**earning.

A point-conditioned recognition model for gross-anatomy spot exams (*땡시*):
given a dissection photo `I` and a pin coordinate `q`, identify the structure
`y` under the pin — `p(y | I, q)` — or **abstain** (with top-k).

> **v2 note:** the name keeps "Synthetic", but from v2 the data is **real, not
> synthetic**. The synthetic mesh-render pipeline (BodyParts3D / Z-Anatomy /
> Open3D) is **removed** — it produced images a human couldn't even orient in
> (garbage in → garbage out). Data now comes from **real BlueLink QuizLink
> dissection PDFs**. The recognition engine is modality-agnostic and unchanged.

## Why the pivot

Surface mesh renders have no dissection-depth axis and lost the region's shape
and topology — fragments on a void that neither a human nor the relational
expert could read. Domain randomization fixes *appearance*, not *missing shape*.
So instead of *generating* shape (and failing), v2 starts from real photos where
shape is **already guaranteed**. **Acceptance rule:** if a human can't tell the
body region in ≤0.5 s, the data is rejected.

## Architecture

```
scalpel/                      recognition engine (kept, modality-agnostic)
  config.py                   hyperparameters (PipelineCfg; SynthCfg removed)
  perception.py               frozen DINOv2 + point poolers + PatchSegmenter
  scene_graph.py              typed scene graph
  relational_gnn.py           R-GCN relational expert
  heads.py                    PrototypicalHead, TemperatureScaler, ProductOfExperts
  pipeline.py                 ScalpelPipeline.predict
  loops.py                    fewshot_adapt, evaluate, (re)pretrain hooks
  data/                       ★ NEW real-data pipeline (replaces synth.py)
    crawl.py                  QuizLink fetch  (Playwright headless + gdown)
    parse.py                  QuizLink PDF -> (clean_I, q, y)   ★ core
    augment.py                photometric augmentation (q follows the transform)
    loader.py                 specimen-level splits -> few-shot gallery / testset
    vocab.py                  label normalization + closed vocabulary 𝒱
```

The MVP is the **appearance expert alone** (DINOv2 frozen + GaussianPool +
prototypical few-shot — almost no training). The relational expert returns later
on SSL over-segmentation (DINOv2 patch clustering) weakly supervised by the
sparse pin labels; PoE fusion + temperature calibration + abstention stay.

## Data: BlueLink QuizLink (verified PDF structure)

Each QuizLink PDF page = one dissection photo as a single baked JPEG2000 image
`Im0` (3000×2250) containing the photo + blue leader lines + label boxes + the
**answer text**, all baked into pixels. Click-to-reveal is an AcroForm button
overlay that *masks* the answer — so extracting the raw `Im0` already shows the
answer (no click simulation needed). Parsing:

| info | where | how |
|---|---|---|
| answer `y` | baked text in a label box | crop box → **OCR** |
| box position | **button widget rect** (structured) | PDF→`Im0` pixel transform |
| leader line | blue line in `Im0` (solid=point, dashed=region) | **HSV segmentation** |
| pin `q` | tissue-side end of the leader (or region centroid) | trace from box |

**Critical (label leak):** the model input `clean_I` must have the answer text
**and** leader lines removed/inpainted, or the model cheats by reading the text.

**Crawl reality:** the QuizLink index renders Drive links only via JavaScript
(static fetch returns navigation only — verified twice), so a **headless
browser (Playwright) is required**; the ~95 MB Drive PDFs need a token-handling
downloader (**gdown**). Polite policy: concurrency 1, ≥2 s between requests,
cache, robots.

## Install

```bash
pip install -r requirements.txt
playwright install chromium     # one-time, for crawl
# parse needs the tesseract binary: brew install tesseract
python tests/smoke_test.py      # M0 acceptance (engine; zero external assets)
```

## Milestones (v2)

`M0` scaffold+smoke ✅ → **WIPE** (remove synth) → `M1'` crawl → `M2'` parse →
`M3'` loader/augment/vocab → `M4'` appearance MVP → `M5'` calibration+abstention
→ `M6'` relational expert (later).

## Ethics & license (hard requirements)

Personal, **non-commercial educational** use only — the range BlueLink permits.

- **Attribution (kept in all outputs/README):** *BlueLink, © B. Kathleen Alsup
  & Glenn M. Fox, University of Michigan — used for non-commercial educational
  purposes.*
- **Notify authors** (recommended): `bluelinkanatomy@gmail.com`.
- **Polite crawl** (concurrency 1, ≥2 s, cache, robots), **donor dignity**
  (private storage, no redistribution, no commercial use), and **never use real
  exam-question photos** anywhere (train/eval/gallery).

Removed (copyright): commercial atlases. The synthetic-mesh sources
(BodyParts3D / Z-Anatomy, CC BY-SA) are no longer used in v2.
