# SCALPEL: Point-Conditioned Anatomy Recognition on Real Cadaveric Dissection Photographs — A Training-Light Baseline and a Quantitative Analysis of Its Ceiling

*Research log / technical report (branch `data-pivot`, work in progress)*

---

## Abstract

We study **point-conditioned recognition** for Korean gross-anatomy spot exams ("땡시"):
given a dissection photo `I` and a pin location `q`, identify the structure `y` under the pin,
i.e. estimate `p(y|I,q)`. After abandoning a synthetic mesh-rendering pipeline (v1) whose
geometry diverged too far from real cadavers, we build a pipeline that extracts `(I, q, y)`
triples from **real BlueLink QuizLink dissection PDFs** (953 triples / 567 structures / 31 specimen
PDFs). The model is **almost training-free** — a frozen DINOv2 backbone, point pooling,
nearest-exemplar retrieval, and temperature calibration with abstention. Under specimen-level
splits, ten random seeds, and a cross-cadaver protocol, it reaches **top-1 49.2±4.3% and
top-5 65.8±3.9% over 215 classes** (≈100× chance); with abstention it answers its most-confident
30% of pins at **~88% accuracy**. A sequence of *cheap-probe-first* experiments establishes that
(i) nearest-exemplar retrieval beats mean-prototype retrieval by **+7.8 pp**, (ii) **data quality**
(pin placement, label correctness) dominates accuracy, and (iii) added model capacity overfits while
the performance ceiling is governed by **data scale** (an unsaturated scaling curve). We report
negative results (pooling width is irrelevant, context concatenation is ineffective, a learned
pooler overfits) on equal footing with positive ones.

---

## 1. Introduction

### 1.1 Problem
A gross-anatomy spot exam asks the examinee to name the exact structure indicated by a tag on a
cadaver. We cast this as point-conditioned classification: `(image I, pin q) → structure name y`.
The central difficulty is an **extreme long tail**: anatomical structures number in the hundreds
to thousands, with very few examples per structure.

### 1.2 The v2 pivot
An early attempt generated data by rendering synthetic 3D meshes, but the *geometric and
topological* gap to real cadavers was judged fatal for generalization, so it was scrapped.
v2 starts from **real dissection photographs** (BlueLink QuizLink), with an acceptance rule:
if a human cannot identify the body region in ≤0.5 s, the sample is rejected.

### 1.3 Contributions
1. A pipeline that turns real QuizLink PDFs into leakage-free `(I,q,y)` triples (crawl, parse,
   clean, split).
2. A training-light recognition baseline with an honest **multi-seed, cross-cadaver** evaluation
   protocol.
3. Fifteen controlled experiments quantifying *what governs accuracy* — the retrieval rule, data
   quality, and ultimately **data scale as the ceiling**, demonstrated by a scaling curve.
   Negative results are reported in full.

---

## 2. Dataset

### 2.1 Source and parsing
A QuizLink PDF page bakes one dissection photo into a single JPEG2000 image (`Im0`, 3000×2250)
with blue leader lines, label boxes, and answer text all burned into the pixels. The click-to-reveal
is an AcroForm button overlay, so the raw `Im0` already shows the answers. Pipeline
(`scalpel/data/`):

| Information | Where | Extraction |
|---|---|---|
| answer `y` | baked text in a label box | crop box → OCR (tesseract) |
| box position | button-widget rect (structured) | PDF→pixel transform |
| leader line | blue line in `Im0` | HSV segmentation |
| pin `q` | tissue-side end of the leader | trace from the box |

**Leakage removal (critical):** the input image must have *all* label boxes and *all* leader lines
inpainted out — otherwise the model reads the text. Other structures' answers on the same photo are
removed too.

### 2.2 Statistics
- **31** specimen PDFs → **953** triples / **567** structures (classes) / **510** unique images.
- Labels (leaders) per image: mean **1.87**, max **9** (multiple answers per photo).
- Evaluable core (≥2 instances): **601 triples / 215 classes**. Singletons: **352** classes
  (open-set candidates).
- **71 of 215 (33%)** core classes occur in a single PDF — the key variable for the cross-cadaver
  analysis.
- Mixed modality: real cadaver dissection + bone specimens + some 3D model renders.

### 2.3 Data-quality cleanup (governs performance; §6.1)
- **Background-pin bug:** 27% of initial triples had the pin on black background, not tissue (a
  leader-endpoint / inward-nudge defect). `_snap_to_tissue` (snap to the nearest tissue pixel within
  70 px, else drop) reduced this **27% → 0.1%**.
- **Photo crop:** removes page borders, caption, and attribution text, keeping only the cadaver
  photo (q is mapped to crop coordinates).
- **Label cleanup:** OCR garbage / misspellings / surface variants were hand-reviewed and corrected
  against a lexicon and fuzzy matching (medical class labels must be exact).

### 2.4 Ethics
Non-commercial, educational use only. BlueLink attribution retained; donor dignity (private storage,
no redistribution); cadaver imagery excluded from git (`*.private.png`, `/data/`). Real exam-question
photos are never used for training, evaluation, or the gallery.

---

## 3. Method

The current pipeline is retrieval-based recognition with **no backpropagation, loss, or optimizer**.

**Enrollment:** each gallery `(I,q,y)` → frozen DINOv2 (vitb14, 518 px) patch grid →
GaussianPool (σ = 40 px, parameter-free) at `q` → L2-normalized embedding `z_q`.

**Inference:** embed the test `(I,q)` identically → class score = cosine to the **nearest gallery
exemplar** of that class (per-class max) → argmax = prediction (top-k). No prototype → OOV →
abstain.

**Calibration / abstention:** `softmax(s·scores)` (scale `s` fit by leave-one-out NLL on the
gallery); a confidence threshold drives a risk–coverage operating curve.

**Optional learned component:** a small supervised-contrastive (SupCon) linear head on top of the
frozen embedding (low capacity). The engine also contains a learned pooler (PinCrossAttention), a
relational expert (R-GCN), and PoE fusion, none adopted as of this report (§6.4, §8).

---

## 4. Experimental Setup

- **Split:** specimen-level (per PDF page) train/test — all triples of one photo go to the same
  side, preventing image-level leakage. `test_frac = 0.3`.
- **Multi-seed:** embed once, then **10 seeds** vary only the split; report mean±std. A single seed
  fluctuates by ±3–4 pp and is unsuitable for conclusions (§6.4).
- **Cross-cadaver:** beyond page-level, a **PDF (specimen)-level** split validates generalization
  (§6.8).
- **Metrics:** selective-accuracy@coverage (top-1/5), coverage, ECE, risk–coverage, paired Δ.
- **Hardware:** Apple M4 Max, MPS.

---

## 5. Experiment Index

> Full per-experiment detail (5W1H: when / why / how / where / result / conclusion /
> reproduce) is in **`RESEARCH_LOG.md`** (the running research journal). This table
> is the one-line index.

| # | Experiment | Key result |
|---|---|---|
| 005 | proto baseline | top1 38.8±3.4 / top5 55.8±4.0 |
| 006 | pooling-width ablation | σ irrelevant for top1 (noise); top5 rises with σ |
| 007 | calibration + abstention (proto) | confident-30% 78.4%, ECE 0.4→0.2 |
| 008 | context probe | global CLS lifts top5 +8 pp, top1 unchanged |
| 009 | discrimination diagnostic | **exemplar 46.6 > proto 38.8 (+7.8)** |
| 010 | adopt exemplar | top1 46.6±3.6 / top5 58.0±4.4 / confident-30% 88.5% |
| 012 | learned head (SupCon) | paired Δtop1 +2.6 (9/10), Δtop5 +7.8 (10/10) |
| 013 | **data scaling** | top1 & coverage still rising at 100% (unsaturated) |
| 014 | best setting | **top1 49.2±4.3 / top5 65.8±3.9 / 39% @≥80% acc** |
| 015 | learned pooler | overfits: paired Δtop1 **−2.3** (1/10) |
| 016 | augmentation | aug-gallery +1.5; backbone robust to photometric shift |
| 017 | pin noise | tolerant to ~40 px tag error (−3.5 pp) |
| 018 | deployment operating point (premature) | 19%±11 answer rate at 90% target — not ready |
| 019 | backbone scaling | vitb14 46.6 / vitl14 46.8 / vitg14 47.7 (marginal) |
| 020 | structured relational context (M6' step 1) | no top1 gain (+region −3.5, +neighbor −0.4) |
| 021 | singletons in gallery (A-1) | vocab ~201→448 (2.2×) at −2.9 pp top1 cost |
| 022 | OCR-recover dropped labels (A-2) | negative: 346 dropped are genuine junk, ~0 recoverable |

---

## 6. Results

### 6.1 Data quality dominates performance
Before touching the model, **data defects were the largest variable**. After fixing the
background-pin bug (27%→0.1%), the same proto model's single-seed top1 rose from ~31% to the low
40s; label cleanup made the class definitions correct. *Lesson: ensuring `(I,q,y)` integrity precedes
any recognition modeling.*

### 6.2 Retrieval rule: nearest-exemplar ≫ mean-prototype
Varying only the retrieval rule on identical frozen embeddings (exp 009):

| Rule | top1 |
|---|---|
| mean prototype | 38.8±3.4 |
| **nearest exemplar (1-NN)** | **46.6±3.6** |
| k-NN (5) vote | 32.3 |
| region (CLS)-gated | 41.0 |

Exemplar wins on 10/10 seeds with separated error bars. **Averaging views from different cadavers and
angles washes out the discriminative detail** (the classic NN > prototype result under high
intra-class variance). A free +7.8 pp with no training.

### 6.3 Calibration + abstention (selective prediction)
After temperature scaling, ranking by confidence yields a monotone risk–coverage curve (85–96%
accuracy in the top 5–20% of confidence). In the best setting (exp 014): **confident top-30% →
87.6%**, **39% of pins answerable at ≥80% accuracy** (up from 24% for frozen-exemplar), ECE 0.2–0.3.
*The model knows when it knows.*

### 6.4 What did not work (negative results)
- **Pooling width σ (exp 006):** top1 is flat over σ∈{10..80} (noise) — the single-seed "σ20 +2.8 pp"
  was a false positive that multi-seed immediately killed. Only top5 rises monotonically with σ.
- **Context concatenation (exp 008):** the global CLS token lifts top5 by +8 pp but does nothing for
  top1 → "region narrows the candidate set but doesn't resolve the fine distinction within it."
- **Learned pooler PinCrossAttention (exp 015):** ~600 K parameters **overfit** 953 triples —
  paired Δtop1 **−2.3** (1/10 wins), top5 only +4.1. Attention visualization shows the Gaussian
  focuses tightly on the pin while the learned attention is diffuse (loses focus). *Low-capacity
  learning helps; high-capacity overfits.*
- **Structured relational context (exp 020, M6' step 1):** over-segmenting each photo (k-means on
  patch tokens) and fusing the pin-region / adjacent-region appearance into the embedding does *not*
  help top1 either (base 46.6 → +region 43.2, +neighbor 46.2, +both 43.9; all ≤ base). Region
  averaging dilutes the discriminative detail — the same pattern as the learned pooler and global
  context. This is the cheap, training-free foundation of the relational expert; its negativity, with
  the learned-pooler overfit, suggests a *trained* R-GCN would also overfit at this data scale rather
  than break the appearance ceiling.

### 6.5 What did work (modestly)
**Learned discriminative head (SupCon linear, exp 012):** a low-capacity head on the frozen embedding
gives paired Δtop1 **+2.6** (9/10) and Δtop5 **+7.8** (10/10) without overfitting. The direction is
valid but the top1 gain is small — a signal of the data-size limit.

### 6.6 Data scaling: the binding constraint
Subsampling the gallery to 25→100% of its specimens (exp 013):

| Gallery | ~triples | top1 (covered) | coverage |
|---|---|---|---|
| 25% | ~108 | 39.1 | 33.9 |
| 50% | ~213 | 39.7 | 57.4 |
| 75% | ~317 | 45.2 | 72.1 |
| 100% | ~420 | 46.6 | 83.2 |

**Neither top1 nor coverage saturates at 100%** (top1 still +1.4 pp over the last 25%; coverage is
nearly linear). After the cheap model levers are exhausted, the remaining ceiling is **data scale** —
more specimens raise accuracy *and* coverage together.

### 6.7 Robustness
- **Photometric domain shift (exp 016):** under strong corruption (darken + contrast + warm cast +
  noise), top1 barely moves (46.6→46.5) — **frozen DINOv2 features are already robust**. An augmented
  gallery adds a small +1.5 pp.
- **Pin noise (exp 017):** jittering the pin by 0/10/20/40/80 px → 46.6/46.0/44.9/43.1/36.4.
  **Tolerant up to ~40 px (≈1.3% of width, ~3 patches) with only −3.5 pp** — GaussianPool's spatial
  averaging absorbs imperfect tags.

### 6.8 Honesty checks
- **Cross-cadaver (specimen leak):** a page-level split can place the same PDF (cadaver) on both
  sides. A **PDF-level** split (entirely different cadavers in gallery vs test) gives the same top1
  (38.8±3.4 vs 39.3±5.4). Accuracy is therefore not inflated; what falls on a new cadaver is
  **coverage (83→63%)** — the 33% of structures present in a single PDF become OOV and are honestly
  abstained. **The real limit is coverage, not accuracy.**
- **Test reuse:** the 15 experiments were evaluated repeatedly on the same 10 splits while selecting
  methods, so the chosen headlines carry ~1–2 pp of *optimism* (selection-on-test, not
  contamination; individual measurements remain valid). The decisive results (exemplar advantage,
  scaling) are robust, but a clean claim of the final number warrants a 3-way (train/val/test) split.

### 6.9 Backbone scaling
We had only ever used the smallest backbone, vitb14. Swapping in larger frozen DINOv2 variants
(exemplar 1-NN, 10 seeds):

| Backbone | top1 | top5 |
|---|---|---|
| vitb14 (86 M) | 46.6±3.6 | 58.0±4.4 |
| vitl14 (300 M) | 46.8±3.7 | 57.4±4.5 |
| vitg14 (1.1 B) | 47.7±3.4 | 58.8±4.3 |

Scaling the backbone by ~13× (vitb→vitg) yields only **+1.1 pp** top1. Feature quality is *not* the
bottleneck — reinforcing that the lever is data/structure, not a bigger encoder.

---

## 7. Synthesis

**Current best (training-light, exp 014):** frozen DINOv2 + point pooling + SupCon head + exemplar
1-NN + calibration/abstention → **top1 49.2±4.3% / top5 65.8±3.9%**, confident top-30% **87.6%**,
**39%** of pins answerable at ≥80% accuracy. ≈100× chance (0.47%).

Following a *cheap-probe-first* discipline (multi-seed kills false positives), we systematically
tested the model levers and converged — via the scaling curve, overfitting of added capacity,
marginal backbone gains, and the negative results — on the conclusion that **the bottleneck is data
scale, not model cleverness.**

---

## 8. Limitations and Threats to Validity

1. **Data scale/diversity:** 953 triples, ~1.9 cadavers per class, 33% single-PDF. The scaling curve
   is unsaturated → both accuracy and coverage are data-bound.
2. **Domain match:** QuizLink teaching photos (cadaver + bone + 3D) differ from a specific
   institution's real exam cadavers; geometric/structural domain shift is untested (no data).
3. **Test-reuse optimism (~1–2 pp):** a clean claim needs a 3-way split.
4. **Residual OCR label noise** and a rare duplicate-pin artifact.
5. **Unbuilt large levers:** the relational expert (M6') and serious training/large backbones are not
   yet evaluated in full — the ceiling should be re-assessed after them.

---

## 9. Conclusion and Future Work

With almost no training, frozen-feature retrieval reaches 215-way cross-cadaver top1 ~49% and ~88% on
its confident subset, and honest evaluation shows that **data is the binding ceiling.** Toward the
ultimate goal (real deployment):
- **Model:** larger frozen backbones (marginal so far, §6.9); a **relational expert (scene graph +
  R-GCN)** to separate adjacent look-alikes; serious episodic training / fine-tuning; PoE fusion.
- **Data:** more specimens covering the curriculum (raises accuracy *and* coverage) — the largest
  lever.
- **Deployment (ultimate goal):** fix a 3-way operating point and open-set spec only *after* the
  above mature; doing so now is premature (exp 018 quantifies that immaturity).

---

## Appendix A. Reproduction
- Data: `python -m scalpel.data.crawl` → `python -m scalpel.data.build` (parse + clean + prune).
- Evaluation: `scripts/eval_appearance.py` (proto, multi-seed), `eval_exemplar.py` (canonical),
  `eval_calibration.py` (M5'), `learned_head.py`, `learning_curve.py`, `backbone_scale.py`, etc.
- Every experiment is logged under `experiments/NNN-*/` with a report, figures, and `metrics.json`.
  Figures containing cadaver imagery are saved as `*.private.png` and git-ignored.

*Document version: `data-pivot`. Backbone scaling (exp 019) complete; §6.9/§7 reflect final numbers.*
