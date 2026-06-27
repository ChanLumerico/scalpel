# SCALPEL — Detailed Research Log

This is the project's running research journal. Every experiment is recorded with
**When / Why (motivation & hypothesis) / What & How (method & setup) / Where (data
& conditions) / Result / Conclusion (what it led to) / Reproduce (script)**.
For the polished narrative see `PAPER.md`; for per-run Korean reports + figures see
`experiments/NNN-*/`.

**Standing dataset (unless noted):** 31 QuizLink specimen PDFs → 953 `(I,q,y)`
triples, 567 structures, 510 images. Evaluable core (≥2 instances): 601 triples /
215 classes. Backbone frozen `dinov2_vitb14` @518px; pin pooling = GaussianPool
σ=40 px; retrieval = cosine; split = specimen-level (per PDF page, leak-free);
reporting = **10-seed mean±std** (embed once, vary split). Hardware: Apple M4 Max
(MPS).

---

## Phase 0 — Data pipeline & quality (prerequisite to any modeling)

### D1 — Pin background-leak fix (snap-to-tissue)
- **When:** 2026-06-26.
- **Why:** Visual inspection of prediction montages showed pins sitting on black
  background, not tissue. Hypothesis: a non-trivial fraction of `q` is mislocated,
  which would poison both gallery prototypes and test queries (DINO features pooled
  at background are meaningless).
- **What & How:** Measured, for every triple, the brightness in a window around `q`;
  flagged "background" if near-black. Root-caused it (not the crop — verified — but
  leader endpoints / `_pull_inward` landing in dark gaps). Added `_snap_to_tissue`:
  if `q` is on near-black, snap to the nearest tissue pixel within 70 px; if none,
  **drop** the triple (an untrustworthy pin is worse than none).
- **Where:** All 31 PDFs, full parse.
- **Result:** Background-pin fraction **27.0% → 0.1%**. Verified visually that pins
  now land on the leader's tissue endpoint (deltoid/pec/serratus, etc.).
- **Conclusion:** Data integrity was the single largest lever before any modeling.
  Triple count 1182 → after rebuild settled to the cleaned set. Two parse bugs were
  also fixed here: `cv2.inpaint` could stall for minutes on bluish pages (capped at
  ≤1400 px + composite), and `_leader_pin` was O(n_components × pixels) and hung on
  noisy pages (vectorized: 12.8 s/box → 0.12 s/box).
- **Reproduce:** `scalpel/data/parse.py` (`_snap_to_tissue`); `python -m scalpel.data.build`.

### D2 — Label cleaning + canonicalization
- **When:** 2026-06-26.
- **Why:** OCR over the baked label boxes yields garbage ("ee", "ma"), misspellings
  ("aterventrieslar"=interventricular), and surface variants; medical class labels
  must be exact, and junk inflates the class count and starves classes of instances.
- **What & How:** `is_valid()` junk filter; greedy frequency-priority fuzzy
  clustering (precision-guarded so distinct structures never merge); plural/abbrev
  normalization (mm→muscles, nn→nerves, …); a curated single-word structure lexicon
  (tibia, uvula, …) so real one-word bones survive; then a **hand review** of OCR
  corrections.
- **Where:** All extracted triples.
- **Result:** 1606 raw → 953 clean triples; 567 distinct, well-formed classes.
- **Conclusion:** Clean labels are a precondition for a meaningful benchmark; the
  cleaning is integrated into `build` so the on-disk dataset is always cleaned.
- **Reproduce:** `scalpel/data/clean.py`, `vocab.py`.

---

## Phase 1 — Appearance baseline & ablations

### 005 — Proto baseline (multi-seed)
- **When:** 2026-06-26.
- **Why:** Establish a training-free reference: can frozen DINOv2 + point pooling +
  **mean-prototype** few-shot identify the pinned structure above chance?
- **What & How:** Embed core once; per seed build one mean prototype per class from
  the gallery; classify a test pin by nearest prototype (cosine). 10 seeds.
- **Where:** ≥2 core (215 classes), specimen split, test_frac 0.3.
- **Result:** **top1 38.8±3.4%, top5 55.8±4.0%, coverage 83%** (≈78× the 0.47%
  chance).
- **Conclusion:** The appearance hypothesis holds — but single-seed runs swing
  ±3-4 pp, so multi-seed mean±std becomes the standard (a single-seed "σ20 win" was
  later shown to be noise, exp 006).
- **Reproduce:** `scripts/eval_appearance.py`.

### 006 — Pooling-width ablation
- **When:** 2026-06-26.
- **Why:** Does the pooling radius σ (how much neighbourhood the point embedding
  sees) matter? A single-seed run had suggested σ=20 beats σ=40 by +2.8 pp.
- **What & How:** Cache DINO grids once; re-pool at bilinear and σ∈{10,20,40,60,80};
  exemplar/prototype eval; 10 seeds with error bars.
- **Where:** Core, as above.
- **Result:** **top1 flat across σ** (38.6–41.2, all error bars overlap) → the σ20
  "win" was noise. **top5 rises monotonically with σ** (54→59); bilinear lowest.
- **Conclusion:** Keep σ=40 (top1 σ-invariant; balances top1/top5). Multi-seed paid
  for itself by killing a false positive on first use.
- **Reproduce:** `scripts/ablate_pooling.py`.

### 007 — Calibration + abstention (on proto)
- **When:** 2026-06-26.
- **Why:** Raw accuracy is one number; deployment value comes from *answering only
  when confident*. Can we calibrate confidence and trade coverage for accuracy?
- **What & How:** Turn cosine sims into `softmax(s·sims)`; fit scale `s` by
  leave-one-out NLL on the gallery (no test leak); rank covered test pins by
  confidence; report the risk–coverage curve, ECE, and operating points. 10 seeds.
- **Where:** Core, proto retrieval.
- **Result:** **confident top-30% → 78.4±7.5%** (vs 38.8% answer-all); **ECE
  0.4→0.2** after calibration; s≈13. Risk–coverage is cleanly monotone.
- **Conclusion:** The model "knows when it knows"; abstention is the lever that makes
  it usable. This evaluation frame is reused for every later model.
- **Reproduce:** `scripts/eval_calibration.py`.

### 008 — Context probe
- **When:** 2026-06-26.
- **Why:** The errors are adjacent look-alikes; would *context* (region / global
  information) help? Cheap test before the heavy relational expert.
- **What & How:** Augment the point embedding with multi-scale pooling and the global
  CLS token (each sub-vector L2-normed, concatenated, re-normed). 4 variants, 10
  seeds.
- **Where:** Core, frozen exemplar.
- **Result:** top1 within noise (40.7–41.0 vs 38.8); **the global CLS token lifts
  top5 by +8 pp (64.0)**.
- **Conclusion:** Region context narrows the candidate set (top5) but does not
  resolve the fine top1 distinction — structured relation, not concatenation, would
  be needed.
- **Reproduce:** `scripts/probe_context.py`.

---

## Phase 2 — The retrieval-rule discovery

### 009 — Discrimination diagnostic
- **When:** 2026-06-26.
- **Why:** Is the top1 ceiling a *feature* limit or a *retrieval-rule* limit? Test
  training-free alternatives on identical embeddings.
- **What & How:** Compare mean-prototype vs nearest single exemplar (1-NN) vs k-NN(5)
  vote vs region(CLS)-gated retrieval. 10 seeds, paired.
- **Where:** Core, frozen embeddings.
- **Result:** **exemplar 46.6±3.6 vs proto 38.8 (+7.8 pp, 10/10 seeds)**; knn5 worse
  (32.3); region-gating marginal/unstable.
- **Conclusion:** Mean-prototype averaging washes out discriminative detail under
  high intra-class variance (different cadavers/angles, few shots) — the classic
  NN>prototype regime. A free +8 pp; adopt exemplar retrieval.
- **Reproduce:** `scripts/probe_discrimination.py`.

### 010 — Adopt exemplar (canonical) + calibration
- **When:** 2026-06-26.
- **Why:** Consolidate the exemplar win into the canonical model with the full
  profile (accuracy + abstention).
- **What & How:** Per-class max-exemplar score; temperature from gallery LOO;
  risk–coverage. 10 seeds.
- **Where:** Core.
- **Result:** **top1 46.6±3.6, top5 58.0±4.4, coverage 83.2%, confident top-30%
  88.5%, ECE 0.5→0.2.**
- **Conclusion:** New training-free canonical baseline; practically useful (~88% on
  its confident subset) for a study assistant.
- **Reproduce:** `scripts/eval_exemplar.py`.

---

## Phase 3 — Learning & scaling

### 012 — Learned discriminative head (first training)
- **When:** 2026-06-26.
- **Why:** Everything so far is zero-training. Can a small learned metric separate
  the look-alikes the frozen features confuse?
- **What & How:** `Dropout(0.2)→Linear(768→256)` trained with supervised-contrastive
  loss on the frozen embeddings, **retrained per seed on the gallery**, evaluated
  cross-cadaver with exemplar 1-NN. Adam lr 1e-3, wd 1e-3, 300 steps. PAIRED
  comparison (same split) — the correct test vs two ±3.6 distributions.
- **Where:** Core.
- **Result:** **paired Δtop1 +2.6 (9/10 seeds), Δtop5 +7.8 (10/10).** No overfit
  (test improves).
- **Conclusion:** Low-capacity learning helps, modestly — a signal that the limit is
  data, not optimization.
- **Reproduce:** `scripts/learned_head.py`.

### 013 — Data-scaling curve (the decisive diagnostic)
- **When:** 2026-06-26.
- **Why:** Model tweaks give small top1 gains — is the bottleneck **data scale**?
- **What & How:** Subsample the gallery to 25/50/75/100% of its specimens; frozen
  exemplar; fixed test; measure top1(covered) and coverage. 10 seeds.
- **Where:** Core.
- **Result:** top1 **39.1→39.7→45.2→46.6**; coverage **33.9→57.4→72.1→83.2**.
  **Neither saturates at 100%** (top1 still +1.4 over the last 25%; coverage nearly
  linear).
- **Conclusion:** Data is the binding ceiling — more specimens would raise accuracy
  *and* coverage. This reframes the whole project.
- **Reproduce:** `scripts/learning_curve.py`.

### 014 — Best setting (full stack, measured jointly)
- **When:** 2026-06-27.
- **Why:** Report the definitive number with all positive levers combined.
- **What & How:** frozen DINOv2 → GaussianPool → **SupCon head** → **exemplar 1-NN**
  → temperature calibration → risk–coverage abstention. 10 seeds.
- **Where:** Core.
- **Result:** **top1 49.2±4.3, top5 65.8±3.9, coverage 83.2%, confident top-30%
  87.6%, answerable at ≥80% accuracy = 39% of pins, ECE 0.3.**
- **Conclusion:** Current best achievable on 953 triples; ≈100× chance.
- **Reproduce:** `scripts/best_setting.py`.

---

## Phase 4 — Model-lever exhaustion (negative results)

### 015 — Learned pooler (PinCrossAttention)
- **When:** 2026-06-27.
- **Why:** A *learned* pooler could focus on the structure rather than a fixed
  Gaussian — potentially bigger than a post-hoc head.
- **What & How:** Train the cross-attention pooler (4 heads, attn_dim 256, dropout
  0.1) with SupCon, class-balanced batches, 250 steps, per seed; exemplar eval;
  PAIRED vs GaussianPool. Also visualized each pooler's patch weights.
- **Where:** Core, cached grids.
- **Result:** **paired Δtop1 −2.3 (only 1/10 win)**, Δtop5 +4.1. The ~600 K-param
  pooler **overfits**; its attention is diffuse while the Gaussian stays tightly on
  the pin.
- **Conclusion:** High-capacity learning overfits 601 triples; the parameter-free
  Gaussian's inductive bias wins at this data scale.
- **Reproduce:** `scripts/learned_pooler.py`.

### 019 — Backbone scaling
- **When:** 2026-06-27.
- **Why:** We had only used the smallest backbone; would a bigger frozen encoder lift
  the ceiling?
- **What & How:** Re-embed and exemplar-eval with `vitb14` / `vitl14` / `vitg14`.
  10 seeds.
- **Where:** Core.
- **Result:** **46.6 / 46.8 / 47.7** top1 — a ~13× bigger encoder buys only +1.1 pp.
- **Conclusion:** Feature quality is not the bottleneck.
- **Reproduce:** `scripts/backbone_scale.py`.

### 020 — Structured relational context (M6' step 1)
- **When:** 2026-06-27.
- **Why:** exp 008 showed global context helps only top5; would *structured*
  neighbour-region context help top1? Cheap foundation for the relational expert.
- **What & How:** Over-segment each photo (k-means on DINO patch tokens + position,
  K=12); fuse the pin-region / adjacent-region appearance into the embedding;
  exemplar eval. 10 seeds.
- **Where:** Core.
- **Result:** base 46.6 → **+region 43.2, +neighbor 46.2, +both 43.9** (all ≤ base).
- **Conclusion:** Region averaging dilutes detail; structured concat doesn't help
  top1 either. With the learned-pooler overfit, a *trained* R-GCN would likely
  overfit too — every model lever points to data.
- **Reproduce:** `scripts/relational_context.py`.

---

## Phase 5 — Robustness & deployment readiness

### 016 — Augmentation (accuracy + domain robustness)
- **When:** 2026-06-27.
- **Why:** Use the unused `augment.py` to (a) thicken the gallery and (b) test
  robustness to camera/lighting shift — both deployment-relevant.
- **What & How:** K=4 augmented views as extra gallery exemplars; test-time
  augmentation; and a strong photometric corruption of the test (dark + contrast +
  warm cast + noise). 10 seeds.
- **Where:** Core, frozen exemplar.
- **Result:** clean top1 base 46.6 → **aug-gallery 48.0 (+1.5)**, tta 46.6; on the
  **corrupted** test, base barely drops (46.6→46.5) → **DINOv2 features are already
  robust to photometric shift.**
- **Conclusion:** Lighting/camera-colour risk is largely absorbed by the backbone;
  augmentation gives a small accuracy/robustness bump.
- **Reproduce:** `scripts/augment_eval.py`.

### 017 — Pin-noise robustness
- **When:** 2026-06-27.
- **Why:** A real tag/click is imperfect; the backbone can't absorb a mislocated pin.
- **What & How:** Jitter the test pin by Gaussian noise (cached grids, re-pool at the
  shifted q); also test a jitter-augmented gallery. 10 seeds.
- **Where:** Core.
- **Result:** jitter 0/10/20/40/80 px → **46.6/46.0/44.9/43.1/36.4** — tolerant to
  ~40 px (≈1.3% of width) with only −3.5 pp; jitter-aug gallery didn't help.
- **Conclusion:** GaussianPool's spatial averaging absorbs imperfect tags; together
  with 016, two deployment-stability axes look good.
- **Reproduce:** `scripts/pin_robustness.py`.

### 018 — Deployment operating point + open-set (premature)
- **When:** 2026-06-27.
- **Why:** Test whether a clean 3-way operating point + open-set rejection is
  achievable (and resolve the test-reuse concern).
- **What & How:** 3-way specimen split; pick τ* on val for a 90% accuracy target;
  report on the untouched test; OOV = singleton pins. 10 seeds.
- **Where:** Full dataset (core + singletons for OOV).
- **Result:** at the 90% target, test acc 81.2±27.8 but **answer rate only 19±11%**
  (one seed 0%); OOV reject 95%; AUROC 67.7.
- **Conclusion:** **Defining a deployment spec now is premature** — the model isn't
  ready; exp 018 quantifies that immaturity. Deployment is the *ultimate* goal, not
  a current task.
- **Reproduce:** `scripts/deployment.py`.

---

## Phase 6 — Data-expansion investigation (honest dead-ends)

### DX1 — Honesty check: cross-cadaver split
- **When:** 2026-06-27.
- **Why:** Is the 38.8% inflated because page-level splits share a cadaver (PDF)?
- **What & How:** Compare page-level vs PDF(specimen)-level split. 10 seeds.
- **Result:** top1 identical (38.8±3.4 vs 39.3±5.4); only **coverage drops 83→63%**
  on held-out cadavers (33% of classes are single-PDF → OOV → abstained).
- **Conclusion:** Accuracy is *not* specimen-leak-inflated; the real limit is
  coverage. The model generalizes to new cadavers.

### DX2 — Is BlueLink QuizLink exhausted?
- **When:** 2026-06-27.
- **Why:** Data is the lever — are there more QuizLink dissection PDFs?
- **What & How:** Headless-crawled the whole BlueLink curriculum tree (~200 pages) →
  646 new Drive ids. First pilot (30 PDFs across sections) → 0 button-format (wrong
  early conclusion "exhausted"). Thorough widget-format scan of all ~426 candidate
  PDFs → 10 button-format found. **Perceptual page-image hashing vs the existing 31**
  → all 10 are duplicates (67–100% page match; curriculum re-links the same PDFs
  under different Drive ids).
- **Result:** **31 QuizLink PDFs is the complete unique set / 953 triples.** Web
  search confirmed no open-access real-cadaver surface-dissection point-labeled
  dataset exists. The "labeled-images" sections are color-segmentation osteology
  images (artificial colour; usable only via the plain pair + manual color→name
  labeling).
- **Conclusion:** The easy data source is tapped out. Real growth needs manual
  labeling of the color-seg set, external CC images, or the deploying institution's
  cadaver photos. (The overlap check was essential — it prevented a same-dissection
  train/test leak.)

---

## Phase 7 — Squeezing more from existing data (in progress)

### 021 — A-1: singletons in the gallery (vocabulary vs accuracy)
- **When:** 2026-06-27.
- **Why:** The 352 singleton classes are excluded from the core eval, but a deployment
  gallery should hold all labelled data so the model can recognise them too. Cost?
- **What & How:** core-test top1/top5 with gallery = core-only vs core+singletons
  (singletons only from non-test pages, no image leak). 10-seed paired.
- **Where:** Full dataset (953 triples), page-level split.
- **Result:** recognizable vocabulary **~201 → ~448 classes (2.2×)** at a cost of
  **Δtop1 −2.9 pp** on the core test (45.2 → 42.3; 9/10 seeds drop a little).
- **Conclusion:** Including singletons in the *deployment* gallery roughly doubles
  the structures the model can attempt, for a small accuracy cost on the structures
  it already knew — a sensible deployment trade that directly attacks the coverage
  bottleneck (DX1). Singleton accuracy itself remains unmeasurable (1 instance).
- **Reproduce:** `scripts/singleton_gallery.py`.

### 023 — Train the head on AUGMENTED embeddings
- **When:** 2026-06-27.
- **Why:** exp 016 augmented only the gallery (retrieval). The learned head (exp 012,
  +2.6) was trained on just 601 un-augmented embeddings — data-starved. Does giving
  it K augmented views per triple as extra training signal help?
- **What & How:** Embed each core triple + K=4 augmented views once; per seed train
  the SupCon head on (a) originals only vs (b) originals+augmented; project & exemplar
  1-NN on originals; PAIRED. 10 seeds.
- **Where:** ≥2 core.
- **Result:** head-raw top1 49.5 → **head-aug 50.6 (paired Δ+1.1, 7/10 seeds)**,
  Δtop5 +0.9 (6/10).
- **Conclusion:** A small, marginal gain. Augmentation adds no new anatomical
  diversity (same specimens), so it acts as light regularization, not a ceiling
  breaker — consistent with exp 013 (data scale is the limit) and exp 016
  (gallery-aug +1.5). The augmentation lever is largely tapped; best stack now
  ~50.6 top1.
- **Reproduce:** `scripts/aug_head.py`.

### 022 — A-2: OCR-recover dropped labels
- **When:** 2026-06-27.
- **Why:** Some triples were dropped as OCR junk that are actually real structures
  with OCR errors; fuzzy-recovery against the clean lexicon could regain them.
- **What & How:** Re-parsed all 31 PDFs to collect raw OCR labels; identified the
  ones `clean` dropped; high-threshold (≥90 token-sort, len≥6) fuzzy-matched them to
  the 567-label clean vocabulary to surface recovery CANDIDATES — deliberately not
  auto-adding (medical labels must be exact; a wrong recovery corrupts data).
- **Where:** All 31 PDFs.
- **Result:** 346 dropped triples (264 distinct labels); **only 1 high-confidence
  recovery candidate** (`cartilages`→`cartilage`, a plural). The dropped labels are
  too garbled to near-match any real structure.
- **Conclusion:** Negative — the dropped labels are genuine OCR garbage, not
  recoverable structures; the label cleaning (D2 + hand review) was already
  thorough. Fuzzy-recovery adds ~0 usable triples (lowering the threshold would
  over-recover wrong labels). Better OCR (upscale/denoise re-OCR) might read a few
  more but is low-yield and out of scope.
- **Reproduce:** `scripts/recover_labels.py`.

---

## Phase 8 — Task-tailored theory (using the dataset's structure)

### 024 — Feature-coherent pooling
- **When:** 2026-06-27.
- **Why:** GaussianPool blends the pinned structure with adjacent tissue (bad for
  look-alikes). Pool by spatial × *feature-similarity to the pin patch* so the
  embedding captures the structure (e.g. the vessel along its length), param-free.
- **What & How:** `w_i = softmax(-d²/2σ² + cos(token_i, seed)/τ)`, wide σ so the
  feature term selects; sweep τ; exemplar 1-NN; PAIRED vs Gaussian. 10 seeds.
- **Where:** ≥2 core.
- **Result:** gauss 46.6 vs coherent {τ0.1 42.7, τ0.2 44.4, τ0.3 46.4, σ120 44.8} —
  all ≤ Gaussian on top1 (best Δ−0.2); **top5 rises (60.7)**.
- **Conclusion:** Negative for top1 — DINO patch tokens are already structure-local,
  so feature-coherent selection adds top5 context but no top1 discrimination. Same
  "top5↑, top1 flat" pattern as pooling-width / context / structured-context.
- **Reproduce:** `scripts/coherent_pool.py`.

### DX3 — Is there an artery/vein colour cue?
- **When:** 2026-06-27.
- **Why:** The biggest confusion is artery↔vein; injected cadavers (red artery / blue
  vein) would give a strong colour cue DINO might underuse.
- **What & How:** Mean RGB in a window at each pin, per tissue type.
- **Result:** artery R−B = +46, vein +33 (difference only **+13**, overlapping;
  muscle/nerve also +52/+53). The specimens are **not strongly colour-injected**.
- **Conclusion:** The colour cue is weak — won't cleanly separate artery from vein.

### 025 — Region-conditioned Bayesian prior
- **When:** 2026-06-27.
- **Why:** The global CLS (region) lifted top5 (exp 008); use it as a *prior*
  `score(y)=app(y)+λ·cos(CLS_test, region_proto(y))` to fix cross-region confusions.
- **What & How:** region_proto = mean CLS per gallery class; fixed-λ sweep (no test
  tuning); exemplar 1-NN; PAIRED. 10 seeds.
- **Where:** ≥2 core.
- **Result:** λ=0 46.6 → λ=0.1 **47.2 (Δ+0.5, 6/10)**, higher λ hurts top1; **top5
  rises 58→65 with λ**.
- **Conclusion:** Marginal for top1 — the dominant confusions are *same-region* fine
  look-alikes (artery↔vein, adjacent muscles), which a region prior can't separate.
  But it meaningfully improves the top-5 candidate set → useful for the deployable
  top-k + abstention product. Same "top5↑, top1 flat" signature as every coarse
  signal.
- **Reproduce:** `scripts/region_prior.py`.

### Synthesis of Phase 8 (task-tailored theory)
Across feature-coherent pooling (024), region prior (025), the colour cue (DX3),
and laterality (too rare, 6%) — every task-tailored signal **improves top5 but not
top1**. Combined with Phases 1-4, the conclusion is robust: the top1 ceiling (~46-50%)
is set by **same-region fine-grained discrimination** that frozen appearance features
+ few shots cannot resolve. The right answer is reliably in the top-5 (~58-66%); the
deployable form is therefore **top-k + calibrated abstention** (improvable — region
prior raises top5 to ~65%), while raising top-1 needs more data or a non-overfitting
structured/relational model.

### DX4 — SAM point-prompted segmentation (does masking help?)
- **When:** 2026-06-27.
- **Why:** Idea: SAM is *point-prompted* (matches our pin q) — `SAM(I,q)` could give
  the pinned structure's mask → sharp mask-pooled features + shape descriptors,
  fixing the "pool the structure not the surrounding" problem that DINO-coherent
  pooling (024) couldn't.
- **What & How:** Installed `segment-anything` (vit_b), point-prompted at q on
  diverse pins (deltoid, basilar artery, liver, facial/femoral structures), inspected
  all 3 granularity masks.
- **Where:** Sample core triples (visual).
- **Result:** Masks are either the **whole visually-coherent region** (deltoid → the
  entire shoulder block 53%; basilar artery → the whole brain base 61%) or a **tiny
  color patch** (trapezius 0.2%, facial artery near-0). No granularity isolates the
  anatomical structure.
- **Conclusion:** **Negative — SAM segments *visual* boundaries, but in a dissection
  anatomical boundaries are often NOT visual** (adjacent muscles share colour, vessels
  are embedded in fat). So SAM can't isolate internal structures — the same reason
  look-alikes are hard. Mask-pooling over a whole-region mask would be *worse* than
  the Gaussian. A bigger SAM (vit_h) or MedSAM (radiology-tuned) wouldn't fix the
  visual≠anatomical mismatch. The "use segmentation" idea is sound in principle but no
  off-the-shelf segmenter fits soft-tissue dissection; the only true anatomical
  segmentation available (BlueLink color-seg) is osteology.

### 026 — Multi-layer DINO features
- **When:** 2026-06-27.
- **Why:** Only the last (semantic) DINO layer is pooled; early layers carry texture
  (vessel-wall striation, fibre direction) that might separate same-region look-alikes.
- **What & How:** Pool several layer-sets at the pin (via `get_intermediate_layers`),
  concat, exemplar 1-NN; PAIRED vs last-layer. 10 seeds.
- **Where:** ≥2 core.
- **Result:** last(L11) 46.6 → all multi-layer sets WORSE (L8+L11 43.6, L2+L5+L8+L11
  42.4, L2+L11 43.8; best Δ−2.9, 0/10); top5 also drops.
- **Conclusion:** Negative — early-layer texture is generic/noisy and dilutes the
  strong semantic last-layer features. Last-layer-only is best for retrieval.
- **Reproduce:** `scripts/multilayer.py`.

---

## Phase 9 — Idea-menu sweep: knowledge, loss, texture, aggregation, ensemble

Motivation: the user asked for a batch of *diverse, task-tailored* attempts beyond
appearance/pooling/context. Five orthogonal angles, each a cheap probe, each
10-seed paired against the canonical exemplar baseline (top1 46.6±3.6 on ≥2 core).
Every one is a **negative**, but each kills a different hypothesis — together they
triangulate the same ceiling from five new directions.

### 027 — BiomedCLIP (vision-language anatomical knowledge)
- **When:** 2026-06-27.
- **Why:** Same-region look-alikes (artery vs vein) may be separable not by more
  appearance but by *anatomical knowledge*. BiomedCLIP (medical CLIP, image+text,
  PubMed-pretrained) carries that knowledge in its text encoder — a signal DINO has
  no access to. Hypothesis: a knowledge prior re-ranks appearance ties correctly.
- **What & How:** On a pin-centred crop, four methods — (a) DINO exemplar 1-NN
  (baseline), (b) BiomedCLIP-image exemplar 1-NN, (c) zero-shot text ("a dissection
  photo of a {class}") vs crop, (d) `score = sim_dino + λ·sim_text` with per-seed
  best λ∈{.3,.6,1} (an optimistic UPPER bound on the knowledge signal). 10 seeds.
- **Where:** ≥2 core, 601/215.
- **Result:** dino 46.6 / **bmc-img 36.9** / **bmc-text 2.0** (≈4× the 0.47% random
  floor but useless) / **dino+textλ 47.3** (+0.7, and that is the *tuned upper bound*).
- **Conclusion:** Negative. BiomedCLIP is trained on published figures, so dissection
  photos are OOD for it — its image features are markedly weaker than DINO and its
  text↔crop alignment collapses to near-chance. Even an oracle-λ knowledge prior buys
  <1 point. The missing information is not "knowledge the text encoder has"; it is
  not present in these images at all. Reconfirms the data ceiling from the knowledge
  angle. (Note: random page captions were NOT used; only class-name prompts.)
- **Reproduce:** `scripts/biomedclip.py`.

### 028 — Angular-margin head (ArcFace-style, contrastive)
- **When:** 2026-06-27.
- **Why:** The SupCon head helped (+~3) by clustering same-structure embeddings, but
  plain SupCon enforces no *margin* — positives need only beat negatives, not beat
  them by a gap. A hard additive angular margin `cos(θ+m)` on positives should carve
  a cleaner boundary between fine look-alikes IF the boundary is loss-underdetermined.
- **What & How:** Add an additive angular margin to the positive terms inside the
  supervised-contrastive objective (ArcFace's cos(θ+m), applied contrastively so it
  stays few-shot compatible — no fixed classifier weights). m=0 reproduces plain
  SupCon → clean paired ablation. Retrain head per seed; exemplar 1-NN in learned
  space; 10 seeds; m∈{0, .1, .2, .3}.
- **Where:** ≥2 core.
- **Result:** frozen 46.6 → **m0 (SupCon) 49.8** (+3.2, replicates the head gain) →
  m.1 49.5, m.2 49.3, m.3 48.7. Best-vs-m0 Δ−0.3 (2/10). Margin monotonically *hurts*
  as it grows.
- **Conclusion:** Negative for the margin. The head's clustering is the lever; adding
  a harder margin gives nothing because the classes it must separate are not separable
  by a sharper decision surface — the discriminating information is absent, not just
  un-margined. The loss *shape* is not the bottleneck; the data is.
- **Reproduce:** `scripts/arcface_head.py`.

### 029 — Local-orientation / texture descriptor at the pin (fibre grain)
- **When:** 2026-06-27.
- **Why:** DINO's semantic features wash out micro-texture, but anatomical "grain" is
  locally discriminative: vessel wall (smooth) vs nerve (striated cable) vs muscle
  (anisotropic fibre). A hand-built descriptor might re-inject what DINO discards.
- **What & How:** On a grayscale patch at the pin, compute a multi-scale (r=24,48)
  HOG-style orientation histogram (gradient angle mod 180°, magnitude-weighted, 9
  bins) + structure-tensor coherence/anisotropy + dominant-orientation (cos/sin 2θ);
  24-d, L2-normalized. Fuse `score = sim_dino + λ·sim_orient` (both exemplar
  class-max), fixed λ∈{0,.2,.4,.7,1}. Also orient-only. 10 seeds.
- **Where:** ≥2 core.
- **Result:** orient-only top1 **11.2±2.4** (≈24× the 0.47% floor — the grain *does*
  carry weak signal), but every fusion λ>0 only *hurts*: λ.2 −4.0, λ.4 −7.8, λ1.0
  −12.4 (all 0/10). top5 drops too.
- **Conclusion:** Negative. The local grain is weakly discriminative in isolation but
  entirely subsumed by DINO; concatenating a low-rank hand descriptor only adds noise
  and dilutes the strong feature. A single pin-point's texture is not the missing
  same-region signal.
- **Reproduce:** `scripts/orientation.py`.

### 030 — Multi-prototype & soft aggregation per class
- **When:** 2026-06-27.
- **Why:** Our rule is exemplar-max ≫ mean (mean washes out detail), but max-over-all
  is sensitive to one noisy gallery item. Between them: k-means sub-prototypes
  (capture multi-view modes) and a soft log-sum-exp (temperature interpolates
  mean↔max). Does any middle ground beat plain max?
- **What & How:** Per class, score by: mean centroid; k-means K=2/3 (max over
  sub-centroids); exemplar (max over all); LSE τ=.1/.05 over all gallery sims. 10
  seeds, paired vs exemplar.
- **Where:** ≥2 core.
- **Result:** mean 38.8 < kmeans-2 43.6 < kmeans-3 45.9 < lse-τ.05 45.2 < **exemplar
  46.6**. Best non-exemplar Δ−0.8 (0/10). Accuracy rises *monotonically* as the
  aggregator approaches max.
- **Conclusion:** Negative — exemplar-max stays best, and the monotone ordering is a
  clean re-derivation of the core lesson: with small per-class galleries (2-core), any
  smoothing destroys the few discriminative details that survive. No aggregation lever
  left.
- **Reproduce:** `scripts/multiproto.py`.

### 031 — Diverse-backbone ensemble (DINO ⊕ BiomedCLIP-image)
- **When:** 2026-06-27.
- **Why:** bmc-img (36.9) is weaker than DINO but trained on a disjoint distribution —
  its *errors* might be complementary, so a fused score could beat either alone even
  with one weak member (the classic ensemble argument).
- **What & How:** `score = sim_dino + λ·sim_bmc` (both exemplar class-max), fixed
  λ∈{0,.3,.6,1,1.5}, 10 seeds, paired vs DINO.
- **Where:** ≥2 core.
- **Result:** best λ=1.0 top1 47.7 (**Δ+1.0 but only 5/10 seeds win**); top5 flat
  (58.0→58.6). Not consistent (paired bar is ≥8/10).
- **Conclusion:** Negative/marginal. The +1 is within seed noise and not consistent —
  the two backbones are not complementary enough on dissection photos (bmc-img is too
  weak and OOD to correct DINO's errors). A cheap ensemble is not a real lever here.
- **Reproduce:** `scripts/ensemble.py`.

### Synthesis of Phase 9
Five independent angles — external knowledge (027), loss geometry (028), hand
texture (029), prototype aggregation (030), backbone diversity (031) — each fails to
move top1 beyond seed noise. The SupCon head's +3 (replicated in 028) remains the
only training lever, and it is a *representation* gain, not a *boundary-sharpening*
one (margin adds nothing). Every negative points the same way: the residual
same-region confusions carry no extra signal in the local image — not in a stronger
backbone's features, not in medical-text knowledge, not in micro-texture, not in a
cleverer metric. **The binding constraint is data coverage/instances, exactly as the
scaling curve (013) showed.** Model-side ideas are now exhausted across nine phases.

### 032 — Class-aware adaptive SAM masking (revisiting DX4)
- **When:** 2026-06-27.
- **Why:** DX4 killed SAM at a single setting; the user proposed making the mask SCALE
  *class-aware* — thin tubular structures (artery/vein/nerve/duct) want a tight small
  mask, bulk tissue (muscle/bone/gland) wants a large mask — then pool DINO tokens
  over the chosen mask instead of an isotropic Gaussian.
- **What & How:** SAM-ViT-B point-prompted at the pin with `multimask_output` gives 3
  masks at 3 scales; pool DINO patch tokens (masked-mean, back off to the pin patch if
  the mask covers <½ patch). Policies: gauss (baseline) / sam-best (highest-IoU mask)
  / sam-small / sam-large / **class-aware (ORACLE thin→small, bulk→large)**. Cheap
  probe = oracle routing (best possible); if the oracle can't beat gauss, no router is
  worth building. 10-seed paired, plus a thin-vs-bulk accuracy breakdown.
- **Where:** ≥2 core (thin 264 / bulk 337).
- **Result:** gauss **46.6** ≫ sam-small 41.5 > class-aware 39.7 > sam-best 37.0 >
  sam-large 34.3 (best non-gauss Δ−5.2, 1/10). Breakdown: thin gauss 47.0 vs
  class-aware(small) **39.4**; bulk gauss 46.3 vs class-aware(large) **39.9** — the
  idea loses on BOTH halves. Notably class-aware (39.7) is *worse* than always-small
  (41.5) because the bulk→large half is the most harmful (large 34.3).
- **Conclusion:** Negative, and the breakdown refutes the specific intuition: "big
  mask for muscle" is the single most damaging choice, and even a tight mask on a thin
  vessel underperforms the Gaussian. The granularity of routing is not the problem —
  any *hard uniform mask-average* dilutes the pin-concentrated signal, and the
  Gaussian's *soft pin-centred weighting* beats it at every scale. The oracle
  upper-bound failing closes the class-aware-routing direction. (SAM masks on
  dissected thin structures are themselves unreliable — visual ≠ anatomical boundary,
  the DX4 finding — so the masks are bad AND the averaging is wrong.)
- **Reproduce:** `scripts/sam_classaware.py`.

### 033 — Thin-gated SAM pooling (the final SAM verdict)
- **When:** 2026-06-27.
- **Why:** Inspection of the masks (rendered as `*.private.png`) converged the design
  away from exp 032's failure: a whole/large mask dilutes (hard average), and forcing
  a bulk mask small just yields an arbitrary circle (≈ Gaussian). The honest synthesis:
  SAM only adds a usable region where the structure is a DISTINCT object — i.e. **thin
  tubular structures** (vessel/nerve/duct). So gate ONLY thin items by the mask, and
  fall bulk back to the plain Gaussian. Pre-registered a strict adoption rule to avoid
  the "pretty mask, unchanged embedding" trap that fooled 008/024/026.
- **What & How:** Per item, pooling weight = `feather(SAM mask) × pin-Gaussian` for
  thin (SAM's smallest multimask output; if even that is loose, >6% of image — e.g. a
  fat vein blob — fall back to Gaussian); plain pin-Gaussian for bulk. 'Treatment' thus
  differs from 'baseline' (plain Gaussian) only on thin items. exemplar 1-NN, 10-seed,
  paired, split by the test item's coarse type. **Pre-registered: ADOPT iff thin
  paired Δ>0 AND ≥7/10 seeds.**
- **Where:** ≥2 core (thin 249 / bulk 352); thin masked 216, loose-fallback 33.
- **Result:** thin **44.5±5.1 vs 46.5±6.6 → Δ−2.0 (1/10)**; bulk 47.3±3.3 vs 46.6±3.2
  → +0.7 (6/10); all 46.2±3.5 vs 46.6±3.6 → −0.4 (3/10). (bulk is not identical because
  a bulk query matches against the whole gallery, whose thin items' embeddings changed
  — a small, noisy side-effect.)
- **Conclusion:** **REJECTED by the pre-registered rule** — mask-gating *actively hurts*
  thin recognition (−2.0). Even where SAM masks look anatomically correct on vessels/
  nerves, gating the pool by them does not improve (and degrades) the embedding: the
  Gaussian's soft reach was using context the tight mask discards, and SAM's masks are
  not consistent enough to help. This is the 008/024/026 pattern at its sharpest —
  prettier localization, no (negative) accuracy. **The SAM/segmentation direction is
  now closed in all forms** (point-prompt DX4, class-aware oracle 032, thin-gated 033).
  The ceiling remains data, not pooling.
- **Reproduce:** `scripts/sam_thingate.py`.

---

## Phase 10 — The last model axis: visual prompting (q at the backbone INPUT)

### 034 — Visual prompting
- **When:** 2026-06-27.
- **Why:** Every method so far (pooling, SAM) conditions on q at READOUT time — the
  backbone always sees the CLEAN image and q only selects/weights tokens afterwards.
  The one untouched, orthogonal axis is injecting q at the backbone INPUT: draw a
  marker at q on the image so DINO itself "sees" the pin. If even this fails, the model
  axis is exhausted *outside* the pooling plane too, locking the data-ceiling claim;
  if it wins, the ceiling was secretly a q-conditioning-locus problem and the whole
  interpretation flips. Either way maximal information for half a day's work.
- **What & How:** Render a marker at q on the 518px image, run frozen DINO on the
  MARKED image, read out: (a) CLS token (global, location-aware via the marker —
  canonical visual prompting) or (b) GaussianPool at q on the marked grid. Markers:
  red filled dot r8, red hollow ring r18 / r30 (no occlusion), lime ring r18 (colour).
  Baseline = GaussianPool on the CLEAN image (46.6). exemplar 1-NN, 10-seed, paired.
  Pre-registered: ADOPT iff best variant Δtop1>0 AND ≥8/10 (≈9 variants → strict,
  Holm-aware).
- **Where:** ≥2 core.
- **Result:** best VP = red-dot8-gpool **46.5 (Δ−0.1, 4/10)** → fails. CLS readouts:
  clean-cls 30.3, marked-cls ~38–40 — i.e. **the marker DOES inject location** (CLS
  jumps +9pp when the pin is drawn) and marked-cls gives the project-best **top5 ~66**,
  but CLS top1 stays far below local pooling (≈39 vs 46.6). gpool-on-marked ≈ baseline
  (the marker neither helps local pooling nor matters; top5 slips ~2pp from occlusion).
- **Conclusion:** REJECTED. Visual prompting verifiably encodes the pin location into
  the backbone, yet does not beat readout-time Gaussian pooling on top1 — CLS is a
  region signal (great top5, poor fine top1, exactly the exp 008 split), and a local
  marker can't add fine discriminability that isn't in the image. **The model axis is
  now exhausted both inside the pooling plane (008/015/020/024/030) and outside it
  (034).** The "data is the binding ceiling" conclusion is locked from both directions.
  Next (per ROI): a human-ceiling study to separate data-limit from intrinsic label
  ambiguity before any expensive data expansion.
- **Reproduce:** `scripts/visual_prompt.py`.

---

## Phase 11 — Optimization on fixed data (reliability / coverage; OPT_HANDOUT)

With the model axis closed, the centre of gravity moves to reliability and coverage —
pushing the risk–coverage curve out on the fixed 953 triples (all frozen-embedding
post-processing). Recommended ROI order (OPT_HANDOUT.md): M-opt0 gate → 037 conformal+
KDE → 038 shrinkage → 039 coverage curve → 040 ensemble (gated). 035 (human ceiling)
runs in parallel, awaiting the pilot.

### 036 — M-opt0: evaluation purification (the gate before any optimization)
- **When:** 2026-06-27.
- **Why:** Before trusting any "+pp", quantify HP-SELECTION leakage — σ40/exemplar/
  calibration were chosen while looking at core's eval; is the reported ~50 a *validation*
  number? (Split leakage is already closed by DX1's specimen-level split.)
- **What & How:** PDF-level nested protocol, 5 folds. Each fold seals ~20% of PDFs as
  holdout; SELECT (σ∈{10,20,40,60,80}, rule∈{exemplar,proto}) on dev via dev-internal
  specimen splits; EVALUATE the dev-selected config AND the fixed canonical (σ40,exemplar)
  on the sealed holdout. **Decompose** the dev−holdout gap into (i) HP-selection optimism
  = gap_selected − gap_canonical, and (ii) cross-cadaver shift = canonical gap.
- **Where:** 601 core / 31 PDFs.
- **Result:** selection dev 45.7 → holdout 37.7 (gap 8.0); canonical(σ40,ex) dev 44.1 →
  holdout 37.6 (gap 6.5). Selection always picked (σ80, exemplar) but its holdout (37.7)
  ≈ canonical holdout (37.6). **Decomposition: HP-selection leakage = 1.5pp 🟢;
  cross-cadaver gap = 6.5pp (significant).** Holdout is noisy (±5, only 6 PDFs/fold).
- **Conclusion:** (1) **Gate PASSED** — HP-selection optimism ~1.5pp: choosing on eval
  added ~0 real generalization (σ80 over σ40 is dev-noise that evaporates on holdout),
  so the **paired Δ comparisons across the 35 experiments stand** and σ40/exemplar are
  robust (consistent with 006/009). (2) **[⚠️ RETRACTED by exp 038]** I initially read
  the holdout drop (~37–40) as a cross-cadaver *accuracy* gap from same-cadaver leakage.
  **That was an artifact of the protocol here:** the holdout was evaluated with a gallery
  drawn from *within the 6 holdout PDFs* (a tiny gallery), so the low top1 reflects
  GALLERY SIZE (exp 013 scaling), not cadaver shift. exp 038, with the proper setup
  (gallery = all dev PDFs, query = unseen PDF), finds cross-cadaver top1 **46.5 ≈
  page-split 46.6 — no gap** — and that the nearest gallery exemplar is same-PDF only
  **0.3%** of the time (no same-cadaver leakage). DX1 was right: cross-cadaver *accuracy
  is invariant*; what drops is *coverage* (83→57%, a cadaver's unique structures go OOV).
  The honest headline is ~46–50, NOT ~37–40.
- **Reproduce:** `scripts/mopt0.py`. (Cross-cadaver claim corrected by `scripts/cadaver_invariant.py` / exp 038.)

### 037 — KDE posterior + Conformal + OOD (reliability/coverage layer)
- **When:** 2026-06-27.
- **Why:** Replace the heuristic softmax-cosine confidence with a principled density
  model: KDE posterior p(y|z)∝π(y)Σ_e exp((cos−1)/h²) for a *calibrated* posterior, a
  split-conformal layer for *guaranteed* prediction sets, and the marginal density p(z)
  as an OOD/OOV score. Primary axis = AURC (per OPT_HANDOUT §5). Pitfall #2 guard: verify
  KDE ECE vs the global-temperature baseline PAIRED. 🔴 key: does a conformal set fit on
  page-split hold its 1−α coverage on cross-cadaver (exchangeability break)?
- **What & How:** Prediction = the SAME exemplar argmax for both methods (top1 identical,
  isolating *confidence* quality). h fit on gallery-LOO likelihood; global-temp s likewise.
  3-way specimen split (gallery 50% / cal 20% / test 30%), 10 seeds page-split PRIMARY +
  5-fold unseen-PDF cross-cadaver report. α=0.1.
- **Where:** ≥2 core (top1 40.8 on the 50% gallery; lower than 50.4 LOO by construction).
- **Result:** ECE kde **0.181** vs base 0.367 (kde better **10/10**) — KDE clearly better
  *calibrated*. BUT AURC kde 0.344 vs base **0.304** (kde better **1/10**) — the
  selective-prediction *ranking* does NOT improve; global-temp is better. OOD AUROC kde
  0.611 vs base **0.687** (0/10) — plain max-cosine separates OOV better than KDE density.
  Conformal: coverage holds (kde 0.92, base 0.916 ≈ target 0.9) but **average set size
  ≈110 of ~172 classes** (vs top5=5) — useless. 🔴 cross-cadaver conformal coverage 0.877
  (violation only **2.3pp**) — the guarantee mostly survives the shift.
- **Conclusion:** Largely NEGATIVE — the handout's pitfall #2 ("elegant theory, flat
  measurement") realized. The principled density improves *absolute* calibration (ECE) but
  not the actionable axes: selective-prediction ranking (AURC) is no better than the
  existing global-temperature heuristic, conformal sets are far too large to be useful at
  this accuracy (the "guaranteed-but-huge" caveat), and max-cosine is a better OOD score
  than KDE density. The KDE prior π(y) likely distorts the confidence ranking (over-trusts
  frequent classes). Net: the existing global-temp + abstention is already near-optimal for
  selective prediction; KDE's only win (calibrated probabilities) is non-actionable on
  AURC. *Good* news from the 🔴 check: conformal's coverage guarantee degrades only ~2–3pp
  cross-cadaver, so it is not same-cadaver-only. Reliability axis adds no operating-point
  gain; the open lever is COVERAGE (038/039), not confidence.
- **Reproduce:** `scripts/conformal_kde.py`.

### 038 — Cross-cadaver gap decomposition (A3) → cadaver-invariant normalization (A1)
- **When:** 2026-06-27.
- **Why:** The cross-cadaver gap M-opt0 reported (~6.5pp) looked like the biggest open
  lever (bigger than reliability), so attack it — but its *cause* decides if it is
  closeable: per-cadaver colour/lighting (fixable by normalization) vs anatomical
  variation between people (a 953 limit). Decompose first (A3); normalize only if colour
  dominates (A1). PRIMARY axis raised to cross-cadaver (the deployment-honest number; the
  reason this lever was never seen — every prior experiment used page-split only).
- **What & How:** A3-1 colour: re-embed Reinhard colour-normalized images (per-cadaver
  colour removed toward a global LAB reference); measure gap recovery. A3-2: fraction of
  page-split test pins whose nearest gallery exemplar is the SAME PDF, and accuracy same-
  vs diff-cadaver. A1: z'=z−λ(μ_cadaver−μ_global), μ_cadaver = CLASS-UNIFORM exemplar
  mean (★037 lesson: no frequency weighting), λ∈{0,.3,.5,.7,1}, gallery+query, primary =
  cross-cadaver (gallery=dev PDFs, query=unseen PDF) 5-fold, page-split 10-seed reported.
- **Where:** 601 core / 31 PDFs.
- **Result (a correction):** **the gap is ~0.** Proper cross-cadaver (large dev gallery,
  unseen-PDF query) top1 **46.5 ≈ page-split 46.6**; coverage drops to 57% (OOV). Nearest
  gallery exemplar is same-PDF only **0.3%** of the time → no same-cadaver leakage. A3-1
  colour recovery −0.2pp (gap is 0.1) → pre-registered STOP. A1 λ-sweep confirms nothing
  to remove: best λ0.3 cross +0.4 (2/5 folds), page −0.8, net −0.4 → not adopted.
- **Conclusion:** **This retracts the M-opt0 cross-cadaver claim.** The "6.5pp gap" was a
  small-holdout-gallery artifact (gallery size, exp 013), not cadaver shift. Cross-cadaver
  *accuracy is invariant* (DX1 confirmed); the only cross-cadaver cost is COVERAGE
  (cadaver-unique structures become OOV). So cadaver-invariant normalization is moot, and
  the headline ~46–50 is the honest generalization accuracy. The real open lever is
  COVERAGE (039), exactly as DX1 and the handout's 🟢 coverage track said — not appearance
  normalization. A clean example of the protocol deciding the conclusion (a fair cross-
  cadaver eval needs a *full* gallery, not a holdout-internal one).
- **Reproduce:** `scripts/cadaver_invariant.py`.

## Phase 12 — Relational-reasoning axis (a wholly new inference paradigm)

### 040 / M-rel0 — Relational-reasoning feasibility gate (precursor, decisive)
- **When:** 2026-06-27.
- **Why (motivation/hypothesis):** Every method to date (027–038) classifies each pin
  *independently*, p(y|I,q), and the dominant confusion (artery↔vein, DX3) is unsplittable
  by appearance. A new axis: jointly infer all pins under an anatomical knowledge graph,
  p(y₁..yₙ|I,{qᵢ},G), so *relative position + anatomical rules* (e.g. NAVEL: nerve-artery-
  vein lateral→medial order) correct appearance. The handout (exp-040, rev2) named three
  failure "cracks": #1 stage-1 = the original ~50% problem (bypass with oracle pins), #2
  image≠anatomy (2D projection / L-R flip breaks "lateral"), #3 LLM knowledge reliability.
  Its fate-deciding question: "does the relation *correct* appearance or only *amplify* a
  wrong stage-1?", to be settled cheaply by an oracle pre-verification (M-rel1), mirroring
  how 032 closed SAM with an oracle. **Before** building any graph/inference, I measured a
  precursor the handout did not enumerate — **crack #0: is a relational neighbour even
  PRESENT on the page?** The relational term can only *fire* on a pin if another pin on the
  same page is a graph-neighbour of it.
- **What & How:** (a) Pure-metadata structural probe — per-page pin-count distribution and
  the fraction of pins with a co-present relational neighbour (shared anatomical modifier,
  or a NAVEL artery/vein/nerve bundle-mate). (b) Model-based **ceiling**: current best
  engine (frozen dinov2_vitb14@518 → GaussianPool σ40 → exemplar class-max cosine),
  10-seed page-split confusion matrix; for every *error*, check whether the true label has
  a co-present same-page neighbour that a positional graph rule could use, and compute the
  max global top1 gain a *perfect oracle* relational term could yield. Tightened the
  "resolvable" predicate in stages to kill false positives: near-duplicate labels (`l
  phrenic nerve ↔ phrenic nerve` = same structure), coincidental generic-modifier sharing
  (`internal jugular vein ↔ internal oblique muscle` = neck vs abdomen), and cranial-nerve
  markers ("cn" + roman numerals — the CN analog of the tissue word "nerve", not a region
  name; two CNs sharing "cn" are as unresolvable as two arteries sharing "artery"). Then
  restricted to what the term can *actually* fix: it is a tie-breaker that must not
  overpower appearance (§2.3), so it only flips an error when the model swapped true↔partner,
  the partner is co-present, AND appearance kept true within top-3 (rank≤3).
- **Where:** 601 core / 215 classes / 369 pages / 31 PDFs.
- **Result:** Structural — **58% of pages are single-pin** (213/369), a NAVEL bundle is
  co-present for only **13% of vessel/nerve pins (4.8% of all)**; the femoral-triangle
  worked example (N+A+V on one page) is the *exception*, not the rule. Ceiling — collapses
  through four honest stages: loosest proxy **+9.8pp** → "true has any resolvable neighbour"
  **+7.0pp** → NAVEL bundle present **+3.0pp** → pred=partner co-present (textbook swap)
  **+0.8pp** → **REALISTIC (swap AND true≤rank3) = +0.4pp = 0.6 pins/seed.** Only **5**
  genuinely-resolvable confusion pairs exist (lateral/medial condyle, s3/s4 ventral ramus,
  pubic symphysis/tubercle, sup/inf gluteal artery, ext/int oblique), and **3/5 are
  direction-dependent** (lateral/medial, sup/inf, ext/int) — exactly what crack #2 attacks;
  invariant fraction 40%.
- **Conclusion:** 🔴 **stop-but-hold (pre-registered).** A *perfect* oracle relational term
  (oracle pins, oracle alignment, oracle graph) gains **+0.4pp ≈ 0.6 pins/seed**, fully
  buried in the σ=3.6pp split noise — so M-rel1 cannot produce a trustworthy positive; the
  handout's §5 "oracle flat ⇒ abandon" verdict is reached one stage *earlier and cheaper*,
  at M-rel0. Crucially the bottleneck is **crack #0 (relational neighbours are structurally
  absent: one structure pinned per photo) and crack #2 (the few real relations are
  direction-dependent)** — both are functions of *data structure*, not of the model or the
  graph, so no inference sophistication can beat them. This matches the project through-line
  (data is the ceiling, §2): the relational axis is exhausted on the current 953 *exactly
  like* the model axis (027–034) and reliability axis (037) — **but unlike them it is the
  one axis that data expansion would directly revive** (multi-pin / bundle-co-labelled pages
  dissolve crack #0). → Not killed; *held* for re-evaluation after data expansion. The graph
  (`anat_graph.json`) and inference (`graph_inference.py`) builds in the handout are deferred,
  not invalidated.
- **Reproduce:** `scripts/confusion_pairs.py` (handout exp-040 §2.1 + §4 M-rel0).

## Phase 13 — Data expansion (BlueLink Images) + leak-corrected re-evaluation

### Data pipeline — BlueLink labeled-slide harvest → clean merged dataset
- **When:** 2026-06-27.
- **Why:** Every axis (model 027–034, reliability 037, cross-cadaver 038, relational 040)
  converged on the same conclusion — the ceiling is DATA. The proven lever (CLAUDE.md §2,§6)
  is more (I,q,y). BlueLink's "Labeled BlueLink Images" curriculum (teaching slides with
  baked labels) is a large, MULTI-LABEL-per-photo source (dissolves exp 040 crack #0).
- **What & How:** (1) Crawl — user's `data/bluelink_html/crawler.py`; fixed 3 bugs:
  filename-only dedup lost 36% (same SlideN across themes are *different* images → dedup by
  original_url), saved-HTML CDN tokens are session-expired (403) → added `--live` (re-fetch
  via canonical URL for fresh tokens), batch-then-download expired tokens mid-run → per-page
  concurrent download. 462/462 slides, 3000×2250, 0 fail. (2) Extract (`bluelink_extract.py`)
  — blue leader-line trace (HSV H119–124) to its tissue tip = q (PCA-direction march, robust
  to crossing lines), OCR box (psm6 both polarities) = label, bottom-left title = region;
  inpaint all annotations + logo + pin-aware crop (never cut a pin) = leak-free I. 1644
  triples / 100% q-rate / 3.8 labels/slide. (3) Merge with QuizLink 953 (`clean_merge.py`):
  OCR fixes (lac→iliac, brachil→brachii), state-strip (cut/reflected), depluralize, abbrev
  (brs→branches), double-label (X artery vein→X artery). (4) Dedup
  (`dedup_union.py`): discovered **49% of QuizLink photos are the same image as a BlueLink
  slide** (QuizLink quizzes are built FROM BlueLink photos); strict photo-identity (exact
  hash ∪ corr≥0.99) merge + pin union (scale-transform, validated) → one specimen per photo
  (leak-safe). (5) QuizLink images were never cleaned → logo/title/©/margins remained;
  `clean_quizlink.py` removes them + pin-aware crop + q-offset, value-preserving (off-tissue
  FLAGGED not dropped — all 79 flags were valid dark structures like coccyx/foramina; only 1
  junk "tee mae" dropped).
- **Where:** final `data/merged_final` — **711 photos / 2230 triples / 502 core(≥2) / 710
  specimens**, 0 missing / 0 q-oob / 0 w-h-mismatch / 0 dup; 531 multi-pin images (max 22).
  vs original QuizLink 953/215 → **2.3× core**. (Intermediate dirs deleted by user on purpose.)
- **Reproduce:** `data/bluelink_html/crawler.py --live`, `scripts/{bluelink_extract,clean_merge,
  dedup_union,clean_quizlink}.py`.

### 041 — Precise re-evaluation (leak-safe) — a major baseline correction
- **When:** 2026-06-27.
- **Why:** Measure the data-expansion payoff under the iron rules, and whether BlueLink helps
  the deployment target (QuizLink-like 땡시).
- **What & How:** frozen dinov2_vitb14@518 → GaussianPool σ40 → exemplar class-max cosine,
  10-seed, **photo-twin BLOCK split** (exact∪corr≥0.90) so the 49% QuizLink↔BlueLink photo
  overlap can't leak across train/test. core(≥2) basis for comparability.
- **Result (a correction):** ⚠️ **the established ~46–49 baseline was LEAK-INFLATED.** Honest
  leak-free QuizLink top1 = **21.5±4.7** (block-split); image-split (leaky) 29.7; un-deduped
  original page-split gave the old 46.6. Cause: **49% of QuizLink photos have a near-twin**;
  the original page-split separated twins → leak. Independently-extracted **BlueLink also
  gives ~27 leak-free** — confirming the honest level is ~21–27, not 46–49. ⭐ **BlueLink
  expansion HELPS QuizLink: gallery QL+BlueLink vs QL-only → Δtop1 +8.9pp (10/10), Δcoverage
  +10.1pp (10/10)**, paired, leak-safe — the deployment-target win. Merged 502-way leak-free:
  top1 31.6 / cov 72.6 / end-to-end 22.9 (vs QuizLink-only end-to-end 15.4 = 1.5×). q
  integrity verified on 32 visual spot-checks (all on-structure; crop+dedup preserved pins).
  Two negatives: **letterbox (aspect-preserving) resize HURTS −2.2pp (0/10)** vs squish
  (squish fills the frame → more effective resolution; distortion is consistent gallery↔query
  so harmless); **cropping cost ~2.5pp** (magnification/scale-inconsistency) — minor.
- **Conclusion:** Coverage is the data-expansion lever (exp 038 confirmed): +10pp coverage and
  +8.9pp top1 on the deployment target from BlueLink. The model's *real* leak-free performance
  is ~21–31 (not the leak-inflated 46–49) — a §1-style honesty correction enabled by dedup +
  twin-grouped split. Data expansion is validated as THE lever; the pipeline (squish) is optimal.
- **Reproduce:** `scripts/eval_merged.py`.

### 042 — EDA: geometry of the class space in DINO embedding space
- **When:** 2026-06-28.
- **Why/How:** embed every core pin, take per-class centroids, t-SNE the 502 centroids to 2D;
  colour by tissue type and by region; quantify within- vs across-group centroid cosine, and
  the artery↔vein same-region paired centroid distance. Added per-tissue & per-region Gaussian
  KDE density heatmaps (mirrors the pipeline's p(z|y), exp037) over an instance t-SNE.
- **Result:** **DINO-space is organised by REGION, not tissue type.** Tissue separation
  within−across ≈ **−0.005 (zero)** — same-tissue classes are NOT closer than different-tissue;
  region separation = **+0.10** (orbit/pelvis/cranial/oral form clean density clusters).
  artery↔vein same-region paired centroid cos = **0.878** (n=13) — near-identical. Intra-class
  cohesion 0.905.
- **Conclusion:** geometric root of the "top5-good / top1-bad" signature — the model places a
  pin in the right REGION (top5) but cannot resolve fine intra-region identity (top1), and
  artery/vein/nerve are entangled (DX3). The lever is data or a fundamentally finer
  representation, not the readout.
- **Reproduce:** `scripts/eda_dino_space.py` (+ embedding cache `data/merged_final/_dino_cache.npy`).

### 043 — Model-methodology sweep on the clean merged data (leak-safe) — model axis re-confirmed exhausted
- **When:** 2026-06-28.
- **Why:** the old "exemplar≫mean / learned-head modest" verdicts were on the 953/leak-inflated
  data — re-check every aggregation/learned lever on 2.3× cleaner, leak-safe 502-way.
- **What & How:** cached σ40 embeddings. **Nested multi-seed 3-way (§1.7):** fixed sealed dev/test
  photo-block split (`scripts/split_devtest.py`, seed 20260628, test 20% = dev 1214 / test 337),
  select on dev 10-seed CV, report final ONCE on the sealed test (bootstrap CI). Methods: mean-proto,
  exemplar(max), kNN-3/5, multi-proto, LSE, KDE, and a trained SupCon head then exemplar.
- **Result:** dev-CV: **exemplar best top1 28.9±3.0** (multi-proto 28.3, mean 24.4, kNN-3 26.0,
  kNN-5 21.1, LSE 21.8, KDE 22.4, **SupCon 27.4 — did NOT help**; SupCon gave +2.6 on old leaky
  data, nothing here → old gain partly leak-driven). **Sealed TEST (final, dev-selected exemplar):
  top1 33.5 (95% CI 27.5–39.4), cov 79.8** — higher than dev-CV (the "−4.6pp" is a gallery-size
  effect, full-dev gallery vs 70% CV folds, NOT selection leak → no leak). Wide CI = the 20%-sealed
  (337-query) tradeoff. Diagnostics: **44% of errors are same-tissue** (intra-tissue
  confusion); per-tissue top1 muscle 43 ≫ vein 22 (vein hardest, DX3); **shot paradox — 2-shot
  38 > 6+ 27** (frequent classes are the common, confusable vessels/muscles, intrinsically harder,
  not a data-per-class deficit); risk-coverage: abstaining to 30–40% coverage lifts selective top1
  to ~52% (deployment operating point).
- **Conclusion:** the model axis is **re-confirmed exhausted on clean leak-safe data** — no
  aggregation or learned-head trick beats exemplar 1-NN. The ceiling is intra-region/intra-tissue
  fine identity (DX3), consistent with exp 042's geometry. Forward path = data (validated +8.9/+10.1
  lever) or a finer representation; not readout tricks.
- **Reproduce:** `scripts/model_sweep.py`.

## Phase 14 — Representation axis (the only un-closed model lever): resolution gate first

### 045 — M-rep0 multiscale high-res local + M-rep0b tissue-oracle / colour probe — the first ceiling crack
- **When:** 2026-06-28.
- **Why:** 043 closed the readout axis and 042/044 showed *why* — tissue types overlap in frozen-DINO
  space. Before reshaping the space (contrastive/LoRA, which is the failed-SupCon trap), the *most
  upstream* question (handout §2): is the fine-identity cue even **in the input**? Our pipeline squishes
  the whole image to 518 px + σ40-pools, so a small pinned structure's discriminating cue (vessel
  wall / lumen / colour) may be destroyed at the *resolution* step — in which case any spatial
  reshaping is moot. Hypothesis: the bottleneck is partly **input resolution**, not just space layout.
- **What & How (training-free gates, run together):**
  **M-rep0** — for each pin, crop a tight high-res box around q from the native-resolution image
  (256 & 512 px, q-centred, out-of-image padded black), resize each to 518, DINO→σ40-pool at the crop
  **centre** → `z_local`. Unit-norm each block, concat with the global embedding, exemplar 1-NN.
  Variants: global / global+L256 / global+L512 / global+L256+L512 / local-only. **No training** — just
  re-embed; paired Δ vs global, clean 502, dev 10-seed CV select + sealed test once (§1.7).
  **M-rep0b(a)** tissue-oracle: restrict candidates to the true tissue → ceiling of a tissue gate.
  **M-rep0b(b)** colour probe: low-level RGB/HSV/texture (no DINO) artery-vs-vein 5-fold AUC, plus
  the same AUC on DINO-global features (is the cue *in the input* vs *captured by DINO*).
- **Where:** `data/merged_final` clean 502 (dev 1214 / test 337, sealed), `_local{256,512}_cache.npy`.
- **Result:** **M-rep0 ADOPTED.** dev-CV top1: global 28.9±3.0 → **global+L256 33.5±2.9, paired Δ +4.65,
  10/10** (global+L512 +3.18, global+L256+L512 +3.45, all 10/10; **local-only −1.75, 2/10** → global
  region-context still required, so multiscale = both). **Sealed TEST: global+L256 top1 36.1
  (CI 30.1–42.0) vs global 33.5** — the first leak-safe gain since the 041 correction.
  **M-rep0b(a):** tissue-oracle 35.3 vs 28.9 → **Δ +6.4 pp** (a tissue gate is worth up to +6.4 → M-rep2
  has headroom). **M-rep0b(b):** artery-vs-vein AUC **colour/texture 0.771 | DINO-global 0.762** (n=430)
  — both ≫ 0.5, so (i) the cue **is in the input** (not the DX3 information dead-end), and (ii) DINO
  *does* encode artery/vein **linearly** (0.76), the failure in 042 was the *centroid-proximity /
  nearest-exemplar readout* (cos 0.88, NN crosses regions), not absence of the axis.
- **Conclusion:** the representation axis is **open** — bottleneck includes **input resolution**, and
  high-res local zoom is a real, leak-safe lever (+4.65 dev / +2.6 sealed, 10/10). New best =
  **global+L256**. Two more green gates: a **tissue-aware readout** (oracle +6.4 → M-rep2 soft-gate
  hierarchical retrieval) and **colour is exploitable** (AUC 0.77 → M-rep3 / a learned tissue axis).
  The artery/vein problem was mis-framed as "DINO can't see it" — DINO sees it (0.76), the *readout*
  throws it away. Next: stack the tissue-aware readout on top of global+L256.
- **Reproduce:** `scripts/multiscale_local.py`.

### 046 — M-rep2: tissue-aware soft-gate readout (realizing the +6.4 oracle) — honest negative
- **When:** 2026-06-28.
- **Why:** 045 showed the tissue cue IS in DINO (artery/vein AUC 0.76) and a tissue-oracle has +6.4pp
  headroom. Realize it with a REAL (imperfect) tissue classifier, SOFT-gated (a hard gate propagates
  Stage-1 error). Tissue-level avoids the class-level SupCon trap (367 samples/tissue, not 4.4/class).
- **What & How:** `final(c) = s_exemplar(c) + λ·log P(tissue(c)|q)` on global+L256; P from a 6-way
  LogReg head (artery/vein/nerve/muscle/bone/other) trained **per-fold on the train gallery only**
  (leak-safe); λ tuned on dev 10-seed CV. Variants: plain soft, confidence-modulated (λ·maxP), hard
  top-1, hard top-2. Additivity tracked: global → +L256 → +tissue.
- **Result:** **no adoption.** dev-CV: global+L256 33.5 → soft-gate (λ=0.05) **33.4 (Δ−0.17, 3/10)**;
  confidence-gate 33.9 (Δ+0.35, **6/10** — right direction, fails the ≥7/10 bar); hard top-1 **26.5
  (−7.0)**, top-2 30.9 (−2.6) = error propagation, exactly what soft was meant to avoid. **Stage-1
  tissue acc 65.5%.** λ-curve monotonically decreasing from λ=0. Sealed test: global 33.5 → +L256
  36.1 → +soft 35.7 (no gain).
- **Conclusion:** the +6.4 oracle assumed *perfect* tissue; a 65.5% classifier injects more error than
  signal. Deeper reason: the exemplar on DINO **already exploits the implicit tissue axis** (AUC 0.76),
  so an explicit, *less* accurate gate is redundant and only adds noise — and when it's confidently
  wrong it down-weights the true class. **The artery/vein lever is NOT a post-hoc tissue gate.** soft≫
  hard confirms the design (avoid hard gates); the signal just isn't there at this classifier quality.
  Forward: a *learned* representation that reshapes same-tissue together (M-rep1, tissue contrastive/
  LoRA — untested), or more resolution/data; not an explicit readout gate.
- **Reproduce:** `scripts/multiscale_readout.py`.

### 047 — M-rep0c: relational-axis revival (040 re-run on multi-pin data) — crack #0 solved, axis still capped
- **When:** 2026-06-28.
- **Why:** 040 closed the relational axis 🔴 but the cause was crack #0 (58% of pages single-pin → no
  co-present relational neighbour), explicitly *held* for data expansion. merged_final has multi-pin
  66.6% — crack #0 should be dissolved. Re-run the *identical* 040 oracle to test if the realistic
  relational ceiling is now measurable.
- **What & How:** same training-free oracle (predicates imported verbatim from `confusion_pairs.py`):
  of the engine's errors, the share where a co-present same-image pin is a graph-partner of the true
  label AND the model swapped to it AND appearance kept true ≤ rank-3 (a tie-breaker can only flip it
  then). New data + new best engine (global+L256) + image co-presence + leak-safe block split. Same
  pre-registered gate as 040.
- **Result:** **crack #0 dissolved** (multi-pin 42%→**66.6%**, single-pin 58%→33%). TIGHT ceiling rose
  040 +7.0 → 047 **+18.5pp** (resolvable neighbours now usually present). But the **realistic ceiling**
  (pred=co-present partner & true≤rank3) is only 040 +0.4 → **+0.8pp (1.8 pins/seed)**, still under
  σ=2.9pp. pred-is-partner just 1.7pp; direction-dependent share 38.1% (crack #2 persists). Found the
  real NAVEL pairs (suprascapular a.↔n. ×4, renal a.↔v. ×3) but they're rare. 🔴 **STOP (still).**
- **Conclusion:** dissolving crack #0 doubled the realistic ceiling but it remains buried — because the
  model's errors are mostly **not** co-present-partner swaps with true near the top; they're cross-region
  look-alikes a positional rule can't touch, and crack #2 (direction-dependence, not invariant under 2D
  projection) erodes 38% of what's left. The relational axis is **not the lever even with multi-pin
  data**; it would need bundle-colabeled pages targeted on purpose (femoral/renal/suprascapular triangles),
  not generic multi-pin density. Honest negative; hold (not discard).
- **Reproduce:** `scripts/multiscale_relational.py`.

### 048 — M-rep0 refinement: the resolution lever is saturated at 045 (clean negative)
- **When:** 2026-06-28.
- **Why:** extend the one delivering lever (M-rep0). 045 found stacking *more* scales dilutes
  (global+L256+L512 < global+L256), so the lever is a *better single local crop*, not more scales.
  Four cheap probes on top of global+L256: (a) α-weighted fusion `unit([z_g ; α·z_l])`, (b) tighter
  L128 zoom, (c) fraction crop (0.25·min(H,W), consistent zoom across 39–2995 px images), (d) local
  CLS pooling vs σ40-at-centre.
- **What & How:** re-embed L128 / frac0.25 / L256-CLS (cached), α-sweep free on cached global+L256.
  dev 10-seed CV select, sealed test once. Baseline = global+L256 (045).
- **Result:** **no additional gain.** best = global+α·L256 (α=0.7) dev **33.8 (Δ+0.25 vs L256, 6/10)** —
  fails the ≥7/10 bar; sealed test 35.3 < L256's 36.1 (within CI). All other refinements WORSE:
  L128 29.3 (−4.26), L256-CLS 28.7 (−4.86, σ40-centre ≫ CLS — the structure is at the crop centre),
  frac0.25 33.0 (−0.54), L128+L256 30.5 (−3.04, stacking dilutes again).
- **Conclusion:** the resolution lever **captured essentially all its gain in one shot (045, global+L256,
  sealed 36.1) and is now saturated.** A 256-px q-centred crop, σ40-pooled at centre, fused equal-weight
  with the global embedding, is the single-local sweet spot; tighter / CLS / fraction / extra-scale all
  regress. The α=0.7 hint is inside split noise. Resolution axis closed. Forward narrows to a *learned*
  representation (M-rep1 tissue-contrastive/LoRA, untested) or the validated data lever — not more
  resolution tuning.
- **Reproduce:** `scripts/multiscale_refine.py`.

### 049 — M-rep1: learned representation reshape (SupCon head) — the last representation lever, negative
- **When:** 2026-06-28.
- **Why:** the only representation lever never tried — *learn* a reshape so the nearest exemplar stops
  crossing regions/tissues. 046 warned the exemplar already uses DINO's implicit tissue axis (AUC 0.76),
  so a head must ADD beyond it; the class-level SupCon trap (~1.7 samples/fold-class) is real. Mitigate
  per handout §0.2: train at TISSUE level (367/tissue) and a HIERARCHICAL tissue+class objective, and
  preserve frozen region structure by also evaluating [frozen ; head] concat.
- **What & How:** SupCon head (MLP 1536→512→128) on the frozen global+L256, trained **per-fold on the
  gallery only** (leak-safe), τ=0.1, 120 epochs. Objectives tissue / class / hierarchical; spaces
  head-only and frozen⊕head. Exemplar 1-NN, dev 10-seed CV select, sealed test once. Baseline = frozen
  global+L256 (045: dev 33.5 / sealed 36.1).
- **Result:** **every variant loses.** best = class:frozen+head dev **31.2 (Δ−2.35, 0/10)**;
  hier:frozen+head 29.6 (−3.96, 1/10); class:head 28.5 (−5.0); tissue-only collapses (head 6.6, −27 —
  pulling all arteries together across regions destroys class identity); tissue:frozen+head 26.3 (−7.2).
  Sealed: frozen 36.1 → best 34.2 (CI 28.6–40.1). No objective, no space ≥ frozen.
- **Conclusion:** a learned reshape does **not** beat the frozen exemplar — 046's lesson confirmed at the
  *learning* level: the frozen DINO+L256 already captures what is separable on this data, and a
  contrastive head overfits / destroys the region structure (top5) rather than adding within-region
  tissue id. The SupCon trap holds even at tissue level and with region-preserving concat. **The
  representation axis is exhausted except the single resolution win (045, global+L256, sealed 36.1).**
  The remaining validated lever is **data** (project through-line §2 — data is the ceiling): readout (043),
  reliability (037), cross-cadaver (038), relational (040/047), and now representation reshaping (046/049)
  all converge there; only data scale (041: +8.9/+10.1) and input resolution (045: +2.6) ever moved it.
- **Reproduce:** `scripts/learned_reshape.py`.

### 050 — M-rep1 (LoRA): backbone last-block reshape — overfits; a textbook sealed-test validation
- **When:** 2026-06-28.
- **Why:** 049 trained a head on the *frozen* pooled vector (can only re-project). LoRA is the distinct,
  more-expressive half the handout named — low-rank adapters on the last DINO block change the patch
  tokens *before* pooling, so the pooled embedding itself can carry new information. Test if backbone
  adaptation (not a head) beats frozen.
- **What & How:** manual LoRA (rank 8) on the last block {qkv, proj, fc1, fc2} + a linear class head,
  class cross-entropy. Cost control: cache the input to the last block once (frozen partial forward),
  LoRA trains only the last block on cached tokens (batch 16, 12 epochs, ~1 min/epoch once memory was
  freed — the first attempt thrashed at batch 64 + a 1.3 GB RAM cache, pushing the 36 GB box to 39 GB).
  **Leak design:** LoRA trains on **dev only** → the **sealed test is clean** (test never seen); dev-CV
  (LoRA trained on all dev) is the *optimistic* upper bound (gallery+query both in dev).
- **Result:** **dev-CV (optimistic): LoRA-only 45.5 (Δ+11.95, 10/10), LoRA+L256 45.2 (Δ+11.71, 10/10)** —
  looks like a huge win. **Sealed test (clean): frozen 36.1 → LoRA-only 24.5, LoRA+L256 30.5** — LoRA
  generalizes *worse* than frozen. The **17.6 pp gap** between optimistic (+12) and clean (−5.6) **is the
  overfitting**, made visible.
- **Conclusion:** 🔴 LoRA does not beat frozen — it memorizes the dev classes (4.4 samples/class) and the
  reshaping doesn't transfer, exactly §2's "added capacity overfits." **M-rep1 is fully negative (head
  049 + LoRA 050); the representation axis is closed except the one resolution win (045).** Equally
  important, this is a **textbook validation of the sealed-test protocol (§1.7)**: the optimistic dev-CV
  would have falsely declared a +12 pp breakthrough; only the sealed test exposed the −5.6 pp reality.
  Across the whole program, only data scale (041: +8.9/+10.1) and input resolution (045: +2.6) ever moved
  the leak-safe ceiling. The remaining validated lever is **data**.
- **Reproduce:** `scripts/lora_reshape.py`.

## Phase 15 — Autonomous search: training-free retrieval & embedding theory (overnight)

### 051 — retrieval re-ranking (CSLS / AQE / local-scaling) — CSLS adopted, a small principled win
- **When:** 2026-06-28 (autonomous).
- **Why:** the readout is gallery retrieval; the re-ID literature has training-free rerankers for two
  known kNN failure modes — hubness (CSLS: some gallery vectors are everyone's neighbour) and
  non-reciprocity (AQE/k-reciprocal). Pure linear algebra on cached embeddings (cheap-probe-first).
- **What & How:** modify the instance-instance similarity, keep the class-max readout. CSLS(q,g) =
  2cos − r_q − r_g (r = mean of top-k sims); AQE (augment query with top-m gallery mean); local-scaling
  (self-tuning by kNN scale). Swept k/m. dev 10-seed CV paired vs cosine baseline (global+L256, 33.5).
- **Result:** **CSLS k=5 adopted** — dev-CV **34.1 (Δ+0.61, 7/10)**; sealed test **36.1 → 38.3**. CSLS
  k=8/15 also positive but <7/10. AQE hurt (m=3 −2.32, m=7 −3.45 — query expansion pulls in wrong-class
  neighbours). local-scaling marginal (+0.37, 6/10).
- **Conclusion:** mild hubness exists and CSLS corrects it for a small, principled gain — the second
  leak-safe lever after resolution. New best = global+L256 + CSLS(k=5) readout, **sealed 38.3** (effect
  small, CI overlaps 36.1, so report as modest not decisive). AQE's failure confirms the errors are not
  fixed by neighbourhood averaging (consistent with 047 — partners aren't co-retrieved).
- **Reproduce:** `scripts/rerank_retrieval.py`.
