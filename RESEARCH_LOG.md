# SCALPEL — Detailed Research Log (5W1H per experiment)

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
