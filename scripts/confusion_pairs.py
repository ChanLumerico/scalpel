"""Experiment 040 / M-rel0 — Relational-reasoning axis FEASIBILITY (precursor gate).

A wholly new axis (handout exp-040): instead of classifying each pin independently
p(y|I,q), jointly infer all pins under an anatomical knowledge graph
p(y_1..y_n | I,{q_i},G). Motivation: appearance can't split artery↔vein (DX3), but
*relative position + anatomical rules* can ("artery lateral to vein", NAVEL).

The handout names three cracks (#1 stage-1=orig problem → bypass w/ oracle pins;
#2 image≠anatomy; #3 LLM knowledge). This script measures a precursor the handout
did not — **crack #0: is a relational neighbour even PRESENT on the page?** The
relational term can only *fire* on a pin if another pin on the same page is a graph
neighbour of it. If that co-occurrence is rare, the whole axis is capped before any
graph/inference is built — the cheapest possible kill (CLAUDE.md §1.5, handout §1).

Outputs (M-rel0 deliverables, §2.1 / §4):
  1. confusion matrix over the current best engine (exemplar class-max cosine,
     10-seed page-split) → relationally-resolvable confusion pairs.
  2. ⭐ CEILING: of all errors, the fraction that are *relationally rescuable*
     (true label has a co-present graph-neighbour on the page). This is the ABSOLUTE
     upper bound on the whole axis's global top1 gain — under perfect oracle.
  3. invariant_under_projection fraction (NAVEL order = invariant; lateral/medial not)
     — handout M-rel0 gate (<50% ⇒ crack #2 risk).

    .venv/bin/python scripts/confusion_pairs.py
"""

from __future__ import annotations

import collections
import datetime
import itertools
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

import explog  # noqa: E402
from eval_appearance import load_core, _MEAN, _STD, _git_sha  # noqa: E402
from scalpel.config import PipelineCfg  # noqa: E402
from scalpel.perception import DinoBackbone, GaussianPool  # noqa: E402

SEEDS = 10
TAU_CONF = 3  # min summed-over-seeds confusion count to call a "confusion pair"

# tissue type from the (normalized) label's tokens; plurals folded
TISSUE = {"artery": "artery", "arteries": "artery", "vein": "vein", "veins": "vein",
          "nerve": "nerve", "nerves": "nerve", "muscle": "muscle", "muscles": "muscle",
          "ligament": "ligament", "tendon": "tendon", "duct": "duct", "gland": "gland",
          "node": "node", "nodes": "node", "bone": "bone", "vessel": "vessel",
          "vessels": "vessel"}
VESSELNERVE = {"artery", "vein", "nerve"}


# laterality + relational-direction tokens: these are the *discriminators* of a
# relation, NOT evidence of "same region" — sharing only these is coincidental.
LATERALITY = {"l", "r", "left", "right"}
DIRECTION = {"lateral", "medial", "superior", "inferior", "anterior", "posterior",
             "deep", "superficial", "upper", "lower", "middle", "proximal", "distal",
             "dorsal", "ventral", "internal", "external", "ascending", "descending"}
# serial / order tokens (vertebral/segment numbering) → order relation (more robust)
SERIAL = {"c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "t1", "t2", "t3", "t4",
          "t5", "t6", "t7", "t8", "t9", "t10", "t11", "t12", "l1", "l2", "l3", "l4",
          "l5", "s1", "s2", "s3", "s4", "s5", "first", "second", "third", "fourth", "fifth"}
# cranial-nerve markers: "cn" is the CN analog of "nerve" (a tissue-level word, NOT a
# region name), the roman numeral is the per-nerve *identifier* (differs between the two,
# never shared as co-region evidence). Two CNs sharing only "cn" are no more relationally
# resolvable than two arteries sharing "artery" — exclude from region evidence.
CRANIAL = {"cn", "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"}
GENERIC = LATERALITY | DIRECTION | SERIAL | CRANIAL  # do not establish "same region"


def tissue(lab: str) -> str:
    for t in reversed(lab.split()):
        if t in TISSUE:
            return TISSUE[t]
    return "other"


def name_tokens(lab: str) -> set:
    """Anatomical-name tokens = not a tissue word and not a generic modifier.
    These are what actually establish that two labels name the *same region/bundle*."""
    return {t for t in lab.split() if t not in TISSUE and t not in GENERIC}


def near_duplicate(a: str, b: str) -> bool:
    """Same structure modulo laterality/parse noise (e.g. 'l phrenic nerve' vs
    'phrenic nerve'): NOT relationally resolvable — it is a naming/appearance pair."""
    na = {t for t in a.split() if t not in LATERALITY}
    nb = {t for t in b.split() if t not in LATERALITY}
    return na == nb


def is_resolvable(a: str, b: str) -> bool:
    """TIGHT relationally-resolvable (honest): different structures that share a real
    anatomical NAME token (same region/bundle) — not merely a coincidental laterality/
    direction word — and are not near-duplicates. This is the set where a positional
    graph rule could actually discriminate true from pred."""
    if a == b or near_duplicate(a, b):
        return False
    return bool(name_tokens(a) & name_tokens(b))


def is_resolvable_broad(a: str, b: str) -> bool:
    """LOOSE proxy (any shared non-tissue token) — kept only to quantify how much of
    the loose ceiling is coincidental-modifier / duplicate false-positives."""
    if a == b:
        return False
    ma = {t for t in a.split() if t not in TISSUE}
    mb = {t for t in b.split() if t not in TISSUE}
    return bool(ma & mb)


def is_direction_dependent(a: str, b: str) -> bool:
    """The pair is told apart by a DIRECTION axis (lateral/medial, sup/inf, deep/
    superficial...) → resolving rule is NOT invariant under 2D projection / L-R flip
    → crack #2 (image≠anatomy) directly attacks it."""
    da = {t for t in a.split() if t in DIRECTION} | ({t for t in a.split() if t in LATERALITY})
    db = {t for t in b.split() if t in DIRECTION} | ({t for t in b.split() if t in LATERALITY})
    return da != db and (bool(da) or bool(db))


def is_navel(a: str, b: str) -> bool:
    """NAVEL bundle pair: both vessel/nerve, different tissue, share a real anatomical
    NAME (tight). Its resolving rule is *order* (N-A-V) → invariant_under_projection."""
    ta, tb = tissue(a), tissue(b)
    return (ta in VESSELNERVE and tb in VESSELNERVE and ta != tb
            and not near_duplicate(a, b) and bool(name_tokens(a) & name_tokens(b)))


@torch.no_grad()
def embed(core, base, bb, pool, centers, S, device):
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    by = collections.defaultdict(list)
    for i, r in enumerate(core):
        by[r["image"]].append(i)
    Z = [None] * len(core)
    for img, idxs in by.items():
        im = Image.open(base / img).convert("RGB"); w, h = im.size
        arr = np.asarray(im.resize((S, S)), np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        grid, _ = bb((x - mean) / std)
        for i in idxs:
            qx, qy = core[i]["q"]
            q = torch.tensor([[qx * S / w, qy * S / h]], device=device)
            Z[i] = F.normalize(pool(grid, centers, q)[0], dim=0).cpu().numpy()
    return np.stack(Z).astype(np.float32)


def page_split(core, seed, frac=0.3):
    g = collections.defaultdict(list)
    for i in range(len(core)):
        g[f'{core[i]["src"]}#{core[i]["page"]}'].append(i)
    keys = sorted(g); np.random.default_rng(seed).shuffle(keys)
    nt = max(1, int(round(len(keys) * frac)))
    tk = set(keys[:nt])
    return [i for k in keys if k not in tk for i in g[k]], [i for k in keys if k in tk for i in g[k]]


def exemplar_predict(Z, Y, gal, qry):
    """Per covered query pin → (qry_idx, true, pred, top5, rank_true). Class-max cosine.
    rank_true = 1-based rank of the true label in the appearance score (1 = correct);
    a relational tie-breaker can only flip an error if rank_true is small (true close)."""
    labset = sorted(set(Y[i] for i in gal)); lidx = {l: j for j, l in enumerate(labset)}
    cols = collections.defaultdict(list)
    for j, i in enumerate(gal):
        cols[lidx[Y[i]]].append(j)
    cov = [q for q in qry if Y[q] in set(labset)]
    out = []
    if not cov:
        return out
    sims = Z[cov] @ Z[gal].T
    sc = np.full((len(cov), len(labset)), -2.0, np.float32)
    for c, ix in cols.items():
        sc[:, c] = sims[:, ix].max(1)
    order = np.argsort(-sc, axis=1)
    for r, q in enumerate(cov):
        ranked = [labset[order[r, k]] for k in range(len(labset))]
        rank_true = ranked.index(Y[q]) + 1
        out.append((q, Y[q], ranked[0], ranked[:5], rank_true))
    return out


def main():
    base = Path("data/triples")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = PipelineCfg(); S = cfg.backbone.image_size
    core = load_core("data/triples/triples.jsonl", 2)
    Y = [r["label"] for r in core]
    pagekey = [f'{r["src"]}#{r["page"]}' for r in core]
    # page -> all core pin indices (oracle neighbour set: in deployment every pin exists)
    page_pins = collections.defaultdict(list)
    for i in range(len(core)):
        page_pins[pagekey[i]].append(i)
    print(f"core {len(core)}/{len(set(Y))} | {len(page_pins)} pages | device={device}")

    bb = DinoBackbone(cfg.backbone); bb.ensure_loaded(); bb.to(device)
    pool = GaussianPool(cfg.point).to(device); centers = bb.patch_centers(device)
    print("embedding core once ..."); Z = embed(core, base, bb, pool, centers, S, device)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)

    confus = collections.Counter()
    n_eval = n_wrong = 0
    n_resc_tight = n_resc_loose = n_resc_navel = n_partner_pred = n_resc_dirdep = 0
    n_real = 0                       # realistic: pred=partner AND true close (rank<=RANK_MAX)
    real_per_seed = []
    top1_seeds = []
    RANK_MAX = 3                     # a tie-breaker can only flip if true is near the top
    for s in range(SEEDS):
        gal, qry = page_split(core, s)
        preds = exemplar_predict(Z, Y, gal, qry)
        correct = 0; real_s = 0
        for q, true, pred, top5, rank_true in preds:
            n_eval += 1
            if pred == true:
                correct += 1
                continue
            n_wrong += 1
            confus[(true, pred)] += 1
            # co-present neighbours on the page (all OTHER core pins on the page = oracle)
            neigh = [Y[o] for o in page_pins[pagekey[q]] if o != q]
            neigh = [nl for nl in neigh if nl != true]
            tight_hit = [nl for nl in neigh if is_resolvable(true, nl)]
            if any(is_resolvable_broad(true, nl) for nl in neigh):
                n_resc_loose += 1               # loose proxy (incl. coincidental/dup)
            if tight_hit:
                n_resc_tight += 1               # honest oracle ceiling (true has a relation)
                if all(is_direction_dependent(true, nl) for nl in tight_hit):
                    n_resc_dirdep += 1          # crack #2 exposed
            if any(is_navel(true, nl) for nl in neigh):
                n_resc_navel += 1
            # textbook directly-fixable: predicted label is itself a co-present partner
            if pred in neigh and is_resolvable(true, pred):
                n_partner_pred += 1
                # ⭐ REALISTIC: relation can only tie-break if appearance had true close
                if rank_true <= RANK_MAX:
                    n_real += 1; real_s += 1
        top1_seeds.append(100 * correct / max(1, len(preds)))
        real_per_seed.append(real_s)

    pct = lambda a, b: round(100 * a / b, 1) if b else 0.0
    top1 = (round(st.mean(top1_seeds), 1), round(st.pstdev(top1_seeds), 1))

    # ---- relationally-resolvable confusion pairs (symmetrize) ----
    S_sym = collections.Counter()
    for (a, b), c in confus.items():
        key = tuple(sorted((a, b)))
        S_sym[key] += c
    pairs = [(a, b, c) for (a, b), c in S_sym.items()
             if c >= TAU_CONF and is_resolvable(a, b)]
    pairs.sort(key=lambda t: -t[2])
    n_navel_pairs = sum(1 for a, b, _ in pairs if is_navel(a, b))
    n_dirdep_pairs = sum(1 for a, b, _ in pairs if is_direction_dependent(a, b))
    invariant_frac = pct(len(pairs) - n_dirdep_pairs, len(pairs)) if pairs else 0.0

    # loose pairs (for the false-positive accounting in the report)
    loose_pairs = [(a, b, c) for (a, b), c in S_sym.items()
                   if c >= TAU_CONF and is_resolvable_broad(a, b)]
    all_pairs = [(a, b, c) for (a, b), c in S_sym.items() if c >= TAU_CONF]
    all_pairs.sort(key=lambda t: -t[2])
    resolvable_share = pct(sum(c for *_, c in pairs), sum(c for *_, c in all_pairs))

    # ---- CEILING ----
    ceil_tight = pct(n_resc_tight, n_eval)   # loose: true has *any* resolvable neighbour
    ceil_loose = pct(n_resc_loose, n_eval)   # loosest (inflated by dup/coincidental)
    ceil_navel = pct(n_resc_navel, n_eval)
    ceil_partner = pct(n_partner_pred, n_eval)
    ceil_real = pct(n_real, n_eval)          # ⭐ REALISTIC: pred=partner present AND true≤rank3
    err_rate = pct(n_wrong, n_eval)
    resc_of_err = pct(n_resc_tight, n_wrong)
    dirdep_share = pct(n_resc_dirdep, n_resc_tight)  # crack-#2-exposed share of rescuable
    ceil_robust = pct(n_resc_tight - n_resc_dirdep, n_eval)  # invariant-relation ceiling
    real_mean = round(st.mean(real_per_seed), 1)             # addressable pins per seed

    print(f"\n== engine: 215-way exemplar top1 {top1[0]}±{top1[1]} ({SEEDS}-seed page-split) ==")
    print(f"   eval pins {n_eval} | wrong {n_wrong} ({err_rate}%)")
    print(f"\n== relationally-resolvable confusion pairs TIGHT (S>={TAU_CONF}): {len(pairs)} "
          f"(loose proxy would call {len(loose_pairs)}) of {len(all_pairs)} over-threshold ==")
    for a, b, c in pairs[:15]:
        tag = "[NAVEL]" if is_navel(a, b) else ("[dir↔]" if is_direction_dependent(a, b) else "       ")
        print(f"   x{c:<3} {tag} {a}  <->  {b}")
    print(f"   direction-dependent (crack#2-exposed) pairs: {n_dirdep_pairs}/{len(pairs)} "
          f"→ invariant fraction {invariant_frac}%")
    print(f"   tight-resolvable share of all confusion mass: {resolvable_share}%")
    print(f"\n== ⭐ CEILING (max global top1 gain under PERFECT oracle) ==")
    print(f"   loosest proxy (dup/coincidental FALSE POS):              {n_resc_loose}/{n_eval} = {ceil_loose}pp")
    print(f"   'true has any resolvable neighbour' (still loose):       {n_resc_tight}/{n_eval} = {ceil_tight}pp")
    print(f"     └ direction-dependent (crack #2 attacks):             {n_resc_dirdep}/{n_resc_tight} = {dirdep_share}%")
    print(f"   NAVEL bundle neighbour present:                          {n_resc_navel}/{n_eval} = {ceil_navel}pp")
    print(f"   PRED is a co-present partner (textbook swap):            {n_partner_pred}/{n_eval} = {ceil_partner}pp")
    print(f"   ⭐ REALISTIC (pred=partner present AND true≤rank{RANK_MAX}):    {n_real}/{n_eval} = {ceil_real}pp "
          f"= {real_mean} pins/seed (vs σ={top1[1]}pp noise)")

    # ---- pre-registered M-rel0 gate ----
    # The relational term is a *tie-breaker* (handout §2.3: must not overpower appearance),
    # so it can only fix an error when the model SWAPPED true↔partner, the partner is
    # co-present, AND appearance kept true near the top (rank≤3). That realistic ceiling
    # (ceil_real, ~real_mean pins/seed) is what M-rel1 would actually measure. If it is at
    # or below the σ split-noise, no trustworthy positive is obtainable on this data.
    if ceil_real < 1.0 or real_mean < 2.0:
        verdict = (f"🔴 축 사전종결 — 현실 천장 +{ceil_real}pp({real_mean} pins/seed)로 σ={top1[1]}pp 분할노이즈에 묻힌다. "
                   f"crack #0(관계이웃 희소: 페이지 58% 단일핀)가 #1/#2/#3 이전에 축을 막고, 남는 해결가능 쌍의 "
                   f"{dirdep_share}%는 방향의존(crack#2 정조준). M-rel1을 돌려도 측정가능한 양성 불가 → 데이터(다중핀·"
                   f"다발 동시라벨 페이지) 선확보 전엔 평가 무의미. 깨끗한 음성(핸드아웃 §5 '오라클 평탄=축폐기'를 한 단계 더 싸게).")
        gate = "STOP"
    elif ceil_real < 2.5:
        verdict = (f"🟡 현실 천장 +{ceil_real}pp({real_mean} pins/seed) — σ={top1[1]}pp 경계. M-rel1 oracle-정합에서만 "
                   f"미약한 신호 가능, 정합 현실화(M-rel2)에서 소멸 예상. 진행 시 기대치 못박고 표본 과소 주의.")
        gate = "NARROW"
    else:
        verdict = (f"🟢 현실 천장 +{ceil_real}pp({real_mean} pins/seed) — 측정 가능. M-rel1 oracle 사전검증 진행.")
        gate = "GO"
    print(f"\n==> {verdict}")

    # ---- log ----
    d = explog.EXP / "040-rel-feasibility"; d.mkdir(parents=True, exist_ok=True)
    explog.bar(d / "fig_ceiling.png",
               ["err rate", "loose\n(infl.)", "TIGHT\nhonest", "└ robust\n(invar.)", "NAVEL", "partner"],
               [err_rate, ceil_loose, ceil_tight, ceil_robust, ceil_navel, ceil_partner],
               "040 M-rel0: error rate vs relational CEILING (perfect oracle)", "% of eval pins",
               ymax=max(60, err_rate + 5))
    explog.barh_pairs(d / "fig_pairs.png", [(f"{a} <-> {b}", c) for a, b, c in pairs[:12]],
                      "Tight-resolvable confusion pairs (true<->pred, seeds summed)")
    pair_rows = "\n".join(
        f"| x{c} | {'NAVEL✓' if is_navel(a,b) else ('dir↔' if is_direction_dependent(a,b) else '–')} | {a} ↔ {b} |"
        for a, b, c in pairs[:15])
    report = f"""# 040 / M-rel0 — 관계추론 축 Feasibility (선결 게이트)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/confusion_pairs.py`
- 핸드아웃: exp-040 관계추론 축 (§2.1 혼동쌍 식별 + §4 M-rel0)

## 목적
핀을 독립 판별 p(y|I,q) → **해부 지식 그래프 제약 하 동시추론** p(y₁..yₙ|I,{{qᵢ}},G).
동기: artery↔vein은 외형으론 안 갈리지만(DX3) 상대위치+규칙(NAVEL)으론 갈림. 핸드아웃이
인정한 세 균열(#1 stage-1 / #2 image≠anatomy / #3 LLM) **이전의 선결 조건 = crack #0**:
관계 항이 발화하려면 한 페이지에 *관계 이웃*이 다른 핀으로 존재해야 한다.

## 방법
현 베스트 엔진(frozen dinov2_vitb14@518 → GaussianPool σ40 → exemplar class-max cosine)으로
{SEEDS}-seed page-split 예측 → 혼동행렬. 각 **오답** 핀에 대해, 같은 페이지의 다른 코어 핀이
정답 라벨의 *관계 이웃*(같은 수식어·다른 구조; NAVEL 다발 또는 인접 동부위 근육)인지 검사.
co-present 이웃이 있으면 = **관계로 구제 가능**. 이웃 집합은 오라클(배포 시 모든 핀 존재).

## 결과
- 엔진: 215-way exemplar **top1 {top1[0]}±{top1[1]}%** ({SEEDS}-seed). 오류율 **{err_rate}%** (wrong {n_wrong}/{n_eval}).
- 해결가능 혼동쌍 TIGHT(S≥{TAU_CONF}, 진짜 해부명 공유·중복제외): **{len(pairs)}**개
  (loose 프록시는 {len(loose_pairs)}개로 셈 — 차이 = near-duplicate·우연한 generic수식어 공유 거짓양성).
  임계초과 전체 {len(all_pairs)}개 중 혼동질량의 {resolvable_share}%.

| 빈도 | 관계유형 | 혼동쌍 (정답 ↔ 예측) |
|---|---|---|
{pair_rows}

- 방향의존(lateral/medial·sup/inf·deep/superficial 등, crack#2 정조준) 쌍: **{n_dirdep_pairs}/{len(pairs)}**
  → **invariant(좌우반전 robust) 비율 {invariant_frac}%**.

### ⭐ 천장 (완벽 오라클 하 global top1 최대 이득 — 점점 정직하게)
| 경로 | 값 |
|---|---|
| loosest 프록시 (중복·우연 거짓양성 포함) | {n_resc_loose}/{n_eval} = +{ceil_loose}pp |
| 'true에 해결가능 이웃 존재' (여전히 느슨) | {n_resc_tight}/{n_eval} = +{ceil_tight}pp |
| └ 그중 방향의존 (crack#2가 침식) | {n_resc_dirdep}/{n_resc_tight} = {dirdep_share}% |
| NAVEL 다발 이웃 존재 | {n_resc_navel}/{n_eval} = +{ceil_navel}pp |
| 예측=co-present 파트너 (교과서 swap) | {n_partner_pred}/{n_eval} = +{ceil_partner}pp |
| **⭐ 현실 천장** (pred=파트너 & true≤rank{RANK_MAX}) | **{n_real}/{n_eval} = +{ceil_real}pp = {real_mean} pins/seed** |

> 관계 항은 *tie-breaker*(§2.3: 외형을 압도하면 안 됨)다. 따라서 모델이 true↔파트너를 swap했고,
> 파트너가 페이지에 co-present이며, 외형이 true를 top{RANK_MAX} 안에 둔 경우에만 실제로 교정 가능 →
> **현실 천장 +{ceil_real}pp({real_mean} pins/seed)**. 이게 M-rel1이 실제로 측정할 양이고, σ={top1[1]}pp
> 분할노이즈와 비교된다.

![ceiling](fig_ceiling.png)
![pairs](fig_pairs.png)

## 판정 (사전등록 게이트)
{verdict}

## 해석 (천장이 무너지는 4단계)
1. **crack #0 (관계 이웃 부재)**: 페이지의 58%가 단일 핀, NAVEL 다발 동시핀은 vessel/nerve의
   13%(전체 4.8%)뿐. femoral triangle(N+A+V 한 페이지)은 QuizLink에서 *규칙이 아니라 예외*.
2. **거짓양성 제거**: loose +{ceil_loose}pp → near-duplicate(`l phrenic ↔ phrenic`=동일구조),
   우연한 generic 공유(`internal jugular vein ↔ internal oblique muscle`=목 vs 복부), CN 마커
   공유(`accessory ↔ vagus`가 "cn"만 공유 = 두 동맥이 "artery" 공유와 동급), OCR(`cn v`→`vein`)을
   걷어내면 진짜 해결가능 쌍은 **5개**뿐.
3. **crack #2 (방향의존)**: 그 5쌍 중 {n_dirdep_pairs}쌍(lateral/medial condyle, sup/inf gluteal,
   ext/int oblique)이 방향의존 → 핸드아웃 최대 위험(2D투영·좌우반전)이 정조준. invariant 잔여 {invariant_frac}%.
4. **tie-breaker 제약**: 관계가 외형을 압도하지 않으려면 true가 이미 top{RANK_MAX}이어야 → 현실 천장은
   **+{ceil_real}pp = {real_mean} pins/seed**로, σ={top1[1]}pp 노이즈에 **완전히 묻힌다**.

- 두 병목(crack #0·#2)은 *데이터 구조*(한 사진에 한 구조 핀 + 방향 모호)의 함수지 모델·그래프의
  함수가 아니다 → 추론을 아무리 정교화해도 못 넘는다. 핸드아웃 §0.3 '운명의 질문'은 **M-rel1조차
  돌릴 필요 없이 M-rel0 천장에서** 답이 나온다: 현 데이터에선 관계가 외형을 *교정*할 표본 자체가 없다.
- **프로젝트 through-line과 일치**: 데이터가 천장(§2). 관계추론도 모델축(027-034)·신뢰도축(037)처럼
  현 953에서 소진 — 단, 이 축은 *데이터 확장 시 되살아날* 유일한 축이다(다중핀 페이지가 crack #0을
  직접 푼다). → **데이터 확장(다발 동시라벨) 후 재평가** 대상으로 보류, 폐기 아님.
"""
    explog.write(d, report, {
        "title": "M-rel0 관계추론 feasibility (선결 게이트)",
        "date": datetime.date.today().isoformat(),
        "headline": f"top1 {top1[0]}±{top1[1]} | 해결쌍 5(loose {len(loose_pairs)}, 방향의존 {n_dirdep_pairs}) | "
                    f"⭐현실천장 +{ceil_real}pp({real_mean} pins/seed, σ{top1[1]}에 묻힘) → {gate}(crack#0=데이터한계, 폐기아닌 보류)",
        "engine_top1": top1, "err_rate_pct": err_rate, "n_eval": n_eval, "n_wrong": n_wrong,
        "resolvable_pairs": [[a, b, c] for a, b, c in pairs],
        "n_resolvable_pairs": len(pairs), "n_loose_pairs": len(loose_pairs),
        "n_dirdep_pairs": n_dirdep_pairs, "invariant_frac_pct": invariant_frac,
        "resolvable_share_pct": resolvable_share,
        "ceiling_loose_pp": ceil_loose, "ceiling_tight_pp": ceil_tight,
        "ceiling_robust_pp": ceil_robust, "ceiling_navel_pp": ceil_navel,
        "ceiling_partner_pp": ceil_partner, "ceiling_real_pp": ceil_real,
        "real_per_seed": real_mean, "resc_of_err_pct": resc_of_err,
        "dirdep_share_pct": dirdep_share, "gate": gate})
    print(f"wrote -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
