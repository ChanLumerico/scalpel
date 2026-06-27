"""Experiment 047 — M-rep0c: relational-axis revival on the clean merged data (040 re-run).

040 pre-closed the relational axis with a 🔴 STOP, but the cause was crack #0 — 58% of QuizLink
pages were single-pin, so a relational neighbour was rarely co-present. The kill was explicitly
"hold, not discard: revive on data expansion (multi-pin pages directly dissolve crack #0)". The
BlueLink harvest did exactly that: merged_final has 531/710 multi-pin images (single-pin 58%→25%).
This re-runs the *identical* 040 oracle ceiling on the new leak-safe data + the new best engine
(global+L256, exp 045) to test whether the realistic relational ceiling is now measurable.

Same metric (training-free oracle): of the engine's errors, the share that are relationally
rescuable = pred is a co-present graph-partner of true AND appearance kept true within rank-3
(a relational tie-breaker can only flip it then). Pre-registered gate identical to 040.

    .venv/bin/python scripts/multiscale_relational.py
"""

from __future__ import annotations

import collections
import datetime
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json  # noqa: E402
import numpy as np  # noqa: E402

import explog  # noqa: E402
from eval_appearance import _git_sha  # noqa: E402
from eval_merged import load, block_split, BASE  # noqa: E402
from split_devtest import get_split  # noqa: E402
from confusion_pairs import (  # noqa: E402  reuse the exact 040 predicates
    exemplar_predict, is_resolvable, is_resolvable_broad, is_navel,
    is_direction_dependent, tissue)

SEEDS = 10
TAU_CONF = 3
RANK_MAX = 3


def unit(Z):
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)


def main():
    rows = load(); Y = [r["label"] for r in rows]
    split = get_split()
    cnt = collections.Counter(Y)
    core = [i for i in range(len(rows)) if cnt[Y[i]] >= 2]
    dev = [i for i in core if split[rows[i]["image"]] == "dev"]
    blk = json.loads((BASE / "_blocks.json").read_text())
    block = [blk[r["image"]] for r in rows]
    zg = unit(np.load(BASE / "_dino_cache.npy").astype(np.float32))
    zl = unit(np.load(BASE / "_local256_cache.npy").astype(np.float32))
    Z = unit(np.concatenate([zg, zl], 1))   # 045 best engine

    # oracle neighbour set = all OTHER core pins on the same IMAGE (deployment: all present)
    img_pins = collections.defaultdict(list)
    for i in core:
        img_pins[rows[i]["image"]].append(i)
    npin = collections.Counter(len(v) for v in img_pins.values())
    single = sum(1 for v in img_pins.values() if len(v) == 1)
    multi_frac = round(100 * (1 - single / len(img_pins)), 1)
    print(f"core {len(core)}/{len(set(Y[i] for i in core))} | images {len(img_pins)} | "
          f"single-pin {single} ({round(100*single/len(img_pins),1)}%) → multi-pin {multi_frac}% "
          f"(040 was 42% multi)")

    confus = collections.Counter()
    n_eval = n_wrong = 0
    n_loose = n_tight = n_navel = n_partner = n_dirdep = n_real = 0
    real_per_seed, top1_seeds = [], []
    for s in range(SEEDS):
        gal, qry = block_split(dev, block, s)            # leak-safe photo-block split on dev
        preds = exemplar_predict(Z, Y, gal, qry)
        correct = real_s = 0
        for q, true, pred, top5, rank_true in preds:
            n_eval += 1
            if pred == true:
                correct += 1; continue
            n_wrong += 1
            confus[(true, pred)] += 1
            neigh = [Y[o] for o in img_pins[rows[q]["image"]] if o != q and Y[o] != true]
            if any(is_resolvable_broad(true, nl) for nl in neigh):
                n_loose += 1
            tight_hit = [nl for nl in neigh if is_resolvable(true, nl)]
            if tight_hit:
                n_tight += 1
                if all(is_direction_dependent(true, nl) for nl in tight_hit):
                    n_dirdep += 1
            if any(is_navel(true, nl) for nl in neigh):
                n_navel += 1
            if pred in neigh and is_resolvable(true, pred):
                n_partner += 1
                if rank_true <= RANK_MAX:
                    n_real += 1; real_s += 1
        top1_seeds.append(100 * correct / max(1, len(preds)))
        real_per_seed.append(real_s)

    pct = lambda a, b: round(100 * a / b, 1) if b else 0.0
    top1 = (round(st.mean(top1_seeds), 1), round(st.pstdev(top1_seeds), 1))
    Ssym = collections.Counter()
    for (a, b), c in confus.items():
        Ssym[tuple(sorted((a, b)))] += c
    pairs = sorted([(a, b, c) for (a, b), c in Ssym.items() if c >= TAU_CONF and is_resolvable(a, b)],
                   key=lambda t: -t[2])
    loose_pairs = [(a, b, c) for (a, b), c in Ssym.items() if c >= TAU_CONF and is_resolvable_broad(a, b)]
    all_pairs = [(a, b, c) for (a, b), c in Ssym.items() if c >= TAU_CONF]
    n_dirdep_pairs = sum(1 for a, b, _ in pairs if is_direction_dependent(a, b))
    invariant_frac = pct(len(pairs) - n_dirdep_pairs, len(pairs)) if pairs else 0.0
    resolvable_share = pct(sum(c for *_, c in pairs), sum(c for *_, c in all_pairs))

    ceil_loose = pct(n_loose, n_eval); ceil_tight = pct(n_tight, n_eval)
    ceil_navel = pct(n_navel, n_eval); ceil_partner = pct(n_partner, n_eval)
    ceil_real = pct(n_real, n_eval); err_rate = pct(n_wrong, n_eval)
    dirdep_share = pct(n_dirdep, n_tight)
    ceil_robust = pct(n_tight - n_dirdep, n_eval); real_mean = round(st.mean(real_per_seed), 1)

    print(f"\n== engine: global+L256 exemplar top1 {top1[0]}±{top1[1]} ({SEEDS}-seed block-split) | "
          f"wrong {n_wrong}/{n_eval} ({err_rate}%) ==")
    print(f"== resolvable pairs TIGHT(S>={TAU_CONF}): {len(pairs)} (loose {len(loose_pairs)}) of {len(all_pairs)}; "
          f"invariant {invariant_frac}% ==")
    for a, b, c in pairs[:12]:
        tag = "[NAVEL]" if is_navel(a, b) else ("[dir↔]" if is_direction_dependent(a, b) else "       ")
        print(f"   x{c:<3} {tag} {a}  <->  {b}")
    print(f"\n== ⭐ CEILING (perfect oracle) ==")
    print(f"   loose proxy:                 {ceil_loose}pp")
    print(f"   TIGHT (true has neighbour):  {ceil_tight}pp  (└ dir-dep {dirdep_share}%)")
    print(f"   NAVEL bundle present:        {ceil_navel}pp")
    print(f"   PRED=co-present partner:     {ceil_partner}pp")
    print(f"   ⭐ REALISTIC (pred=partner & true≤rank{RANK_MAX}): {n_real}/{n_eval} = {ceil_real}pp "
          f"= {real_mean} pins/seed (vs σ={top1[1]}pp)")

    if ceil_real < 1.0 or real_mean < 2.0:
        gate, vtag = "STOP", "🔴"
        verdict = (f"🔴 여전히 사전종결 — 다중핀이 {multi_frac}%로 늘어도 현실 천장 +{ceil_real}pp({real_mean} pins/seed)가 "
                   f"σ={top1[1]}pp에 묻힌다. crack#0은 풀렸으나 crack#2(방향의존 {dirdep_share}%)·tie-breaker 제약이 남아 "
                   f"관계가 외형을 교정할 표본이 부족. 데이터 더, 또는 다발 동시라벨 페이지 타게팅 필요.")
    elif ceil_real < 2.5:
        gate, vtag = "NARROW", "🟡"
        verdict = (f"🟡 현실 천장 +{ceil_real}pp({real_mean} pins/seed) — σ={top1[1]}pp 경계. crack#0 해소로 040 대비 상승. "
                   f"M-rel1 oracle 사전검증은 가능하나 정합 현실화(M-rel2)에서 소멸 주의. 방향의존 {dirdep_share}%.")
    else:
        gate, vtag = "GO", "🟢"
        verdict = (f"🟢 현실 천장 +{ceil_real}pp({real_mean} pins/seed) — 측정 가능! crack#0 해소(다중핀 {multi_frac}%)가 "
                   f"040 STOP을 되살렸다. M-rel1 oracle 사전검증 진행. 방향의존 {dirdep_share}%는 invariant NAVEL로 우회.")
    print(f"\n==> {verdict}")

    d = explog.EXP / "047-relational-revival"; d.mkdir(parents=True, exist_ok=True)
    explog.bar(d / "fig1_ceiling.png",
               ["err rate", "loose", "TIGHT", "robust", "NAVEL", "partner", "⭐real"],
               [err_rate, ceil_loose, ceil_tight, ceil_robust, ceil_navel, ceil_partner, ceil_real],
               f"047 relational ceiling (perfect oracle) — multi-pin {multi_frac}%", "% of eval pins",
               ymax=max(60, err_rate + 5))
    explog.bar(d / "fig2_crack0.png", ["040 (old 953)", f"047 (merged)"],
               [42.0, multi_frac], "047 crack#0 dissolved: multi-pin image %", "% multi-pin images", ymax=100)
    if pairs:
        explog.barh_pairs(d / "fig3_pairs.png", [(f"{a} <-> {b}", c) for a, b, c in pairs[:12]],
                          "047 tight-resolvable confusion pairs (true<->pred)")
    pair_rows = "\n".join(
        f"| x{c} | {'NAVEL✓' if is_navel(a,b) else ('dir↔' if is_direction_dependent(a,b) else '–')} | {a} ↔ {b} |"
        for a, b, c in pairs[:12]) or "| – | – | (none over threshold) |"
    report = f"""# 047 — M-rep0c: 관계축 부활 (040 재실행, clean merged + global+L256)

- 날짜: {datetime.date.today().isoformat()} · 커밋 `main @ {_git_sha()}` · `scripts/multiscale_relational.py`
- 040은 crack#0(페이지 58% 단일핀)로 🔴 STOP했으나 "데이터 확장 시 부활" 보류였다. BlueLink 수확으로
  **다중핀 이미지 {multi_frac}%**(단일핀 58%→{round(100*single/len(img_pins),1)}%) — crack#0 직접 해소.
- 동일 oracle 천장(학습 0)을 새 데이터·새 엔진(global+L256)으로 재측정. 사전등록 게이트 040과 동일.

## 결과
- 엔진 global+L256 exemplar **top1 {top1[0]}±{top1[1]}%** (block-split), 오류율 {err_rate}% ({n_wrong}/{n_eval}).
- 해결가능 쌍 TIGHT(S≥{TAU_CONF}): **{len(pairs)}** (loose {len(loose_pairs)}), invariant {invariant_frac}%, 방향의존 {n_dirdep_pairs}.

| 빈도 | 관계유형 | 혼동쌍 (정답 ↔ 예측) |
|---|---|---|
{pair_rows}

### ⭐ 천장 (완벽 오라클)
| 경로 | 040 (old) | 047 (merged) |
|---|---|---|
| TIGHT (true에 이웃 존재) | +7.0pp | +{ceil_tight}pp |
| NAVEL 다발 이웃 | +3.0pp | +{ceil_navel}pp |
| 예측=co-present 파트너 | +0.8pp | +{ceil_partner}pp |
| **⭐ 현실 천장** (pred=파트너 & true≤rank{RANK_MAX}) | **+0.4pp (0.6/seed)** | **+{ceil_real}pp ({real_mean}/seed)** |

![ceiling](fig1_ceiling.png)
![crack0](fig2_crack0.png)
{'![pairs](fig3_pairs.png)' if pairs else ''}

## 판정 (사전등록 게이트, 040과 동일)
{vtag} **{gate}** — {verdict}

## 핵심
- crack#0(다중핀 부재)은 {multi_frac}%로 **해소**됐다 — 040 보류 조건 충족.
- 현실 천장 040 +0.4pp → 047 +{ceil_real}pp ({real_mean} pins/seed). {'σ에 묻힘 — 잔여 crack#2/tie-breaker가 병목.' if gate=='STOP' else '040 대비 상승 — M-rel1 진행 후보.'}
"""
    explog.write(d, report, {
        "title": "관계축 부활 (040 재실행, merged+L256)", "date": datetime.date.today().isoformat(),
        "headline": f"multi-pin 42%→{multi_frac}% (crack#0 해소) | 현실천장 040 +0.4 → 047 +{ceil_real}pp "
                    f"({real_mean}/seed, σ{top1[1]}) | 해결쌍 {len(pairs)} 방향의존 {dirdep_share}% → {gate}",
        "engine_top1": list(top1), "err_rate_pct": err_rate, "n_eval": n_eval, "n_wrong": n_wrong,
        "multi_pin_pct": multi_frac, "single_pin_pct": round(100 * single / len(img_pins), 1),
        "n_resolvable_pairs": len(pairs), "n_loose_pairs": len(loose_pairs),
        "resolvable_pairs": [[a, b, c] for a, b, c in pairs[:20]],
        "n_dirdep_pairs": n_dirdep_pairs, "invariant_frac_pct": invariant_frac,
        "ceiling_loose_pp": ceil_loose, "ceiling_tight_pp": ceil_tight, "ceiling_navel_pp": ceil_navel,
        "ceiling_partner_pp": ceil_partner, "ceiling_real_pp": ceil_real, "real_per_seed": real_mean,
        "dirdep_share_pct": dirdep_share, "gate": gate})
    print(f"wrote -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
