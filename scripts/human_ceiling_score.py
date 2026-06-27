"""Experiment 035 (score) — grade a human-ceiling answer file and run the error-overlap.

Reads the blind answers + the hidden manifest, and produces the PRE-REGISTERED 1st
output: the human-vs-model 2×2 with the two decision cells —
  P(model wrong ∩ human right)  = learnable headroom (data/method room) 🟢
  P(both wrong)                 = intrinsic ambiguity (data won't fix)   🔴
with 95% bootstrap CIs, plus human MC top1/top5, free-recall top1, vs model 46.6/58,
and the pre-registered self-pilot interpretation.

    .venv/bin/python scripts/human_ceiling_score.py [answers.json] [evaluator-label]

Default answers = data/human_ceiling/answers.json. Writes experiments/035-human-ceiling/
report.md + metrics.json (numbers only — no cadaver imagery).
"""

from __future__ import annotations

import collections
import datetime
import difflib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

DATA = Path("data/human_ceiling")
EXP = Path("experiments/035-human-ceiling")


def norm(s):
    s = re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())
    return re.sub(r"\s+", " ", s).strip()


def free_correct(free, true):
    f, t = norm(free), norm(true)
    if not f:
        return False
    if f == t:
        return True
    ft, tt = set(f.split()), set(t.split())
    if tt and tt <= ft:                                   # all true content words present
        return True
    return difflib.SequenceMatcher(None, f, t).ratio() >= 0.85


def boot(mask, fn, B=5000):
    mask = np.asarray(mask)
    rng = np.random.default_rng(0)
    n = len(mask)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    stats = [fn(mask[rng.integers(0, n, n)]) for _ in range(B)]
    return (float(round(100 * fn(mask), 1)),
            float(round(100 * np.percentile(stats, 2.5), 1)),
            float(round(100 * np.percentile(stats, 97.5), 1)))


def main():
    ans_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "answers.json"
    evaluator = sys.argv[2] if len(sys.argv) > 2 else "self-pilot (creator — contaminated upper bound)"
    if not ans_path.exists():
        print(f"no answer file at {ans_path} — run the quiz first (data/human_ceiling/quiz.html)")
        return 1
    answers = {a["id"]: a for a in json.loads(ans_path.read_text(encoding="utf-8"))}
    man = {}
    for line in open(DATA / "manifest_answers.jsonl", encoding="utf-8"):
        m = json.loads(line); man[m["id"]] = m

    ids = [i for i in man if i in answers]
    if not ids:
        print("no overlap between answers and manifest"); return 1
    hm1, hm5, hf1, mm1, mm5, tissue = [], [], [], [], [], []
    for i in ids:
        a, m = answers[i], man[i]
        dk = a.get("dontknow")
        mc = a.get("mc", [])
        hm1.append(0 if dk or not mc else int(norm(mc[0]) == norm(m["true"])))
        hm5.append(0 if dk else int(any(norm(g) == norm(m["true"]) for g in mc[:5])))
        hf1.append(0 if dk else int(free_correct(a.get("free", ""), m["true"])))
        mm1.append(int(m["model_top1_correct"])); mm5.append(int(m["model_top5_correct"]))
        tissue.append(m["tissue"])
    hm1, hm5, hf1, mm1, mm5 = map(np.array, (hm1, hm5, hf1, mm1, mm5))
    n = len(ids)

    h1 = boot(hm1, np.mean); h5 = boot(hm5, np.mean); f1 = boot(hf1, np.mean)
    m1 = boot(mm1, np.mean); m5 = boot(mm5, np.mean)
    both_right = boot((hm1 & mm1), np.mean)
    headroom = boot(((1 - mm1) & hm1), np.mean)          # model wrong, human right
    ambiguity = boot(((1 - mm1) & (1 - hm1)), np.mean)   # both wrong
    model_only = boot((mm1 & (1 - hm1)), np.mean)        # model right, human wrong

    # per-tissue human vs model top1
    per_t = {}
    for t in sorted(set(tissue)):
        idxs = [k for k in range(n) if tissue[k] == t]
        per_t[t] = {"n": len(idxs), "human_top1": round(100 * hm1[idxs].mean(), 1),
                    "model_top1": round(100 * mm1[idxs].mean(), 1)}

    # pre-registered self-pilot interpretation
    hv = h1[0]
    if "self" in evaluator.lower():
        if hv <= 55:
            interp = ("🔴 본인(오염된 상한)도 낮음 → 가장 강한 결론. 진짜 사람값 ≤ 이 값. "
                      "외부 평가는 확증용. P(둘 다 오답) 높으면 본질적 모호성 확정.")
        elif hv >= 80:
            interp = "🟡 본인 높음 → 기억 효과 배제 불가. 외부(의대생/전문가) 평가 필수, 본인만으론 결론 불가."
        else:
            interp = "🟡 중간 → 외부 평가로 확정 필요."
    else:
        interp = f"외부 평가자({evaluator}) — 기억 오염 없음, 결론에 직접 사용 가능."

    print(f"\n== human ceiling (n={n}, evaluator={evaluator}) ==")
    print(f"  human  MC top1 {h1[0]} [{h1[1]},{h1[2]}]   top5 {h5[0]}   free-recall top1 {f1[0]}")
    print(f"  model  top1    {m1[0]} [{m1[1]},{m1[2]}]   top5 {m5[0]}   (canonical 46.6/58)")
    print(f"  2x2 (human MC vs model, top1):")
    print(f"    both right            {both_right[0]}%")
    print(f"    🟢 model✗ human✓ HEAD {headroom[0]} [{headroom[1]},{headroom[2]}]")
    print(f"    🔴 both wrong  AMBIG  {ambiguity[0]} [{ambiguity[1]},{ambiguity[2]}]")
    print(f"    model✓ human✗         {model_only[0]}%")
    print(f"  per-tissue: " + " | ".join(f"{t} n{v['n']} H{v['human_top1']}/M{v['model_top1']}" for t, v in per_t.items()))
    print(f"  => {interp}")

    EXP.mkdir(parents=True, exist_ok=True)
    pt = "\n".join(f"| {t} | {v['n']} | {v['human_top1']}% | {v['model_top1']}% |" for t, v in per_t.items())
    report = f"""# 035 — Human ceiling (결과)

- 날짜: {datetime.date.today().isoformat()}
- 평가자: **{evaluator}**  | n={n}
- 스크립트: `scripts/human_ceiling_build.py` (패킷) · `scripts/human_ceiling_score.py` (채점)
- 사전등록: `PREREG.md`

## 사람 vs 모델 (215-way, top1/top5; 95% bootstrap CI)
| | top1 | top5 |
|---|---|---|
| 사람 (객관식) | {h1[0]}% [{h1[1]}, {h1[2]}] | {h5[0]}% |
| 사람 (자유회상) | {f1[0]}% | — |
| 모델 (LOO) | {m1[0]}% [{m1[1]}, {m1[2]}] | {m5[0]}% |

## 오답 겹침 2×2 (사람 객관식 vs 모델, top1) — 1차 출력
| | 모델 정답 | 모델 오답 |
|---|---|---|
| **사람 정답** | {both_right[0]}% | 🟢 **{headroom[0]}%** [{headroom[1]}, {headroom[2]}] (학습가능 헤드룸) |
| **사람 오답** | {model_only[0]}% | 🔴 **{ambiguity[0]}%** [{ambiguity[1]}, {ambiguity[2]}] (본질적 모호성) |

## 조직별 top1 (사람/모델)
| 조직 | n | 사람 | 모델 |
|---|---|---|---|
{pt}

## 해석 (사전등록 규칙)
{interp}
"""
    (EXP / "report.md").write_text(report, encoding="utf-8")
    (EXP / "metrics.json").write_text(json.dumps({
        "title": "Human ceiling", "date": datetime.date.today().isoformat(), "evaluator": evaluator, "n": n,
        "headline": f"human MC top1 {h1[0]} vs model {m1[0]} | headroom {headroom[0]}% ambiguity {ambiguity[0]}%",
        "human_mc_top1": h1, "human_mc_top5": h5[0], "human_free_top1": f1[0],
        "model_top1": m1, "model_top5": m5[0],
        "overlap": {"both_right": both_right[0], "headroom_model_wrong_human_right": headroom,
                    "ambiguity_both_wrong": ambiguity, "model_right_human_wrong": model_only[0]},
        "per_tissue": per_t}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote -> {EXP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
