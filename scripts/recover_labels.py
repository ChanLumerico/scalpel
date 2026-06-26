"""Experiment A-2 — recover OCR-dropped labels (conservative, for verification).

clean.py drops labels that fail the junk filter; some are real structures with OCR
errors ("aterventrieslar branch"=interventricular). We re-parse the PDFs, find the
dropped labels, and HIGH-THRESHOLD fuzzy-match them against the clean vocabulary to
surface RECOVERY CANDIDATES. Medical labels must be exact, so we do NOT auto-add —
we report candidates for human verification and quantify the upside.

    .venv/bin/python scripts/recover_labels.py
"""

from __future__ import annotations

import collections
import datetime
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rapidfuzz import fuzz, process  # noqa: E402

import explog  # noqa: E402
from eval_appearance import _git_sha  # noqa: E402
from scalpel.data import clean as C  # noqa: E402
from scalpel.data.parse import parse_quizlink  # noqa: E402
from scalpel.data.vocab import Vocab  # noqa: E402

THRESH = 90      # conservative; only near-exact OCR fixes


def main():
    clean_vocab = list(json.load(open("data/triples/vocab.json")).keys())
    cv_set = set(clean_vocab)
    v = Vocab()
    raw = collections.Counter()
    pdfs = sorted(glob.glob("data/quizlink/*.pdf"))
    print(f"re-parsing {len(pdfs)} PDFs to collect raw OCR labels...")
    for k, pdf in enumerate(pdfs):
        try:
            for t in parse_quizlink(pdf, v):
                raw[t.label] += 1
        except Exception as e:  # noqa: BLE001
            print(f"  skip {pdf}: {type(e).__name__}")
        if (k + 1) % 8 == 0:
            print(f"  ...{k+1}/{len(pdfs)}")

    kept = sum(c for l, c in raw.items() if l in cv_set)
    dropped = {l: c for l, c in raw.items() if l not in cv_set and not C.is_valid(l)}
    print(f"raw labels: {sum(raw.values())} ({len(raw)} distinct) | kept-in-vocab {kept} | "
          f"dropped {sum(dropped.values())} ({len(dropped)} distinct)")

    # high-threshold recovery candidates (len>=6, near a real label)
    cand = []
    for l, c in sorted(dropped.items(), key=lambda x: -x[1]):
        if len(l) < 6:
            continue
        m = process.extractOne(l, clean_vocab, scorer=fuzz.token_sort_ratio)
        if m and m[1] >= THRESH:
            cand.append((l, m[0], round(m[1]), c))
    rec_tri = sum(c for *_, c in cand)
    print(f"\nRECOVERY CANDIDATES (>= {THRESH} sim, len>=6): {len(cand)} labels / {rec_tri} triples")
    for bad, good, s, c in cand[:30]:
        print(f"   {bad!r:36s} -> {good!r:28s} {s}  (x{c})")

    d = explog.next_dir("recover-labels")
    rows = "\n".join(f"| `{bad}` | `{good}` | {s} | {c} |" for bad, good, s, c in cand[:40])
    report = f"""# A-2: OCR 드롭 라벨 복구 후보 (recover-labels)

- 날짜: {datetime.date.today().isoformat()}
- 커밋: `data-pivot @ {_git_sha()}`
- 스크립트: `scripts/recover_labels.py`

## 목적
clean이 OCR 쓰레기로 버린 라벨 중 *진짜 구조물의 OCR 오타*를 회수. **의료 라벨은 정확해야 하므로
자동 추가하지 않고**, 정제 어휘(567)에 고임계(≥{THRESH}) 퍼지매칭되는 **복구 후보만 검증용으로** 제시.

## 규모
- raw 라벨 {sum(raw.values())} ({len(raw)} distinct) | 어휘 내 유지 {kept} | **드롭 {sum(dropped.values())} ({len(dropped)} distinct)**
- **복구 후보: {len(cand)} 라벨 / {rec_tri} 트리플** (≥{THRESH} 유사, 길이≥6)

## 후보 (상위 40, OCR오타 → 정규형)
| dropped | → candidate | sim | n |
|---|---|---|---|
{rows}

## 판정 / 다음
- 회수 가능 상한 ~{rec_tri} 트리플 ({100*rec_tri/953:.0f}% of 953). 작지 않으면 **수작업 검증 후** 매핑을
  clean에 반영해 재빌드. 단 오복구(다른 구조물로 잘못 매핑) 위험이 있어 **사람이 확인 필수**.
- 임계값을 낮추면 회수↑·정밀도↓. 현 {THRESH}는 near-exact OCR 오타에 한정(보수적).
"""
    explog.write(d, report, {
        "title": "OCR 드롭 라벨 복구 후보", "date": datetime.date.today().isoformat(),
        "headline": f"dropped {sum(dropped.values())} tri; recoverable candidates {len(cand)} labels / {rec_tri} triples (>= {THRESH} sim, needs manual verify)",
        "dropped_triples": sum(dropped.values()), "candidates": len(cand), "candidate_triples": rec_tri})
    print(f"wrote -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
