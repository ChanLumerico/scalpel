# 037 — KDE posterior + Conformal + OOD (신뢰도·coverage 계층)

- 날짜: 2026-06-27
- 커밋: `data-pivot @ 1493b05`
- 스크립트: `scripts/conformal_kde.py`  (PRIMARY page-split 10-seed + cross-cadaver 병기)

예측은 **두 방법 공통 exemplar**(top1 40.786%, 3-way split 갤러리 50%) — KDE는 *신뢰도*만 비교.

## PRIMARY (page-split, paired vs global-temp baseline)
| 지표 | KDE | baseline(글로벌온도) | KDE 우세 |
|---|---|---|---|
| **ECE** ↓ | 0.181 | 0.367 | 10/10 |
| **AURC** ↓ [primary] | 0.344 | 0.304 | 1/10 |
| OOD AUROC ↑ | 0.611 | 0.687 | 0/10 |
| conformal cover (목표 0.9) | 0.92 | 0.916 | — |
| conformal **평균 집합크기** | 114.4 | 108.555 | (~172클래스 중, top5=5) |

- 함정#2(ECE): KDE가 글로벌온도 대비 paired 10/10 → **개선(이론=실측)**.
- AURC(주축): 1/10 → **동률/미달 — 보정만 개선, 선택예측 순위는 동급**.
- OOD: baseline max-cos가 OOV 분리 우세(0/10).

## 병기 (cross-cadaver, unseen PDF 5-fold; ±5 노이즈, 6 PDF 불안정)
| top1 | ECE | AURC | conf cover kde | conf cover base |
|---|---|---|---|---|
| 43.309% | 0.234 | 0.298 | 0.877 | 0.866 |

## 🔴 conformal 보장의 cross-cadaver 검증 (핵심)
- page-split 적합 집합 → unseen PDF 커버리지: **kde 0.877 (위반 2.3pp)**, base 0.866 (위반 3.4pp). 목표 0.9.
- 판정: 보장 대체로 유지.

## 판정 / 다음
- ECE는 KDE가 명백히 개선(보정 가치 실재). AURC가 동급이면 → **순위(선택예측)는 글로벌온도로 충분**,
  KDE의 가치는 *절대 신뢰도 보정(ECE)+OOD*에 한정. conformal 집합크기가 top5보다 크면(약한 모델)
  핸드아웃의 "보장되나 큰 집합" 시나리오 — 정직 보고. 다음 038(shrinkage, coverage).
