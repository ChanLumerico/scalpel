# 043 — 모델 방법론 스윕 (clean merged, 누수안전)

- 날짜: 2026-06-28
- 커밋: `data-pivot @ 14d41b5`
- 스크립트: `scripts/model_sweep.py` · 데이터 `data/merged_final` (1551 core / 502 cls)
- 엔진: frozen dinov2_vitb14@518 → GaussianPool σ40 (캐시), 10-seed photo-block split

## 방법 비교
| 방법 | top1 | top5 | coverage |
|---|---|---|---|
| mean-proto | 26.7±3.3 | 45.3 | 72.6 |
| exemplar | 31.6±4.1 | 47.6 | 72.6 |
| kNN-3 | 28.3±3.4 | 40.7 | 72.6 |
| kNN-5 | 22.2±2.6 | 45.1 | 72.6 |
| multi-proto | 30.2±4.0 | 47.3 | 72.6 |
| LSE | 24.0±2.3 | 40.4 | 72.6 |
| KDE | 24.8±2.6 | 42.1 | 72.6 |
| SupCon+exemplar | 30.4±2.4 | 46.5 | 72.6 |

![methods](fig1_methods.png)

## 진단 — 어디서 막히나
### 조직형별 (DX3: 혈관/신경 낮음)
| 조직형 | n | top1 | top5 |
|---|---|---|---|
| muscle | 589 | 43 | 62 |
| artery | 816 | 33 | 53 |
| nerve | 557 | 29 | 40 |
| other | 1314 | 28 | 43 |
| bone | 51 | 24 | 31 |
| vein | 181 | 22 | 34 |

![tissue](fig2_by_tissue.png)
![region](fig3_by_region.png)
![shot](fig4_by_shot.png)
![risk-coverage](fig5_risk_coverage.png)
![confusions](fig6_confusions.png)
![confidence](fig7_confidence.png)

## 핵심
- **오류의 44%가 같은 조직형 내 혼동** — 외형으로 조직형은 OK, 조직형 *내* 미세정체성이 천장(DX3, exp042 기하와 일치).
- 집계방법: best = exemplar. (옛 953에서 exemplar≫mean였는데 누수안전 502에서 재확인.)
- SupCon 학습헤드: top1 30.4 vs exemplar 31.6.
- shot↑일수록 정확도↑ (long-tail 레버 = 데이터).
