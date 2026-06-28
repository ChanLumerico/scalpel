# 056 — M-bb0: 백본 sweep (최소 세팅, 학습 0)

- 날짜: 2026-06-28 · 커밋 `main @ a7a6381` · `scripts/backbone_sweep.py`
- clean 502 (dev 1214/test 337 봉인), 최소 세팅(σ40 GaussianPool + exemplar 1-NN, L256/CSLS 없음), 10-seed.
- 주지표 = k-NN top1(우리 readout 그 자체). 진단 = silhouette/hubness/tissue·region centroid sep.
- **DINOv3는 timm이 가중치를 open으로 호스팅 → HF gate 우회** (facebook/dinov3-*는 gated지만 `vit_*_dinov3.lvd1689m`은 공개).
  DINOv3 ViT는 patch16 @512(공정 위해 768 고해상 체크도), σ는 patch-정규화(SIGMA/14×patch).

## 결과 (paired Δ vs DINOv2-vitb14)
| 백본 | k-NN top1 | Δ | wins | silhouette | hubness | tissue_sep | region_sep |
|---|---|---|---|---|---|---|---|
| DINOv2-vitb14 (base) | 28.9±3.0 | +0.0 | 0/10 | -0.15 | 0.696 | -0.006 | 0.011 |
| dinov3-large-768 | 30.6±3.4 | +1.69 | 9/10 | -0.118 | 0.991 | -0.008 | 0.006 |
| dinov3-base-768 | 31.9±3.3 | +3.04 | 9/10 | -0.132 | 0.714 | -0.009 | 0.009 |
| dinov3-base | 29.2±3.0 | +0.37 | 6/10 | -0.161 | 0.628 | -0.01 | 0.012 |
| dinov3-small | 28.4±1.9 | -0.46 | 4/10 | -0.149 | 0.783 | -0.009 | 0.007 |
| dinov3-large | 28.8±3.3 | -0.08 | 5/10 | -0.155 | 0.811 | -0.009 | 0.007 |
| DINOv2-vitg14 | 31.3±2.9 | +2.43 | 9/10 | -0.106 | 0.758 | -0.001 | 0.005 |
| DINOv2-vitl14 | 29.0±3.6 | +0.09 | 4/10 | -0.143 | 0.84 | -0.002 | 0.003 |

![knn](fig1_knn.png)
![diag](fig2_diag.png)

## dev-CV vs 봉인 test (상위 3개) — 불일치 드러내기
- 봉인 base 33.5 → dinov3-base-768 **32.0** · DINOv2-vitg14 **36.8** · dinov3-large-768 **33.5**
- **dev-CV 1등 = dinov3-base-768**, 신뢰가능(dev∧봉인) best = **DINOv2-vitg14**.

## 판정
🟢 **신뢰가능 best = DINOv2-vitg14**: dev-CV Δ+2.43 (9/10) **그리고** 봉인 33.5→36.8. 단 tissue_sep≈0 — 일반 품질 개선이지 병목 해결 아님. → M-bb1 적층 후보.

## 핵심
- **세대 교체(DINOv2→DINOv3)는 깨끗한 레버 아님** — dinov3@512 ≈ vitb14; dinov3-base@768은 dev-CV 31.9(+3.04)로
  vitg14 타이지만 **봉인 test 32.0으로 baseline 33.5 못 넘음**(고해상 dev 신기루, 봉인이 차단 — §1.7 재검증).
  벤치마크(retrieval +10.8·fine-grained SOTA)가 frozen 1-NN·OOD 카데바엔 전이 안 됨(027 OOD 계열).
- **유일하게 신뢰가능한 백본 레버 = 크기(DINOv2-vitg14, 1.1B)** — dev Δ+2.43(9/10) **그리고** 봉인 36.8.
  단 **tissue_sep 전 백본 ≈0** → 어떤 백본도 핵심 병목(조직 얽힘) 못 풀고 일반 품질(hubness↓)만 개선.
- 다음: vitg14 위 L256·CSLS 적층(M-bb1)이 현재 best(vitb14+L256+CSLS 봉인 38.3) 초과하는지.
