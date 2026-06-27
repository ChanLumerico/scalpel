# SCALPEL — Optimization Handout (rev1, SSOT for the post-034 phase)

> 모델 축 종결(034까지) 이후, **데이터를 늘리지 않고** 고정 953 트리플 위에서 짜낼 수 있는
> 개선을 정리. 무게중심 = **coverage·신뢰도**. top1 레버(앙상블·two-stage·LLM re-ranker)는
> *조건부*(사전검증 통과 시). 모든 접근 = frozen 임베딩 위 후처리(과적합 무관).
> (이 파일은 사용자 제공 핸드아웃 rev1의 충실한 사본 — 이 phase의 단일 진실원.)

## 0. 전제
- 데이터 953/567/510, 코어(≥2) 601/215. 엔진 frozen dinov2_vitb14@518 → GaussianPool σ40 →
  exemplar 1-NN(cos), specimen-split, 10-seed.
- 모델 축 종결(008/015/020/024/026/030/032/033 + 034 visual prompting). top1 천장 ≈46–50%.
- 운영점: top1(cov)~50, top5~58–66, cov~83%, confident-30%~88%, ECE~0.2–0.3.
- 세 축은 곡선의 트레이드오프 → 목표 = **risk–coverage 곡선을 바깥으로**.
- 기대치 못박음: top1 짜내도 +2~5pp 상한. coverage·신뢰도는 깨끗한 레버.

## 1. 선결 — 평가 정화 (M-opt0, 모든 최적화의 게이트)
두 누수 구분:
- **split 입도 누수** = ✅ DX1이 닫음(specimen/page-level split; top1 불변 38.8 vs 39.3).
- **HP-selection 누수** = ⚠️ 미확정 — 30개 실험 HP를 보고셋에서 골랐나. **M-opt0 대상.**

처방: ①split 구현 검증(group-key=page/specimen) ②953을 PDF단위 dev/holdout=80/20, holdout 봉인
③모든 스윕·채택은 dev에서만 ④holdout 최종확인 1회. 보고~50과 holdout 차 = 과적합 양.
**사전등록 게이트(측정 전 고정)**: ≤1–2pp🟢 신뢰 / 3–5pp🟡 절대값 낮춰읽되 paired Δ 유효 /
>5pp🔴 best-stack 재검증. (conformal/temp/h/λ는 적합 파라미터 → dev/LOO에서만.)

## 2. 통합틀 — 하나의 밀도모델 p(y,z)=p(z|y)p(y)
- p(z|y) KDE/Gaussian → 정확도+신뢰도(posterior). p(y) → 불균형. p(z) 낮음 → OOD/OOV(coverage).
- 밀도 하나 잘하면 세 축 동시. **단 top1은 안 올림**(가치=신뢰도·coverage 원리적 통합).

## 3. 접근 (축별)
### 3.1 신뢰도
- 🟢 **(A) KDE retrieval**: p(y|z)∝π(y)Σ exp(−‖z−z_e‖²/2h²). calibrated posterior + OOD(=p(z)).
  ⚠️ h→0이 exemplar-max(현 best); KDE는 top1 레버 아님(030). h는 gallery-LOO로 적합.
- 🟢 **(B) Conformal**: P(y∈C(z))≥1−α, 분포가정 없이 유한표본 보장. nonconformity=−log p̂.
  적응적 집합(쉬운핀 1/어려운핀 N). 제품(top-k+기권)에 정합. cal셋 test와 독립 필수.
- 🟡 **(C) per-region temperature** / **(D) entropy-gated 기권**: 경량 보조.
### 3.2 Coverage
- 🟢 **(E) Hierarchical shrinkage**: μ̂_y=λ(n)μ_y^obs+(1−λ)μ_parent. James–Stein, 해부 계층(FMA/
  region/tissue). singleton coverage↑, 021 순진포함(−2.9)보다 손실↓. **계층 소스(region vs FMA)
  먼저 dev 비교.** λ=shot 의존.
- 🟡 **(F) 속성/zero-shot 임베딩**: tissue+region 속성공간 → OOV 부분인식. 라벨접미/페이지 공짜.
- 🟡 **(G) EVT/OpenMax**: 꼬리 Weibull로 OOV reject 원리화(018 AUROC 67.7 개선).
- 🟢 **(H) Coverage 곡선**: singleton 포함비율 {0,25,50,75,100%} 스윕 → end-to-end(top1×cov) 스윗스팟.
### 3.3 마지막 top1 레버 (조건부 — 사전검증 필수)
- 🟡 **(I) 강한 멤버 앙상블**: late fusion s(y)=Σ w_m s_m(y) + Borda. **사전: 오답 상관행렬** —
  비상관쌍 있어야 이득(∝1−상관). 상관↑면 즉시 중단(=top1 종결 최종증거). 멤버=014/025/023/016.
- 🟡 **(J) Two-stage**: 215→K 좁혀 fine 재비교. **사전: recall@K 곡선**(1단계가 떨구면 복구불가).
- 🟡 **(K) LLM 혼동쌍 re-ranker(label-only)**: 이미지 안 줌(027 OOD 우회), 위치불변 해부규칙으로
  top-k tie-break. **사전: 혼동이 지식으로 풀리나 사진속위치로만 풀리나**(DX3: artery↔vein은 위치).
  강한 회의 — 040 멤버/tie-breaker로만.

## 4. 권장 순서 (ROI)
```
M-opt0 평가정화(holdout) 🔴 게이트
 → 037 Conformal+KDE posterior 🟢 신뢰도+OOD(=측정도구)
 → 038 Hierarchical shrinkage 🟢 coverage
 → 039 Coverage 곡선(singleton 스윕) 🟢 end-to-end
 → 040 강한멤버 앙상블(오답상관 사전검증) 🟡 마지막 top1 / 또는 종결증거
 → (041 two-stage/EVT/속성/LLM-K) 🟡 040 보고 결정
```

## 5. 평가 규율 (공통)
- 적합 파라미터는 dev/LOO에서만. 10-seed paired, specimen-split. holdout 최종 1회.
- 사전등록 채택: `ADOPT iff Δ주지표>0 AND ≥⌈0.8·seeds⌉/seeds`, N변형 시 Holm-Bonferroni.
- **단일 비교축 = AURC**(방법 간). 보조: selective-acc@cov / 평균 집합크기 / ECE / end-to-end.
- 함정#1 "예쁜 localization, 평탄 정확도"(008/024/032/033) → 임베딩 코사인 sanity 먼저.
- 함정#2 "우아한 이론, 평탄 실측"(037도 변종 위험) → 글로벌온도/고정top-k 베이스라인 paired 검증.
- 오답겹침 교차표 1차 출력 유지.

## 6. Definition of Done
- M-opt0 dev/holdout 분리·gap 보고. 037 conformal 집합<top-k & KDE ECE↓ & OOD AUROC↑.
  038/039 shrinkage 손실↓ & 곡선 스윗스팟. 040 오답상관 사전검증→채택 또는 종결확정.

## 부록 B — 정직한 기대치
| # | 접근 | 축 | 기대 | 위험 |
|---|------|----|------|:---:|
| A | KDE posterior | 신뢰+OOD | ECE↓, top1≈불변 | 🟢 |
| B | Conformal | 신뢰(보장) | 보장됨, 단 집합 클 수도 | 🟢* |
| C/D | region-temp/entropy | 신뢰 | ECE소폭↓ / 위험핀거름 | 🟢 |
| E | hier. shrinkage | coverage | 어휘↑ 손실↓ | 🟡 |
| F/G | 속성 / EVT | coverage/OOV | 부분인식 / AUROC↑ | 🟡 |
| H | coverage 곡선 | end-to-end | +1~3pp 스윗스팟 | 🟢 |
| I | 앙상블 | top1 | +2~4pp 또는 0 | 🟡 |
| J | two-stage | top1 | +0~3pp(가설) | 🟡 |
| K | LLM 혼동쌍 | top1(혼동쌍) | 거의 0(회의) | 🔴 |

> *Conformal: 위험을 낮추진 않음 — 약한 모델이면 90%집합 평균크기 8~10 클 수 있음. 가치는
> 고정 top-k 대비 *평균 집합 축소*가 실제로 일어날 때. top1은 거의 닫혔고, 확실한 가치는
> 신뢰도(보장 calibration)+coverage(원리적 OOV)로 risk–coverage 곡선을 바깥으로 미는 것.
