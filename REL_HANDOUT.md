# SCALPEL — Relational-Reasoning Axis Handout (exp 040, SSOT)

> **[exp 040 / M-rel0 결과 — 2026-06-27]** 🔴 **stop-but-hold (사전등록 게이트).**
> 완벽 오라클(핀·정합·그래프 모두 정답) 하 현실 천장 = **+0.4pp ≈ 0.6 pins/seed**, σ=3.6pp
> 분할노이즈에 묻힘 → M-rel1을 돌려도 측정가능한 양성 불가. 원인은 핸드아웃이 열거한 세 균열
> *이전*의 **crack #0 (관계 이웃 부재: 페이지 58%가 단일 핀, NAVEL 다발 동시핀 4.8%)** + 남은
> 해결가능 5쌍 중 3쌍이 **방향의존(crack #2 정조준)**. 둘 다 *데이터 구조*의 함수 → 추론 정교화로
> 못 넘음. **폐기 아님 — 데이터 확장(다중핀·다발 동시라벨 페이지) 후 되살아날 유일한 축으로 보류.**
> 측정: `scripts/confusion_pairs.py`, 기록: `experiments/040-rel-feasibility/`, RESEARCH_LOG Phase 12.
> 아래 §2(그래프/추론)·§4(M-rel1~3) 빌드는 **데이터 선확보 시 그대로 재개**한다(무효 아님, 보류).

> (이 파일은 사용자 제공 exp-040 rev2 핸드아웃의 충실한 condensation — 이 축의 단일 진실원.)

---

## 0. 가설과 균열
**핵심 가설:** 외형은 artery↔vein를 못 가르지만(DX3), **상대 위치(vision 쉬움) + 해부 규칙(LLM)**
이면 "외측=artery"(NAVEL)로 교정 가능. 핀을 독립 판별 p(y|I,q) → **지식 그래프 제약 하 동시추론**
p(y₁..yₙ | I,{qᵢ},G).

**세 균열 (실패 가능 지점):**
- **#1 stage-1 = 원래 문제** — vision이 노드를 깔아야 하는데 그게 못 푸는 ~50% → 🟢 oracle 핀으로 우회.
- **#2 image ≠ anatomy** — LLM 규칙은 3D 표준해부, 사진은 2D투영·박리변형·개체변이 → 🔴 진짜 위험.
- **#3 LLM 지식 신뢰** — 🟡 고신뢰 관계만 + Atlas 교차검증.

**[040이 추가로 발견]** **crack #0** — 관계 항이 *발화하려면* 페이지에 관계 이웃이 다른 핀으로 있어야
한다. QuizLink는 한 사진에 한 구조를 핀하므로 이게 희소(#1/#2/#3 이전에 축을 막음).

**운명의 질문:** "관계 제약이 외형을 *교정*하나, 틀린 1단계를 *증폭*만 하나?" → oracle 사전검증으로 싸게 판정.

## 1. 정밀도 원칙
feasibility는 *상한*을 본다 → 각 단계에 최선 도구(oracle·frontier)를 써서 "방법이 틀린 건지 도구가
약한 건지"를 가른다(032 oracle 논리). 로컬 36GB는 *배포* 제약이지 feasibility 제약이 아니다.
**핵심 단순화:** 핀이 주어지면 상대 위치는 *좌표 기하*로 계산 → vision detection 불필요(crack #1 회피).

## 2. 파이프라인
### 2.1 지식 그래프 (수동 우선, hallucination 0)
혼동행렬 C에서 *모델이 실제 틀리는* 쌍만 그래프에 (전체 215는 신호 희석). 기준: S=C+Cᵀ, S≥τ,
same_region, is_relational_resolvable(artery↔vein, N-A-V, 인접근육). Atlas(Gray's/Netter)로 ~15-25쌍
수동 작성 → `anat_graph.json`. **`invariant_under_projection` 플래그가 핵심**(crack #2 대응): "lateral"은
2D반전에 불변 아님(false), "N-A-V *순서*"는 불변(true). c<0.7 또는 검증실패 엣지 제외.

### 2.2 상대 위치 (좌표 기하, vision 불필요)
oracle 핀 {qᵢ}에서 쌍별 horiz/vert/dist/angle, 삼중 순서는 PCA 1축 사영 정렬(NAVEL용). ⚠️ 이미지
"right"가 해부 "lateral"과 같다는 보장 없음(crack #2) → *순서*가 반전에 robust.

### 2.3 추론 (명시적, LLM 아님)
목적함수(log-space): Ŷ = argmax Σᵢ log p_vis(yᵢ|I,qᵢ) + λ Σ_(i,j) φ(yᵢ,yⱼ | rel_ij^img, G).
외형 항 = 기존 exemplar score의 temperature softmax 로그(스택 그대로 재사용, 그 위에 더함).
**관계 항 φ** = ±m·c·a (그래프관계 존재 & img일치=+, 모순=−, 무관=0). m=margin, c=엣지신뢰도,
a=정합신뢰도(invariant=true면 1, 방향성이면 a₀<1). ⚠️ λm이 크면 외형 압도(032 교훈) → 작은 값부터
스윕, 외형 top1 안 깎으며 혼동쌍만 교정하는 지점. 추론: 각 핀 top-K(K=5)로 후보 제한 → n≤6 exact
(Kⁿ), 큰 n은 max-product BP. 결정론적·검증가능.

### 2.4 정합 (crack #2 핵심): (a) 상대 순서만(반전 불변, 기본·a=1) (b) 페이지 측위 라벨 (c) oracle 정합
(feasibility 한정, crack #2 상한 분리).

## 3. 도구: feasibility=수동그래프+oracle핀+좌표기하+기존 exemplar+exact/BP. 배포=RAG+로컬 LLM
(Qwen3.6-35B-A3B MLX, BGE-M3, FAISS)로 215클래스 그래프 자동화. vision detection 어디에도 불필요.

## 4. 실험 게이트
- **M-rel0** — 혼동쌍 그래프 + invariant 비율. (※040에서 여기에 **천장 측정**을 더해 선결 판정.)
- **M-rel1 ⭐ (운명결정)** — oracle 핀 + oracle 정합 + 고신뢰 관계로 혼동쌍 재순위. **혼동쌍 핀 한정**
  paired baseline(외형만) vs treatment(외형+φ). 사전등록: oracle정합(a=1)에서도 Δtop1≤0 → 🔴 crack#2
  치명 축폐기 / Δ>0 & ≥7/10 seed → 🟢 M-rel2.
- **M-rel2** — 정합 현실화(oracle정합 제거), crack #2 비용 정량화.
- **M-rel3** — 핀 노이즈 ablation, crack #1 민감도(detection 불필요).
- 전역 top1 병기(혼동쌍만 돕고 나머지 무해 확인). 10-seed paired, page-split primary(038).

## 5. 결과 분기 (사전 해석)
- M-rel1 oracle 양성 + M-rel2 정합 유지 → 🟢 외형-밖 첫 돌파.
- M-rel1 양성 + M-rel2 정합 붕괴 → 🟡 원리는 맞으나 image≠anatomy 실전 차단.
- M-rel1 oracle 평탄 → 🔴 축폐기(032식 상한실패), 깨끗한 음성. **← 040이 M-rel0 천장에서 한 단계 더 싸게 도달.**

## 6. Definition of Done
- [x] **M-rel0** — 혼동쌍 + **천장 측정**(040: realistic +0.4pp, σ에 묻힘 → stop-but-hold).
- [ ] **M-rel1~3** — *데이터 확장(다중핀 페이지) 후* 재개. crack #0 해소가 선결.

## 부록 — 기존 실험과 다른 점
020(외형 이웃 관계)·025(region 빈도)·034(단일 핀 q 주입)·K(LLM 사후 tie-break) 어느 것과도 다른
*핀들의 동시 structured prediction*. 단, 040 M-rel0가 보였듯 **현 데이터에선 관계 이웃 자체가 희소**해
원리적 상한이 잡음에 묻힌다 → 데이터가 이 축의 선결 조건.
