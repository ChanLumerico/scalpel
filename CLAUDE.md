# CLAUDE.md — SCALPEL project rules & direction

이 파일은 이 저장소에서 작업할 때 **반드시 지켜야 하는 철칙**과 프로젝트의 주제·방향을 정리한다.
(매 세션 컨텍스트로 로드됨. 한글 OK. `PAPER.md`만 영어.)

## 0. 프로젝트 주제
**SCALPEL** — 한국 육안해부 실기시험("땡시")을 위한 **점-조건부 해부구조 인식** `p(y|I,q)`:
박리 사진 `I`와 핀 좌표 `q`가 주어지면 그 자리의 구조 `y`를 식별(또는 기권).
- **v2 (현재):** 합성 메시 렌더 폐기 → **실제 BlueLink QuizLink 박리 PDF**가 유일한 데이터 소스.
- **궁극 목표 = 실배포 가능한 모델.** (단, §6 — 배포는 *목표*지 지금 할 일이 아님.)

## 1. 평가 철칙 (가장 중요 — 어기면 결론이 거짓이 됨)
1. **항상 multi-seed (≥10), mean±std 보고.** 단일 seed는 ±3–4%p 노이즈 → 단일 seed로 결론 금지.
   (실제로 단일 seed의 "σ20 +2.8%p"는 거짓양성이었고 multi-seed가 즉시 기각했다.)
2. **photo-twin 블록 단위 분할 (이미지 누수 차단).** 한 사진의 모든 트리플은 같은 fold로 묶는다. ★
   **거의 같은 사진도 같은 fold로** — merged 데이터는 QuizLink의 49%가 BlueLink 슬라이드와 *동일 사진*이라
   (exp 041), 단순 specimen(페이지) split은 누수돼 ~46→21로 인플레됐다. **photo-twin 블록(exact∪corr≥0.90)
   단위 분할 필수** (`scripts/split_devtest.py`의 블록).
3. **방법 비교는 paired** (같은 분할에서 A vs B). 비대응 std로 작은 차이 논하지 말 것.
4. **일반화 주장엔 cross-cadaver (PDF 단위) 분할로 교차검증.**
5. **싼 프로브 먼저, 무거운 빌드 나중.** 효과를 싸게 확인하고 투자.
6. **음성 결과도 동등하게 기록·보고.** (과적합·무효도 중요한 발견.)
7. **봉인 test + 멀티시드 = 중첩 3-way (둘 중 하나가 아니라 둘 다).** 멀티시드만으론 여러 방법/HP를 *같은*
   eval로 고를 때 HP-선택 누수(exp 036서 ~1.5pp 실측)가 끼고, 단일 3-way는 ±3-4pp 노이즈로 결론 불안정.
   → **dev/test를 한 번 고정 봉인**(`scripts/split_devtest.py`, seed 고정·test 20%·photo-block), **선택·튜닝은
   dev에서만 10-seed CV**(mean±std), **최종 수치만 봉인 test 1회**(부트스트랩 CI). 순수 paired 방법비교(튜닝
   없음)는 dev 멀티시드로 충분. test를 반복 들여다보면 점진 누수 — 한 번만.
8. 무작위 기준선(=1/클래스수)과 비교해 배수로 의미 부여.

## 2. 데이터 품질이 모델보다 먼저
- **`(I,q,y)` 정합성 선결:** 핀은 반드시 **조직 위**(배경 금지), 라벨은 **의학적으로 정확**해야 한다
  (클래스 라벨이 틀리면 안 됨 — 손-검수). 누수 제거: 모든 리더선·라벨박스 inpaint.
- **데이터가 결정적 천장임이 실측으로 확립됨**(스케일링 곡선 미포화, 모델 용량 추가는 과적합).
  → 성능 개선의 *진짜 레버는 데이터*. 모델 trick으로 천장 못 넘는다.

## 3. 윤리 (하드 — 협상 불가)
- 비상업·교육 목적 한정. BlueLink 출처 표기 유지.
- 시신 존엄: 비공개 저장, 재배포 금지. **카데바 이미지는 git에 절대 커밋 금지**
  (`/data/` 와 `*.private.png` gitignore — figure에 카데바가 들어가면 `*.private.png`로 저장).
- **실시험 문제 사진은 학습/평가/갤러리 어디에도 사용 금지.**

## 4. 모델 / 방법 (현재 베스트와 교훈)
- **현재 베스트:** frozen DINOv2(vitb14) + 핀 GaussianPool(σ40) + **exemplar 1-NN**.
  ⚠️ 옛 "215-way top1 49.2"는 **누수 인플레**였음(exp 041 — 사진 49% 중복). **정직 수치(누수안전 merged
  502-way, exp 043, 봉인 test §1.7):** dev-CV exemplar **28.9±3.0**, **봉인 test top1 33.5 (CI 27.5–39.4)**,
  확신30% **~52%**. 데이터 확장이 검증된 레버
  (+BlueLink → QuizLink Δtop1 +8.9/Δcov +10.1, 둘 다 10/10).
- 교훈: **최근접 exemplar ≫ 평균 프로토타입**(누수안전서도 31.6 vs 26.7로 재확인). **SupCon 헤드는 누수안전
  데이터선 도움 안 됨**(옛 +2.6은 부분 누수기반). 백본 키워도 한계, 맥락 concat·SAM·visual prompting·관계추론
  전부 음성. **DINO-space는 부위로 조직화되지 조직형(동맥/정맥/신경)은 못 가름**(exp 042, artery↔vein cos 0.88).
  → 천장 = 부위내 미세정체성; 모델 trick 아닌 데이터 또는 더 미세한 표현이 레버.

## 5. 로깅 규율 (연구 일지 — 빠짐없이)
- **모든 실험 → `experiments/NNN-*/`**: 한글 `report.md` + figure + `metrics.json`. `explog.py`로 생성.
- **`RESEARCH_LOG.md`(영어)가 정식 연구 일지** — 실험이 끝나면 **반드시 5W1H로 매우 상세히** 항목 추가:
  **When / Why(동기·가설) / What & How(방법·설정) / Where(데이터·조건) / Result(수치) /
  Conclusion(의미·다음으로 이어진 것) / Reproduce(스크립트).** 한 줄 요약 금지.
- **`PAPER.md`(영어)** 는 다듬은 논문 — §5 색인 + §6 주제별 결과 갱신.
- 카데바 이미지가 든 figure는 `*.private.png`(gitignore). 수치 차트는 커밋 가능.
- 커밋 메시지에 핵심 수치 요약. 끝나면 push.

## 6. 단계/순서 (넘겨짚지 말 것)
- **배포(운영점·open-set 스펙 확정)는 궁극 목표지 지금 작업이 아니다.** 먼저 모델·데이터를 충분히 키운다.
- 현재 우선순위: **데이터 확장**(증명된 레버) → 그 위에서 학습형 레버·관계추론 재평가.

## 7. 환경
- **통합 `.venv/`** (pyenv Python 3.12.7). 실행: `.venv/bin/python`.
- parse/build/OCR엔 tesseract 필요: `export PATH="/opt/homebrew/bin:$PATH"`.
- 데이터셋 빌드: `python -m scalpel.data.build` (parse→clean→prune 자동). `data/`는 gitignore.

## 8. 현재 상태 / 다음
- M0→M5 완료, exp 001–020 기록됨. 모델 레버 소진(데이터로 수렴).
- **진행 중: 데이터 확장** — BlueLink 커리큘럼 트리(하위 페이지 ~600개)를 크롤해 QuizLink PDF를 더 확보
  (현재 31개 → 더). coverage·정확도 동시 상승이 목표.
