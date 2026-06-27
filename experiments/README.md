# 실험 인덱스 (experiments)

각 실험은 서브폴더에 한글 `report.md` + figure + `metrics.json`.

| ID | 제목 | 날짜 | 핵심 결과 |
|---|---|---|---|
| [001-baseline](001-baseline/report.md) | 베이스라인 (M4' 외형 MVP) | 2026-06-26 | top1 31.3% / top5 44.7% @cov92.7% (229-way) |
| [002-pooling-ablation](002-pooling-ablation/report.md) | 점 풀링 애블레이션 | 2026-06-26 | best=sigma20 top1 34.1% (σ40=31.3%) |
| [005-baseline-mseed](005-baseline-mseed/report.md) | 베이스라인 (baseline-mseed) | 2026-06-26 | top1 38.8±3.4% / top5 55.8±4.0% @cov83.2% (200-way, 10 seeds) |
| [006-pooling-ablation](006-pooling-ablation/report.md) | 점 풀링 애블레이션 | 2026-06-26 | best=sigma10 top1 41.2±5.0% (σ40=38.8%) |
| [007-calibration](007-calibration/report.md) | 보정 + 기권 (M5') | 2026-06-26 | acc@cov100=38.8% → 상위30%만 답 78.4% | ECE 0.4→0.2 (10 seeds) |
| [008-context-probe](008-context-probe/report.md) | 관계/맥락 프로브 | 2026-06-26 | best=ms_cls top1 41.0±2.9% (local 38.8±3.4, Δ2.2) → 노이즈 안 — 단순 맥락 concat은 무효 |
| [009-discrimination-probe](009-discrimination-probe/report.md) | 판별 진단 | 2026-06-26 | best=exemplar top1 46.6±3.6% (proto 38.8) → 학습/구조 보상 신호 있음 |
| [010-baseline-exemplar](010-baseline-exemplar/report.md) | 정식 모델: exemplar + 보정 | 2026-06-26 | exemplar top1 46.6±3.6% / top5 58.0±4.4% @cov83.2% | 상위30% 88.5% |
| [012-learned-head](012-learned-head/report.md) | 학습형 판별 헤드 (첫 학습) | 2026-06-26 | learned top1 49.2 vs frozen 46.6 | paired Δtop1 +2.6(9/10) Δtop5 +7.8(10/10) → 학습이 일관되게 도움 (paired) |
| [013-learning-curve](013-learning-curve/report.md) | 데이터 스케일링 곡선 | 2026-06-26 | top1 39.1→46.6% as gallery 25→100%; last-25% Δ1.4 → 데이터가 레버 — 100%에서도 top1 상승 중 (더 모으면 오름) |
| [014-best-setting](014-best-setting/report.md) | 최고 세팅 최종 성능 | 2026-06-27 | top1 49.2±4.3% / top5 65.8±3.9% @cov83.2% | 확신30% 87.6% | ECE 0.3 |
| [015-learned-pooler](015-learned-pooler/report.md) | 학습형 풀러 vs GaussianPool | 2026-06-27 | gauss 46.6 vs learned 44.4 | paired Δtop1 -2.3(1/10) Δtop5 4.1(10/10) → 효과 불명확/노이즈 |
| [016-augmentation](016-augmentation/report.md) | 증강 (정확도+견고성) | 2026-06-27 | clean base 46.6→aug-gal 48.0; CORRUPT base 46.5→aug 47.0 (Δ0.5) |
| [017-pin-robustness](017-pin-robustness/report.md) | 핀 노이즈 강건성 | 2026-06-27 | top1 46.6%(0px)→43.1%(40px); aug-gal 42.5% @40px |
| [018-deployment](018-deployment/report.md) | 배포 운영점 + open-set | 2026-06-27 | @90%: test acc 81.2% answer 19.3% | OOV reject 95.1% AUROC 67.7 |
| [019-backbone-scale](019-backbone-scale/report.md) | 백본 스케일링 | 2026-06-27 | vitb14 top1 46.6% | vitl14 top1 46.8% | vitg14 top1 47.7% |
| [020-relational-context](020-relational-context/report.md) | M6' 구조적 이웃 맥락 | 2026-06-27 | best=neighbor Δtop1 -0.4(4/10) → 구조적 맥락도 top1 무효 → 데이터/학습이 레버 |
| [021-singleton-gallery](021-singleton-gallery/report.md) | 싱글톤 갤러리 포함 | 2026-06-27 | vocab 201→448, core top1 45.2→42.3 (Δ-2.9) |
| [022-recover-labels](022-recover-labels/report.md) | OCR 드롭 라벨 복구 후보 | 2026-06-27 | dropped 346 tri; recoverable candidates 1 labels / 1 triples (>= 90 sim, needs manual verify) |
| [023-aug-head](023-aug-head/report.md) | 증강 임베딩으로 헤드 학습 | 2026-06-27 | head-raw 49.5 vs head-aug 50.6 | paired Δtop1 1.1(7/10) → 효과 불명확 (증강은 새 다양성 없음) |
| [024-coherent-pool](024-coherent-pool/report.md) | Feature-coherent 풀링 | 2026-06-27 | best=coherent σ80 τ0.3 Δtop1 -0.2(4/10) → 효과 불명확/노이즈 |
| [025-region-prior](025-region-prior/report.md) | 부위-조건부 사전 | 2026-06-27 | best λ=0.1 Δtop1 +0.5(6/10) → 효과 불명확 (교차-부위 혼동은 소수) |
| [026-multilayer](026-multilayer/report.md) | 다층 DINO 특징 | 2026-06-27 | best=L2+L11 (tex+sem) Δtop1 -2.9(0/10) → 효과 불명확/노이즈 |
| [027-biomedclip](027-biomedclip/report.md) | BiomedCLIP 지식 | 2026-06-27 | dino 46.6 | bmc-img 36.9 | bmc-text 2.0 | dino+textλ 47.3 |
| [028-arcface-head](028-arcface-head/report.md) | 각마진 헤드 | 2026-06-27 | best m=0.1 vs-SupCon Δtop1 -0.3(2/10) → 각마진 추가 이득 없음 (SupCon로 충분) |
| [029-orientation](029-orientation/report.md) | 국소 방향/질감 기술자 | 2026-06-27 | best λ=0.2 Δtop1 -4.0(0/10) → 방향 기술자 추가 이득 없음 | orient-only 11.2 |
| [030-multiproto](030-multiproto/report.md) | 다중 프로토타입/soft 집계 | 2026-06-27 | best non-ex kmeans-3 Δtop1 -0.8(0/10) → exemplar-max가 여전히 최선 |
| [031-ensemble](031-ensemble/report.md) | 다양백본 앙상블 | 2026-06-27 | best λ=1.0 Δtop1 +1.0(5/10) → 상보성 부족 — 앙상블 무효 |
| [032-sam-classaware](032-sam-classaware/report.md) | 클래스-인지 SAM 마스킹 | 2026-06-27 | best sam-small Δtop1 -5.2(1/10) → 마스킹 무효 — Gaussian이 최선 (DX4 재확인) | thin g47.0/ca39.4 bulk g46.3/ca39.9 |
| [033-sam-thingate](033-sam-thingate/report.md) | thin-게이팅 SAM 풀링 (최종 판정) | 2026-06-27 | thin Δtop1 -2.0(1/10) → 기각; SAM 전 형태 종결(DX4/032/033) |
| [034-visual-prompt](034-visual-prompt/report.md) | Visual prompting | 2026-06-27 | best red-dot8-gpool Δtop1 -0.1(4/10) → 기각 |
| [036-mopt0-purify](036-mopt0-purify/report.md) | M-opt0 평가 정화 | 2026-06-27 | HP-selection 누수 1.5pp → 🟢 누수 거의 없음 — paired Δ·30개 숫자 신뢰 | cross-cadaver 갭 6.5pp (page-split ~44.1 vs unseen-PDF ~37.6) |
| [037-conformal-kde](037-conformal-kde/report.md) | KDE posterior + Conformal + OOD | 2026-06-27 | ECE kde 0.181 vs base 0.367 (10/10) | AURC 0.344 vs 0.304 (1/10) | conf size kde 114.4 base 108.555 | xcadaver conf cover 0.877 (viol 2.3pp) |
| [038-cadaver-invariant](038-cadaver-invariant/report.md) | Cross-cadaver 갭 분해 → cadaver-invariant | 2026-06-27 | 갭 0.1pp | 색조recovery -0.2pp (A3 stop) | A1 best λ0.3 cross +0.4(2/5) net -0.4 → 🔴  |
| [040-rel-feasibility](040-rel-feasibility/report.md) | M-rel0 관계추론 feasibility (선결 게이트) | 2026-06-27 | top1 46.6±3.6 | 해결쌍 5(loose 13, 방향의존 3) | ⭐현실천장 +0.4pp(0.6 pins/seed, σ3.6에 묻힘) → STOP(crack#0=데이터한계, 폐기아닌 보류) |
| [041-merged-eval](041-merged-eval/report.md) | 정밀 재평가 (clean merged) | 2026-06-27 | merged 502-way top1 31.6±4.1 cov 72.6 ee 22.9 | QL 195-way top1 21.5 cov 71.5 | +BlueL gallery Δtop1 8.9(10/10) Δcov 10.1(10/10) |
| [042-dino-space-eda](042-dino-space-eda/report.md) | EDA: DINO-space 클래스 중심점 기하 | 2026-06-28 | 502 core 중심점 2D | 조직형 분리 -0.005(≈0) vs 부위 분리 0.098(강) | artery↔vein cos 0.878(DX3) → DINO는 부위로 조직화 |
| [043-model-sweep](043-model-sweep/report.md) | 모델 방법론 스윕 (중첩 멀티시드 3-way) | 2026-06-28 | dev-선택 best=exemplar → 봉인 TEST top1 33.5 (CI 27.5–39.4) cov 79.8 | dev-CV 28.9 (낙관 -4.6pp) | SupCon test 31.6 | errors same-tissue 42% |
