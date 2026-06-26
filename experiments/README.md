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
