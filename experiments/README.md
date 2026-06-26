# 실험 인덱스 (experiments)

각 실험은 서브폴더에 한글 `report.md` + figure + `metrics.json`.

| ID | 제목 | 날짜 | 핵심 결과 |
|---|---|---|---|
| [001-baseline](001-baseline/report.md) | 베이스라인 (M4' 외형 MVP) | 2026-06-26 | top1 31.3% / top5 44.7% @cov92.7% (229-way) |
| [002-pooling-ablation](002-pooling-ablation/report.md) | 점 풀링 애블레이션 | 2026-06-26 | best=sigma20 top1 34.1% (σ40=31.3%) |
| [005-baseline-mseed](005-baseline-mseed/report.md) | 베이스라인 (baseline-mseed) | 2026-06-26 | top1 38.8±3.4% / top5 55.8±4.0% @cov83.2% (200-way, 10 seeds) |
