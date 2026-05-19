# QQQ Defensive Timing Research Decision: qqq_sma190_buffer_p0p5_hold0_to_spy

## 1. Summary Of The Research Path

- ML ETF rotation struggled versus QQQ.
- The QQQ-core overlay was diagnostic because it showed how much of the gap was QQQ exposure, but it is not the desired final direction.
- Pure weighted allocation did not improve the QQQ gap enough to justify continuing that path now.
- The QQQ timing path produced a defensive candidate worth preserving for validation.

## 2. Current Candidate Decision

- Candidate: `qqq_sma190_buffer_p0p5_hold0_to_spy`.
- Decision: `keep_as_best_qqq_defensive_timing_candidate`.
- Research status: `research_only_defensive_candidate`.
- Production-ready: `false`.
- Full-period result: strategy total return `3.7883` vs QQQ buy-and-hold `3.7950`.
- Risk-adjusted trade-off: return-to-drawdown `11.3985` vs QQQ `10.8050`.

## 3. Why QQQ Buy-And-Hold Remains The Primary Benchmark

- The research question is whether any ETF rotation or timing method improves the QQQ investor experience.
- This candidate still trails QQQ buy-and-hold on raw full-period total return.
- The observed strategy-vs-QQQ gap is `-0.0067`, so the bar has not moved.
- SPY and cash are useful references, but they are not the hurdle rate for a QQQ defensive timing candidate.

## 4. Why This Candidate Is Useful Anyway

- It gives the research program a simple, interpretable defensive QQQ timing baseline.
- It nearly captured QQQ return while allocating to SPY about `0.2871` of the time.
- It improved the full-period drawdown profile and return-to-drawdown, which is the intended defensive use case.
- It was most useful in: bear_2022.
- It is a concrete candidate for longer-history and walk-forward validation.

## 5. Explicit No-Go Decisions

- No QQQ-core continuation for now.
- No pure weighted allocation continuation for now.
- No tree_screen from this result.
- No MLP from this result.
- No benchmark_gap_features_v4 from this result.
- No production export.
- No S3 workflow.
- Do not modify the Market Data repository.

## 6. Recommended Next Task

- Run longer-history validation if data allows.
- Add walk-forward robustness validation for this exact QQQ defensive timing candidate.
- Keep the result research-only until those validations and paper-tracking design exist.
