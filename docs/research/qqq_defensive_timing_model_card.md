# QQQ Defensive Timing Model Card: qqq_sma190_buffer_p0p5_hold0_to_spy

## 1. Status

| Field | Value |
| --- | --- |
| Strategy name | qqq_sma190_buffer_p0p5_hold0_to_spy |
| Strategy type | QQQ defensive timing / risk-on-risk-off |
| Family | total_return_spy_defensive |
| Risk-on asset | QQQ |
| Risk-off asset | SPY |
| Signal name | qqq_sma_price_filter |
| Current decision | keep_as_best_qqq_defensive_timing_candidate |
| Research status | research_only_defensive_candidate |
| Production-ready | production_ready = false |

## 2. Strategy Rule

- Hold QQQ when QQQ close > SMA190 * 1.005.
- Otherwise hold SPY.
- Signal from close T applies next trading day.
- Minimum hold days: `0`.
- Transaction cost assumption: `10.0000` bps per turnover unit.

## 3. Dataset And Benchmarks

- Dataset: Tiingo-only adjusted ETF research data.
- Range: `2019-01-02` to `2026-05-15`.
- Trading days in replay: `1853`.
- Benchmarks: QQQ buy-and-hold, SPY buy-and-hold, and cash.

## 4. Full-Period Metrics

QQQ buy-and-hold comparison remains the primary readout: the strategy nearly matched QQQ return, but did not beat it on raw total return.

| Metric | Value |
| --- | --- |
| Strategy total return | 3.7883 |
| QQQ buy-and-hold total return | 3.7950 |
| SPY buy-and-hold total return | 2.2962 |
| Strategy vs QQQ | -0.0067 |
| Strategy vs SPY | 1.4921 |
| Annualized return | 0.2374 |
| Daily volatility | 0.0139 |
| Sharpe-like | 1.0746 |
| Strategy max drawdown | -0.3324 |
| QQQ buy-and-hold max drawdown | -0.3512 |
| SPY buy-and-hold max drawdown | -0.3370 |
| Drawdown improvement vs QQQ | 0.0189 |
| Return-to-drawdown | 11.3985 |
| QQQ buy-and-hold return-to-drawdown | 10.8050 |
| QQQ return capture | 0.9982 |
| Transaction count | 23 |
| Exposure rate to QQQ | 0.7129 |
| Exposure rate to SPY | 0.2871 |
| Estimated transaction cost | 0.0460 |

## 5. Subperiod Metrics

| period_name | period_start | period_end | strategy_total_return | qqq_buy_hold_total_return | spy_buy_hold_total_return | strategy_vs_qqq | strategy_vs_spy | strategy_max_drawdown | qqq_max_drawdown | spy_max_drawdown | exposure_rate_to_qqq | exposure_rate_to_spy | transaction_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | 2019-01-01 | 2026-05-15 | 3.7883 | 3.7950 | 2.2962 | -0.0067 | 1.4921 | -0.3324 | -0.3512 | -0.3370 | 0.7129 | 0.2871 | 23.0000 |
| covid_crash_recovery | 2020-01-01 | 2020-12-31 | 0.4309 | 0.4618 | 0.1728 | -0.0309 | 0.2581 | -0.3324 | -0.2856 | -0.3370 | 0.9130 | 0.0870 | 4.0000 |
| bear_2022 | 2022-01-01 | 2022-12-31 | -0.2237 | -0.3322 | -0.1864 | 0.1086 | -0.0372 | -0.2795 | -0.3483 | -0.2450 | 0.0518 | 0.9482 | 3.0000 |
| ai_bull_2023_2024 | 2023-01-01 | 2024-12-31 | 0.8203 | 0.9578 | 0.5826 | -0.1375 | 0.2377 | -0.1356 | -0.1356 | -0.0997 | 0.9602 | 0.0398 | 7.0000 |
| recent_2025_2026 | 2025-01-01 | 2026-05-15 | 0.3505 | 0.3981 | 0.2827 | -0.0477 | 0.0678 | -0.2197 | -0.2277 | -0.1876 | 0.8192 | 0.1808 | 8.0000 |

- COVID crash/recovery: strategy trails QQQ slightly and has worse drawdown than QQQ.
- Bear 2022: strategy beats QQQ materially by reducing QQQ exposure.
- AI bull 2023-2024: strategy trails QQQ because QQQ bull market strength dominates.
- Recent 2025-2026: strategy trails QQQ mildly while still beating SPY.

## 6. What Worked

- The candidate preserved most QQQ upside with QQQ return capture of `0.9982`.
- It improved full-period max drawdown versus QQQ by `0.0189`.
- It improved return-to-drawdown versus QQQ: `11.3985` vs `10.8050`.
- It materially helped in QQQ-relative terms during: bear_2022.

## 7. What Failed

- The strategy did not beat QQQ buy-and-hold on full-period raw return; strategy vs QQQ was `-0.0067`.
- The COVID crash/recovery window had worse drawdown than QQQ.
- Strong QQQ bull markets remain hard for the defensive switch to match.
- Weakest QQQ-relative subperiod: ai_bull_2023_2024 (`-0.1375`).

## 8. Risks And Limitations

- History is short: the selected Tiingo-only replay starts in 2019.
- The rule was selected from research sweeps and needs out-of-sample robustness checks.
- It may whipsaw around the SMA threshold, especially with `min_hold_days = 0`.
- SPY is still equity risk, not a cash or bond hedge.
- Transaction costs and slippage are simplified.
- The result is sensitive to QQQ trend regimes and may lag during strong QQQ advances.

## 9. Why It Is Not Production-Ready

- `production_ready = false`.
- This is research documentation, not a runtime signal specification.
- No live paper-tracking process has been designed for this exact candidate.
- No longer-history validation or walk-forward robustness result is attached yet.
- No production signal export should be created or updated from this replay.
- This is not evidence that QQQ buy-and-hold has been beaten.

## 10. Current Decision

- Decision: `keep_as_best_qqq_defensive_timing_candidate`.
- Research status: `research_only_defensive_candidate`.
- Keep as best QQQ defensive timing candidate.
- Keep research-only.
- Do not mark production-ready.

## 11. Next Validation Steps

- Longer-history validation if data allows.
- Walk-forward robustness.
- Live paper-tracking design.
- Transaction and slippage sensitivity.
- No production promotion yet.
