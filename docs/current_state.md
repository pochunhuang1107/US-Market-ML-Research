# Current State: QQQ-Only RF Timing Research Paper-Tracking

Last updated: 2026-05-29

## Executive Summary

The main research path is now QQQ-only random-forest timing. The selected
direction is a research-only paper-tracking setup for locked QQQ/cash RF
candidates confirmed out of time.

This is not production promotion, not a production signal export, not a model
card, and not a broad sweep. `production_ready` remains `false` for every row.

QQQ buy-and-hold remains the primary benchmark. Tradable holdings are only QQQ
and cash. SPY, XLE, ARKK, XLK, and other ETFs may appear only as features or
reference inputs; they are not holdings.

## Latest Artifact Sources

- Out-of-time confirmation report:
  `artifacts/reports/qqq_only_rf_out_of_time_confirmation/qqq_only_rf_out_of_time_confirmation_report.md`
- Out-of-time confirmation decision matrix:
  `artifacts/reports/qqq_only_rf_out_of_time_confirmation/qqq_only_rf_out_of_time_confirmation_decision_matrix.csv`
- Paper-tracking package:
  `artifacts/reports/qqq_only_rf_paper_tracking/qqq_only_rf_paper_tracking_report.md`
- Paper-tracking candidate registry:
  `artifacts/reports/qqq_only_rf_paper_tracking/qqq_only_rf_paper_tracking_candidate_registry.csv`
- Paper-tracking initial signal snapshot:
  `artifacts/reports/qqq_only_rf_paper_tracking/qqq_only_rf_paper_tracking_initial_signal.csv`

## Locked Paper-Tracking Registry

| role | candidate_id | threshold | min_holding_days | status | production_ready |
| --- | --- | ---: | ---: | --- | --- |
| Primary paper-tracking candidate | `primary_rf_threshold_050` | 0.50 | 0 | Start research paper-tracking | false |
| Secondary paper-tracking reference | `secondary_rf_threshold_045_hold10` | 0.45 | 10 | Keep beside the primary for comparison | false |
| Rejected out-of-time reference | `rejected_rf_threshold_045_hold0` | 0.45 | 0 | Deprioritized after out-of-time confirmation | false |

All three rows use:

- `model_type = random_forest`
- `training_mode = rolling_5y`
- `target_name = risk_on_20d`
- `feature_set_name = qqq_self_cross_market`
- `exposure_mode = binary_0_100`
- `rebalance_frequency = weekly`
- `rebalance_weekday = FRI`
- `random_state = 42`
- `n_estimators = 60`
- `max_depth = default/None`
- `min_samples_leaf = 5`
- `switch_margin = 0.0`

## Interpretation

The out-of-time confirmation changed the ranking. The threshold 0.50,
min_holding_days 0 row is now the primary paper-tracking candidate. The
threshold 0.45, min_holding_days 10 row passed out of time with a smaller edge
and should remain a secondary reference. The threshold 0.45, min_holding_days 0
row is deprioritized after out-of-time confirmation and should not become the
primary paper-tracking candidate.

## Explicit No-Go Decisions

- Do not mark any candidate production-ready.
- Do not create a production signal export.
- Do not create a model card from this package.
- Do not write anything under `artifacts/model_signals`.
- Do not introduce SPY core/sleeve behavior.
- Do not introduce ETF rotation holdings.
- Do not run another broad sweep yet.

## Recommended Next Work

- Start research paper-tracking for `primary_rf_threshold_050`.
- Keep `secondary_rf_threshold_045_hold10` as a secondary reference.
- Monitor QQQ-relative returns, drawdowns, raw score behavior, exposure flips,
  and turnover against QQQ buy-and-hold.
- Revisit the candidate only after enough forward paper-tracking evidence has
  accumulated.
- Keep the next phase as research paper-tracking, not production.
