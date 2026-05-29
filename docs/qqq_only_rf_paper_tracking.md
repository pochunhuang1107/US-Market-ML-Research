# QQQ-Only RF Paper-Tracking Notes

This package tracks locked QQQ/cash random-forest timing candidates after the
out-of-time confirmation. It is research-only and does not create production
signals, production model artifacts, or a model card.

## Candidate Registry

The registry contains exactly three rows:

| candidate_id | role | threshold | min_holding_days | production_ready |
| --- | --- | ---: | ---: | --- |
| `primary_rf_threshold_050` | primary paper-tracking candidate | 0.50 | 0 | false |
| `secondary_rf_threshold_045_hold10` | secondary paper-tracking reference | 0.45 | 10 | false |
| `rejected_rf_threshold_045_hold0` | rejected out-of-time reference | 0.45 | 0 | false |

All rows are QQQ/cash only. Other ETFs and reference series are features only.
QQQ buy-and-hold remains the primary benchmark.

## Output Package

Default output directory:

`artifacts/reports/qqq_only_rf_paper_tracking`

Required files:

- `qqq_only_rf_paper_tracking_candidate_registry.csv`
- `qqq_only_rf_paper_tracking_candidate_registry.json`
- `qqq_only_rf_paper_tracking_initial_signal.csv`
- `qqq_only_rf_paper_tracking_initial_signal.json`
- `qqq_only_rf_paper_tracking_replay_summary.csv`
- `qqq_only_rf_paper_tracking_decision_matrix.csv`
- `qqq_only_rf_paper_tracking_report.md`

Required charts:

- `qqq_only_rf_paper_tracking_equity_curve.png`
- `qqq_only_rf_paper_tracking_drawdown_curve.png`
- `qqq_only_rf_paper_tracking_exposure_timeline.png`
- `qqq_only_rf_paper_tracking_holdout_comparison.png`

## Update Workflow

Default update command:

`bash scripts/update_qqq_only_rf_paper_tracking.sh`

Default update output directory:

`artifacts/reports/qqq_only_rf_paper_tracking_updates`

The update workflow reads the locked package registry from
`artifacts/reports/qqq_only_rf_paper_tracking`, recomputes only the three locked
QQQ/cash RF rows, compares the latest signal snapshot with the prior update
history or initial package signal, and updates a local history CSV.

Required update files:

- `qqq_only_rf_paper_tracking_update_signal.csv`
- `qqq_only_rf_paper_tracking_update_signal.json`
- `qqq_only_rf_paper_tracking_update_summary.csv`
- `qqq_only_rf_paper_tracking_update_history.csv`
- `qqq_only_rf_paper_tracking_update_decision_matrix.csv`
- `qqq_only_rf_paper_tracking_update_report.md`

Optional update charts:

- `qqq_only_rf_paper_tracking_update_exposure_snapshot.png`
- `qqq_only_rf_paper_tracking_update_signal_history.png`

Update decision labels are `maintain_tracking`, `exposure_change_watch`,
`data_missing`, and `guardrail_failed`.

## Guardrails

- Research-only.
- No production signal export.
- No model card.
- No writes under `artifacts/model_signals`.
- No SPY core/sleeve.
- No ETF rotation holdings.
- `production_ready` is always `false`.
- Do not run another broad sweep yet.
