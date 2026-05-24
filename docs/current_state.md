# Current State: Long-History QQQ-Relative Core-Sleeve Research

Last updated: 2026-05-23

## Executive Summary

The long-history QQQ-relative research cycle is complete through rich replay.
It produced research winners and tracking candidates only. No candidate is
production-ready, and no production signal export, model card, or live trading
artifact should be created from this state.

QQQ buy-and-hold remains the primary benchmark. SPY is used as the core sleeve
asset in blended portfolios and as a secondary reference, not as the primary
benchmark.

The next serious work should be tracking or forward-style validation, signal
review, and report synthesis around the selected core-sleeve candidates. Do not
run another broad model zoo yet.

## Latest Artifact Sources

- Decision report:
  `artifacts/reports/long_history_qqq_core_sleeve_decision/long_history_qqq_core_sleeve_decision_report.md`
- Decision recommendations:
  `artifacts/reports/long_history_qqq_core_sleeve_decision/long_history_qqq_core_sleeve_decision_recommendations.csv`
- Rich replay report:
  `artifacts/reports/long_history_qqq_core_sleeve_rich_replay/long_history_qqq_core_sleeve_rich_replay_report.md`
- Rich replay summary:
  `artifacts/reports/long_history_qqq_core_sleeve_rich_replay/long_history_qqq_core_sleeve_rich_replay_summary.csv`
- Rich replay symbol exposure:
  `artifacts/reports/long_history_qqq_core_sleeve_rich_replay/long_history_qqq_core_sleeve_rich_replay_symbol_exposure.csv`

## Completed Research Cycle

The completed cycle covered:

1. Long-history QQQ-relative model zoo.
2. Shortlist validation.
3. Candidate deep dive.
4. Concentration-control sweep.
5. SPY core plus candidate sleeve study.
6. Core-sleeve refinement with realistic accounting.
7. Core-sleeve decision report.
8. Core-sleeve rich replay report.

## Candidate Role Definitions

- Raw return reference: standalone sleeve kept to understand maximum historical
  return potential and signal behavior; not a tracking or production candidate.
- Signal-review reference: standalone sleeve kept mainly to inspect model
  selection behavior; not a tracking or production candidate.
- Primary tracking candidate: selected core-sleeve blend to monitor in
  tracking/forward-style validation.
- Secondary tracking reference: selected alternate core-sleeve blend to keep
  beside the primary rows for comparison.
- Production-ready candidate: none. `production_ready = false` for every row.

## Selected Tracking Candidates

| role | candidate_id | SPY core | candidate sleeve | accounting mode | total_return | QQQ buy-hold total_return | strategy_vs_qqq | max_drawdown | drawdown_improvement_vs_qqq | return_to_drawdown | top sleeve symbol | top3 sleeve exposure | production_ready |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| Primary return-oriented tracking candidate | `candidate_0001` | 35% | 65% | `candidate_rebalance_aligned_weights` | 4.8565 | 3.7950 | +1.0615 | -0.3512 | +0.000017 | 13.8278 | XLE, 43.9% | 86.4% | false |
| Primary balanced/risk-controlled tracking candidate | `candidate_0003` | 25% | 75% | `candidate_rebalance_aligned_weights` | 4.7054 | 3.7950 | +0.9104 | -0.3468 | +0.0044 | 13.5661 | ARKK, 34.1% | 72.8% | false |
| Secondary tracking reference | `candidate_0001` | 40% | 60% | `candidate_rebalance_aligned_weights` | 4.6413 | 3.7950 | +0.8463 | -0.3499 | +0.0013 | 13.2650 | XLE, 43.9% | 86.4% | false |
| Secondary tracking reference | `candidate_0003` | 40% | 60% | `candidate_rebalance_aligned_weights` | 4.2682 | 3.7950 | +0.4732 | -0.3227 | +0.0285 | 13.2252 | ARKK, 34.1% | 72.8% | false |

## Standalone References

| role | candidate_id | SPY core | candidate sleeve | accounting mode | total_return | strategy_vs_qqq | max_drawdown | drawdown_improvement_vs_qqq | return_to_drawdown | production_ready |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Raw return reference only | `candidate_0001` | 0% | 100% | `daily_rebalanced_fixed_weights` | 6.4379 | +2.6429 | -0.3603 | -0.0091 | 17.8669 | false |
| Signal-review reference only | `candidate_0003` | 0% | 100% | `daily_rebalanced_fixed_weights` | 5.3627 | +1.5677 | -0.3956 | -0.0443 | 13.5575 | false |

## Interpretation

`candidate_0001` is the primary return-oriented tracking direction. The 35% SPY
core / 65% candidate sleeve row keeps meaningful QQQ-relative excess return
while bringing drawdown roughly in line with QQQ. Its sleeve is concentrated in
XLE, XLU, VUG, and XLRE, so tracking work should inspect whether that behavior
is stable and economically sensible.

`candidate_0003` is the primary balanced/risk-controlled tracking direction.
The 25% SPY core / 75% candidate sleeve row gives up some return versus
`candidate_0001` but modestly improves QQQ-relative drawdown. Its sleeve is led
by ARKK, XLK, and MTUM, so signal review should focus on whether that exposure
is intentional, repeatable, and acceptable.

The standalone rows are references only. They are not production candidates and
should not be used as direct deployment targets.

## Explicit No-Go Decisions

- Do not mark any candidate production-ready.
- Do not create a production signal export from these rows.
- Do not create a model card as part of this current-state update.
- Do not run a new sweep or train models from this state.
- Do not run another broad model zoo yet.
- Do not treat SPY as the primary benchmark; QQQ remains the primary benchmark.

## Recommended Next Work

- Create a tracking/forward-validation report around the two primary
  core-sleeve candidates.
- Review `candidate_0001` signal behavior around XLE, XLU, VUG, and XLRE
  exposure.
- Review `candidate_0003` signal behavior around ARKK, XLK, and MTUM exposure.
- Consider paper-trading-style daily or quarterly signal tracking before any
  production signal export.
- Do not promote to production yet.
- Do not run another broad model zoo yet.
