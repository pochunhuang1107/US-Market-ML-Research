# Long-History QQQ Core-Sleeve Summary

This note summarizes the completed long-history QQQ-relative core-sleeve
research cycle. It is documentation only and does not imply production
readiness.

## Bottom Line

The rich replay selected two primary tracking candidates:

- `candidate_0001`, 35% SPY core / 65% candidate sleeve,
  `candidate_rebalance_aligned_weights`: primary return-oriented tracking
  candidate.
- `candidate_0003`, 25% SPY core / 75% candidate sleeve,
  `candidate_rebalance_aligned_weights`: primary balanced/risk-controlled
  tracking candidate.

Two 40% SPY / 60% sleeve rows remain secondary tracking references. Standalone
`candidate_0001` is a raw return reference only. Standalone `candidate_0003` is
a signal-review reference only. `production_ready = false` for all candidates.

QQQ remains the primary benchmark. SPY is the core sleeve asset inside selected
portfolios, not the primary benchmark.

## Primary Tracking Candidates

| candidate_id | practical role | SPY core | candidate sleeve | total_return | strategy_vs_qqq | max_drawdown | drawdown_improvement_vs_qqq | return_to_drawdown | concentration note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `candidate_0001` | Primary return-oriented tracking candidate | 35% | 65% | 4.8565 | +1.0615 | -0.3512 | +0.000017 | 13.8278 | XLE top symbol at about 43.9%; top three sleeve symbols about 86.4% |
| `candidate_0003` | Primary balanced/risk-controlled tracking candidate | 25% | 75% | 4.7054 | +0.9104 | -0.3468 | +0.0044 | 13.5661 | ARKK top symbol at about 34.1%; top three sleeve symbols about 72.8% |

Both rows use `candidate_rebalance_aligned_weights` and have
`production_ready = false`.

## Secondary References

| candidate_id | role | SPY core | candidate sleeve | total_return | strategy_vs_qqq | max_drawdown | drawdown_improvement_vs_qqq | production_ready |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `candidate_0001` | Secondary tracking reference | 40% | 60% | 4.6413 | +0.8463 | -0.3499 | +0.0013 | false |
| `candidate_0003` | Secondary tracking reference | 40% | 60% | 4.2682 | +0.4732 | -0.3227 | +0.0285 | false |

## Standalone References

| candidate_id | role | SPY core | candidate sleeve | total_return | strategy_vs_qqq | max_drawdown | production_ready |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `candidate_0001` | Raw return reference only | 0% | 100% | 6.4379 | +2.6429 | -0.3603 | false |
| `candidate_0003` | Signal-review reference only | 0% | 100% | 5.3627 | +1.5677 | -0.3956 | false |

## Artifacts

- `artifacts/reports/long_history_qqq_core_sleeve_decision/long_history_qqq_core_sleeve_decision_report.md`
- `artifacts/reports/long_history_qqq_core_sleeve_decision/long_history_qqq_core_sleeve_decision_recommendations.csv`
- `artifacts/reports/long_history_qqq_core_sleeve_rich_replay/long_history_qqq_core_sleeve_rich_replay_report.md`
- `artifacts/reports/long_history_qqq_core_sleeve_rich_replay/long_history_qqq_core_sleeve_rich_replay_summary.csv`
- `artifacts/reports/long_history_qqq_core_sleeve_rich_replay/long_history_qqq_core_sleeve_rich_replay_symbol_exposure.csv`

## Recommended Next Work

- Create a tracking/forward-validation report around the two primary
  core-sleeve candidates.
- Review `candidate_0001` signal behavior around XLE, XLU, VUG, and XLRE.
- Review `candidate_0003` signal behavior around ARKK, XLK, and MTUM.
- Consider paper-trading-style daily or quarterly signal tracking before any
  production signal export.
- Do not promote to production yet.
- Do not run another broad model zoo yet.
