# Research Log

## 2026-05-23: Long-History QQQ-Relative Core-Sleeve Cycle Complete

The long-history QQQ-relative model zoo, shortlist validation, candidate deep
dive, concentration-control sweep, SPY core plus candidate sleeve study,
realistic-accounting refinement, decision report, and rich replay are complete.

Current source of truth:

- `docs/current_state.md`
- `docs/long_history_qqq_core_sleeve_summary.md`
- `artifacts/reports/long_history_qqq_core_sleeve_decision/long_history_qqq_core_sleeve_decision_report.md`
- `artifacts/reports/long_history_qqq_core_sleeve_rich_replay/long_history_qqq_core_sleeve_rich_replay_report.md`

Decision:

- Primary tracking candidates: `candidate_0001` 35% SPY / 65% sleeve and
  `candidate_0003` 25% SPY / 75% sleeve, both using
  `candidate_rebalance_aligned_weights`.
- Secondary tracking references: `candidate_0001` 40% SPY / 60% sleeve and
  `candidate_0003` 40% SPY / 60% sleeve.
- Standalone `candidate_0001` remains a raw return reference only.
- Standalone `candidate_0003` remains a signal-review reference only.
- QQQ remains the primary benchmark.
- SPY is the core sleeve asset, not the primary benchmark.
- `production_ready = false` for every candidate.

Next work:

- Create a tracking/forward-validation report around the two primary
  core-sleeve candidates.
- Review `candidate_0001` signal behavior around XLE, XLU, VUG, and XLRE.
- Review `candidate_0003` signal behavior around ARKK, XLK, and MTUM.
- Consider paper-trading-style daily or quarterly signal tracking before any
  production signal export.
- Do not promote to production.
- Do not run another broad model zoo yet.
