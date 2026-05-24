#!/usr/bin/env bash
set -euo pipefail

append_repeated_args() {
  local flag="$1"
  local values="$2"
  local value
  values="${values//,/ }"
  for value in $values; do
    if [[ -n "$value" ]]; then
      cmd+=("$flag" "$value")
    fi
  done
}

main() {
  local root_dir
  root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  cd "$root_dir"

  SPY_CORE_SLEEVE_RESULTS="${SPY_CORE_SLEEVE_RESULTS:-artifacts/sweeps/long_history_qqq_spy_core_sleeve/long_history_qqq_spy_core_sleeve_results.csv}"
  SPY_CORE_SLEEVE_DAILY_EQUITY="${SPY_CORE_SLEEVE_DAILY_EQUITY:-artifacts/sweeps/long_history_qqq_spy_core_sleeve/long_history_qqq_spy_core_sleeve_daily_equity.csv}"
  CANDIDATE_DEEP_DIVE_DAILY_EQUITY="${CANDIDATE_DEEP_DIVE_DAILY_EQUITY:-artifacts/deep_dives/long_history_qqq_candidate_deep_dive/long_history_qqq_candidate_deep_dive_daily_equity.csv}"
  CANDIDATE_DEEP_DIVE_SUMMARY="${CANDIDATE_DEEP_DIVE_SUMMARY:-artifacts/deep_dives/long_history_qqq_candidate_deep_dive/long_history_qqq_candidate_deep_dive_summary.csv}"
  CANDIDATE_DEEP_DIVE_DECISION_MATRIX="${CANDIDATE_DEEP_DIVE_DECISION_MATRIX:-artifacts/deep_dives/long_history_qqq_candidate_deep_dive/long_history_qqq_candidate_deep_dive_decision_matrix.csv}"
  OUTPUT_DIR="${OUTPUT_DIR:-artifacts/sweeps/long_history_qqq_core_sleeve_refinement}"
  REPORT_OUTPUT_DIR="${REPORT_OUTPUT_DIR:-artifacts/reports/long_history_qqq_core_sleeve_refinement}"
  START_DATE="${START_DATE:-2019-01-01}"
  END_DATE="${END_DATE:-2026-05-15}"
  CANDIDATE_IDS="${CANDIDATE_IDS:-candidate_0001 candidate_0003}"
  PORTFOLIO_ACCOUNTING_MODES="${PORTFOLIO_ACCOUNTING_MODES:-daily_rebalanced_fixed_weights monthly_rebalanced_weights quarterly_rebalanced_weights candidate_rebalance_aligned_weights}"
  DRY_RUN="${DRY_RUN:-0}"

  mkdir -p "$OUTPUT_DIR" "$REPORT_OUTPUT_DIR"

  local cmd=(
    python -m us_market_ml.cli.run_long_history_qqq_core_sleeve_refinement
    --spy-core-sleeve-results "$SPY_CORE_SLEEVE_RESULTS"
    --spy-core-sleeve-daily-equity "$SPY_CORE_SLEEVE_DAILY_EQUITY"
    --candidate-deep-dive-daily-equity "$CANDIDATE_DEEP_DIVE_DAILY_EQUITY"
    --candidate-deep-dive-summary "$CANDIDATE_DEEP_DIVE_SUMMARY"
    --candidate-deep-dive-decision-matrix "$CANDIDATE_DEEP_DIVE_DECISION_MATRIX"
    --output-dir "$OUTPUT_DIR"
    --report-output-dir "$REPORT_OUTPUT_DIR"
    --start-date "$START_DATE"
    --end-date "$END_DATE"
  )

  append_repeated_args "--candidate-id" "$CANDIDATE_IDS"
  append_repeated_args "--portfolio-accounting-mode" "$PORTFOLIO_ACCOUNTING_MODES"

  printf 'Candidate IDs: %s\n' "$CANDIDATE_IDS"
  printf 'Accounting modes: %s\n' "$PORTFOLIO_ACCOUNTING_MODES"
  printf 'candidate_0001 SPY core grid: 0.00 0.30 0.35 0.40 0.45 0.50 0.55 0.60 1.00\n'
  printf 'candidate_0003 SPY core grid: 0.00 0.20 0.25 0.30 0.35 0.40 0.45 0.50 1.00\n\n'
  printf 'Local-only guardrails: no S3/cloud, no broad model zoo, no MLP, no model card, no production signal export, no Market Data repo modification.\n\n'

  printf 'Equivalent Python command:\n'
  printf '  %q' "${cmd[@]}"
  printf '\n\n'

  printf 'Running refinement preflight and printing selected candidates...\n'
  "${cmd[@]}" --dry-run

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '\nDRY_RUN=1: selected candidates, accounting modes, and fine grid written; no study artifacts generated.\n'
    exit 0
  fi

  printf '\nStarting focused SPY core plus candidate sleeve refinement. This uses existing local daily returns and should run quickly.\n'
  "${cmd[@]}"

  printf '\nLong-history QQQ core-sleeve refinement complete.\n'
  printf 'Sweep artifacts are in: %s\n' "$OUTPUT_DIR"
  printf 'Report artifacts are in: %s\n' "$REPORT_OUTPUT_DIR"
  printf 'Generated artifacts remain uncommitted unless you explicitly stage them.\n'
}

main "$@"
