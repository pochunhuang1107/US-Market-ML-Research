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

  CANDIDATE_DEEP_DIVE_DAILY_EQUITY="${CANDIDATE_DEEP_DIVE_DAILY_EQUITY:-artifacts/deep_dives/long_history_qqq_candidate_deep_dive/long_history_qqq_candidate_deep_dive_daily_equity.csv}"
  CANDIDATE_DEEP_DIVE_DAILY_DRAWDOWN="${CANDIDATE_DEEP_DIVE_DAILY_DRAWDOWN:-artifacts/deep_dives/long_history_qqq_candidate_deep_dive/long_history_qqq_candidate_deep_dive_daily_drawdown.csv}"
  CANDIDATE_DEEP_DIVE_SUMMARY="${CANDIDATE_DEEP_DIVE_SUMMARY:-artifacts/deep_dives/long_history_qqq_candidate_deep_dive/long_history_qqq_candidate_deep_dive_summary.csv}"
  CANDIDATE_DEEP_DIVE_DECISION_MATRIX="${CANDIDATE_DEEP_DIVE_DECISION_MATRIX:-artifacts/deep_dives/long_history_qqq_candidate_deep_dive/long_history_qqq_candidate_deep_dive_decision_matrix.csv}"
  CANDIDATE_DEEP_DIVE_SELECTED_SYMBOLS="${CANDIDATE_DEEP_DIVE_SELECTED_SYMBOLS:-artifacts/deep_dives/long_history_qqq_candidate_deep_dive/long_history_qqq_candidate_deep_dive_selected_symbols.csv}"
  OUTPUT_DIR="${OUTPUT_DIR:-artifacts/sweeps/long_history_qqq_spy_core_sleeve}"
  REPORT_OUTPUT_DIR="${REPORT_OUTPUT_DIR:-artifacts/reports/long_history_qqq_spy_core_sleeve}"
  START_DATE="${START_DATE:-2019-01-01}"
  END_DATE="${END_DATE:-2026-05-15}"
  CANDIDATE_IDS="${CANDIDATE_IDS:-candidate_0001 candidate_0003}"
  SPY_CORE_WEIGHTS="${SPY_CORE_WEIGHTS:-0.00 0.20 0.30 0.40 0.50 0.60 0.70 0.80 1.00}"
  DRY_RUN="${DRY_RUN:-0}"

  mkdir -p "$OUTPUT_DIR" "$REPORT_OUTPUT_DIR"

  local cmd=(
    python -m us_market_ml.cli.run_long_history_qqq_spy_core_sleeve
    --candidate-deep-dive-daily-equity "$CANDIDATE_DEEP_DIVE_DAILY_EQUITY"
    --candidate-deep-dive-daily-drawdown "$CANDIDATE_DEEP_DIVE_DAILY_DRAWDOWN"
    --candidate-deep-dive-summary "$CANDIDATE_DEEP_DIVE_SUMMARY"
    --candidate-deep-dive-decision-matrix "$CANDIDATE_DEEP_DIVE_DECISION_MATRIX"
    --candidate-deep-dive-selected-symbols "$CANDIDATE_DEEP_DIVE_SELECTED_SYMBOLS"
    --output-dir "$OUTPUT_DIR"
    --report-output-dir "$REPORT_OUTPUT_DIR"
    --start-date "$START_DATE"
    --end-date "$END_DATE"
  )

  append_repeated_args "--candidate-id" "$CANDIDATE_IDS"
  append_repeated_args "--spy-core-weight" "$SPY_CORE_WEIGHTS"

  printf 'Candidate IDs: %s\n' "$CANDIDATE_IDS"
  printf 'SPY core weight grid: %s\n\n' "$SPY_CORE_WEIGHTS"

  printf 'Equivalent Python command:\n'
  printf '  %q' "${cmd[@]}"
  printf '\n\n'

  printf 'Running SPY core-sleeve preflight and printing selected candidates...\n'
  "${cmd[@]}" --dry-run

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '\nDRY_RUN=1: selected candidates and core weight grid written; no study artifacts generated.\n'
    exit 0
  fi

  printf '\nStarting focused SPY core plus candidate sleeve study. This is local-only and uses existing deep-dive daily returns.\n'
  "${cmd[@]}"

  printf '\nRefreshing Markdown report from CSV artifacts...\n'
  python -m us_market_ml.cli.create_long_history_qqq_spy_core_sleeve_report \
    --sweep-output-dir "$OUTPUT_DIR" \
    --output-dir "$REPORT_OUTPUT_DIR"

  printf '\nLong-history QQQ SPY core-sleeve study complete.\n'
  printf 'Sweep artifacts are in: %s\n' "$OUTPUT_DIR"
  printf 'Report artifacts are in: %s\n' "$REPORT_OUTPUT_DIR"
  printf 'Generated artifacts remain uncommitted unless you explicitly stage them.\n'
}

main "$@"
