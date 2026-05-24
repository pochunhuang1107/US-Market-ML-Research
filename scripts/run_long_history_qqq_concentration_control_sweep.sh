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

  MODEL_ZOO_RESULTS="${MODEL_ZOO_RESULTS:-artifacts/sweeps/long_history_qqq_relative_model_zoo/long_history_qqq_relative_model_zoo_results.csv}"
  CANDIDATE_DEEP_DIVE_SUMMARY="${CANDIDATE_DEEP_DIVE_SUMMARY:-artifacts/deep_dives/long_history_qqq_candidate_deep_dive/long_history_qqq_candidate_deep_dive_summary.csv}"
  CANDIDATE_DEEP_DIVE_DECISION_MATRIX="${CANDIDATE_DEEP_DIVE_DECISION_MATRIX:-artifacts/deep_dives/long_history_qqq_candidate_deep_dive/long_history_qqq_candidate_deep_dive_decision_matrix.csv}"
  CANDIDATE_DEEP_DIVE_SELECTED_SYMBOLS="${CANDIDATE_DEEP_DIVE_SELECTED_SYMBOLS:-artifacts/deep_dives/long_history_qqq_candidate_deep_dive/long_history_qqq_candidate_deep_dive_selected_symbols.csv}"
  MARKET_DATASET_GLOB="${MARKET_DATASET_GLOB:-/Users/casper/Projects/Research/Market Data/data/research/etf_daily_prices_tiingo_only/**/*.parquet}"
  OUTPUT_DIR="${OUTPUT_DIR:-artifacts/sweeps/long_history_qqq_concentration_control_sweep}"
  REPORT_OUTPUT_DIR="${REPORT_OUTPUT_DIR:-artifacts/reports/long_history_qqq_concentration_control_sweep}"
  START_DATE="${START_DATE:-2019-01-01}"
  END_DATE="${END_DATE:-2026-05-15}"
  CANDIDATE_IDS="${CANDIDATE_IDS:-candidate_0001 candidate_0003}"
  DRY_RUN="${DRY_RUN:-0}"
  MIN_HISTORY_ROWS="${MIN_HISTORY_ROWS:-252}"
  MIN_TRAIN_ELIGIBLE_ROWS="${MIN_TRAIN_ELIGIBLE_ROWS:-1000}"

  mkdir -p "$OUTPUT_DIR" "$REPORT_OUTPUT_DIR"

  local cmd=(
    python -m us_market_ml.cli.run_long_history_qqq_concentration_control_sweep
    --model-zoo-results "$MODEL_ZOO_RESULTS"
    --candidate-deep-dive-summary "$CANDIDATE_DEEP_DIVE_SUMMARY"
    --candidate-deep-dive-decision-matrix "$CANDIDATE_DEEP_DIVE_DECISION_MATRIX"
    --candidate-deep-dive-selected-symbols "$CANDIDATE_DEEP_DIVE_SELECTED_SYMBOLS"
    --market-dataset-glob "$MARKET_DATASET_GLOB"
    --output-dir "$OUTPUT_DIR"
    --report-output-dir "$REPORT_OUTPUT_DIR"
    --start-date "$START_DATE"
    --end-date "$END_DATE"
    --min-history-rows "$MIN_HISTORY_ROWS"
    --min-train-eligible-rows "$MIN_TRAIN_ELIGIBLE_ROWS"
  )

  append_repeated_args "--candidate-id" "$CANDIDATE_IDS"

  printf 'Equivalent Python command:\n'
  printf '  %q' "${cmd[@]}"
  printf '\n\n'

  printf 'Running concentration-control preflight and printing selected source candidates...\n'
  "${cmd[@]}" --dry-run

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '\nDRY_RUN=1: selected source candidates and control count written; no replay started.\n'
    exit 0
  fi

  printf '\nStarting targeted concentration-control sweep. This is local-only and much smaller than the model zoo.\n'
  "${cmd[@]}"

  printf '\nRefreshing Markdown report from CSV artifacts...\n'
  python -m us_market_ml.cli.create_long_history_qqq_concentration_control_report \
    --sweep-output-dir "$OUTPUT_DIR" \
    --output-dir "$REPORT_OUTPUT_DIR"

  printf '\nLong-history QQQ concentration-control sweep complete.\n'
  printf 'Sweep artifacts are in: %s\n' "$OUTPUT_DIR"
  printf 'Report artifacts are in: %s\n' "$REPORT_OUTPUT_DIR"
  printf 'Generated artifacts remain uncommitted unless you explicitly stage them.\n'
}

main "$@"
