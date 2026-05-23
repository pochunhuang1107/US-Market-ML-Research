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
  MODEL_ZOO_DECISION_MATRIX="${MODEL_ZOO_DECISION_MATRIX:-artifacts/reports/long_history_qqq_relative_model_zoo/long_history_qqq_relative_model_zoo_decision_matrix.csv}"
  MARKET_DATASET_GLOB="${MARKET_DATASET_GLOB:-/Users/casper/Projects/Research/Market Data/data/research/etf_daily_prices_tiingo_only/**/*.parquet}"
  OUTPUT_DIR="${OUTPUT_DIR:-artifacts/validation/long_history_qqq_shortlist_validation}"
  START_DATE="${START_DATE:-2019-01-01}"
  END_DATE="${END_DATE:-2026-05-15}"
  TOP_N="${TOP_N:-10}"
  DRY_RUN="${DRY_RUN:-0}"
  MIN_HISTORY_ROWS="${MIN_HISTORY_ROWS:-252}"
  MIN_TRAIN_ELIGIBLE_ROWS="${MIN_TRAIN_ELIGIBLE_ROWS:-1000}"
  CANDIDATE_ID="${CANDIDATE_ID:-}"
  INCLUDE_DECISION="${INCLUDE_DECISION:-}"

  mkdir -p "$OUTPUT_DIR"

  local cmd=(
    python -m us_market_ml.cli.run_long_history_qqq_shortlist_validation
    --model-zoo-results "$MODEL_ZOO_RESULTS"
    --model-zoo-decision-matrix "$MODEL_ZOO_DECISION_MATRIX"
    --market-dataset-glob "$MARKET_DATASET_GLOB"
    --output-dir "$OUTPUT_DIR"
    --start-date "$START_DATE"
    --end-date "$END_DATE"
    --top-n "$TOP_N"
    --min-history-rows "$MIN_HISTORY_ROWS"
    --min-train-eligible-rows "$MIN_TRAIN_ELIGIBLE_ROWS"
  )

  append_repeated_args "--candidate-id" "$CANDIDATE_ID"
  append_repeated_args "--include-decision" "$INCLUDE_DECISION"

  printf 'Equivalent Python command:\n'
  printf '  %q' "${cmd[@]}"
  printf '\n\n'

  printf 'Running shortlist preflight and printing selected candidates...\n'
  "${cmd[@]}" --dry-run

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '\nDRY_RUN=1: selected candidates and preflight written; no validation replay started.\n'
    exit 0
  fi

  printf '\nStarting focused validation replay. This is local-only and should be much faster than the model zoo.\n'
  "${cmd[@]}"

  printf '\nRefreshing Markdown report from CSV artifacts...\n'
  python -m us_market_ml.cli.create_long_history_qqq_shortlist_validation_report \
    --output-dir "$OUTPUT_DIR"

  printf '\nLong-history QQQ shortlist validation complete.\n'
  printf 'Artifacts are in: %s\n' "$OUTPUT_DIR"
  printf 'Generated artifacts remain uncommitted unless you explicitly stage them.\n'
}

main "$@"
