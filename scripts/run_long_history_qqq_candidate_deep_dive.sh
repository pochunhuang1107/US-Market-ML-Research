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
  SHORTLIST_CANDIDATES="${SHORTLIST_CANDIDATES:-artifacts/validation/long_history_qqq_shortlist_validation/long_history_qqq_shortlist_validation_candidates.csv}"
  SHORTLIST_DECISION_MATRIX="${SHORTLIST_DECISION_MATRIX:-artifacts/validation/long_history_qqq_shortlist_validation/long_history_qqq_shortlist_validation_decision_matrix.csv}"
  MARKET_DATASET_GLOB="${MARKET_DATASET_GLOB:-/Users/casper/Projects/Research/Market Data/data/research/etf_daily_prices_tiingo_only/**/*.parquet}"
  OUTPUT_DIR="${OUTPUT_DIR:-artifacts/deep_dives/long_history_qqq_candidate_deep_dive}"
  START_DATE="${START_DATE:-2019-01-01}"
  END_DATE="${END_DATE:-2026-05-15}"
  CANDIDATE_IDS="${CANDIDATE_IDS:-candidate_0001 candidate_0003}"
  DRY_RUN="${DRY_RUN:-0}"
  MIN_HISTORY_ROWS="${MIN_HISTORY_ROWS:-252}"
  MIN_TRAIN_ELIGIBLE_ROWS="${MIN_TRAIN_ELIGIBLE_ROWS:-1000}"
  INCLUDE_DEFAULT_CANDIDATES="${INCLUDE_DEFAULT_CANDIDATES:-0}"

  mkdir -p "$OUTPUT_DIR"

  local cmd=(
    python -m us_market_ml.cli.run_long_history_qqq_candidate_deep_dive
    --model-zoo-results "$MODEL_ZOO_RESULTS"
    --shortlist-candidates "$SHORTLIST_CANDIDATES"
    --shortlist-decision-matrix "$SHORTLIST_DECISION_MATRIX"
    --market-dataset-glob "$MARKET_DATASET_GLOB"
    --output-dir "$OUTPUT_DIR"
    --start-date "$START_DATE"
    --end-date "$END_DATE"
    --min-history-rows "$MIN_HISTORY_ROWS"
    --min-train-eligible-rows "$MIN_TRAIN_ELIGIBLE_ROWS"
  )

  append_repeated_args "--candidate-id" "$CANDIDATE_IDS"
  if [[ "$INCLUDE_DEFAULT_CANDIDATES" == "1" ]]; then
    cmd+=(--include-default-candidates)
  fi

  printf 'Equivalent Python command:\n'
  printf '  %q' "${cmd[@]}"
  printf '\n\n'

  printf 'Running deep-dive preflight and printing selected candidates...\n'
  "${cmd[@]}" --dry-run

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '\nDRY_RUN=1: selected candidates and preflight written; no replay started.\n'
    exit 0
  fi

  printf '\nStarting rich candidate replay. This is local-only and should finish faster than the model zoo.\n'
  "${cmd[@]}"

  printf '\nRefreshing Markdown report from CSV artifacts...\n'
  python -m us_market_ml.cli.create_long_history_qqq_candidate_deep_dive_report \
    --output-dir "$OUTPUT_DIR"

  printf '\nLong-history QQQ candidate deep dive complete.\n'
  printf 'Artifacts are in: %s\n' "$OUTPUT_DIR"
  printf 'Generated artifacts remain uncommitted unless you explicitly stage them.\n'
}

main "$@"
