#!/usr/bin/env bash
set -euo pipefail

append_values_arg() {
  local flag="$1"
  local values="$2"
  local parts=()
  read -r -a parts <<< "$values"
  cmd+=("$flag")
  if [[ "${#parts[@]}" -eq 0 ]]; then
    cmd+=("$values")
  else
    local part
    for part in "${parts[@]}"; do
      cmd+=("$part")
    done
  fi
}

main() {
  local root_dir
  root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  cd "$root_dir"

  MARKET_DATASET_GLOB="${MARKET_DATASET_GLOB:-/Users/casper/Projects/Research/Market Data/data/research/etf_daily_prices_tiingo_only/**/*.parquet}"
  OUTPUT_DIR="${OUTPUT_DIR:-artifacts/replays/tiingo_selected_candidate_rich_replay_v2}"
  START_DATE="${START_DATE:-2019-01-01}"
  END_DATE="${END_DATE:-2026-05-15}"
  MAX_WORKERS="${MAX_WORKERS:-4}"
  STEP_MONTHS="${STEP_MONTHS:-3}"
  TRAIN_WINDOW_YEARS="${TRAIN_WINDOW_YEARS:-3 5}"
  VALIDATION_WINDOW_MONTHS="${VALIDATION_WINDOW_MONTHS:-12}"
  TEST_WINDOW_MONTHS="${TEST_WINDOW_MONTHS:-3 6 12}"
  TRANSACTION_COST_BPS="${TRANSACTION_COST_BPS:-10}"
  FEATURE_SET_VERSION="${FEATURE_SET_VERSION:-benchmark_gap_features_v2}"
  ALLOCATION_OVERLAY_NAME="${ALLOCATION_OVERLAY_NAME:-equal_weight_top5_baseline}"
  REFERENCE_FEATURE_SET_VERSIONS="${REFERENCE_FEATURE_SET_VERSIONS:-default_features_v1 benchmark_gap_features_v1}"
  MIN_HISTORY_ROWS="${MIN_HISTORY_ROWS:-252}"
  MIN_TRAIN_ELIGIBLE_ROWS="${MIN_TRAIN_ELIGIBLE_ROWS:-5000}"
  INCLUDE_REFERENCES="${INCLUDE_REFERENCES:-0}"
  DRY_RUN="${DRY_RUN:-0}"

  mkdir -p "$OUTPUT_DIR"

  local cmd=(
    python -m us_market_ml.cli.run_tiingo_selected_candidate_rich_replay
    --market-dataset-glob "$MARKET_DATASET_GLOB"
    --output-dir "$OUTPUT_DIR"
    --start-date "$START_DATE"
    --end-date "$END_DATE"
    --max-workers "$MAX_WORKERS"
    --step-months "$STEP_MONTHS"
    --transaction-cost-bps "$TRANSACTION_COST_BPS"
    --feature-set-version "$FEATURE_SET_VERSION"
    --allocation-overlay-name "$ALLOCATION_OVERLAY_NAME"
    --min-history-rows "$MIN_HISTORY_ROWS"
    --min-train-eligible-rows "$MIN_TRAIN_ELIGIBLE_ROWS"
  )
  append_values_arg --train-window-years "$TRAIN_WINDOW_YEARS"
  append_values_arg --validation-window-months "$VALIDATION_WINDOW_MONTHS"
  append_values_arg --test-window-months "$TEST_WINDOW_MONTHS"
  append_values_arg --reference-feature-set-versions "$REFERENCE_FEATURE_SET_VERSIONS"

  if [[ "$INCLUDE_REFERENCES" == "0" ]]; then
    cmd+=(--no-include-references)
  else
    cmd+=(--include-references)
  fi

  cleanup_on_interrupt() {
    printf '\nInterrupt received; rich replay should terminate with this shell.\n' >&2
    trap - INT TERM
    exit 130
  }
  trap cleanup_on_interrupt INT TERM

  printf 'Equivalent Python command:\n'
  printf '  %q' "${cmd[@]}"
  printf '\n\n'

  printf 'Planning benchmark-gap v2 rich replay and printing fold count/output paths before starting...\n'
  "${cmd[@]}" --dry-run

  if [[ "$DRY_RUN" == "1" ]]; then
    trap - INT TERM
    printf '\nDRY_RUN=1; stopping after plan.\n'
    exit 0
  fi

  printf '\nStarting local Tiingo benchmark-gap v2 selected-candidate rich replay.\n'
  printf 'No S3/cloud/GPU, no MLP, no tree models, and no production signal export are used.\n\n'
  "${cmd[@]}"

  trap - INT TERM
  printf '\nRich replay complete.\n'
  printf 'Output directory:\n'
  printf '  %s\n' "$OUTPUT_DIR"
  printf 'Report:\n'
  printf '  %s/rich_replay_report.md\n' "$OUTPUT_DIR"
  printf 'Reference comparison:\n'
  printf '  %s/reference_performance.csv\n' "$OUTPUT_DIR"
  printf '  %s/feature_set_comparison.csv\n' "$OUTPUT_DIR"
}

if [[ "${TIINGO_SELECTED_CANDIDATE_RICH_REPLAY_TEST_HELPERS:-0}" != "1" ]]; then
  main "$@"
fi
