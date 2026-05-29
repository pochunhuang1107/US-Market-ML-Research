#!/usr/bin/env bash
set -euo pipefail

main() {
  local root_dir
  root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  cd "$root_dir"

  MARKET_DATASET_GLOB="${MARKET_DATASET_GLOB:-/Users/casper/Projects/Research/Market Data/data/research/etf_daily_prices_tiingo_only/**/*.parquet}"
  REFERENCE_SERIES_GLOB="${REFERENCE_SERIES_GLOB:-/Users/casper/Projects/Research/Market Data/data/research/reference_series/**/*.parquet}"
  OUTPUT_DIR="${OUTPUT_DIR:-artifacts/reports/qqq_only_rf_paper_tracking}"
  START_DATE="${START_DATE:-2019-01-01}"
  END_DATE="${END_DATE:-2026-05-15}"
  DRY_RUN="${DRY_RUN:-0}"
  MIN_TRAIN_ROWS="${MIN_TRAIN_ROWS:-252}"

  OUT_OF_TIME_SUMMARY="${OUT_OF_TIME_SUMMARY:-artifacts/reports/qqq_only_rf_out_of_time_confirmation/qqq_only_rf_out_of_time_confirmation_summary.csv}"
  OUT_OF_TIME_DECISION_MATRIX="${OUT_OF_TIME_DECISION_MATRIX:-artifacts/reports/qqq_only_rf_out_of_time_confirmation/qqq_only_rf_out_of_time_confirmation_decision_matrix.csv}"
  OUT_OF_TIME_RESULTS="${OUT_OF_TIME_RESULTS:-artifacts/sweeps/qqq_only_rf_out_of_time_confirmation/qqq_only_rf_out_of_time_confirmation_results.csv}"
  OUT_OF_TIME_EXPOSURE_LOG="${OUT_OF_TIME_EXPOSURE_LOG:-artifacts/sweeps/qqq_only_rf_out_of_time_confirmation/qqq_only_rf_out_of_time_confirmation_exposure_log.csv}"

  local cmd=(
    python -m us_market_ml.cli.create_qqq_only_rf_paper_tracking_package
    --market-dataset-glob "$MARKET_DATASET_GLOB"
    --reference-series-glob "$REFERENCE_SERIES_GLOB"
    --output-dir "$OUTPUT_DIR"
    --start-date "$START_DATE"
    --end-date "$END_DATE"
    --min-train-rows "$MIN_TRAIN_ROWS"
    --out-of-time-summary "$OUT_OF_TIME_SUMMARY"
    --out-of-time-decision-matrix "$OUT_OF_TIME_DECISION_MATRIX"
    --out-of-time-results "$OUT_OF_TIME_RESULTS"
    --out-of-time-exposure-log "$OUT_OF_TIME_EXPOSURE_LOG"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    cmd+=(--dry-run)
  fi

  printf 'QQQ-only RF paper-tracking package guardrails:\n'
  printf '  local-only: true\n'
  printf '  reads local data and prior artifacts only: true\n'
  printf '  S3/cloud: false\n'
  printf '  broad sweep: false\n'
  printf '  production signal export: false\n'
  printf '  model card: false\n'
  printf '  tradable holdings: QQQ/cash only\n'
  printf '  SPY core/sleeve: false\n'
  printf '  ETF rotation holdings: false\n'
  printf '  output under artifacts/model_signals: false\n'
  printf '  generated artifacts remain uncommitted\n\n'

  printf 'Equivalent Python command:\n'
  printf '  %q' "${cmd[@]}"
  printf '\n\n'

  "${cmd[@]}"
}

main "$@"
