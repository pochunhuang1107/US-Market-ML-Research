#!/usr/bin/env bash
set -euo pipefail

main() {
  local root_dir
  root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  cd "$root_dir"

  MARKET_DATASET_GLOB="${MARKET_DATASET_GLOB:-/Users/casper/Projects/Research/Market Data/data/research/etf_daily_prices_tiingo_only/**/*.parquet}"
  REFERENCE_SERIES_GLOB="${REFERENCE_SERIES_GLOB:-/Users/casper/Projects/Research/Market Data/data/research/reference_series/**/*.parquet}"
  PACKAGE_DIR="${PACKAGE_DIR:-artifacts/reports/qqq_only_rf_paper_tracking}"
  OUTPUT_DIR="${OUTPUT_DIR:-artifacts/reports/qqq_only_rf_paper_tracking_updates}"
  START_DATE="${START_DATE:-2019-01-01}"
  END_DATE="${END_DATE:-2026-05-15}"
  DRY_RUN="${DRY_RUN:-0}"
  MIN_TRAIN_ROWS="${MIN_TRAIN_ROWS:-252}"

  local cmd=(
    python -m us_market_ml.cli.update_qqq_only_rf_paper_tracking
    --market-dataset-glob "$MARKET_DATASET_GLOB"
    --reference-series-glob "$REFERENCE_SERIES_GLOB"
    --package-dir "$PACKAGE_DIR"
    --output-dir "$OUTPUT_DIR"
    --start-date "$START_DATE"
    --end-date "$END_DATE"
    --min-train-rows "$MIN_TRAIN_ROWS"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    cmd+=(--dry-run)
  fi

  printf 'QQQ-only RF paper-tracking update guardrails:\n'
  printf '  local-only: true\n'
  printf '  reads locked package registry and local data only: true\n'
  printf '  S3/cloud: false\n'
  printf '  broad sweep: false\n'
  printf '  production signal export: false\n'
  printf '  model card: false\n'
  printf '  production_ready always false\n'
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
