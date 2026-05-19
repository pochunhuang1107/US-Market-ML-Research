#!/usr/bin/env bash
set -euo pipefail

main() {
  local root_dir
  root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  cd "$root_dir"

  MARKET_DATASET_GLOB="${MARKET_DATASET_GLOB:-/Users/casper/Projects/Research/Market Data/data/research/etf_daily_prices_tiingo_only/**/*.parquet}"
  OUTPUT_DIR="${OUTPUT_DIR:-artifacts/sweeps/tiingo_qqq_defensive_timing_sweep}"
  REPORT_OUTPUT_DIR="${REPORT_OUTPUT_DIR:-artifacts/reports/tiingo_qqq_defensive_timing_sweep}"
  START_DATE="${START_DATE:-2019-01-01}"
  END_DATE="${END_DATE:-2026-05-15}"
  TRANSACTION_COST_BPS="${TRANSACTION_COST_BPS:-10}"
  DRY_RUN="${DRY_RUN:-0}"
  CREATE_REPORT="${CREATE_REPORT:-1}"

  local terminal_log results_csv preflight_md
  terminal_log="$OUTPUT_DIR/tiingo_qqq_defensive_timing_sweep_terminal.log"
  results_csv="$OUTPUT_DIR/tiingo_qqq_defensive_timing_sweep_results.csv"
  preflight_md="$OUTPUT_DIR/tiingo_qqq_defensive_timing_sweep_preflight.md"

  mkdir -p "$OUTPUT_DIR" "$REPORT_OUTPUT_DIR"

  local cmd=(
    python -m us_market_ml.cli.run_tiingo_qqq_defensive_timing_sweep
    --market-dataset-glob "$MARKET_DATASET_GLOB"
    --output-dir "$OUTPUT_DIR"
    --start-date "$START_DATE"
    --end-date "$END_DATE"
    --transaction-cost-bps "$TRANSACTION_COST_BPS"
  )

  printf 'Equivalent Python command:\n' | tee -a "$terminal_log"
  printf '  %q' "${cmd[@]}" | tee -a "$terminal_log"
  printf '\n\n' | tee -a "$terminal_log"
  printf 'Local-only guardrails: no S3/cloud/GPU, no tree_screen, no tree models, no MLP, no feature v4, no production signal export.\n' | tee -a "$terminal_log"
  printf 'Required symbols: QQQ, SPY. Optional defensive symbols: IEF, TLT, SHY.\n' | tee -a "$terminal_log"

  {
    printf '\n==== %s preflight tiingo_qqq_defensive_timing_sweep ====\n' "$(date '+%Y-%m-%d %H:%M:%S')"
    printf 'Command:'
    printf ' %q' "${cmd[@]}" --dry-run
    printf '\n'
  } >> "$terminal_log"

  "${cmd[@]}" --dry-run 2>&1 | tee -a "$terminal_log"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '\nDRY_RUN=1: preflight complete; no sweep outputs generated.\n' | tee -a "$terminal_log"
    printf 'Preflight Markdown: %s\n' "$preflight_md" | tee -a "$terminal_log"
    exit 0
  fi

  {
    printf '\n==== %s starting tiingo_qqq_defensive_timing_sweep ====\n' "$(date '+%Y-%m-%d %H:%M:%S')"
    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
  } >> "$terminal_log"

  printf '\nStarting focused Tiingo QQQ defensive timing sweep.\n' | tee -a "$terminal_log"
  "${cmd[@]}" 2>&1 | tee -a "$terminal_log"

  if [[ "$CREATE_REPORT" != "0" ]]; then
    printf '\nCreating Tiingo QQQ defensive timing sweep report...\n' | tee -a "$terminal_log"
    python -m us_market_ml.cli.create_tiingo_qqq_defensive_timing_sweep_report \
      --results-csv "$results_csv" \
      --market-dataset-glob "$MARKET_DATASET_GLOB" \
      --output-dir "$REPORT_OUTPUT_DIR" 2>&1 | tee -a "$terminal_log"
  fi

  printf '\nTiingo QQQ defensive timing sweep complete.\n' | tee -a "$terminal_log"
  printf 'Useful output commands:\n' | tee -a "$terminal_log"
  printf '  tail -n 160 %q\n' "$terminal_log" | tee -a "$terminal_log"
  printf '  sed -n %q %q\n' '1,280p' "$preflight_md" | tee -a "$terminal_log"
  printf '  find %q -maxdepth 1 -type f | sort\n' "$OUTPUT_DIR" | tee -a "$terminal_log"
  printf '  find %q -maxdepth 1 -type f | sort\n' "$REPORT_OUTPUT_DIR" | tee -a "$terminal_log"
}

main "$@"
