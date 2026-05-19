#!/usr/bin/env bash
set -euo pipefail

main() {
  local root_dir
  root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  cd "$root_dir"

  MARKET_DATASET_GLOB="${MARKET_DATASET_GLOB:-/Users/casper/Projects/Research/Market Data/data/research/etf_daily_prices_tiingo_only/**/*.parquet}"
  OUTPUT_DIR="${OUTPUT_DIR:-artifacts/replays/tiingo_qqq_sma190_defensive_replay}"
  START_DATE="${START_DATE:-2019-01-01}"
  END_DATE="${END_DATE:-2026-05-15}"
  TRANSACTION_COST_BPS="${TRANSACTION_COST_BPS:-10}"
  STRATEGY_NAME="${STRATEGY_NAME:-qqq_sma190_buffer_p0p5_hold0_to_spy}"
  SLOW_SMA_DAYS="${SLOW_SMA_DAYS:-190}"
  BUFFER_PCT="${BUFFER_PCT:-0.005}"
  MIN_HOLD_DAYS="${MIN_HOLD_DAYS:-0}"
  RISK_OFF_ASSET="${RISK_OFF_ASSET:-SPY}"
  DRY_RUN="${DRY_RUN:-0}"
  CREATE_REPORT="${CREATE_REPORT:-1}"

  local terminal_log report_path preflight_md
  terminal_log="$OUTPUT_DIR/tiingo_qqq_sma190_defensive_replay_terminal.log"
  report_path="$OUTPUT_DIR/tiingo_qqq_sma190_defensive_replay_report.md"
  preflight_md="$OUTPUT_DIR/tiingo_qqq_sma190_defensive_replay_preflight.md"

  mkdir -p "$OUTPUT_DIR"
  : > "$terminal_log"

  local cmd=(
    python -m us_market_ml.cli.run_tiingo_qqq_defensive_timing_selected_replay
    --market-dataset-glob "$MARKET_DATASET_GLOB"
    --output-dir "$OUTPUT_DIR"
    --start-date "$START_DATE"
    --end-date "$END_DATE"
    --transaction-cost-bps "$TRANSACTION_COST_BPS"
    --strategy-name "$STRATEGY_NAME"
    --slow-sma-days "$SLOW_SMA_DAYS"
    --buffer-pct "$BUFFER_PCT"
    --min-hold-days "$MIN_HOLD_DAYS"
    --risk-off-asset "$RISK_OFF_ASSET"
  )

  printf 'Selected QQQ defensive timing replay configuration:\n' | tee -a "$terminal_log"
  printf '  STRATEGY_NAME=%s\n' "$STRATEGY_NAME" | tee -a "$terminal_log"
  printf '  risk_on_asset=QQQ\n' | tee -a "$terminal_log"
  printf '  RISK_OFF_ASSET=%s\n' "$RISK_OFF_ASSET" | tee -a "$terminal_log"
  printf '  signal_name=qqq_sma_price_filter\n' | tee -a "$terminal_log"
  printf '  SLOW_SMA_DAYS=%s\n' "$SLOW_SMA_DAYS" | tee -a "$terminal_log"
  printf '  BUFFER_PCT=%s\n' "$BUFFER_PCT" | tee -a "$terminal_log"
  printf '  MIN_HOLD_DAYS=%s\n' "$MIN_HOLD_DAYS" | tee -a "$terminal_log"
  printf '  TRANSACTION_COST_BPS=%s\n' "$TRANSACTION_COST_BPS" | tee -a "$terminal_log"
  printf 'Required symbols: QQQ, SPY.\n' | tee -a "$terminal_log"
  printf 'Local-only guardrails: no S3/cloud/GPU, no tree_screen, no tree models, no MLP, no feature v4, no production signal export.\n' | tee -a "$terminal_log"
  printf 'Generated artifacts should remain uncommitted.\n' | tee -a "$terminal_log"
  printf 'Equivalent Python command:\n' | tee -a "$terminal_log"
  printf '  %q' "${cmd[@]}" | tee -a "$terminal_log"
  printf '\n\n' | tee -a "$terminal_log"

  "${cmd[@]}" --dry-run 2>&1 | tee -a "$terminal_log"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '\nDRY_RUN=1: preflight complete; no replay CSV outputs generated.\n' | tee -a "$terminal_log"
    printf 'Preflight Markdown: %s\n' "$preflight_md" | tee -a "$terminal_log"
    exit 0
  fi

  printf '\nStarting selected Tiingo QQQ defensive timing replay.\n' | tee -a "$terminal_log"
  "${cmd[@]}" 2>&1 | tee -a "$terminal_log"

  if [[ "$CREATE_REPORT" != "0" ]]; then
    printf '\nCreating selected Tiingo QQQ defensive timing replay report...\n' | tee -a "$terminal_log"
    python -m us_market_ml.cli.create_tiingo_qqq_defensive_timing_selected_replay_report \
      --output-dir "$OUTPUT_DIR" \
      --report-path "$report_path" 2>&1 | tee -a "$terminal_log"
  fi

  printf '\nSelected Tiingo QQQ defensive timing replay complete.\n' | tee -a "$terminal_log"
  printf 'Useful output commands:\n' | tee -a "$terminal_log"
  printf '  find %q -maxdepth 1 -type f | sort\n' "$OUTPUT_DIR" | tee -a "$terminal_log"
  printf '  sed -n %q %q\n' '1,260p' "$report_path" | tee -a "$terminal_log"
}

main "$@"
