#!/usr/bin/env bash
set -euo pipefail

main() {
  local root_dir
  root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  cd "$root_dir"

  RF_CONFIRMATION_RESULTS="${RF_CONFIRMATION_RESULTS:-artifacts/sweeps/qqq_only_rf_confirmation/qqq_only_rf_confirmation_results.csv}"
  RF_CONFIRMATION_DAILY_EQUITY="${RF_CONFIRMATION_DAILY_EQUITY:-artifacts/sweeps/qqq_only_rf_confirmation/qqq_only_rf_confirmation_daily_equity.csv}"
  RF_CONFIRMATION_DAILY_DRAWDOWN="${RF_CONFIRMATION_DAILY_DRAWDOWN:-artifacts/sweeps/qqq_only_rf_confirmation/qqq_only_rf_confirmation_daily_drawdown.csv}"
  RF_CONFIRMATION_EXPOSURE_LOG="${RF_CONFIRMATION_EXPOSURE_LOG:-artifacts/sweeps/qqq_only_rf_confirmation/qqq_only_rf_confirmation_exposure_log.csv}"
  RF_CONFIRMATION_SUMMARY="${RF_CONFIRMATION_SUMMARY:-artifacts/reports/qqq_only_rf_confirmation/qqq_only_rf_confirmation_summary.csv}"
  RF_CONFIRMATION_DECISION_MATRIX="${RF_CONFIRMATION_DECISION_MATRIX:-artifacts/reports/qqq_only_rf_confirmation/qqq_only_rf_confirmation_decision_matrix.csv}"
  OUTPUT_DIR="${OUTPUT_DIR:-artifacts/reports/qqq_only_rf_signal_review}"
  DRY_RUN="${DRY_RUN:-0}"

  local cmd=(
    python -m us_market_ml.cli.create_qqq_only_rf_signal_review
    --results "$RF_CONFIRMATION_RESULTS"
    --daily-equity "$RF_CONFIRMATION_DAILY_EQUITY"
    --daily-drawdown "$RF_CONFIRMATION_DAILY_DRAWDOWN"
    --exposure-log "$RF_CONFIRMATION_EXPOSURE_LOG"
    --summary "$RF_CONFIRMATION_SUMMARY"
    --decision-matrix "$RF_CONFIRMATION_DECISION_MATRIX"
    --output-dir "$OUTPUT_DIR"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    cmd+=(--dry-run)
  fi

  printf 'QQQ-only RF signal review guardrails:\n'
  printf '  local-only: true\n'
  printf '  reads existing artifacts only: true\n'
  printf '  S3/cloud: false\n'
  printf '  new model training: false\n'
  printf '  broad sweep: false\n'
  printf '  production signal export: false\n'
  printf '  model card: false\n'
  printf '  tradable holdings: QQQ/cash only\n'
  printf '  generated artifacts remain uncommitted\n\n'

  printf 'Equivalent Python command:\n'
  printf '  %q' "${cmd[@]}"
  printf '\n\n'

  "${cmd[@]}"
}

main "$@"
