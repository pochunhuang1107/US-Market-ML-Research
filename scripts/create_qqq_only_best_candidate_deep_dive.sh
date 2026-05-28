#!/usr/bin/env bash
set -euo pipefail

main() {
  local root_dir
  root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  cd "$root_dir"

  MODEL_ZOO_RESULTS="${MODEL_ZOO_RESULTS:-artifacts/sweeps/qqq_only_cross_market_timing_model_zoo/qqq_only_cross_market_timing_model_zoo_results.csv}"
  MODEL_ZOO_DAILY_EQUITY="${MODEL_ZOO_DAILY_EQUITY:-artifacts/sweeps/qqq_only_cross_market_timing_model_zoo/qqq_only_cross_market_timing_model_zoo_daily_equity.csv}"
  MODEL_ZOO_DAILY_DRAWDOWN="${MODEL_ZOO_DAILY_DRAWDOWN:-artifacts/sweeps/qqq_only_cross_market_timing_model_zoo/qqq_only_cross_market_timing_model_zoo_daily_drawdown.csv}"
  MODEL_ZOO_EXPOSURE_LOG="${MODEL_ZOO_EXPOSURE_LOG:-artifacts/sweeps/qqq_only_cross_market_timing_model_zoo/qqq_only_cross_market_timing_model_zoo_exposure_log.csv}"
  MODEL_ZOO_SUMMARY="${MODEL_ZOO_SUMMARY:-artifacts/reports/qqq_only_cross_market_timing_model_zoo/qqq_only_cross_market_timing_model_zoo_summary.csv}"
  MODEL_ZOO_DECISION_MATRIX="${MODEL_ZOO_DECISION_MATRIX:-artifacts/reports/qqq_only_cross_market_timing_model_zoo/qqq_only_cross_market_timing_model_zoo_decision_matrix.csv}"
  OUTPUT_DIR="${OUTPUT_DIR:-artifacts/reports/qqq_only_best_candidate_deep_dive}"
  TOP_N_CANDIDATES="${TOP_N_CANDIDATES:-10}"
  DRY_RUN="${DRY_RUN:-0}"

  local cmd=(
    python -m us_market_ml.cli.create_qqq_only_best_candidate_deep_dive
    --model-zoo-results "$MODEL_ZOO_RESULTS"
    --model-zoo-daily-equity "$MODEL_ZOO_DAILY_EQUITY"
    --model-zoo-daily-drawdown "$MODEL_ZOO_DAILY_DRAWDOWN"
    --model-zoo-exposure-log "$MODEL_ZOO_EXPOSURE_LOG"
    --model-zoo-summary "$MODEL_ZOO_SUMMARY"
    --model-zoo-decision-matrix "$MODEL_ZOO_DECISION_MATRIX"
    --output-dir "$OUTPUT_DIR"
    --top-n-candidates "$TOP_N_CANDIDATES"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    cmd+=(--dry-run)
  fi

  printf 'QQQ-only best-candidate deep dive guardrails:\n'
  printf '  local-only: true\n'
  printf '  reads existing artifacts only: true\n'
  printf '  S3/cloud: false\n'
  printf '  new model training: false\n'
  printf '  broad sweep: false\n'
  printf '  production signal export: false\n'
  printf '  model card: false\n'
  printf '  generated artifacts remain uncommitted\n\n'

  printf 'Equivalent Python command:\n'
  printf '  %q' "${cmd[@]}"
  printf '\n\n'

  "${cmd[@]}"
}

main "$@"
