#!/usr/bin/env bash
set -u

OUT_DIR="artifacts/sweeps"

TREE_RUN="tabular_tree_purged_comparison"
MLP_RUN="mlp_refinement_purged_scaled"

TREE_TOTAL=4000
MLP_TOTAL=13824

mkdir -p "$OUT_DIR"

ensure_no_sweep_running() {
  if pgrep -fl "us-market-ml-sweep" >/dev/null; then
    echo "Another us-market-ml-sweep process is already running:"
    pgrep -fl "us-market-ml-sweep"
    echo "Stop it first with: pkill -f 'us-market-ml-sweep'"
    exit 1
  fi
}

progress_monitor() {
  local run_name="$1"
  local total="$2"
  local label="$3"

  while true; do
    DONE=$(wc -l < "$OUT_DIR/${run_name}_completed_keys.txt" 2>/dev/null | tr -d ' ')
    FAILED=$(wc -l < "$OUT_DIR/${run_name}_failed_keys.txt" 2>/dev/null | tr -d ' ')
    DONE=${DONE:-0}
    FAILED=${FAILED:-0}
    REMAINING=$((total - DONE))

    echo "[$label] [$DONE/$total] completed | failed=$FAILED | remaining=$REMAINING | $(date)"
    sleep 60
  done
}

run_tree_sweep() {
  echo ""
  echo "=================================================="
  echo "Starting TREE sweep at $(date)"
  echo "Run name: $TREE_RUN"
  echo "=================================================="

  progress_monitor "$TREE_RUN" "$TREE_TOTAL" "TREE" &
  MONITOR_PID=$!

  us-market-ml-sweep \
    --run-name "$TREE_RUN" \
    --output-dir "$OUT_DIR" \
    --universe core:artifacts/cache/universe_core_2018_2026.parquet \
    --universe broad_macro:artifacts/cache/universe_broad_macro_2018_2026.parquet \
    --universe sector:artifacts/cache/universe_sector_2018_2026.parquet \
    --universe style_growth:artifacts/cache/universe_style_growth_2018_2026.parquet \
    --universe full:artifacts/cache/universe_full_2018_2026.parquet \
    --fold fold_2022:2019-01-01:2020-12-31:2021-01-01:2021-12-31:2022-01-01:2022-12-31 \
    --fold fold_2023:2019-01-01:2021-12-31:2022-01-01:2022-12-31:2023-01-01:2023-12-31 \
    --fold fold_2024:2019-01-01:2022-12-31:2023-01-01:2023-12-31:2024-01-01:2024-12-31 \
    --fold fold_2025_2026:2019-01-01:2023-12-31:2024-01-01:2024-12-31:2025-01-01:2026-05-08 \
    --model-family baseline \
    --baseline-models ridge,linear_regression,random_forest,gradient_boosting,extra_trees \
    --top-k 1,2,3,5 \
    --seeds 42,7,11,21,99,123,2024,2025,314,777 \
    --label-horizon-days 20

  EXIT_CODE=$?
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true

  echo "TREE sweep finished at $(date) with exit code $EXIT_CODE"
  return "$EXIT_CODE"
}

run_mlp_sweep() {
  echo ""
  echo "=================================================="
  echo "Starting MLP refinement sweep at $(date)"
  echo "Run name: $MLP_RUN"
  echo "=================================================="

  progress_monitor "$MLP_RUN" "$MLP_TOTAL" "MLP" &
  MONITOR_PID=$!

  us-market-ml-sweep \
    --run-name "$MLP_RUN" \
    --output-dir "$OUT_DIR" \
    --universe broad_macro:artifacts/cache/universe_broad_macro_2018_2026.parquet \
    --universe core:artifacts/cache/universe_core_2018_2026.parquet \
    --universe full:artifacts/cache/universe_full_2018_2026.parquet \
    --fold fold_2022:2019-01-01:2020-12-31:2021-01-01:2021-12-31:2022-01-01:2022-12-31 \
    --fold fold_2023:2019-01-01:2021-12-31:2022-01-01:2022-12-31:2023-01-01:2023-12-31 \
    --fold fold_2024:2019-01-01:2022-12-31:2023-01-01:2023-12-31:2024-01-01:2024-12-31 \
    --fold fold_2025_2026:2019-01-01:2023-12-31:2024-01-01:2024-12-31:2025-01-01:2026-05-08 \
    --model-family mlp \
    --top-k 1,2,3,5 \
    --hidden-units 64 \
    --hidden-units 128,64 \
    --hidden-units 256,128 \
    --hidden-units 128,64,32 \
    --learning-rates 0.001,0.0005,0.0002 \
    --dropouts 0.1,0.2,0.3 \
    --seeds 42,7,11,21,99,123,2024,2025 \
    --max-epochs 120 \
    --patience 12 \
    --batch-size 128 \
    --label-horizon-days 20 \
    --scale-target

  EXIT_CODE=$?
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true

  echo "MLP refinement sweep finished at $(date) with exit code $EXIT_CODE"
  return "$EXIT_CODE"
}

cleanup() {
  echo ""
  echo "Stopping overnight sweep..."
  pkill -f "us-market-ml-sweep" 2>/dev/null || true
  exit 130
}

trap cleanup INT TERM

ensure_no_sweep_running

run_tree_sweep
TREE_EXIT=$?

if [[ "$TREE_EXIT" -ne 0 ]]; then
  echo "TREE sweep failed or was interrupted. Not starting MLP sweep."
  exit "$TREE_EXIT"
fi

run_mlp_sweep
MLP_EXIT=$?

echo ""
echo "=================================================="
echo "Overnight run finished at $(date)"
echo "TREE exit code: $TREE_EXIT"
echo "MLP exit code: $MLP_EXIT"
echo "=================================================="

exit "$MLP_EXIT"
