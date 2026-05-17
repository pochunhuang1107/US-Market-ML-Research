#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_NAME="${RUN_NAME:-extended_history_overnight_study}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/sweeps/extended_history_overnight_study}"
MARKET_DATASET_GLOB="${MARKET_DATASET_GLOB:-/Users/casper/Projects/Research/Market Data/data/research/etf_daily_prices_extended_tiingo_split/**/*.parquet}"
MAX_WORKERS="${MAX_WORKERS:-4}"
DRY_RUN="${DRY_RUN:-0}"

RESULTS_CSV="$OUTPUT_DIR/${RUN_NAME}_results.csv"
COMPLETED_KEYS="$OUTPUT_DIR/${RUN_NAME}_completed_keys.txt"
FAILED_KEYS="$OUTPUT_DIR/${RUN_NAME}_failed_keys.txt"
SKIPPED_KEYS="$OUTPUT_DIR/${RUN_NAME}_skipped_keys.txt"
TOTAL_CONFIGS_FILE="$OUTPUT_DIR/${RUN_NAME}_total_configs.txt"
TERMINAL_LOG="$OUTPUT_DIR/${RUN_NAME}_terminal.log"
RANK_CSV="$OUTPUT_DIR/extended_history_candidate_rank.csv"
REPORT_MD="$OUTPUT_DIR/extended_history_report.md"
INVALID_FOLD_REPORT="$OUTPUT_DIR/extended_history_invalid_fold_report.csv"

mkdir -p "$OUTPUT_DIR"

cmd=(
  python -m us_market_ml.cli.run_extended_history_overnight_study
  --run-name "$RUN_NAME"
  --output-dir "$OUTPUT_DIR"
  --market-dataset-glob "$MARKET_DATASET_GLOB"
  --max-workers "$MAX_WORKERS"
)

count_non_empty_lines() {
  local path="$1"
  if [[ -f "$path" ]]; then
    grep -cve '^[[:space:]]*$' "$path" || true
  else
    printf '0'
  fi
}

result_row_count() {
  if [[ -f "$RESULTS_CSV" ]]; then
    local lines
    lines="$(wc -l < "$RESULTS_CSV" | tr -d ' ')"
    if [[ "$lines" -gt 0 ]]; then
      printf '%s' "$((lines - 1))"
    else
      printf '0'
    fi
  else
    printf '0'
  fi
}

total_config_count() {
  if [[ -f "$TOTAL_CONFIGS_FILE" ]]; then
    tr -d '[:space:]' < "$TOTAL_CONFIGS_FILE"
  else
    printf '?'
  fi
}

print_progress() {
  printf '[%s] completed=%s skipped=%s failed=%s result_rows=%s total=%s log=%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" \
    "$(count_non_empty_lines "$COMPLETED_KEYS")" \
    "$(count_non_empty_lines "$SKIPPED_KEYS")" \
    "$(count_non_empty_lines "$FAILED_KEYS")" \
    "$(result_row_count)" \
    "$(total_config_count)" \
    "$TERMINAL_LOG"
}

python - <<'PY' > "$TOTAL_CONFIGS_FILE"
from pathlib import Path
from us_market_ml.cli.run_extended_history_overnight_study import (
    DEFAULT_DATA_END,
    DEFAULT_DATA_START,
    build_extended_history_folds,
    build_extended_history_sweep_config,
)

folds = build_extended_history_folds(DEFAULT_DATA_START, DEFAULT_DATA_END)
config = build_extended_history_sweep_config(
    market_dataset_glob="unused",
    folds=[item.fold for item in folds],
    output_dir=Path("unused"),
    run_name="unused",
    resume=True,
    max_workers=1,
)
print(
    len(config.universes)
    * len(config.folds)
    * len(config.models)
    * len(config.signals)
    * len(config.feature_set_versions)
    * len(config.transaction_cost_bps_values)
)
PY

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'DRY_RUN=1: printing extended-history study plan only.\n'
  printf 'Skipped checkpoint: %s\n' "$SKIPPED_KEYS"
  printf 'Invalid fold report: %s\n' "$INVALID_FOLD_REPORT"
  printf 'Command:\n'
  printf '  %q' "${cmd[@]}" --dry-run
  printf '\n\n'
  "${cmd[@]}" --dry-run
  exit 0
fi

{
  printf '\n==== %s starting %s ====\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$RUN_NAME"
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
} >> "$TERMINAL_LOG"

printf 'Skipped checkpoint: %s\n' "$SKIPPED_KEYS"
printf 'Invalid fold report: %s\n' "$INVALID_FOLD_REPORT"

"${cmd[@]}" >> "$TERMINAL_LOG" 2>&1 &
child_pid=$!

terminate_child() {
  printf '\nInterrupt received; forwarding SIGINT to study process %s.\n' "$child_pid"
  kill -INT "$child_pid" 2>/dev/null || true
  wait "$child_pid" 2>/dev/null || true
  print_progress
  printf 'Resume later with:\n'
  printf '  caffeinate -is env MAX_WORKERS=%q bash scripts/run_extended_history_overnight_study.sh\n' "$MAX_WORKERS"
  exit 130
}

trap terminate_child INT TERM

print_progress
while kill -0 "$child_pid" 2>/dev/null; do
  sleep 60
  if kill -0 "$child_pid" 2>/dev/null; then
    print_progress
  fi
done

if wait "$child_pid"; then
  exit_code=0
else
  exit_code=$?
fi
trap - INT TERM

print_progress
if [[ "$exit_code" -ne 0 ]]; then
  printf 'Study failed with exit code %s. Tail the log with:\n' "$exit_code"
  printf '  tail -n 80 %q\n' "$TERMINAL_LOG"
  exit "$exit_code"
fi

printf '\nExtended-history study complete.\n'
printf 'Useful output commands:\n'
printf '  tail -n 80 %q\n' "$TERMINAL_LOG"
printf '  cat %q\n' "$RANK_CSV"
printf '  sed -n %q %q\n' '1,220p' "$REPORT_MD"
printf '  cat %q\n' "$SKIPPED_KEYS"
printf '  cat %q\n' "$INVALID_FOLD_REPORT"
printf '  find %q -maxdepth 1 -type f | sort\n' "$OUTPUT_DIR"
