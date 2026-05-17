#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_NAME="${RUN_NAME:-expanded_dynamic_candidate_confirmation_study}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/sweeps/expanded_dynamic_candidate_confirmation_study}"
MARKET_DATASET_GLOB="${MARKET_DATASET_GLOB:-/Users/casper/Projects/Research/Market Data/data/research/etf_daily_prices_extended_tiingo_split/**/*.parquet}"
MAX_WORKERS="${MAX_WORKERS:-4}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
MIN_HISTORY_ROWS="${MIN_HISTORY_ROWS:-252}"
MIN_TRAIN_ELIGIBLE_ROWS="${MIN_TRAIN_ELIGIBLE_ROWS:-5000}"
STEP_MONTHS="${STEP_MONTHS:-6}"

RESULTS_CSV="$OUTPUT_DIR/${RUN_NAME}_results.csv"
COMPLETED_KEYS="$OUTPUT_DIR/${RUN_NAME}_completed_keys.txt"
FAILED_KEYS="$OUTPUT_DIR/${RUN_NAME}_failed_keys.txt"
SKIPPED_KEYS="$OUTPUT_DIR/${RUN_NAME}_skipped_keys.txt"
TERMINAL_LOG="$OUTPUT_DIR/${RUN_NAME}_terminal.log"
PREFLIGHT_CSV="$OUTPUT_DIR/expanded_dynamic_candidate_confirmation_preflight.csv"
PREFLIGHT_MD="$OUTPUT_DIR/expanded_dynamic_candidate_confirmation_preflight.md"

mkdir -p "$OUTPUT_DIR"

cmd=(
  python -m us_market_ml.cli.run_expanded_dynamic_candidate_confirmation_study
  --run-name "$RUN_NAME"
  --output-dir "$OUTPUT_DIR"
  --market-dataset-glob "$MARKET_DATASET_GLOB"
  --max-workers "$MAX_WORKERS"
  --min-history-rows "$MIN_HISTORY_ROWS"
  --min-train-eligible-rows "$MIN_TRAIN_ELIGIBLE_ROWS"
  --step-months "$STEP_MONTHS"
)

if [[ "$RESUME" == "0" ]]; then
  cmd+=(--no-resume)
fi

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

print_progress() {
  printf '[%s] completed=%s skipped=%s failed=%s result_rows=%s log=%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" \
    "$(count_non_empty_lines "$COMPLETED_KEYS")" \
    "$(count_non_empty_lines "$SKIPPED_KEYS")" \
    "$(count_non_empty_lines "$FAILED_KEYS")" \
    "$(result_row_count)" \
    "$TERMINAL_LOG"
}

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'DRY_RUN=1: writing expanded dynamic candidate confirmation preflight only.\n'
  printf 'Preflight CSV: %s\n' "$PREFLIGHT_CSV"
  printf 'Preflight Markdown: %s\n' "$PREFLIGHT_MD"
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

printf 'Preflight CSV: %s\n' "$PREFLIGHT_CSV"
printf 'Preflight Markdown: %s\n' "$PREFLIGHT_MD"
printf 'Skipped checkpoint: %s\n' "$SKIPPED_KEYS"

"${cmd[@]}" >> "$TERMINAL_LOG" 2>&1 &
child_pid=$!

terminate_child() {
  printf '\nInterrupt received; forwarding SIGINT to study process %s.\n' "$child_pid"
  kill -INT "$child_pid" 2>/dev/null || true
  wait "$child_pid" 2>/dev/null || true
  print_progress
  printf 'Resume later with:\n'
  printf '  env RESUME=1 MAX_WORKERS=%q MIN_TRAIN_ELIGIBLE_ROWS=%q bash scripts/run_expanded_dynamic_candidate_confirmation_study.sh\n' "$MAX_WORKERS" "$MIN_TRAIN_ELIGIBLE_ROWS"
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

printf '\nExpanded dynamic candidate confirmation study complete.\n'
printf 'Useful output commands:\n'
printf '  tail -n 80 %q\n' "$TERMINAL_LOG"
printf '  sed -n %q %q\n' '1,220p' "$PREFLIGHT_MD"
printf '  cat %q\n' "$SKIPPED_KEYS"
printf '  find %q -maxdepth 1 -type f | sort\n' "$OUTPUT_DIR"
