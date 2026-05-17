#!/usr/bin/env bash
set -euo pipefail

count_non_empty_lines() {
  local path="$1"
  if [[ -f "$path" ]]; then
    grep -cve '^[[:space:]]*$' "$path" || true
  else
    printf '0'
  fi
}

result_row_count() {
  local path="$1"
  if [[ -f "$path" ]]; then
    local lines
    lines="$(wc -l < "$path" | tr -d ' ')"
    if [[ "$lines" -gt 0 ]]; then
      printf '%s' "$((lines - 1))"
    else
      printf '0'
    fi
  else
    printf '0'
  fi
}

collect_descendants() {
  local pid="$1"
  local children child
  children="$(pgrep -P "$pid" 2>/dev/null || true)"
  if [[ -z "$children" ]]; then
    children="$(ps -axo pid=,ppid= | awk -v parent="$pid" '$2 == parent {print $1}')"
  fi
  for child in $children; do
    printf '%s\n' "$child"
    collect_descendants "$child"
  done
}

terminate_process_tree() {
  local pid="$1"
  local signal_name="$2"
  local descendants child
  descendants="$(collect_descendants "$pid" | sort -u || true)"
  for child in $descendants; do
    kill "-$signal_name" "$child" 2>/dev/null || true
  done
  kill "-$signal_name" "$pid" 2>/dev/null || true
}

wait_with_timeout() {
  local pid="$1"
  local timeout_seconds="$2"
  local start_time now
  start_time="$(date +%s)"
  while kill -0 "$pid" 2>/dev/null; do
    now="$(date +%s)"
    if [[ "$((now - start_time))" -ge "$timeout_seconds" ]]; then
      return 1
    fi
    sleep 1
  done
  wait "$pid" 2>/dev/null || true
  return 0
}

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

preflight_count() {
  local path="$1"
  local expression="$2"
  python - "$path" "$expression" <<'PY'
import sys
import pandas as pd

path, expression = sys.argv[1], sys.argv[2]
df = pd.read_csv(path)
if expression == "total":
    print(len(df))
elif expression == "valid":
    print(int(df["valid_config"].sum()))
elif expression == "skipped":
    print(int((~df["valid_config"]).sum()))
else:
    raise SystemExit(f"unknown expression: {expression}")
PY
}

format_elapsed() {
  local seconds="$1"
  printf '%02d:%02d:%02d' "$((seconds / 3600))" "$(((seconds % 3600) / 60))" "$((seconds % 60))"
}

main() {
  local root_dir
  root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  cd "$root_dir"

  RUN_NAME="${RUN_NAME:-tiingo_modern_baseline_sweep}"
  OUTPUT_DIR="${OUTPUT_DIR:-artifacts/sweeps/tiingo_modern_baseline_sweep}"
  MARKET_DATASET_GLOB="${MARKET_DATASET_GLOB:-/Users/casper/Projects/Research/Market Data/data/research/etf_daily_prices_tiingo_only/**/*.parquet}"
  START_DATE="${START_DATE:-2019-01-01}"
  END_DATE="${END_DATE:-2026-05-15}"
  MAX_WORKERS="${MAX_WORKERS:-4}"
  RESUME="${RESUME:-1}"
  DRY_RUN="${DRY_RUN:-0}"
  MIN_HISTORY_ROWS="${MIN_HISTORY_ROWS:-252}"
  MIN_TRAIN_ELIGIBLE_ROWS="${MIN_TRAIN_ELIGIBLE_ROWS:-5000}"
  STEP_MONTHS="${STEP_MONTHS:-3}"
  TRAIN_WINDOW_YEARS="${TRAIN_WINDOW_YEARS:-2 3 5}"
  VALIDATION_WINDOW_MONTHS="${VALIDATION_WINDOW_MONTHS:-6 12}"
  TEST_WINDOW_MONTHS="${TEST_WINDOW_MONTHS:-3 6 12}"

  local results_csv completed_keys failed_keys skipped_keys terminal_log preflight_csv preflight_md
  results_csv="$OUTPUT_DIR/${RUN_NAME}_results.csv"
  completed_keys="$OUTPUT_DIR/${RUN_NAME}_completed_keys.txt"
  failed_keys="$OUTPUT_DIR/${RUN_NAME}_failed_keys.txt"
  skipped_keys="$OUTPUT_DIR/${RUN_NAME}_skipped_keys.txt"
  terminal_log="$OUTPUT_DIR/${RUN_NAME}_terminal.log"
  preflight_csv="$OUTPUT_DIR/tiingo_modern_baseline_sweep_preflight.csv"
  preflight_md="$OUTPUT_DIR/tiingo_modern_baseline_sweep_preflight.md"

  mkdir -p "$OUTPUT_DIR"

  local cmd=(
    python -m us_market_ml.cli.run_tiingo_modern_baseline_sweep
    --market-dataset-glob "$MARKET_DATASET_GLOB"
    --output-dir "$OUTPUT_DIR"
    --run-name "$RUN_NAME"
    --start-date "$START_DATE"
    --end-date "$END_DATE"
    --max-workers "$MAX_WORKERS"
    --min-history-rows "$MIN_HISTORY_ROWS"
    --min-train-eligible-rows "$MIN_TRAIN_ELIGIBLE_ROWS"
    --step-months "$STEP_MONTHS"
  )
  append_values_arg --train-window-years "$TRAIN_WINDOW_YEARS"
  append_values_arg --validation-window-months "$VALIDATION_WINDOW_MONTHS"
  append_values_arg --test-window-months "$TEST_WINDOW_MONTHS"

  if [[ "$RESUME" == "0" ]]; then
    cmd+=(--no-resume)
  else
    cmd+=(--resume)
  fi

  printf 'Equivalent Python command:\n'
  printf '  %q' "${cmd[@]}"
  printf '\n\n'

  {
    printf '\n==== %s preflight %s ====\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$RUN_NAME"
    printf 'Command:'
    printf ' %q' "${cmd[@]}" --dry-run
    printf '\n'
  } >> "$terminal_log"

  printf 'Running preflight before dispatching work...\n'
  "${cmd[@]}" --dry-run 2>&1 | tee -a "$terminal_log"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '\nDRY_RUN=1: preflight complete; no long run started.\n'
    exit 0
  fi

  local total_configs valid_configs skipped_configs started_at
  total_configs="$(preflight_count "$preflight_csv" total)"
  valid_configs="$(preflight_count "$preflight_csv" valid)"
  skipped_configs="$(preflight_count "$preflight_csv" skipped)"
  started_at="$(date +%s)"

  print_progress() {
    local completed failed skipped result_rows now elapsed pending
    completed="$(count_non_empty_lines "$completed_keys")"
    failed="$(count_non_empty_lines "$failed_keys")"
    skipped="$(count_non_empty_lines "$skipped_keys")"
    result_rows="$(result_row_count "$results_csv")"
    now="$(date +%s)"
    elapsed="$((now - started_at))"
    pending="$((total_configs - completed - skipped - failed))"
    if [[ "$pending" -lt 0 ]]; then
      pending=0
    fi
    printf '[%s] completed %s / pending %s / total %s; skipped=%s failed=%s result_rows=%s elapsed=%s log=%s\n' \
      "$(date '+%Y-%m-%d %H:%M:%S')" \
      "$completed" \
      "$pending" \
      "$total_configs" \
      "$skipped" \
      "$failed" \
      "$result_rows" \
      "$(format_elapsed "$elapsed")" \
      "$terminal_log" | tee -a "$terminal_log"
  }

  local study_pid=""
  local shutting_down=0
  cleanup_on_interrupt() {
    local signal_name="${1:-INT}"
    if [[ "$shutting_down" == "1" ]]; then
      return
    fi
    shutting_down=1
    printf '\nInterrupt received; forwarding SIGINT to study process %s ...\n' "$study_pid" | tee -a "$terminal_log"
    if [[ -n "$study_pid" ]] && kill -0 "$study_pid" 2>/dev/null; then
      terminate_process_tree "$study_pid" INT
      if wait_with_timeout "$study_pid" 15; then
        printf 'Study process exited after interrupt.\n' | tee -a "$terminal_log"
        print_progress
        exit 130
      fi
      printf 'Study process did not exit after timeout; sending SIGTERM.\n' | tee -a "$terminal_log"
      terminate_process_tree "$study_pid" TERM
      if wait_with_timeout "$study_pid" 10; then
        printf 'Study process exited after SIGTERM.\n' | tee -a "$terminal_log"
        print_progress
        exit 130
      fi
      printf 'Study process did not exit after timeout; sending SIGKILL.\n' | tee -a "$terminal_log"
      terminate_process_tree "$study_pid" KILL
      wait_with_timeout "$study_pid" 5 || true
    fi
    print_progress
    if [[ "$signal_name" == "TERM" ]]; then
      exit 143
    fi
    exit 130
  }

  trap 'cleanup_on_interrupt INT' INT
  trap 'cleanup_on_interrupt TERM' TERM

  {
    printf '\n==== %s starting %s ====\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$RUN_NAME"
    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    printf 'Preflight CSV: %s\n' "$preflight_csv"
    printf 'Preflight Markdown: %s\n' "$preflight_md"
    printf 'Total configs: %s\n' "$total_configs"
    printf 'Valid configs: %s\n' "$valid_configs"
    printf 'Skipped configs: %s\n' "$skipped_configs"
  } >> "$terminal_log"

  printf '\nStarting Tiingo modern baseline sweep. Progress prints every 60 seconds.\n'
  printf 'Terminal/progress log: %s\n' "$terminal_log"

  "${cmd[@]}" >> "$terminal_log" 2>&1 &
  study_pid=$!

  print_progress
  while kill -0 "$study_pid" 2>/dev/null; do
    sleep 60
    if kill -0 "$study_pid" 2>/dev/null; then
      print_progress
    fi
  done

  local exit_code
  if wait "$study_pid"; then
    exit_code=0
  else
    exit_code=$?
  fi
  trap - INT TERM

  print_progress
  if [[ "$exit_code" -ne 0 ]]; then
    printf 'Study failed with exit code %s. Tail the log with:\n' "$exit_code"
    printf '  tail -n 120 %q\n' "$terminal_log"
    exit "$exit_code"
  fi

  printf '\nTiingo modern baseline sweep complete.\n'
  printf 'Useful output commands:\n'
  printf '  tail -n 120 %q\n' "$terminal_log"
  printf '  sed -n %q %q\n' '1,260p' "$preflight_md"
  printf '  find %q -maxdepth 1 -type f | sort\n' "$OUTPUT_DIR"
}

if [[ "${TIINGO_BASELINE_SWEEP_TEST_HELPERS:-0}" != "1" ]]; then
  main "$@"
fi
