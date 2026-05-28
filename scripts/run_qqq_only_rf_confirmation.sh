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
elif expression == "baseline":
    print(int(df["is_baseline"].sum()))
elif expression == "model":
    print(int((~df["is_baseline"]).sum()))
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

  MARKET_DATASET_GLOB="${MARKET_DATASET_GLOB:-/Users/casper/Projects/Research/Market Data/data/research/etf_daily_prices_tiingo_only/**/*.parquet}"
  REFERENCE_SERIES_GLOB="${REFERENCE_SERIES_GLOB:-/Users/casper/Projects/Research/Market Data/data/research/reference_series/**/*.parquet}"
  OUTPUT_DIR="${OUTPUT_DIR:-artifacts/sweeps/qqq_only_rf_confirmation}"
  REPORT_OUTPUT_DIR="${REPORT_OUTPUT_DIR:-artifacts/reports/qqq_only_rf_confirmation}"
  START_DATE="${START_DATE:-2019-01-01}"
  END_DATE="${END_DATE:-2026-05-15}"
  MAX_WORKERS="${MAX_WORKERS:-2}"
  RESUME="${RESUME:-1}"
  DRY_RUN="${DRY_RUN:-0}"
  MIN_TRAIN_ROWS="${MIN_TRAIN_ROWS:-252}"

  local results_csv daily_equity daily_drawdown exposure_log feature_summary
  local completed_keys failed_keys skipped_keys terminal_log preflight_csv preflight_md
  results_csv="$OUTPUT_DIR/qqq_only_rf_confirmation_results.csv"
  daily_equity="$OUTPUT_DIR/qqq_only_rf_confirmation_daily_equity.csv"
  daily_drawdown="$OUTPUT_DIR/qqq_only_rf_confirmation_daily_drawdown.csv"
  exposure_log="$OUTPUT_DIR/qqq_only_rf_confirmation_exposure_log.csv"
  feature_summary="$OUTPUT_DIR/qqq_only_rf_confirmation_feature_summary.csv"
  completed_keys="$OUTPUT_DIR/qqq_only_rf_confirmation_completed_keys.txt"
  failed_keys="$OUTPUT_DIR/qqq_only_rf_confirmation_failed_keys.txt"
  skipped_keys="$OUTPUT_DIR/qqq_only_rf_confirmation_skipped_keys.txt"
  terminal_log="$OUTPUT_DIR/qqq_only_rf_confirmation_terminal.log"
  preflight_csv="$OUTPUT_DIR/qqq_only_rf_confirmation_preflight.csv"
  preflight_md="$OUTPUT_DIR/qqq_only_rf_confirmation_preflight.md"

  mkdir -p "$OUTPUT_DIR" "$REPORT_OUTPUT_DIR"

  local cmd=(
    python -m us_market_ml.cli.run_qqq_only_rf_confirmation
    --market-dataset-glob "$MARKET_DATASET_GLOB"
    --reference-series-glob "$REFERENCE_SERIES_GLOB"
    --output-dir "$OUTPUT_DIR"
    --report-output-dir "$REPORT_OUTPUT_DIR"
    --start-date "$START_DATE"
    --end-date "$END_DATE"
    --max-workers "$MAX_WORKERS"
    --min-train-rows "$MIN_TRAIN_ROWS"
  )
  if [[ "$RESUME" == "0" ]]; then
    cmd+=(--no-resume)
  else
    cmd+=(--resume)
  fi

  printf 'QQQ-only RF confirmation guardrails:\n'
  printf '  local-only: true\n'
  printf '  focused confirmation: true\n'
  printf '  broad model zoo: false\n'
  printf '  tradable holdings: QQQ,cash\n'
  printf '  S3/cloud: false\n'
  printf '  production signal export: false\n'
  printf '  model card: false\n'
  printf '  generated artifacts remain uncommitted\n\n'

  printf 'Equivalent Python command:\n'
  printf '  %q' "${cmd[@]}"
  printf '\n\n'

  {
    printf '\n==== %s QQQ-only RF confirmation preflight ====\n' "$(date '+%Y-%m-%d %H:%M:%S')"
    printf 'Command:'
    printf ' %q' "${cmd[@]}" --dry-run
    printf '\n'
  } >> "$terminal_log"

  printf 'Running preflight before dispatching work...\n'
  "${cmd[@]}" --dry-run 2>&1 | tee -a "$terminal_log"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '\nDRY_RUN=1: preflight complete; no RF confirmation run started.\n'
    exit 0
  fi

  local total_configs baseline_configs model_configs started_at
  total_configs="$(preflight_count "$preflight_csv" total)"
  baseline_configs="$(preflight_count "$preflight_csv" baseline)"
  model_configs="$(preflight_count "$preflight_csv" model)"
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
    printf '[%s] completed %s / pending %s / total %s; baselines=%s models=%s skipped=%s failed=%s result_rows=%s elapsed=%s log=%s\n' \
      "$(date '+%Y-%m-%d %H:%M:%S')" \
      "$completed" \
      "$pending" \
      "$total_configs" \
      "$baseline_configs" \
      "$model_configs" \
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
    printf '\n==== %s starting QQQ-only RF confirmation ====\n' "$(date '+%Y-%m-%d %H:%M:%S')"
    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    printf 'Preflight CSV: %s\n' "$preflight_csv"
    printf 'Preflight Markdown: %s\n' "$preflight_md"
    printf 'Total configs: %s\n' "$total_configs"
    printf 'Baseline configs: %s\n' "$baseline_configs"
    printf 'Model configs: %s\n' "$model_configs"
  } >> "$terminal_log"

  printf '\nStarting QQQ-only RF confirmation. Progress prints every 60 seconds.\n'
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

  printf '\nQQQ-only RF confirmation complete.\n'
  printf 'Useful output commands:\n'
  printf '  tail -n 120 %q\n' "$terminal_log"
  printf '  sed -n %q %q\n' '1,260p' "$preflight_md"
  printf '  find %q -maxdepth 1 -type f | sort\n' "$OUTPUT_DIR"
  printf '  find %q -maxdepth 1 -type f | sort\n' "$REPORT_OUTPUT_DIR"
  printf 'Core sweep files include:\n'
  printf '  %s\n  %s\n  %s\n  %s\n  %s\n' \
    "$results_csv" "$daily_equity" "$daily_drawdown" "$exposure_log" "$feature_summary"
}

if [[ "${QQQ_ONLY_RF_CONFIRMATION_TEST_HELPERS:-0}" != "1" ]]; then
  main "$@"
fi
