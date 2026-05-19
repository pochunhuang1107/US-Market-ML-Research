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
elif expression == "invalid":
    print(int((~df["valid_config"]).sum()))
else:
    raise SystemExit(f"unknown expression: {expression}")
PY
}

print_preflight_summary() {
  local preflight_csv="$1"
  local completed_keys="$2"
  local failed_keys="$3"
  local resume_flag="$4"
  local max_workers="$5"
  local dataset_glob="$6"
  local output_dir="$7"
  local results_csv="$8"
  local terminal_log="$9"
  python - "$preflight_csv" "$completed_keys" "$failed_keys" "$resume_flag" \
    "$max_workers" "$dataset_glob" "$output_dir" "$results_csv" "$terminal_log" <<'PY'
import sys
from pathlib import Path
import pandas as pd

(
    preflight_csv,
    completed_keys,
    failed_keys,
    resume_flag,
    max_workers,
    dataset_glob,
    output_dir,
    results_csv,
    terminal_log,
) = sys.argv[1:]
df = pd.read_csv(preflight_csv)
valid = df.loc[df["valid_config"], "run_key"].astype(str)
valid_keys = set(valid)

def read_lines(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    return [line.strip().split(",", maxsplit=1)[0] for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

completed = set(read_lines(completed_keys)) if resume_flag != "0" else set()
completed_in_plan = len(valid_keys.intersection(completed))
failed_count = len(read_lines(failed_keys)) if resume_flag != "0" else 0
pending = max(0, len(valid_keys) - completed_in_plan)
print("Run-count summary before starting:")
print(f"  total allocation evaluations: {len(df)}")
print(f"  valid allocation evaluations: {int(df['valid_config'].sum())}")
print(f"  invalid allocation evaluations: {int((~df['valid_config']).sum())}")
print(f"  generated fold count: {df['fold_name'].nunique()}")
print(f"  allocation rule count: {df['allocation_rule_name'].nunique()}")
print(f"  already completed evaluations if RESUME=1: {completed_in_plan}")
print(f"  failed checkpoint count if RESUME=1: {failed_count}")
print(f"  pending allocation evaluations for this batch: {pending}")
print(f"  allocation rule names: {', '.join(df['allocation_rule_name'].dropna().drop_duplicates().astype(str).tolist())}")
print(f"  feature set versions: {sorted(df['feature_set_version'].dropna().astype(str).unique())}")
print(f"  model types: {', '.join(sorted(df['model_type'].dropna().astype(str).unique()))}")
print(f"  top_k values: {sorted(pd.to_numeric(df['top_k']).dropna().astype(int).unique().tolist())}")
print(f"  transaction cost values: {sorted(pd.to_numeric(df['transaction_cost_bps']).dropna().astype(float).unique().tolist())}")
print(f"  max workers: {max_workers}")
print(f"  dataset glob: {dataset_glob}")
print("  dataset source label: tiingo_only")
print(f"  output directory: {output_dir}")
print(f"  results CSV path: {results_csv}")
print(f"  completed checkpoint path: {completed_keys}")
print(f"  failed checkpoint path: {failed_keys}")
print(f"  terminal/progress log path: {terminal_log}")
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
  OUTPUT_DIR="${OUTPUT_DIR:-artifacts/sweeps/tiingo_v3_pure_weighted_allocation_study}"
  RUN_NAME="${RUN_NAME:-tiingo_v3_pure_weighted_allocation_study}"
  START_DATE="${START_DATE:-2019-01-01}"
  END_DATE="${END_DATE:-2026-05-15}"
  MAX_WORKERS="${MAX_WORKERS:-4}"
  RESUME="${RESUME:-1}"
  DRY_RUN="${DRY_RUN:-0}"
  CREATE_REPORT="${CREATE_REPORT:-1}"
  FEATURE_SET_VERSION="${FEATURE_SET_VERSION:-benchmark_gap_features_v3}"
  TRANSACTION_COST_BPS="${TRANSACTION_COST_BPS:-10}"
  MIN_HISTORY_ROWS="${MIN_HISTORY_ROWS:-252}"
  MIN_TRAIN_ELIGIBLE_ROWS="${MIN_TRAIN_ELIGIBLE_ROWS:-5000}"
  STEP_MONTHS="${STEP_MONTHS:-3}"
  TRAIN_WINDOW_YEARS="${TRAIN_WINDOW_YEARS:-3 5}"
  VALIDATION_WINDOW_MONTHS="${VALIDATION_WINDOW_MONTHS:-12}"
  TEST_WINDOW_MONTHS="${TEST_WINDOW_MONTHS:-3 6 12}"
  REPORT_OUTPUT_DIR="${REPORT_OUTPUT_DIR:-artifacts/reports/tiingo_v3_pure_weighted_allocation_study}"

  local results_csv completed_keys failed_keys terminal_log preflight_csv preflight_md
  results_csv="$OUTPUT_DIR/tiingo_v3_pure_weighted_allocation_results.csv"
  completed_keys="$OUTPUT_DIR/tiingo_v3_pure_weighted_allocation_completed_keys.txt"
  failed_keys="$OUTPUT_DIR/tiingo_v3_pure_weighted_allocation_failed_keys.txt"
  terminal_log="$OUTPUT_DIR/tiingo_v3_pure_weighted_allocation_terminal.log"
  preflight_csv="$OUTPUT_DIR/tiingo_v3_pure_weighted_allocation_preflight.csv"
  preflight_md="$OUTPUT_DIR/tiingo_v3_pure_weighted_allocation_preflight.md"

  mkdir -p "$OUTPUT_DIR"

  local cmd=(
    python -m us_market_ml.cli.run_tiingo_v3_pure_weighted_allocation_study
    --market-dataset-glob "$MARKET_DATASET_GLOB"
    --output-dir "$OUTPUT_DIR"
    --run-name "$RUN_NAME"
    --start-date "$START_DATE"
    --end-date "$END_DATE"
    --max-workers "$MAX_WORKERS"
    --min-history-rows "$MIN_HISTORY_ROWS"
    --min-train-eligible-rows "$MIN_TRAIN_ELIGIBLE_ROWS"
    --step-months "$STEP_MONTHS"
    --feature-set-version "$FEATURE_SET_VERSION"
    --transaction-cost-bps "$TRANSACTION_COST_BPS"
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

  printf 'Running preflight before dispatching pure weighted-allocation rules...\n'
  "${cmd[@]}" --dry-run 2>&1 | tee -a "$terminal_log"

  print_preflight_summary "$preflight_csv" "$completed_keys" "$failed_keys" "$RESUME" \
    "$MAX_WORKERS" "$MARKET_DATASET_GLOB" "$OUTPUT_DIR" "$results_csv" "$terminal_log" \
    | tee -a "$terminal_log"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '\nDRY_RUN=1: preflight complete; no long run started.\n'
    exit 0
  fi

  local total_configs valid_configs invalid_configs started_at
  total_configs="$(preflight_count "$preflight_csv" total)"
  valid_configs="$(preflight_count "$preflight_csv" valid)"
  invalid_configs="$(preflight_count "$preflight_csv" invalid)"
  started_at="$(date +%s)"

  print_progress() {
    local completed failed result_rows now elapsed pending
    completed="$(count_non_empty_lines "$completed_keys")"
    failed="$(count_non_empty_lines "$failed_keys")"
    result_rows="$(result_row_count "$results_csv")"
    now="$(date +%s)"
    elapsed="$((now - started_at))"
    pending="$((total_configs - completed - invalid_configs - failed))"
    if [[ "$pending" -lt 0 ]]; then
      pending=0
    fi
    printf '[%s] completed %s / pending %s / total %s; invalid=%s failed=%s result_rows=%s elapsed=%s log=%s\n' \
      "$(date '+%Y-%m-%d %H:%M:%S')" \
      "$completed" \
      "$pending" \
      "$total_configs" \
      "$invalid_configs" \
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
    printf 'Total allocation evaluations: %s\n' "$total_configs"
    printf 'Valid allocation evaluations: %s\n' "$valid_configs"
    printf 'Invalid allocation evaluations: %s\n' "$invalid_configs"
  } >> "$terminal_log"

  printf '\nStarting Tiingo v3 pure weighted-allocation study. Progress prints every 60 seconds.\n'
  printf 'No S3/cloud/GPU, no MLP, no tree models, no feature v4, no production signal export, and no fixed QQQ/SPY core are used.\n'
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

  if [[ "$CREATE_REPORT" != "0" ]]; then
    printf '\nCreating Tiingo v3 pure weighted-allocation report...\n'
    python -m us_market_ml.cli.create_tiingo_v3_pure_weighted_allocation_report \
      --results-csv "$results_csv" \
      --market-dataset-glob "$MARKET_DATASET_GLOB" \
      --output-dir "$REPORT_OUTPUT_DIR" 2>&1 | tee -a "$terminal_log"
  fi

  printf '\nTiingo v3 pure weighted-allocation study complete.\n'
  printf 'Useful output commands:\n'
  printf '  tail -n 120 %q\n' "$terminal_log"
  printf '  sed -n %q %q\n' '1,260p' "$preflight_md"
  printf '  find %q -maxdepth 1 -type f | sort\n' "$OUTPUT_DIR"
}

if [[ "${TIINGO_V3_PURE_WEIGHTED_ALLOCATION_STUDY_TEST_HELPERS:-0}" != "1" ]]; then
  main "$@"
fi
