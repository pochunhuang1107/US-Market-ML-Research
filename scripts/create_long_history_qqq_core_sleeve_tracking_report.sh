#!/usr/bin/env bash
set -euo pipefail

main() {
  local root_dir
  root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  cd "$root_dir"

  CORE_SLEEVE_RICH_REPLAY_DIR="${CORE_SLEEVE_RICH_REPLAY_DIR:-artifacts/reports/long_history_qqq_core_sleeve_rich_replay}"
  OUTPUT_DIR="${OUTPUT_DIR:-artifacts/reports/long_history_qqq_core_sleeve_tracking}"

  mkdir -p "$OUTPUT_DIR"

  local cmd=(
    python -m us_market_ml.cli.create_long_history_qqq_core_sleeve_tracking_report
    --core-sleeve-rich-replay-dir "$CORE_SLEEVE_RICH_REPLAY_DIR"
    --output-dir "$OUTPUT_DIR"
  )

  printf 'Long-history QQQ core-sleeve tracking report\n'
  printf 'Local tracking/report only: reads existing artifacts only, no S3/cloud, no broad sweep, no model training, no MLP, no production signal export, no model card, no production-ready marking.\n\n'
  printf 'Equivalent Python command:\n'
  printf '  %q' "${cmd[@]}"
  printf '\n\n'

  "${cmd[@]}"

  printf '\nTracking report complete.\n'
  printf 'Report artifacts are in: %s\n' "$OUTPUT_DIR"
  printf 'Generated artifacts remain uncommitted unless you explicitly stage them.\n'
}

main "$@"
