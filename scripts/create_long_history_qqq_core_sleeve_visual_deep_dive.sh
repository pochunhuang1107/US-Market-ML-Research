#!/usr/bin/env bash
set -euo pipefail

main() {
  local root_dir
  root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  cd "$root_dir"

  CORE_SLEEVE_RICH_REPLAY_DIR="${CORE_SLEEVE_RICH_REPLAY_DIR:-artifacts/reports/long_history_qqq_core_sleeve_rich_replay}"
  CORE_SLEEVE_SIGNAL_REVIEW_DIR="${CORE_SLEEVE_SIGNAL_REVIEW_DIR:-artifacts/reports/long_history_qqq_core_sleeve_signal_review}"
  CORE_SLEEVE_TRACKING_DIR="${CORE_SLEEVE_TRACKING_DIR:-artifacts/reports/long_history_qqq_core_sleeve_tracking}"
  OUTPUT_DIR="${OUTPUT_DIR:-artifacts/reports/long_history_qqq_core_sleeve_visual_deep_dive}"

  mkdir -p "$OUTPUT_DIR"

  local cmd=(
    python -m us_market_ml.cli.create_long_history_qqq_core_sleeve_visual_deep_dive
    --core-sleeve-rich-replay-dir "$CORE_SLEEVE_RICH_REPLAY_DIR"
    --core-sleeve-signal-review-dir "$CORE_SLEEVE_SIGNAL_REVIEW_DIR"
    --core-sleeve-tracking-dir "$CORE_SLEEVE_TRACKING_DIR"
    --output-dir "$OUTPUT_DIR"
  )

  printf 'Long-history QQQ core-sleeve visual deep-dive report\n'
  printf 'Local research-only visual report: reads existing artifacts only, no S3/cloud, no broad sweep, no model training, no MLP, no production signal export, no model card, no production-ready marking.\n\n'
  printf 'Equivalent Python command:\n'
  printf '  %q' "${cmd[@]}"
  printf '\n\n'

  "${cmd[@]}"

  printf '\nVisual deep-dive report complete.\n'
  printf 'Report artifacts are in: %s\n' "$OUTPUT_DIR"
  printf 'Generated artifacts remain uncommitted unless you explicitly stage them.\n'
}

main "$@"
