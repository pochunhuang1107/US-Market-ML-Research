#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MARKET_DATASET_GLOB="${MARKET_DATASET_GLOB:-/Users/casper/Projects/Research/Market Data/data/research/etf_daily_prices_extended_tiingo_split/**/*.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/reports/extended_history_start_year_viability}"

CSV_REPORT="$OUTPUT_DIR/start_year_viability.csv"
MARKDOWN_REPORT="$OUTPUT_DIR/start_year_viability_report.md"
JSON_SUMMARY="$OUTPUT_DIR/start_year_viability_summary.json"

printf 'Creating extended-history start-year viability report.\n'
printf 'Dataset glob: %s\n' "$MARKET_DATASET_GLOB"
printf 'Output dir: %s\n' "$OUTPUT_DIR"

python -m us_market_ml.cli.create_extended_history_start_year_viability_report \
  --market-dataset-glob "$MARKET_DATASET_GLOB" \
  --output-dir "$OUTPUT_DIR"

printf '\nGenerated output paths:\n'
printf '  CSV: %s\n' "$CSV_REPORT"
printf '  Markdown: %s\n' "$MARKDOWN_REPORT"
printf '  JSON summary: %s\n' "$JSON_SUMMARY"
