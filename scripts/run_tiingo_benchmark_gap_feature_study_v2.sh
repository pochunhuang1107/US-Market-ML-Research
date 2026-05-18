#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MARKET_DATASET_GLOB="${MARKET_DATASET_GLOB:-/Users/casper/Projects/Research/Market Data/data/research/etf_daily_prices_tiingo_only/**/*.parquet}"
export OUTPUT_DIR="${OUTPUT_DIR:-artifacts/sweeps/tiingo_benchmark_gap_feature_study_v2}"
export RUN_NAME="${RUN_NAME:-tiingo_benchmark_gap_feature_study_v2}"
export START_DATE="${START_DATE:-2019-01-01}"
export END_DATE="${END_DATE:-2026-05-15}"
export MAX_WORKERS="${MAX_WORKERS:-4}"
export RESUME="${RESUME:-1}"
export DRY_RUN="${DRY_RUN:-0}"
export FEATURE_SET_VERSIONS="${FEATURE_SET_VERSIONS:-default_features_v1 benchmark_gap_features_v1 benchmark_gap_features_v2}"
export REPORT_MODULE="${REPORT_MODULE:-us_market_ml.cli.create_tiingo_benchmark_gap_feature_v2_report}"
export REPORT_OUTPUT_DIR="${REPORT_OUTPUT_DIR:-artifacts/reports/tiingo_benchmark_gap_feature_study_v2}"

exec bash "$script_dir/run_tiingo_benchmark_gap_feature_study.sh" "$@"
