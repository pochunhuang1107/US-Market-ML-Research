"""Run a real-data smoke test for the baseline E2E workflow."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from us_market_ml.backtest import SignalBacktestConfig, TopKSignalConfig
from us_market_ml.data import (
    S3MarketDataConfig,
    read_local_market_data,
    read_s3_curated_etf_data,
)
from us_market_ml.experiments import (
    BaselineE2EExperimentConfig,
    BaselineExperimentConfig,
    run_baseline_e2e_experiment,
    summarize_baseline_e2e_experiment,
)
from us_market_ml.models import BaselineModelConfig
from us_market_ml.splits import WalkForwardFold


def run_baseline_smoke_from_args(argv: list[str] | None = None) -> int:
    """Run the baseline E2E research workflow from command-line arguments."""

    args = _parse_args(argv)
    if args.data_path is not None:
        market_data = read_local_market_data(args.data_path)
        input_source_lines = [
            "input source: local",
            f"data path: {args.data_path}",
        ]
    else:
        s3_config = S3MarketDataConfig(
            bucket=args.s3_bucket,
            curated_prefix=args.s3_curated_prefix,
            aws_profile=args.aws_profile,
            region_name=args.region_name,
        )
        market_data = read_s3_curated_etf_data(
            symbols=args.symbols,
            years=args.years,
            config=s3_config,
        )
        input_source_lines = [
            "input source: s3",
            f"s3 bucket: {args.s3_bucket}",
            f"s3 curated prefix: {args.s3_curated_prefix}",
            f"symbols: {', '.join(args.symbols)}",
            f"years: {', '.join(str(year) for year in args.years)}",
        ]

    fold = WalkForwardFold(
        name="smoke_fold",
        train_start=pd.Timestamp(args.train_start),
        train_end=pd.Timestamp(args.train_end),
        validation_start=pd.Timestamp(args.validation_start),
        validation_end=pd.Timestamp(args.validation_end),
        test_start=pd.Timestamp(args.test_start),
        test_end=pd.Timestamp(args.test_end),
    )
    experiment_config = BaselineE2EExperimentConfig(
        baseline_config=BaselineExperimentConfig(
            target_col=args.target_col,
            model_config=BaselineModelConfig(model_type=args.model_type),
        ),
        signal_config=TopKSignalConfig(top_k=args.top_k),
        backtest_config=SignalBacktestConfig(initial_equity=args.initial_equity),
    )

    result = run_baseline_e2e_experiment(
        raw_df=market_data,
        folds=[fold],
        config=experiment_config,
    )
    summary = summarize_baseline_e2e_experiment(result)

    print("US Market ML baseline smoke test")
    for line in input_source_lines:
        print(line)
    print(f"loaded row count: {len(market_data)}")
    print(f"symbol count: {market_data['symbol'].nunique()}")
    print(f"min date: {market_data['date'].min().date()}")
    print(f"max date: {market_data['date'].max().date()}")
    print(f"feature count: {len(result.baseline_result.feature_cols)}")
    print()
    print(summary.to_string(index=False))

    return 0


def parse_comma_separated_symbols(value: str) -> list[str]:
    """Parse a comma-separated symbol list for CLI S3 loading."""

    symbols = [symbol.strip().upper() for symbol in value.split(",") if symbol.strip()]
    if not symbols:
        raise ValueError("At least one symbol is required.")
    return symbols


def parse_comma_separated_years(value: str) -> list[int]:
    """Parse a comma-separated year list for CLI S3 loading."""

    years: list[int] = []
    for raw_year in value.split(","):
        year = raw_year.strip()
        if not year:
            continue
        years.append(int(year))

    if not years:
        raise ValueError("At least one year is required.")
    return years


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a smoke test of the baseline E2E ETF rotation research workflow."
        )
    )
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--s3-bucket")
    parser.add_argument("--s3-curated-prefix", default="curated/etf_daily_prices")
    parser.add_argument("--symbols", type=parse_comma_separated_symbols)
    parser.add_argument("--years", type=parse_comma_separated_years)
    parser.add_argument("--aws-profile")
    parser.add_argument("--region-name", default="us-west-2")
    parser.add_argument("--train-start", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--validation-start", required=True)
    parser.add_argument("--validation-end", required=True)
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--target-col", default="excess_return_vs_spy_20d")
    parser.add_argument("--model-type", default="ridge")
    parser.add_argument("--top-k", default=3, type=int)
    parser.add_argument("--initial-equity", default=1.0, type=float)
    args = parser.parse_args(argv)

    if (args.data_path is None) == (args.s3_bucket is None):
        parser.error("Exactly one of --data-path or --s3-bucket must be provided.")

    if args.s3_bucket is not None:
        if args.symbols is None:
            parser.error("--symbols is required when --s3-bucket is provided.")
        if args.years is None:
            parser.error("--years is required when --s3-bucket is provided.")

    return args


if __name__ == "__main__":
    raise SystemExit(run_baseline_smoke_from_args())
