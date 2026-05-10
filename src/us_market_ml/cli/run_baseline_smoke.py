"""Run a local real-data smoke test for the baseline E2E workflow."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from us_market_ml.backtest import SignalBacktestConfig, TopKSignalConfig
from us_market_ml.data import read_local_market_data
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
    market_data = read_local_market_data(args.data_path)
    fold = WalkForwardFold(
        name="smoke_fold",
        train_start=pd.Timestamp(args.train_start),
        train_end=pd.Timestamp(args.train_end),
        validation_start=pd.Timestamp(args.validation_start),
        validation_end=pd.Timestamp(args.validation_end),
        test_start=pd.Timestamp(args.test_start),
        test_end=pd.Timestamp(args.test_end),
    )
    config = BaselineE2EExperimentConfig(
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
        config=config,
    )
    summary = summarize_baseline_e2e_experiment(result)

    print("US Market ML baseline smoke test")
    print(f"data path: {args.data_path}")
    print(f"loaded row count: {len(market_data)}")
    print(f"symbol count: {market_data['symbol'].nunique()}")
    print(f"min date: {market_data['date'].min().date()}")
    print(f"max date: {market_data['date'].max().date()}")
    print(f"feature count: {len(result.baseline_result.feature_cols)}")
    print()
    print(summary.to_string(index=False))

    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a local smoke test of the baseline E2E ETF rotation research workflow."
        )
    )
    parser.add_argument("--data-path", required=True, type=Path)
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
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run_baseline_smoke_from_args())
