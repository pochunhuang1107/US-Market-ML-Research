"""Generate local MLP strategy benchmark comparison reports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from us_market_ml.backtest import (
    BenchmarkConfig,
    SignalBacktestConfig,
    TopKSignalConfig,
    build_buy_and_hold_benchmark,
    build_equal_weight_benchmark,
    combine_strategy_and_benchmarks,
    summarize_equity_series,
)
from us_market_ml.data import read_local_market_data
from us_market_ml.experiments import (
    MLPE2EExperimentConfig,
    MLPExperimentConfig,
    run_mlp_e2e_experiment,
)
from us_market_ml.models import MLPRegressorConfig
from us_market_ml.splits import PurgeConfig, WalkForwardFold
from us_market_ml.visualization import plot_equity_curves


def run_benchmark_report_from_args(argv: list[str] | None = None) -> int:
    """Run one selected local MLP config and export benchmark report files."""

    args = _parse_args(argv)
    market_data = read_local_market_data(args.data_path)
    fold = WalkForwardFold(
        name="benchmark_report_fold",
        train_start=pd.Timestamp(args.train_start),
        train_end=pd.Timestamp(args.train_end),
        validation_start=pd.Timestamp(args.validation_start),
        validation_end=pd.Timestamp(args.validation_end),
        test_start=pd.Timestamp(args.test_start),
        test_end=pd.Timestamp(args.test_end),
    )
    experiment_config = MLPE2EExperimentConfig(
        mlp_config=MLPExperimentConfig(
            target_col=args.target_col,
            model_config=MLPRegressorConfig(
                hidden_units=args.hidden_units,
                learning_rate=args.learning_rate,
                dropout=args.dropout,
                max_epochs=args.max_epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                random_state=args.random_state,
                scale_target=args.scale_target,
            ),
            purge_config=(
                PurgeConfig(label_horizon_days=args.label_horizon_days)
                if args.label_horizon_days > 0
                else None
            ),
        ),
        signal_config=TopKSignalConfig(top_k=args.top_k),
        backtest_config=SignalBacktestConfig(initial_equity=1.0),
    )

    result = run_mlp_e2e_experiment(
        raw_df=market_data,
        folds=[fold],
        config=experiment_config,
    )
    strategy_curve = result.fold_results[0].test_backtest.equity_curve
    test_market_data = _filter_test_period(market_data, args.test_start, args.test_end)

    benchmark_config = BenchmarkConfig(initial_equity=1.0)
    benchmark_curves = [
        build_buy_and_hold_benchmark(
            test_market_data,
            symbol=symbol,
            config=benchmark_config,
        )
        for symbol in args.benchmark_symbols
    ]
    benchmark_curves.append(
        build_equal_weight_benchmark(
            test_market_data,
            symbols=args.universe_symbols,
            config=benchmark_config,
            output_name="equal_weight",
        )
    )

    combined_equity = combine_strategy_and_benchmarks(
        strategy_equity_curve=strategy_curve,
        benchmark_curves=benchmark_curves,
        strategy_name="mlp_strategy",
    )
    metrics = summarize_equity_series(combined_equity)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / f"{args.report_name}_combined_equity.csv"
    metrics_path = output_dir / f"{args.report_name}_metrics.csv"
    plot_path = output_dir / f"{args.report_name}_equity.png"
    combined_equity.to_csv(combined_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    plot_equity_curves(
        combined_equity,
        plot_path,
        title="MLP Strategy vs Benchmark Equity",
    )

    print("US Market ML benchmark report")
    print(f"data path: {args.data_path}")
    print(f"output combined equity CSV: {combined_path}")
    print(f"output metrics CSV: {metrics_path}")
    print(f"output equity PNG: {plot_path}")
    print()
    print(metrics.to_string(index=False))

    return 0


def parse_comma_separated_symbols(value: str) -> list[str]:
    """Parse comma-separated ticker symbols."""

    symbols = [symbol.strip().upper() for symbol in value.split(",") if symbol.strip()]
    if not symbols:
        raise argparse.ArgumentTypeError("At least one symbol is required.")
    return symbols


def parse_hidden_units(value: str) -> tuple[int, ...]:
    """Parse comma-separated hidden layer sizes."""

    try:
        hidden_units = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid hidden units: {value!r}") from exc
    if not hidden_units:
        raise argparse.ArgumentTypeError("At least one hidden unit value is required.")
    if any(unit <= 0 for unit in hidden_units):
        raise argparse.ArgumentTypeError("hidden units must be positive integers.")
    return hidden_units


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one local MLP config and export benchmark comparison files."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-name", default="benchmark_report")
    parser.add_argument("--universe-symbols", type=parse_comma_separated_symbols)
    parser.add_argument(
        "--benchmark-symbols",
        type=parse_comma_separated_symbols,
        default=parse_comma_separated_symbols("SPY,QQQ"),
    )
    parser.add_argument("--train-start", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--validation-start", required=True)
    parser.add_argument("--validation-end", required=True)
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--target-col", default="excess_return_vs_spy_20d")
    parser.add_argument("--top-k", default=3, type=int)
    parser.add_argument(
        "--hidden-units",
        default=parse_hidden_units("256,128"),
        type=parse_hidden_units,
    )
    parser.add_argument("--learning-rate", default=0.0005, type=float)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument(
        "--scale-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Standardize MLP targets using train-only target mean/std.",
    )
    parser.add_argument("--max-epochs", default=120, type=int)
    parser.add_argument("--patience", default=12, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--random-state", default=42, type=int)
    parser.add_argument("--label-horizon-days", default=20, type=int)
    args = parser.parse_args(argv)

    if args.top_k <= 0:
        parser.error("--top-k must be positive.")
    if args.max_epochs <= 0:
        parser.error("--max-epochs must be positive.")
    if args.patience <= 0:
        parser.error("--patience must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.label_horizon_days < 0:
        parser.error("--label-horizon-days must be greater than or equal to 0.")

    return args


def _filter_test_period(
    market_data: pd.DataFrame,
    test_start: str,
    test_end: str,
) -> pd.DataFrame:
    date_values = pd.to_datetime(market_data["date"])
    mask = (date_values >= pd.Timestamp(test_start)) & (date_values <= pd.Timestamp(test_end))
    return market_data.loc[mask].copy()


if __name__ == "__main__":
    raise SystemExit(run_benchmark_report_from_args())
