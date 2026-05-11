"""Generate stitched walk-forward MLP benchmark comparison reports."""

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
    build_daily_selection_summary,
    build_equal_weight_benchmark,
    build_position_contribution_frame,
    build_selected_holdings_frame,
    combine_strategy_and_benchmarks,
    stitch_test_equity_curves,
    summarize_backtest,
    summarize_equity_series,
    summarize_holdings,
    summarize_position_contributions,
)
from us_market_ml.cli.run_benchmark_report import (
    parse_comma_separated_symbols,
    parse_hidden_units,
)
from us_market_ml.data import read_local_market_data
from us_market_ml.experiments import (
    MLPE2EExperimentConfig,
    MLPE2EExperimentResult,
    MLPExperimentConfig,
    run_mlp_e2e_experiment,
)
from us_market_ml.models import MLPRegressorConfig
from us_market_ml.splits import PurgeConfig, WalkForwardFold
from us_market_ml.visualization import plot_equity_curves


def run_walk_forward_benchmark_report_from_args(argv: list[str] | None = None) -> int:
    """Run local MLP walk-forward folds and export a stitched OOS benchmark report."""

    args = _parse_args(argv)
    market_data = read_local_market_data(args.data_path)
    folds = list(args.fold)

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
        folds=folds,
        config=experiment_config,
    )
    stitched_strategy = stitch_test_equity_curves(
        [
            (fold_result.fold_name, fold_result.test_backtest.equity_curve)
            for fold_result in result.fold_results
        ]
    )
    if stitched_strategy.empty:
        raise ValueError("No test equity curve rows were produced for the provided folds.")

    oos_start = pd.Timestamp(stitched_strategy["date"].min())
    oos_end = pd.Timestamp(stitched_strategy["date"].max())
    oos_market_data = _filter_oos_period(market_data, oos_start, oos_end)

    benchmark_config = BenchmarkConfig(initial_equity=1.0)
    benchmark_curves = [
        build_buy_and_hold_benchmark(
            oos_market_data,
            symbol=symbol,
            config=benchmark_config,
        )
        for symbol in args.benchmark_symbols
    ]
    benchmark_curves.append(
        build_equal_weight_benchmark(
            oos_market_data,
            symbols=args.universe_symbols,
            config=benchmark_config,
            output_name="equal_weight",
        )
    )

    combined_equity = combine_strategy_and_benchmarks(
        strategy_equity_curve=stitched_strategy,
        benchmark_curves=benchmark_curves,
        strategy_name="mlp_strategy",
    )
    metrics = summarize_equity_series(combined_equity)
    fold_summary = _build_fold_summary(result, folds)
    selected_holdings = build_selected_holdings_frame(result.fold_results)
    position_contributions = build_position_contribution_frame(result.fold_results)
    holdings_summary = summarize_holdings(selected_holdings)
    contribution_summary = summarize_position_contributions(position_contributions)
    daily_selection_summary = build_daily_selection_summary(selected_holdings)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / f"{args.report_name}_combined_equity.csv"
    metrics_path = output_dir / f"{args.report_name}_metrics.csv"
    plot_path = output_dir / f"{args.report_name}_equity.png"
    fold_summary_path = output_dir / f"{args.report_name}_fold_summary.csv"
    selected_holdings_path = output_dir / f"{args.report_name}_selected_holdings.csv"
    position_contributions_path = output_dir / f"{args.report_name}_position_contributions.csv"
    holdings_summary_path = output_dir / f"{args.report_name}_holdings_summary.csv"
    contribution_summary_path = output_dir / f"{args.report_name}_contribution_summary.csv"
    daily_selection_summary_path = output_dir / f"{args.report_name}_daily_selection_summary.csv"
    combined_equity.to_csv(combined_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    fold_summary.to_csv(fold_summary_path, index=False)
    selected_holdings.to_csv(selected_holdings_path, index=False)
    position_contributions.to_csv(position_contributions_path, index=False)
    holdings_summary.to_csv(holdings_summary_path, index=False)
    contribution_summary.to_csv(contribution_summary_path, index=False)
    daily_selection_summary.to_csv(daily_selection_summary_path, index=False)
    plot_equity_curves(
        combined_equity,
        plot_path,
        title="MLP Walk-Forward OOS Strategy vs Benchmark Equity",
    )

    print("US Market ML walk-forward benchmark report")
    print(f"data path: {args.data_path}")
    print(f"OOS period: {oos_start.date()} to {oos_end.date()}")
    print(f"output combined equity CSV: {combined_path}")
    print(f"output metrics CSV: {metrics_path}")
    print(f"output equity PNG: {plot_path}")
    print(f"output fold summary CSV: {fold_summary_path}")
    print(f"output selected holdings CSV: {selected_holdings_path}")
    print(f"output position contributions CSV: {position_contributions_path}")
    print(f"output holdings summary CSV: {holdings_summary_path}")
    print(f"output contribution summary CSV: {contribution_summary_path}")
    print(f"output daily selection summary CSV: {daily_selection_summary_path}")
    print()
    print(metrics.to_string(index=False))

    return 0


def parse_walk_forward_fold(value: str) -> WalkForwardFold:
    """Parse a named walk-forward fold CLI value."""

    parts = value.split(":")
    if len(parts) != 7 or any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "Fold must use format "
            "name:train_start:train_end:validation_start:validation_end:test_start:test_end."
        )

    name, train_start, train_end, validation_start, validation_end, test_start, test_end = (
        part.strip() for part in parts
    )
    try:
        return WalkForwardFold(
            name=name,
            train_start=pd.Timestamp(train_start),
            train_end=pd.Timestamp(train_end),
            validation_start=pd.Timestamp(validation_start),
            validation_end=pd.Timestamp(validation_end),
            test_start=pd.Timestamp(test_start),
            test_end=pd.Timestamp(test_end),
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run local MLP walk-forward folds and export a stitched out-of-sample "
            "benchmark comparison report."
        )
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-name", default="walk_forward_benchmark_report")
    parser.add_argument(
        "--fold",
        action="append",
        type=parse_walk_forward_fold,
        required=True,
        help=(
            "Repeatable fold spec: "
            "name:train_start:train_end:validation_start:validation_end:test_start:test_end"
        ),
    )
    parser.add_argument(
        "--benchmark-symbols",
        type=parse_comma_separated_symbols,
        default=parse_comma_separated_symbols("SPY,QQQ"),
    )
    parser.add_argument("--universe-symbols", type=parse_comma_separated_symbols)
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


def _filter_oos_period(
    market_data: pd.DataFrame,
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
) -> pd.DataFrame:
    date_values = pd.to_datetime(market_data["date"])
    mask = (date_values >= oos_start) & (date_values <= oos_end)
    return market_data.loc[mask].copy()


def _build_fold_summary(
    result: MLPE2EExperimentResult,
    folds: list[WalkForwardFold],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold_result, mlp_fold_result, fold in zip(
        result.fold_results,
        result.mlp_result.fold_results,
        folds,
        strict=True,
    ):
        equity_curve = fold_result.test_backtest.equity_curve
        metrics = summarize_backtest(equity_curve, SignalBacktestConfig(initial_equity=1.0))
        if equity_curve.empty:
            test_start: object = fold.test_start.date()
            test_end: object = fold.test_end.date()
        else:
            test_start = pd.Timestamp(equity_curve["date"].min()).date()
            test_end = pd.Timestamp(equity_curve["date"].max()).date()
        rows.append(
            {
                "fold_name": fold_result.fold_name,
                "test_start": test_start,
                "test_end": test_end,
                "strategy_total_return": metrics["total_return"],
                "strategy_max_drawdown": metrics["max_drawdown"],
                "strategy_final_equity": metrics["final_equity"],
                "epochs_trained": mlp_fold_result.fit_result.epochs_trained,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "fold_name",
            "test_start",
            "test_end",
            "strategy_total_return",
            "strategy_max_drawdown",
            "strategy_final_equity",
            "epochs_trained",
        ],
    )


if __name__ == "__main__":
    raise SystemExit(run_walk_forward_benchmark_report_from_args())
