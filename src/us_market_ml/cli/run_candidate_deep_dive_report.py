"""Generate local candidate deep-dive diagnostic reports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from us_market_ml.backtest import (
    BenchmarkConfig,
    build_buy_and_hold_benchmark,
    build_concentration_report,
    build_daily_selection_summary,
    build_equal_weight_benchmark,
    build_position_contribution_frame,
    build_selected_holdings_frame,
    combine_strategy_and_benchmarks,
    stitch_test_equity_curves,
    summarize_equity_series,
    summarize_holdings,
    summarize_position_contributions,
)
from us_market_ml.cli.run_benchmark_report import parse_comma_separated_symbols
from us_market_ml.cli.run_candidate_comparison_report import (
    FOLD_SUMMARY_COLUMNS,
    CandidateConfig,
    _build_candidate_fold_summary,
    _build_candidate_rank_table,
    _candidate_curve_to_long,
    _run_candidate_experiment,
    parse_candidate_config,
)
from us_market_ml.cli.run_walk_forward_benchmark_report import (
    _filter_oos_period,
    parse_walk_forward_fold,
)
from us_market_ml.data import read_local_market_data
from us_market_ml.visualization import (
    plot_contribution_by_symbol,
    plot_equity_curves,
    plot_monthly_selection_heatmap,
    plot_selection_frequency,
)


@dataclass(frozen=True)
class _CandidateDeepDiveData:
    candidate: CandidateConfig
    stitched_strategy: pd.DataFrame
    candidate_equity: pd.DataFrame
    fold_summary: pd.DataFrame
    selected_holdings: pd.DataFrame
    position_contributions: pd.DataFrame
    holdings_summary: pd.DataFrame
    contribution_summary: pd.DataFrame
    concentration_report: pd.DataFrame
    daily_selection_summary: pd.DataFrame


def run_candidate_deep_dive_report_from_args(argv: list[str] | None = None) -> int:
    """Run selected local candidates and export full diagnostic packs."""

    args = _parse_args(argv)
    market_data = read_local_market_data(args.data_path)
    folds = list(args.fold)

    candidate_reports: list[_CandidateDeepDiveData] = []
    expected_dates: pd.Series | None = None

    for candidate in args.candidate:
        result = _run_candidate_experiment(candidate, market_data, folds, args)
        stitched_strategy = stitch_test_equity_curves(
            [
                (fold_result.fold_name, fold_result.test_backtest.equity_curve)
                for fold_result in result.fold_results
            ]
        )
        if stitched_strategy.empty:
            raise ValueError(
                f"No test equity curve rows were produced for candidate {candidate.name!r}."
            )

        curve_dates = pd.to_datetime(stitched_strategy["date"]).reset_index(drop=True)
        if expected_dates is None:
            expected_dates = curve_dates
        elif not curve_dates.equals(expected_dates):
            raise ValueError("Candidate test periods do not align; v1 requires identical dates.")

        selected_holdings = build_selected_holdings_frame(result.fold_results)
        position_contributions = build_position_contribution_frame(result.fold_results)
        holdings_summary = summarize_holdings(selected_holdings)
        contribution_summary = summarize_position_contributions(position_contributions)
        candidate_reports.append(
            _CandidateDeepDiveData(
                candidate=candidate,
                stitched_strategy=stitched_strategy,
                candidate_equity=_candidate_curve_to_long(stitched_strategy, candidate.name),
                fold_summary=_build_candidate_fold_summary(candidate, result, folds),
                selected_holdings=selected_holdings,
                position_contributions=position_contributions,
                holdings_summary=holdings_summary,
                contribution_summary=contribution_summary,
                concentration_report=build_concentration_report(
                    holdings_summary,
                    contribution_summary,
                ),
                daily_selection_summary=build_daily_selection_summary(selected_holdings),
            )
        )

    if expected_dates is None or expected_dates.empty:
        raise ValueError("No candidate test equity curve rows were produced.")

    oos_start = pd.Timestamp(expected_dates.min())
    oos_end = pd.Timestamp(expected_dates.max())
    oos_market_data = _filter_oos_period(market_data, oos_start, oos_end)
    benchmark_curves = _build_benchmark_curves(oos_market_data, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for report_data in candidate_reports:
        _write_candidate_report(
            report_data=report_data,
            benchmark_curves=benchmark_curves,
            output_dir=output_dir,
        )

    candidate_equity = (
        pd.concat(
            [report_data.candidate_equity for report_data in candidate_reports],
            ignore_index=True,
        )
        .sort_values(by=["date", "series_name"], kind="mergesort")
        .reset_index(drop=True)
    )
    candidate_metrics = summarize_equity_series(candidate_equity)
    candidate_rank = _build_candidate_rank_table(
        candidate_metrics,
        [report_data.candidate for report_data in candidate_reports],
    )
    candidate_metrics_output = candidate_rank.copy()
    concentration_summary = _build_concentration_summary(candidate_reports)
    fold_summary = pd.DataFrame(
        [
            row
            for report_data in candidate_reports
            for row in report_data.fold_summary.to_dict("records")
        ],
        columns=FOLD_SUMMARY_COLUMNS,
    )

    candidate_equity_path = output_dir / f"{args.report_name}_candidate_equity.csv"
    candidate_metrics_path = output_dir / f"{args.report_name}_candidate_metrics.csv"
    candidate_rank_path = output_dir / f"{args.report_name}_candidate_rank.csv"
    concentration_summary_path = output_dir / f"{args.report_name}_concentration_summary.csv"
    fold_summary_path = output_dir / f"{args.report_name}_fold_summary.csv"
    candidate_equity_plot_path = output_dir / f"{args.report_name}_candidate_equity.png"
    candidate_equity.to_csv(candidate_equity_path, index=False)
    candidate_metrics_output.to_csv(candidate_metrics_path, index=False)
    candidate_rank.to_csv(candidate_rank_path, index=False)
    concentration_summary.to_csv(concentration_summary_path, index=False)
    fold_summary.to_csv(fold_summary_path, index=False)
    plot_equity_curves(
        candidate_equity,
        candidate_equity_plot_path,
        title="Candidate OOS Strategy Equity",
    )

    print("US Market ML candidate deep-dive report")
    print(f"data path: {args.data_path}")
    print(f"OOS period: {oos_start.date()} to {oos_end.date()}")
    print(f"output candidate equity CSV: {candidate_equity_path}")
    print(f"output candidate metrics CSV: {candidate_metrics_path}")
    print(f"output candidate rank CSV: {candidate_rank_path}")
    print(f"output concentration summary CSV: {concentration_summary_path}")
    print(f"output fold summary CSV: {fold_summary_path}")
    print(f"output candidate equity PNG: {candidate_equity_plot_path}")
    for report_data in candidate_reports:
        print(f"output candidate directory: {output_dir / report_data.candidate.name}")
    print()
    print(candidate_rank.to_string(index=False))

    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run selected local walk-forward candidates and export per-candidate "
            "holdings, contribution, benchmark, and visualization diagnostics."
        )
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-name", default="candidate_deep_dive")
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
        "--candidate",
        action="append",
        type=parse_candidate_config,
        required=True,
        help=(
            "Repeatable candidate spec: "
            "mlp:name:top_k:hidden_units:learning_rate:dropout:random_state, "
            "baseline:name:model_type:top_k:random_state, or legacy "
            "name:top_k:hidden_units:learning_rate:dropout:random_state"
        ),
    )
    parser.add_argument(
        "--benchmark-symbols",
        type=parse_comma_separated_symbols,
        default=parse_comma_separated_symbols("SPY,QQQ"),
    )
    parser.add_argument("--universe-symbols", type=parse_comma_separated_symbols)
    parser.add_argument("--target-col", default="excess_return_vs_spy_20d")
    parser.add_argument(
        "--scale-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Standardize MLP targets using train-only target mean/std.",
    )
    parser.add_argument("--max-epochs", default=120, type=int)
    parser.add_argument("--patience", default=12, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--label-horizon-days", default=20, type=int)
    args = parser.parse_args(argv)

    candidate_names = [candidate.name for candidate in args.candidate]
    duplicate_names = sorted(
        {name for name in candidate_names if candidate_names.count(name) > 1}
    )
    if duplicate_names:
        parser.error(f"--candidate names must be unique; duplicates: {', '.join(duplicate_names)}.")
    if args.max_epochs <= 0:
        parser.error("--max-epochs must be positive.")
    if args.patience <= 0:
        parser.error("--patience must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.label_horizon_days < 0:
        parser.error("--label-horizon-days must be greater than or equal to 0.")

    return args


def _build_benchmark_curves(
    oos_market_data: pd.DataFrame,
    args: argparse.Namespace,
) -> list[pd.DataFrame]:
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
    return benchmark_curves


def _write_candidate_report(
    report_data: _CandidateDeepDiveData,
    benchmark_curves: list[pd.DataFrame],
    output_dir: Path,
) -> None:
    candidate_name = report_data.candidate.name
    candidate_dir = output_dir / candidate_name
    candidate_dir.mkdir(parents=True, exist_ok=True)

    combined_equity = combine_strategy_and_benchmarks(
        strategy_equity_curve=report_data.stitched_strategy,
        benchmark_curves=benchmark_curves,
        strategy_name=candidate_name,
    )
    metrics = summarize_equity_series(combined_equity)

    combined_path = candidate_dir / f"{candidate_name}_combined_equity.csv"
    metrics_path = candidate_dir / f"{candidate_name}_metrics.csv"
    fold_summary_path = candidate_dir / f"{candidate_name}_fold_summary.csv"
    selected_holdings_path = candidate_dir / f"{candidate_name}_selected_holdings.csv"
    position_contributions_path = candidate_dir / f"{candidate_name}_position_contributions.csv"
    holdings_summary_path = candidate_dir / f"{candidate_name}_holdings_summary.csv"
    contribution_summary_path = candidate_dir / f"{candidate_name}_contribution_summary.csv"
    concentration_report_path = candidate_dir / f"{candidate_name}_concentration_report.csv"
    daily_selection_summary_path = candidate_dir / f"{candidate_name}_daily_selection_summary.csv"
    equity_path = candidate_dir / f"{candidate_name}_equity.png"
    selection_frequency_path = candidate_dir / f"{candidate_name}_selection_frequency.png"
    contribution_by_symbol_path = candidate_dir / f"{candidate_name}_contribution_by_symbol.png"
    monthly_selection_heatmap_path = (
        candidate_dir / f"{candidate_name}_monthly_selection_heatmap.png"
    )

    combined_equity.to_csv(combined_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    report_data.fold_summary.to_csv(fold_summary_path, index=False)
    report_data.selected_holdings.to_csv(selected_holdings_path, index=False)
    report_data.position_contributions.to_csv(position_contributions_path, index=False)
    report_data.holdings_summary.to_csv(holdings_summary_path, index=False)
    report_data.contribution_summary.to_csv(contribution_summary_path, index=False)
    report_data.concentration_report.to_csv(concentration_report_path, index=False)
    report_data.daily_selection_summary.to_csv(daily_selection_summary_path, index=False)
    plot_equity_curves(
        combined_equity,
        equity_path,
        title=f"{candidate_name} OOS Strategy vs Benchmarks",
    )
    plot_selection_frequency(report_data.holdings_summary, selection_frequency_path)
    plot_contribution_by_symbol(report_data.contribution_summary, contribution_by_symbol_path)
    plot_monthly_selection_heatmap(report_data.selected_holdings, monthly_selection_heatmap_path)


def _build_concentration_summary(
    candidate_reports: list[_CandidateDeepDiveData],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for report_data in candidate_reports:
        metrics = report_data.concentration_report.iloc[0].to_dict()
        rows.append(
            {
                "candidate_name": report_data.candidate.name,
                "candidate_family": report_data.candidate.family,
                "model_type": (
                    report_data.candidate.model_type
                    if report_data.candidate.model_type is not None
                    else "mlp"
                ),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    raise SystemExit(run_candidate_deep_dive_report_from_args())
