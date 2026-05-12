"""Generate local candidate transaction-cost sensitivity reports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from math import isfinite
from pathlib import Path

import pandas as pd

from us_market_ml.backtest import (
    SignalBacktestConfig,
    stitch_test_equity_curves,
    summarize_backtest,
)
from us_market_ml.cli.run_benchmark_report import parse_comma_separated_symbols
from us_market_ml.cli.run_candidate_comparison_report import (
    CandidateConfig,
    _candidate_model_type_label,
    _run_candidate_experiment,
    parse_candidate_config,
)
from us_market_ml.cli.run_walk_forward_benchmark_report import parse_walk_forward_fold
from us_market_ml.data import read_local_market_data
from us_market_ml.splits import WalkForwardFold
from us_market_ml.visualization import plot_cost_sensitivity

SENSITIVITY_COLUMNS = [
    "candidate_name",
    "candidate_family",
    "model_type",
    "top_k",
    "cost_bps",
    "total_return",
    "sharpe_like",
    "max_drawdown",
    "final_equity",
    "total_transaction_cost",
    "mean_daily_turnover",
    "max_daily_turnover",
    "return_decay_vs_0bps",
    "final_equity_decay_vs_0bps",
]

PIVOT_METRIC_COLUMNS = [
    "total_return",
    "final_equity",
    "sharpe_like",
]


def run_cost_sensitivity_report_from_args(argv: list[str] | None = None) -> int:
    """Run selected candidates across transaction-cost assumptions and export a report."""

    args = _parse_args(argv)
    market_data = read_local_market_data(args.data_path)
    folds = list(args.fold)

    sensitivity = _build_cost_sensitivity(
        candidates=args.candidate,
        cost_bps_values=args.cost_bps,
        market_data=market_data,
        folds=folds,
        args=args,
    )
    pivot = _build_cost_sensitivity_pivot(sensitivity, args.cost_bps)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sensitivity_path = output_dir / f"{args.report_name}_cost_sensitivity.csv"
    pivot_path = output_dir / f"{args.report_name}_cost_sensitivity_pivot.csv"
    plot_path = output_dir / f"{args.report_name}_cost_sensitivity.png"

    sensitivity.to_csv(sensitivity_path, index=False)
    pivot.to_csv(pivot_path, index=False)
    plot_cost_sensitivity(
        sensitivity,
        plot_path,
        metric_col="total_return",
        title="Candidate Cost Sensitivity",
    )

    summary = sensitivity.sort_values(
        by=["cost_bps", "total_return", "candidate_name"],
        ascending=[True, False, True],
        kind="mergesort",
    )

    print("US Market ML cost sensitivity report")
    print(f"data path: {args.data_path}")
    print(f"cost bps grid: {', '.join(_format_cost_bps(value) for value in args.cost_bps)}")
    print(f"output sensitivity CSV: {sensitivity_path}")
    print(f"output sensitivity pivot CSV: {pivot_path}")
    print(f"output sensitivity PNG: {plot_path}")
    print()
    print(summary.to_string(index=False))

    return 0


def parse_float_list(value: str) -> list[float]:
    """Parse a comma-separated list of finite float values."""

    parts = [part.strip() for part in value.split(",")]
    if any(not part for part in parts):
        raise argparse.ArgumentTypeError("Float list values must be non-empty.")

    parsed_values: list[float] = []
    for part in parts:
        try:
            parsed_value = float(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid float list value: {part!r}") from exc
        if not isfinite(parsed_value):
            raise argparse.ArgumentTypeError(f"Invalid float list value: {part!r}")
        parsed_values.append(parsed_value)

    return parsed_values


def _build_cost_sensitivity(
    *,
    candidates: list[CandidateConfig],
    cost_bps_values: list[float],
    market_data: pd.DataFrame,
    folds: list[WalkForwardFold],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for candidate in candidates:
        for cost_bps in cost_bps_values:
            cost_args = argparse.Namespace(
                **{**vars(args), "transaction_cost_bps": cost_bps}
            )
            result = _run_candidate_experiment(candidate, market_data, folds, cost_args)
            stitched_strategy = stitch_test_equity_curves(
                [
                    (fold_result.fold_name, fold_result.test_backtest.equity_curve)
                    for fold_result in result.fold_results
                ]
            )
            if stitched_strategy.empty:
                raise ValueError(
                    f"No test equity curve rows were produced for candidate "
                    f"{candidate.name!r} at {cost_bps} bps."
                )

            metrics = summarize_backtest(
                stitched_strategy,
                SignalBacktestConfig(initial_equity=1.0),
            )
            rows.append(
                {
                    "candidate_name": candidate.name,
                    "candidate_family": candidate.family,
                    "model_type": _candidate_model_type_label(candidate),
                    "top_k": candidate.top_k,
                    "cost_bps": cost_bps,
                    "total_return": metrics["total_return"],
                    "sharpe_like": metrics["sharpe_like"],
                    "max_drawdown": metrics["max_drawdown"],
                    "final_equity": metrics["final_equity"],
                    "total_transaction_cost": metrics["total_transaction_cost"],
                    "mean_daily_turnover": metrics["mean_daily_turnover"],
                    "max_daily_turnover": metrics["max_daily_turnover"],
                }
            )

    sensitivity = pd.DataFrame(rows, columns=SENSITIVITY_COLUMNS[:-2])
    return _add_no_cost_decay_columns(sensitivity)


def _add_no_cost_decay_columns(sensitivity: pd.DataFrame) -> pd.DataFrame:
    output = sensitivity.copy()
    no_cost = output.loc[
        output["cost_bps"] == 0.0,
        ["candidate_name", "total_return", "final_equity"],
    ].rename(
        columns={
            "total_return": "no_cost_total_return",
            "final_equity": "no_cost_final_equity",
        }
    )
    output = output.merge(no_cost, on="candidate_name", how="left", validate="many_to_one")
    output["return_decay_vs_0bps"] = output["total_return"] - output["no_cost_total_return"]
    output["final_equity_decay_vs_0bps"] = (
        output["final_equity"] - output["no_cost_final_equity"]
    )
    return output[SENSITIVITY_COLUMNS]


def _build_cost_sensitivity_pivot(
    sensitivity: pd.DataFrame,
    cost_bps_values: list[float],
) -> pd.DataFrame:
    metadata_columns = ["candidate_name", "candidate_family", "model_type"]
    metadata = (
        sensitivity[metadata_columns]
        .drop_duplicates(subset=["candidate_name"], keep="first")
        .sort_values("candidate_name", kind="mergesort")
        .reset_index(drop=True)
    )
    output = metadata.copy()

    for metric in PIVOT_METRIC_COLUMNS:
        pivot = sensitivity.pivot(
            index="candidate_name",
            columns="cost_bps",
            values=metric,
        )
        for cost_bps in cost_bps_values:
            column_name = f"{metric}_{_format_cost_bps(cost_bps)}bps"
            output[column_name] = output["candidate_name"].map(pivot.get(cost_bps))

    return output


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run selected local walk-forward candidates across a grid of "
            "turnover-based transaction-cost assumptions."
        )
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-name", default="cost_sensitivity")
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
        "--cost-bps",
        type=parse_float_list,
        default=parse_float_list("0,1,2,5,10,20,50"),
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
    if not args.cost_bps:
        parser.error("--cost-bps must include at least one value.")
    if any(cost_bps < 0.0 for cost_bps in args.cost_bps):
        parser.error("--cost-bps values must be greater than or equal to 0.")
    if len(set(args.cost_bps)) != len(args.cost_bps):
        parser.error("--cost-bps values must be unique.")
    if args.max_epochs <= 0:
        parser.error("--max-epochs must be positive.")
    if args.patience <= 0:
        parser.error("--patience must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.label_horizon_days < 0:
        parser.error("--label-horizon-days must be greater than or equal to 0.")

    return args


def _format_cost_bps(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


if __name__ == "__main__":
    raise SystemExit(run_cost_sensitivity_report_from_args())
