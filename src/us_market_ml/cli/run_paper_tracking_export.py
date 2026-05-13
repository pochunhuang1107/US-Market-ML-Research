"""Export research-only daily paper-tracking targets for one candidate."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from us_market_ml.cli.run_benchmark_report import parse_comma_separated_symbols
from us_market_ml.cli.run_candidate_comparison_report import (
    CandidateConfig,
    _candidate_model_type_label,
    _run_candidate_experiment,
    parse_candidate_config,
)
from us_market_ml.data import read_local_market_data
from us_market_ml.splits import WalkForwardFold

DAILY_TARGET_COLUMNS = [
    "date",
    "symbol",
    "prediction",
    "selected",
    "signal_rank",
    "target_weight",
    "close",
]

DAILY_SUMMARY_COLUMNS = [
    "date",
    "selected_symbols",
    "selected_count",
    "total_target_weight",
    "previous_selected_symbols",
    "trade_required",
    "turnover",
    "portfolio_return",
    "equity",
]

SWITCH_LOG_COLUMNS = [
    "date",
    "previous_selected_symbols",
    "selected_symbols",
    "turnover",
    "reason",
]

METRIC_COLUMNS = [
    "candidate_name",
    "candidate_family",
    "model_type",
    "tracking_start",
    "tracking_end",
    "total_return",
    "sharpe_like",
    "max_drawdown",
    "final_equity",
    "total_transaction_cost",
    "mean_daily_turnover",
    "max_daily_turnover",
]


def run_paper_tracking_export_from_args(argv: list[str] | None = None) -> int:
    """Run one selected candidate and export daily paper-tracking CSVs."""

    args = _parse_args(argv)
    market_data = read_local_market_data(args.data_path)
    fold = _build_tracking_fold(args)
    result = _run_candidate_experiment(args.candidate, market_data, [fold], args)
    fold_result = result.fold_results[0]

    daily_targets = _build_daily_targets(fold_result.test_signals)
    equity_curve = fold_result.test_backtest.equity_curve
    if daily_targets.empty:
        raise ValueError("No test signal rows were produced for the tracking window.")
    if equity_curve.empty:
        raise ValueError("No test backtest rows were produced for the tracking window.")

    daily_summary = _build_daily_summary(daily_targets, equity_curve)
    switch_log = _build_switch_log(daily_summary)
    metrics = _build_metrics(args.candidate, fold, fold_result.test_backtest.summary_metrics)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_targets_path = output_dir / f"{args.export_name}_daily_targets.csv"
    daily_summary_path = output_dir / f"{args.export_name}_daily_summary.csv"
    switch_log_path = output_dir / f"{args.export_name}_switch_log.csv"
    metrics_path = output_dir / f"{args.export_name}_metrics.csv"

    daily_targets.to_csv(daily_targets_path, index=False)
    daily_summary.to_csv(daily_summary_path, index=False)
    switch_log.to_csv(switch_log_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    print("US Market ML paper-tracking export")
    print("research-only: not live trading advice and not an order/execution system")
    print(f"data path: {args.data_path}")
    print(f"candidate: {args.candidate.name}")
    print(f"tracking period: {fold.test_start.date()} to {fold.test_end.date()}")
    print(f"benchmark symbols: {', '.join(args.benchmark_symbols)}")
    print(f"output daily targets CSV: {daily_targets_path}")
    print(f"output daily summary CSV: {daily_summary_path}")
    print(f"output switch log CSV: {switch_log_path}")
    print(f"output metrics CSV: {metrics_path}")

    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one local candidate over a tracking window and export research-only "
            "daily target, summary, switch-log, and metric CSVs."
        )
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--export-name", default="paper_tracking")
    parser.add_argument(
        "--candidate",
        type=parse_candidate_config,
        required=True,
        help=(
            "Candidate spec: "
            "baseline:name:model_type:top_k:random_state[:min_holding_days:switch_margin] "
            "or mlp:name:top_k:hidden_units:learning_rate:dropout:random_state"
            "[:min_holding_days:switch_margin]."
        ),
    )
    parser.add_argument("--train-start", type=pd.Timestamp, required=True)
    parser.add_argument("--train-end", type=pd.Timestamp, required=True)
    parser.add_argument("--validation-start", type=pd.Timestamp, required=True)
    parser.add_argument("--validation-end", type=pd.Timestamp, required=True)
    parser.add_argument("--tracking-start", type=pd.Timestamp, required=True)
    parser.add_argument("--tracking-end", type=pd.Timestamp, required=True)
    parser.add_argument(
        "--benchmark-symbols",
        type=parse_comma_separated_symbols,
        default=parse_comma_separated_symbols("SPY,QQQ"),
    )
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
    parser.add_argument("--transaction-cost-bps", default=10.0, type=float)
    args = parser.parse_args(argv)

    if args.max_epochs <= 0:
        parser.error("--max-epochs must be positive.")
    if args.patience <= 0:
        parser.error("--patience must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.label_horizon_days < 0:
        parser.error("--label-horizon-days must be greater than or equal to 0.")
    if args.transaction_cost_bps < 0.0:
        parser.error("--transaction-cost-bps must be greater than or equal to 0.")

    return args


def _build_tracking_fold(args: argparse.Namespace) -> WalkForwardFold:
    return WalkForwardFold(
        name="paper_tracking",
        train_start=args.train_start,
        train_end=args.train_end,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        test_start=args.tracking_start,
        test_end=args.tracking_end,
    )


def _build_daily_targets(test_signals: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in DAILY_TARGET_COLUMNS if column not in test_signals]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Test signals are missing required export columns: {missing}")

    daily_targets = test_signals[DAILY_TARGET_COLUMNS].copy()
    daily_targets["date"] = pd.to_datetime(daily_targets["date"])
    daily_targets["selected"] = daily_targets["selected"].fillna(False).astype(bool)
    return daily_targets.sort_values(
        by=["date", "selected", "signal_rank", "symbol"],
        ascending=[True, False, True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def _build_daily_summary(
    daily_targets: pd.DataFrame,
    equity_curve: pd.DataFrame,
) -> pd.DataFrame:
    required_equity_columns = ["date", "turnover", "portfolio_return", "equity"]
    missing_columns = [column for column in required_equity_columns if column not in equity_curve]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Test backtest is missing required export columns: {missing}")

    equity_by_date = equity_curve[required_equity_columns].copy()
    equity_by_date["date"] = pd.to_datetime(equity_by_date["date"])
    equity_by_date = equity_by_date.set_index("date")

    rows: list[dict[str, object]] = []
    previous_selected_symbols = ""
    for date_value, date_group in daily_targets.groupby("date", sort=True, dropna=False):
        date = pd.Timestamp(date_value)
        selected_mask = date_group["selected"].fillna(False).astype(bool)
        selected_group = date_group.loc[selected_mask]
        selected_symbols = _format_selected_symbols(selected_group)
        if date not in equity_by_date.index:
            raise ValueError(f"Backtest equity curve is missing tracking date: {date.date()}")
        equity_row = equity_by_date.loc[date]

        rows.append(
            {
                "date": date,
                "selected_symbols": selected_symbols,
                "selected_count": int(selected_mask.sum()),
                "total_target_weight": float(selected_group["target_weight"].sum()),
                "previous_selected_symbols": previous_selected_symbols,
                "trade_required": selected_symbols != previous_selected_symbols,
                "turnover": float(equity_row["turnover"]),
                "portfolio_return": float(equity_row["portfolio_return"]),
                "equity": float(equity_row["equity"]),
            }
        )
        previous_selected_symbols = selected_symbols

    return pd.DataFrame(rows, columns=DAILY_SUMMARY_COLUMNS)


def _format_selected_symbols(selected_group: pd.DataFrame) -> str:
    if selected_group.empty:
        return ""
    ranked = selected_group.sort_values(
        by=["signal_rank", "symbol"],
        ascending=[True, True],
        na_position="last",
        kind="mergesort",
    )
    return ",".join(ranked["symbol"].astype(str).tolist())


def _build_switch_log(daily_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row_number, row in enumerate(daily_summary.to_dict("records")):
        if row_number == 0:
            continue
        if row["selected_symbols"] == row["previous_selected_symbols"]:
            continue
        rows.append(
            {
                "date": row["date"],
                "previous_selected_symbols": row["previous_selected_symbols"],
                "selected_symbols": row["selected_symbols"],
                "turnover": row["turnover"],
                "reason": "selection_changed",
            }
        )
    return pd.DataFrame(rows, columns=SWITCH_LOG_COLUMNS)


def _build_metrics(
    candidate: CandidateConfig,
    fold: WalkForwardFold,
    summary_metrics: dict[str, float],
) -> pd.DataFrame:
    row = {
        "candidate_name": candidate.name,
        "candidate_family": candidate.family,
        "model_type": _candidate_model_type_label(candidate),
        "tracking_start": fold.test_start.date(),
        "tracking_end": fold.test_end.date(),
        "total_return": summary_metrics["total_return"],
        "sharpe_like": summary_metrics["sharpe_like"],
        "max_drawdown": summary_metrics["max_drawdown"],
        "final_equity": summary_metrics["final_equity"],
        "total_transaction_cost": summary_metrics["total_transaction_cost"],
        "mean_daily_turnover": summary_metrics["mean_daily_turnover"],
        "max_daily_turnover": summary_metrics["max_daily_turnover"],
    }
    return pd.DataFrame([row], columns=METRIC_COLUMNS)


if __name__ == "__main__":
    raise SystemExit(run_paper_tracking_export_from_args())
