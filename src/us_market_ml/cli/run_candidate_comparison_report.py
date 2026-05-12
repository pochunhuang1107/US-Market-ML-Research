"""Generate local candidate comparison reports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from us_market_ml.backtest import (
    BenchmarkConfig,
    SignalBacktestConfig,
    SignalSmoothingConfig,
    TopKSignalConfig,
    build_buy_and_hold_benchmark,
    build_equal_weight_benchmark,
    stitch_test_equity_curves,
    summarize_backtest,
    summarize_equity_series,
)
from us_market_ml.cli.run_benchmark_report import (
    parse_comma_separated_symbols,
    parse_hidden_units,
)
from us_market_ml.cli.run_walk_forward_benchmark_report import parse_walk_forward_fold
from us_market_ml.data import read_local_market_data
from us_market_ml.experiments import (
    BaselineE2EExperimentConfig,
    BaselineE2EExperimentResult,
    BaselineExperimentConfig,
    MLPE2EExperimentConfig,
    MLPE2EExperimentResult,
    MLPExperimentConfig,
    run_baseline_e2e_experiment,
    run_mlp_e2e_experiment,
)
from us_market_ml.models import BaselineModelConfig, MLPRegressorConfig
from us_market_ml.splits import PurgeConfig, WalkForwardFold
from us_market_ml.visualization import plot_equity_curves

FOLD_SUMMARY_COLUMNS = [
    "candidate_name",
    "candidate_family",
    "model_type",
    "fold_name",
    "test_start",
    "test_end",
    "selected_top_k",
    "hidden_units",
    "learning_rate",
    "dropout",
    "random_state",
    "strategy_total_return",
    "strategy_max_drawdown",
    "strategy_final_equity",
    "strategy_total_transaction_cost",
    "strategy_mean_daily_turnover",
    "epochs_trained",
]


@dataclass(frozen=True)
class CandidateConfig:
    """One named strategy candidate selected for side-by-side comparison."""

    family: str
    name: str
    top_k: int
    random_state: int
    model_type: str | None = None
    hidden_units: tuple[int, ...] | None = None
    learning_rate: float | None = None
    dropout: float | None = None


def run_candidate_comparison_report_from_args(argv: list[str] | None = None) -> int:
    """Run selected local candidates and export an OOS comparison report."""

    args = _parse_args(argv)
    market_data = read_local_market_data(args.data_path)
    folds = list(args.fold)

    candidate_curves: list[pd.DataFrame] = []
    fold_summaries: list[pd.DataFrame] = []
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

        candidate_curves.append(_candidate_curve_to_long(stitched_strategy, candidate.name))
        fold_summaries.append(_build_candidate_fold_summary(candidate, result, folds))

    if expected_dates is None or expected_dates.empty:
        raise ValueError("No candidate test equity curve rows were produced.")

    oos_start = pd.Timestamp(expected_dates.min())
    oos_end = pd.Timestamp(expected_dates.max())
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

    combined_equity = pd.concat(
        [*candidate_curves, *(_benchmark_curve_to_long(curve) for curve in benchmark_curves)],
        ignore_index=True,
    ).sort_values(by=["date", "series_name"], kind="mergesort")
    combined_equity = combined_equity.reset_index(drop=True)
    metrics = summarize_equity_series(combined_equity)
    candidate_rank = _build_candidate_rank_table(metrics, args.candidate)
    fold_summary = pd.DataFrame(
        [
            row
            for candidate_fold_summary in fold_summaries
            for row in candidate_fold_summary.to_dict("records")
        ],
        columns=FOLD_SUMMARY_COLUMNS,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / f"{args.report_name}_combined_equity.csv"
    metrics_path = output_dir / f"{args.report_name}_metrics.csv"
    candidate_rank_path = output_dir / f"{args.report_name}_candidate_rank.csv"
    fold_summary_path = output_dir / f"{args.report_name}_fold_summary.csv"
    plot_path = output_dir / f"{args.report_name}_equity.png"
    combined_equity.to_csv(combined_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    candidate_rank.to_csv(candidate_rank_path, index=False)
    fold_summary.to_csv(fold_summary_path, index=False)
    plot_equity_curves(
        combined_equity,
        plot_path,
        title="Candidate Strategy Comparison vs Benchmarks",
    )

    print("US Market ML candidate comparison report")
    print(f"data path: {args.data_path}")
    print(f"OOS period: {oos_start.date()} to {oos_end.date()}")
    print(f"output combined equity CSV: {combined_path}")
    print(f"output metrics CSV: {metrics_path}")
    print(f"output candidate rank CSV: {candidate_rank_path}")
    print(f"output fold summary CSV: {fold_summary_path}")
    print(f"output equity PNG: {plot_path}")
    print()
    print(candidate_rank.to_string(index=False))

    return 0


def parse_candidate_config(value: str) -> CandidateConfig:
    """Parse a candidate spec from CLI text."""

    parts = [part.strip() for part in value.split(":")]
    if any(not part for part in parts):
        raise argparse.ArgumentTypeError("Candidate values must be non-empty.")

    if len(parts) == 7:
        return _parse_mlp_candidate_config(parts, original_value=value)
    if len(parts) == 5:
        return _parse_baseline_candidate_config(parts, original_value=value)
    if len(parts) == 6:
        return _parse_legacy_mlp_candidate_config(parts, original_value=value)

    if parts and parts[0] not in {"mlp", "baseline"}:
        raise argparse.ArgumentTypeError(
            "Candidate family must be one of {'mlp', 'baseline'}."
        )

    raise argparse.ArgumentTypeError(
        "Candidate must use format "
        "mlp:name:top_k:hidden_units:learning_rate:dropout:random_state, "
        "baseline:name:model_type:top_k:random_state, or legacy "
        "name:top_k:hidden_units:learning_rate:dropout:random_state."
    )


def _parse_mlp_candidate_config(
    parts: list[str],
    *,
    original_value: str,
) -> CandidateConfig:
    (
        family,
        name,
        top_k_value,
        hidden_units_value,
        learning_rate_value,
        dropout_value,
        random_state_value,
    ) = parts
    _validate_candidate_family(family)
    return _build_mlp_candidate_config(
        name=name,
        top_k_value=top_k_value,
        hidden_units_value=hidden_units_value,
        learning_rate_value=learning_rate_value,
        dropout_value=dropout_value,
        random_state_value=random_state_value,
        original_value=original_value,
    )


def _parse_legacy_mlp_candidate_config(
    parts: list[str],
    *,
    original_value: str,
) -> CandidateConfig:
    (
        name,
        top_k_value,
        hidden_units_value,
        learning_rate_value,
        dropout_value,
        random_state_value,
    ) = parts
    return _build_mlp_candidate_config(
        name=name,
        top_k_value=top_k_value,
        hidden_units_value=hidden_units_value,
        learning_rate_value=learning_rate_value,
        dropout_value=dropout_value,
        random_state_value=random_state_value,
        original_value=original_value,
    )


def _build_mlp_candidate_config(
    *,
    name: str,
    top_k_value: str,
    hidden_units_value: str,
    learning_rate_value: str,
    dropout_value: str,
    random_state_value: str,
    original_value: str,
) -> CandidateConfig:
    if not name:
        raise argparse.ArgumentTypeError("candidate name must be non-empty.")

    try:
        top_k = int(top_k_value)
        hidden_units = parse_hidden_units(hidden_units_value)
        learning_rate = float(learning_rate_value)
        dropout = float(dropout_value)
        random_state = int(random_state_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid candidate value: {original_value!r}") from exc

    if top_k <= 0:
        raise argparse.ArgumentTypeError("candidate top_k must be positive.")
    if learning_rate <= 0.0:
        raise argparse.ArgumentTypeError("candidate learning_rate must be positive.")
    if not 0.0 <= dropout < 1.0:
        raise argparse.ArgumentTypeError("candidate dropout must satisfy 0 <= dropout < 1.")

    return CandidateConfig(
        family="mlp",
        name=name,
        top_k=top_k,
        random_state=random_state,
        hidden_units=hidden_units,
        learning_rate=learning_rate,
        dropout=dropout,
    )


def _parse_baseline_candidate_config(
    parts: list[str],
    *,
    original_value: str,
) -> CandidateConfig:
    family, name, model_type, top_k_value, random_state_value = parts
    _validate_candidate_family(family)
    if not name:
        raise argparse.ArgumentTypeError("candidate name must be non-empty.")
    if not model_type:
        raise argparse.ArgumentTypeError("baseline candidate model_type must be non-empty.")

    try:
        top_k = int(top_k_value)
        random_state = int(random_state_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid candidate value: {original_value!r}") from exc

    if top_k <= 0:
        raise argparse.ArgumentTypeError("candidate top_k must be positive.")

    return CandidateConfig(
        family="baseline",
        name=name,
        top_k=top_k,
        random_state=random_state,
        model_type=model_type,
    )


def _validate_candidate_family(family: str) -> None:
    if family not in {"mlp", "baseline"}:
        raise argparse.ArgumentTypeError(
            "Candidate family must be one of {'mlp', 'baseline'}."
        )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run selected local walk-forward candidates and export an "
            "out-of-sample benchmark comparison report."
        )
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-name", default="candidate_comparison")
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
    parser.add_argument("--transaction-cost-bps", default=0.0, type=float)
    parser.add_argument("--min-holding-days", default=1, type=int)
    parser.add_argument("--switch-margin", default=0.0, type=float)
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
    if args.transaction_cost_bps < 0.0:
        parser.error("--transaction-cost-bps must be greater than or equal to 0.")
    if args.min_holding_days < 1:
        parser.error("--min-holding-days must be greater than or equal to 1.")
    if args.switch_margin < 0.0:
        parser.error("--switch-margin must be greater than or equal to 0.")

    return args


def _run_candidate_experiment(
    candidate: CandidateConfig,
    market_data: pd.DataFrame,
    folds: list[WalkForwardFold],
    args: argparse.Namespace,
) -> MLPE2EExperimentResult | BaselineE2EExperimentResult:
    if candidate.family == "mlp":
        return run_mlp_e2e_experiment(
            raw_df=market_data,
            folds=folds,
            config=_build_mlp_experiment_config(candidate, args),
        )
    if candidate.family == "baseline":
        return run_baseline_e2e_experiment(
            raw_df=market_data,
            folds=folds,
            config=_build_baseline_experiment_config(candidate, args),
        )
    raise ValueError(f"Unsupported candidate family: {candidate.family!r}")


def _build_mlp_experiment_config(
    candidate: CandidateConfig,
    args: argparse.Namespace,
) -> MLPE2EExperimentConfig:
    if (
        candidate.hidden_units is None
        or candidate.learning_rate is None
        or candidate.dropout is None
    ):
        raise ValueError("MLP candidates require hidden_units, learning_rate, and dropout.")
    return MLPE2EExperimentConfig(
        mlp_config=MLPExperimentConfig(
            target_col=args.target_col,
            model_config=MLPRegressorConfig(
                hidden_units=candidate.hidden_units,
                learning_rate=candidate.learning_rate,
                dropout=candidate.dropout,
                max_epochs=args.max_epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                random_state=candidate.random_state,
                scale_target=args.scale_target,
            ),
            purge_config=_build_purge_config(args),
        ),
        signal_config=TopKSignalConfig(top_k=candidate.top_k),
        backtest_config=SignalBacktestConfig(
            initial_equity=1.0,
            transaction_cost_bps=args.transaction_cost_bps,
        ),
        smoothing_config=_build_smoothing_config(args),
    )


def _build_experiment_config(
    candidate: CandidateConfig,
    args: argparse.Namespace,
) -> MLPE2EExperimentConfig:
    """Build an MLP experiment config for legacy MLP-only report callers."""

    return _build_mlp_experiment_config(candidate, args)


def _build_baseline_experiment_config(
    candidate: CandidateConfig,
    args: argparse.Namespace,
) -> BaselineE2EExperimentConfig:
    if candidate.model_type is None:
        raise ValueError("Baseline candidates require model_type.")
    return BaselineE2EExperimentConfig(
        baseline_config=BaselineExperimentConfig(
            target_col=args.target_col,
            model_config=BaselineModelConfig(
                model_type=candidate.model_type,
                random_state=candidate.random_state,
            ),
            purge_config=_build_purge_config(args),
        ),
        signal_config=TopKSignalConfig(top_k=candidate.top_k),
        backtest_config=SignalBacktestConfig(
            initial_equity=1.0,
            transaction_cost_bps=args.transaction_cost_bps,
        ),
        smoothing_config=_build_smoothing_config(args),
    )


def _build_smoothing_config(args: argparse.Namespace) -> SignalSmoothingConfig | None:
    min_holding_days = getattr(args, "min_holding_days", 1)
    switch_margin = getattr(args, "switch_margin", 0.0)
    if min_holding_days <= 1 and switch_margin <= 0.0:
        return None
    return SignalSmoothingConfig(
        min_holding_days=min_holding_days,
        switch_margin=switch_margin,
    )


def _build_purge_config(args: argparse.Namespace) -> PurgeConfig | None:
    if args.label_horizon_days <= 0:
        return None
    return PurgeConfig(label_horizon_days=args.label_horizon_days)


def _candidate_curve_to_long(
    stitched_strategy: pd.DataFrame,
    candidate_name: str,
) -> pd.DataFrame:
    optional_columns = [
        column
        for column in ["transaction_cost", "turnover"]
        if column in stitched_strategy.columns
    ]
    output_columns = [
        "date",
        "series_name",
        "daily_return",
        "equity",
        *optional_columns,
    ]
    return (
        stitched_strategy[["date", "portfolio_return", "equity", *optional_columns]]
        .copy()
        .rename(columns={"portfolio_return": "daily_return"})
        .assign(series_name=candidate_name)[output_columns]
    )


def _benchmark_curve_to_long(benchmark_curve: pd.DataFrame) -> pd.DataFrame:
    return (
        benchmark_curve[["date", "benchmark_name", "benchmark_return", "benchmark_equity"]]
        .copy()
        .rename(
            columns={
                "benchmark_name": "series_name",
                "benchmark_return": "daily_return",
                "benchmark_equity": "equity",
            }
        )
    )[["date", "series_name", "daily_return", "equity"]]


def _build_candidate_rank_table(
    metrics: pd.DataFrame,
    candidates: list[CandidateConfig] | list[str],
) -> pd.DataFrame:
    candidate_names = [
        candidate.name if isinstance(candidate, CandidateConfig) else candidate
        for candidate in candidates
    ]
    candidate_metrics = metrics.loc[metrics["series_name"].isin(candidate_names)].copy()
    candidate_metrics = candidate_metrics.rename(columns={"series_name": "candidate_name"})
    metadata_rows: list[dict[str, str]] = []
    for candidate in candidates:
        if isinstance(candidate, CandidateConfig):
            metadata_rows.append(
                {
                    "candidate_name": candidate.name,
                    "candidate_family": candidate.family,
                    "model_type": _candidate_model_type_label(candidate),
                }
            )
        else:
            metadata_rows.append(
                {
                    "candidate_name": candidate,
                    "candidate_family": "mlp",
                    "model_type": "mlp",
                }
            )
    candidate_metadata = pd.DataFrame(metadata_rows)
    candidate_metrics = candidate_metrics.merge(
        candidate_metadata,
        on="candidate_name",
        how="left",
        validate="one_to_one",
    )
    candidate_metrics["rank_by_total_return"] = (
        candidate_metrics["total_return"].rank(method="min", ascending=False).astype("Int64")
    )
    candidate_metrics["rank_by_sharpe_like"] = (
        candidate_metrics["sharpe_like"].rank(method="min", ascending=False).astype("Int64")
    )
    candidate_metrics["rank_by_max_drawdown"] = (
        candidate_metrics["max_drawdown"].rank(method="min", ascending=False).astype("Int64")
    )
    optional_metric_columns = [
        column
        for column in [
            "total_transaction_cost",
            "mean_daily_turnover",
            "max_daily_turnover",
        ]
        if column in candidate_metrics.columns
    ]
    return candidate_metrics[
        [
            "candidate_name",
            "candidate_family",
            "model_type",
            "total_return",
            "sharpe_like",
            "max_drawdown",
            "final_equity",
            *optional_metric_columns,
            "rank_by_total_return",
            "rank_by_sharpe_like",
            "rank_by_max_drawdown",
        ]
    ].sort_values(
        by=["rank_by_total_return", "rank_by_sharpe_like", "candidate_name"],
        kind="mergesort",
    )


def _build_candidate_fold_summary(
    candidate: CandidateConfig,
    result: MLPE2EExperimentResult | BaselineE2EExperimentResult,
    folds: list[WalkForwardFold],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    fold_results = _candidate_fold_results(candidate, result)
    fit_results = _candidate_fit_results(candidate, result)
    for fold_result, fit_result, fold in zip(fold_results, fit_results, folds, strict=True):
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
                "candidate_name": candidate.name,
                "candidate_family": candidate.family,
                "model_type": _candidate_model_type_label(candidate),
                "fold_name": fold_result.fold_name,
                "test_start": test_start,
                "test_end": test_end,
                "selected_top_k": candidate.top_k,
                "hidden_units": (
                    ",".join(str(unit) for unit in candidate.hidden_units)
                    if candidate.hidden_units is not None
                    else None
                ),
                "learning_rate": candidate.learning_rate,
                "dropout": candidate.dropout,
                "random_state": candidate.random_state,
                "strategy_total_return": metrics["total_return"],
                "strategy_max_drawdown": metrics["max_drawdown"],
                "strategy_final_equity": metrics["final_equity"],
                "strategy_total_transaction_cost": metrics.get("total_transaction_cost"),
                "strategy_mean_daily_turnover": metrics.get("mean_daily_turnover"),
                "epochs_trained": getattr(fit_result, "epochs_trained", float("nan")),
            }
        )

    return pd.DataFrame(
        rows,
        columns=FOLD_SUMMARY_COLUMNS,
    )


def _candidate_fit_results(
    candidate: CandidateConfig,
    result: MLPE2EExperimentResult | BaselineE2EExperimentResult,
) -> list[Any]:
    if candidate.family == "mlp":
        if not isinstance(result, MLPE2EExperimentResult):
            raise TypeError("MLP candidate received non-MLP result.")
        return [fold_result.fit_result for fold_result in result.mlp_result.fold_results]
    if candidate.family == "baseline":
        if not isinstance(result, BaselineE2EExperimentResult):
            raise TypeError("Baseline candidate received non-baseline result.")
        return [fold_result.fit_result for fold_result in result.baseline_result.fold_results]
    raise ValueError(f"Unsupported candidate family: {candidate.family!r}")


def _candidate_fold_results(
    candidate: CandidateConfig,
    result: MLPE2EExperimentResult | BaselineE2EExperimentResult,
) -> list[Any]:
    if candidate.family == "mlp":
        if not isinstance(result, MLPE2EExperimentResult):
            raise TypeError("MLP candidate received non-MLP result.")
        return list(result.fold_results)
    if candidate.family == "baseline":
        if not isinstance(result, BaselineE2EExperimentResult):
            raise TypeError("Baseline candidate received non-baseline result.")
        return list(result.fold_results)
    raise ValueError(f"Unsupported candidate family: {candidate.family!r}")


def _candidate_model_type_label(candidate: CandidateConfig) -> str:
    return candidate.model_type if candidate.model_type is not None else "mlp"


def _filter_oos_period(
    market_data: pd.DataFrame,
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
) -> pd.DataFrame:
    date_values = pd.to_datetime(market_data["date"])
    mask = (date_values >= oos_start) & (date_values <= oos_end)
    return market_data.loc[mask].copy()


if __name__ == "__main__":
    raise SystemExit(run_candidate_comparison_report_from_args())
