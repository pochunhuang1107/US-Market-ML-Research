"""Run a daily-level rich replay for the Tiingo selected candidate."""

from __future__ import annotations

import argparse
import os
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.backtest import (
    SignalBacktestConfig,
    SignalSmoothingConfig,
    TopKSignalConfig,
    build_holding_periods,
    calculate_max_drawdown,
)
from us_market_ml.cli.create_tiingo_selected_candidate_rich_replay_report import (
    DAILY_HOLDINGS_NAME,
    DAILY_STRATEGY_EQUITY_NAME,
    FOLD_PERFORMANCE_NAME,
    REBALANCE_LOG_NAME,
    REFERENCE_PERFORMANCE_NAME,
    REPORT_PATH_NAME,
    RICH_REPLAY_SUMMARY_NAME,
    SELECTED_SYMBOL_SUMMARY_NAME,
    TRADE_TURNOVER_DIAGNOSTICS_NAME,
    UNDERPERFORMANCE_WINDOWS_NAME,
    RichReplayReportData,
    render_tiingo_selected_candidate_rich_replay_markdown,
)
from us_market_ml.cli.run_tiingo_modern_baseline_sweep import (
    ALLOCATION_MODE,
    DATASET_SOURCE_LABEL,
    DEFAULT_DATASET_GLOB,
    DEFAULT_MAX_WORKERS,
    DEFAULT_MIN_HISTORY_ROWS,
    DEFAULT_MIN_TRAIN_ELIGIBLE_ROWS,
    FEATURE_SELECTION_SCOPE,
    FEATURE_SET_VERSION,
    INCLUDE_PARTIAL_FINAL_TEST,
    LABEL_HORIZON_DAYS,
    MIN_PARTIAL_TEST_DAYS,
    OLD_BRIDGED_DATASET_GLOB,
    RANDOM_STATE,
    TARGET_COL,
    UNIVERSE_MODE,
    _format_command,
    _parse_date,
    _parse_max_workers,
    _parse_non_negative_float_values,
    _parse_positive_int,
    _parse_positive_int_values,
)
from us_market_ml.data import SupervisedDatasetConfig, read_local_market_data
from us_market_ml.experiments import (
    BaselineE2EExperimentConfig,
    BaselineE2EExperimentResult,
    BaselineExperimentConfig,
    run_baseline_e2e_experiment,
)
from us_market_ml.models import BaselineModelConfig
from us_market_ml.splits import WalkForwardFold, make_sliding_window_folds

DEFAULT_OUTPUT_DIR = Path("artifacts/replays/tiingo_selected_candidate_rich_replay")
DEFAULT_START_DATE = "2019-01-01"
DEFAULT_END_DATE = "2026-05-15"
DEFAULT_TRAIN_WINDOW_YEARS = (3, 5)
DEFAULT_VALIDATION_WINDOW_MONTHS = (12,)
DEFAULT_TEST_WINDOW_MONTHS = (3, 6, 12)
DEFAULT_STEP_MONTHS = 3
DEFAULT_TRANSACTION_COST_BPS = 10.0
DEFAULT_BENCHMARK_SYMBOLS = ("SPY", "QQQ")

MODEL_FAMILY = "baseline"
UNIVERSE_NAME = "tiingo_selected_candidate_rich_replay_dynamic"

DAILY_STRATEGY_EQUITY_COLUMNS = [
    "date",
    "fold",
    "test_start",
    "test_end",
    "strategy_equity",
    "strategy_daily_return",
    "strategy_drawdown",
    "spy_equity",
    "spy_daily_return",
    "spy_drawdown",
    "qqq_equity",
    "qqq_daily_return",
    "qqq_drawdown",
]
DAILY_HOLDINGS_COLUMNS = [
    "date",
    "fold",
    "symbol",
    "target_weight",
    "prediction",
    "signal_rank",
    "selected",
    "rebalance_flag",
]
REBALANCE_LOG_COLUMNS = [
    "date",
    "fold",
    "selected_symbols",
    "previous_symbols",
    "added_symbols",
    "removed_symbols",
    "turnover",
    "transaction_cost",
]
TRADE_TURNOVER_DIAGNOSTICS_COLUMNS = [
    "fold",
    "test_start",
    "test_end",
    "rebalance_count",
    "average_turnover",
    "max_turnover",
    "total_transaction_cost",
    "average_holding_days_if_available",
]
FOLD_PERFORMANCE_COLUMNS = [
    "fold",
    "train_window_years",
    "validation_window_months",
    "test_window_months",
    "test_start",
    "test_end",
    "strategy_total_return",
    "spy_total_return",
    "qqq_total_return",
    "strategy_vs_spy",
    "strategy_vs_qqq",
    "strategy_sharpe_like",
    "spy_sharpe_like",
    "qqq_sharpe_like",
    "strategy_max_drawdown",
    "spy_max_drawdown",
    "qqq_max_drawdown",
    "total_transaction_cost",
    "mean_daily_turnover",
]
UNDERPERFORMANCE_WINDOWS_COLUMNS = [
    "fold",
    "benchmark_symbol",
    "test_start",
    "test_end",
    "strategy_total_return",
    "benchmark_total_return",
    "excess_return_vs_benchmark",
    "underperformed_benchmark",
    "underperformance_severity",
]
SELECTED_SYMBOL_SUMMARY_COLUMNS = [
    "symbol",
    "selected_days",
    "selected_fold_count",
    "avg_target_weight",
    "avg_prediction",
    "avg_signal_rank",
    "first_selected_date",
    "last_selected_date",
]
REFERENCE_PERFORMANCE_COLUMNS = [
    "reference_role",
    "reference_label",
    "model_type",
    "top_k",
    "min_holding_days",
    "switch_margin",
    "fold_count",
    "average_total_return",
    "average_sharpe_like",
    "average_max_drawdown",
    "worst_max_drawdown",
    "average_total_transaction_cost",
    "average_mean_daily_turnover",
]


@dataclass(frozen=True)
class RichReplayCandidateSpec:
    """One selected/referenced candidate configuration."""

    role: str
    label: str
    model_type: str
    top_k: int
    min_holding_days: int
    switch_margin: float


@dataclass(frozen=True)
class RichReplayFoldPlan:
    """A replay fold plus the window metadata used to create it."""

    fold: WalkForwardFold
    train_window_years: int
    validation_window_months: int
    test_window_months: int


@dataclass(frozen=True)
class TiingoSelectedCandidateRichReplayArtifacts:
    """All daily-level replay artifact tables."""

    daily_strategy_equity: pd.DataFrame
    daily_holdings: pd.DataFrame
    rebalance_log: pd.DataFrame
    trade_turnover_diagnostics: pd.DataFrame
    fold_performance: pd.DataFrame
    underperformance_windows: pd.DataFrame
    selected_symbol_summary: pd.DataFrame
    rich_replay_summary: pd.DataFrame
    reference_performance: pd.DataFrame
    warnings: tuple[str, ...]


PRIMARY_SPEC = RichReplayCandidateSpec(
    role="selected_candidate",
    label="ridge / top5 / hold20 / margin0.005 / cost10",
    model_type="ridge",
    top_k=5,
    min_holding_days=20,
    switch_margin=0.005,
)
HISTORICAL_CHAMPION_SPEC = RichReplayCandidateSpec(
    role="historical_champion",
    label="linear_regression / top1 / hold10 / margin0.002 / cost10",
    model_type="linear_regression",
    top_k=1,
    min_holding_days=10,
    switch_margin=0.002,
)
RETURN_REFERENCE_SPEC = RichReplayCandidateSpec(
    role="return_reference",
    label="ridge / top1 / hold20 / margin0.005 / cost10",
    model_type="ridge",
    top_k=1,
    min_holding_days=20,
    switch_margin=0.005,
)
DRAWDOWN_GUARDRAIL_SPEC = RichReplayCandidateSpec(
    role="drawdown_guardrail",
    label="ridge / top5 / hold5 / margin0.002 / cost10",
    model_type="ridge",
    top_k=5,
    min_holding_days=5,
    switch_margin=0.002,
)
REFERENCE_SPECS = (
    HISTORICAL_CHAMPION_SPEC,
    RETURN_REFERENCE_SPEC,
    DRAWDOWN_GUARDRAIL_SPEC,
)


def run_tiingo_selected_candidate_rich_replay_from_args(
    argv: list[str] | None = None,
) -> int:
    """Run the selected-candidate rich replay and write all artifacts."""

    args = _parse_args(argv)
    fold_plans = build_tiingo_selected_candidate_rich_replay_folds(
        start_date=args.start_date,
        end_date=args.end_date,
        train_window_years_values=args.train_window_years,
        validation_window_months_values=args.validation_window_months,
        test_window_months_values=args.test_window_months,
        step_months=args.step_months,
    )
    if not fold_plans:
        raise SystemExit("No Tiingo selected-candidate rich replay folds were generated.")

    _print_plan(args=args, fold_plans=fold_plans, dry_run=args.dry_run)
    if args.dry_run:
        return 0

    try:
        artifacts = build_tiingo_selected_candidate_rich_replay(
            market_dataset_glob=args.market_dataset_glob,
            start_date=args.start_date,
            end_date=args.end_date,
            train_window_years=args.train_window_years,
            validation_window_months=args.validation_window_months,
            test_window_months=args.test_window_months,
            step_months=args.step_months,
            transaction_cost_bps=args.transaction_cost_bps,
            include_references=args.include_references,
            min_history_rows=args.min_history_rows,
            min_train_eligible_rows=args.min_train_eligible_rows,
        )
    except KeyboardInterrupt:
        print("Interrupt received; Tiingo selected-candidate rich replay stopped cleanly.")
        return 130

    outputs = write_tiingo_selected_candidate_rich_replay_artifacts(
        artifacts=artifacts,
        output_dir=args.output_dir,
    )

    print()
    print("Rich replay artifacts written:")
    for name, path in outputs.items():
        print(f"{name}: {path}")
    if artifacts.warnings:
        print()
        print("Warnings:")
        for warning in artifacts.warnings:
            print(f"- {warning}")
    print()
    print(artifacts.rich_replay_summary.to_string(index=False))
    return 0


def build_tiingo_selected_candidate_rich_replay_folds(
    *,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    train_window_years_values: Sequence[int] = DEFAULT_TRAIN_WINDOW_YEARS,
    validation_window_months_values: Sequence[int] = DEFAULT_VALIDATION_WINDOW_MONTHS,
    test_window_months_values: Sequence[int] = DEFAULT_TEST_WINDOW_MONTHS,
    step_months: int = DEFAULT_STEP_MONTHS,
) -> list[RichReplayFoldPlan]:
    """Build the selected-candidate replay rolling window grid."""

    fold_plans: list[RichReplayFoldPlan] = []
    for train_window_years in train_window_years_values:
        for validation_window_months in validation_window_months_values:
            for test_window_months in test_window_months_values:
                folds = make_sliding_window_folds(
                    data_start=start_date,
                    data_end=end_date,
                    train_window_years=train_window_years,
                    validation_window_months=validation_window_months,
                    test_window_months=test_window_months,
                    step_months=step_months,
                    include_partial_final_test=INCLUDE_PARTIAL_FINAL_TEST,
                    min_partial_test_days=MIN_PARTIAL_TEST_DAYS,
                )
                for fold in folds:
                    fold_plans.append(
                        RichReplayFoldPlan(
                            fold=fold,
                            train_window_years=train_window_years,
                            validation_window_months=validation_window_months,
                            test_window_months=test_window_months,
                        )
                    )
    return fold_plans


def build_tiingo_selected_candidate_rich_replay(
    *,
    market_dataset_glob: str | Path = DEFAULT_DATASET_GLOB,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    train_window_years: Sequence[int] = DEFAULT_TRAIN_WINDOW_YEARS,
    validation_window_months: Sequence[int] = DEFAULT_VALIDATION_WINDOW_MONTHS,
    test_window_months: Sequence[int] = DEFAULT_TEST_WINDOW_MONTHS,
    step_months: int = DEFAULT_STEP_MONTHS,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
    include_references: bool = True,
    min_history_rows: int = DEFAULT_MIN_HISTORY_ROWS,
    min_train_eligible_rows: int = DEFAULT_MIN_TRAIN_ELIGIBLE_ROWS,
) -> TiingoSelectedCandidateRichReplayArtifacts:
    """Run/replay the selected candidate and build daily-level artifacts."""

    fold_plans = build_tiingo_selected_candidate_rich_replay_folds(
        start_date=start_date,
        end_date=end_date,
        train_window_years_values=train_window_years,
        validation_window_months_values=validation_window_months,
        test_window_months_values=test_window_months,
        step_months=step_months,
    )
    if not fold_plans:
        raise ValueError("At least one fold is required for the rich replay.")

    raw_df = read_local_market_data(market_dataset_glob)
    folds = [fold_plan.fold for fold_plan in fold_plans]
    primary_result = _run_candidate_replay(
        raw_df=raw_df,
        folds=folds,
        spec=PRIMARY_SPEC,
        transaction_cost_bps=transaction_cost_bps,
        min_history_rows=min_history_rows,
    )
    primary_artifacts = _build_primary_artifacts(
        raw_df=raw_df,
        result=primary_result,
        fold_plans=fold_plans,
        transaction_cost_bps=transaction_cost_bps,
    )

    reference_performance = pd.DataFrame(columns=REFERENCE_PERFORMANCE_COLUMNS)
    if include_references:
        reference_performance = _build_reference_performance(
            raw_df=raw_df,
            folds=folds,
            specs=REFERENCE_SPECS,
            transaction_cost_bps=transaction_cost_bps,
            min_history_rows=min_history_rows,
        )

    rich_replay_summary = _build_rich_replay_summary(
        daily_strategy_equity=primary_artifacts.daily_strategy_equity,
        daily_holdings=primary_artifacts.daily_holdings,
        rebalance_log=primary_artifacts.rebalance_log,
        fold_performance=primary_artifacts.fold_performance,
        underperformance_windows=primary_artifacts.underperformance_windows,
        reference_performance=reference_performance,
        train_window_years=train_window_years,
        validation_window_months=validation_window_months,
        test_window_months=test_window_months,
        transaction_cost_bps=transaction_cost_bps,
        min_history_rows=min_history_rows,
        min_train_eligible_rows=min_train_eligible_rows,
    )

    return TiingoSelectedCandidateRichReplayArtifacts(
        daily_strategy_equity=primary_artifacts.daily_strategy_equity,
        daily_holdings=primary_artifacts.daily_holdings,
        rebalance_log=primary_artifacts.rebalance_log,
        trade_turnover_diagnostics=primary_artifacts.trade_turnover_diagnostics,
        fold_performance=primary_artifacts.fold_performance,
        underperformance_windows=primary_artifacts.underperformance_windows,
        selected_symbol_summary=primary_artifacts.selected_symbol_summary,
        rich_replay_summary=rich_replay_summary,
        reference_performance=reference_performance,
        warnings=tuple(_build_warnings(primary_artifacts.fold_performance)),
    )


def write_tiingo_selected_candidate_rich_replay_artifacts(
    *,
    artifacts: TiingoSelectedCandidateRichReplayArtifacts,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write all rich replay CSVs and Markdown report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        DAILY_STRATEGY_EQUITY_NAME: output_dir / DAILY_STRATEGY_EQUITY_NAME,
        DAILY_HOLDINGS_NAME: output_dir / DAILY_HOLDINGS_NAME,
        REBALANCE_LOG_NAME: output_dir / REBALANCE_LOG_NAME,
        TRADE_TURNOVER_DIAGNOSTICS_NAME: output_dir / TRADE_TURNOVER_DIAGNOSTICS_NAME,
        FOLD_PERFORMANCE_NAME: output_dir / FOLD_PERFORMANCE_NAME,
        UNDERPERFORMANCE_WINDOWS_NAME: output_dir / UNDERPERFORMANCE_WINDOWS_NAME,
        SELECTED_SYMBOL_SUMMARY_NAME: output_dir / SELECTED_SYMBOL_SUMMARY_NAME,
        RICH_REPLAY_SUMMARY_NAME: output_dir / RICH_REPLAY_SUMMARY_NAME,
        REFERENCE_PERFORMANCE_NAME: output_dir / REFERENCE_PERFORMANCE_NAME,
        REPORT_PATH_NAME: output_dir / REPORT_PATH_NAME,
    }
    artifacts.daily_strategy_equity.to_csv(outputs[DAILY_STRATEGY_EQUITY_NAME], index=False)
    artifacts.daily_holdings.to_csv(outputs[DAILY_HOLDINGS_NAME], index=False)
    artifacts.rebalance_log.to_csv(outputs[REBALANCE_LOG_NAME], index=False)
    artifacts.trade_turnover_diagnostics.to_csv(
        outputs[TRADE_TURNOVER_DIAGNOSTICS_NAME],
        index=False,
    )
    artifacts.fold_performance.to_csv(outputs[FOLD_PERFORMANCE_NAME], index=False)
    artifacts.underperformance_windows.to_csv(
        outputs[UNDERPERFORMANCE_WINDOWS_NAME],
        index=False,
    )
    artifacts.selected_symbol_summary.to_csv(outputs[SELECTED_SYMBOL_SUMMARY_NAME], index=False)
    artifacts.rich_replay_summary.to_csv(outputs[RICH_REPLAY_SUMMARY_NAME], index=False)
    artifacts.reference_performance.to_csv(outputs[REFERENCE_PERFORMANCE_NAME], index=False)
    report_data = RichReplayReportData(
        daily_strategy_equity=artifacts.daily_strategy_equity,
        daily_holdings=artifacts.daily_holdings,
        rebalance_log=artifacts.rebalance_log,
        trade_turnover_diagnostics=artifacts.trade_turnover_diagnostics,
        fold_performance=artifacts.fold_performance,
        underperformance_windows=artifacts.underperformance_windows,
        selected_symbol_summary=artifacts.selected_symbol_summary,
        rich_replay_summary=artifacts.rich_replay_summary,
        reference_performance=artifacts.reference_performance,
    )
    outputs[REPORT_PATH_NAME].write_text(
        render_tiingo_selected_candidate_rich_replay_markdown(report_data),
        encoding="utf-8",
    )
    return outputs


@dataclass(frozen=True)
class _PrimaryArtifacts:
    daily_strategy_equity: pd.DataFrame
    daily_holdings: pd.DataFrame
    rebalance_log: pd.DataFrame
    trade_turnover_diagnostics: pd.DataFrame
    fold_performance: pd.DataFrame
    underperformance_windows: pd.DataFrame
    selected_symbol_summary: pd.DataFrame


def _run_candidate_replay(
    *,
    raw_df: pd.DataFrame,
    folds: Sequence[WalkForwardFold],
    spec: RichReplayCandidateSpec,
    transaction_cost_bps: float,
    min_history_rows: int,
) -> BaselineE2EExperimentResult:
    config = BaselineE2EExperimentConfig(
        baseline_config=BaselineExperimentConfig(
            target_col=TARGET_COL,
            model_config=BaselineModelConfig(
                model_type=spec.model_type,
                random_state=RANDOM_STATE,
            ),
            dataset_config=SupervisedDatasetConfig(
                forward_horizons=(5, 10, LABEL_HORIZON_DAYS),
                benchmark_symbol="SPY",
                excess_return_horizons=(LABEL_HORIZON_DAYS,),
                rank_target_col=TARGET_COL,
                rank_col=f"rank_{TARGET_COL}",
                feature_set_version=FEATURE_SET_VERSION,
            ),
            prediction_col="prediction",
            universe_mode=UNIVERSE_MODE,
            min_history_rows=min_history_rows,
            feature_selection_scope=FEATURE_SELECTION_SCOPE,
            min_feature_non_null_ratio=0.0,
        ),
        signal_config=TopKSignalConfig(
            prediction_col="prediction",
            top_k=spec.top_k,
            allocation_mode=ALLOCATION_MODE,
        ),
        backtest_config=SignalBacktestConfig(
            initial_equity=1.0,
            transaction_cost_bps=transaction_cost_bps,
        ),
        smoothing_config=SignalSmoothingConfig(
            min_holding_days=spec.min_holding_days,
            switch_margin=spec.switch_margin,
        ),
    )
    return run_baseline_e2e_experiment(raw_df, list(folds), config)


def _build_primary_artifacts(
    *,
    raw_df: pd.DataFrame,
    result: BaselineE2EExperimentResult,
    fold_plans: Sequence[RichReplayFoldPlan],
    transaction_cost_bps: float,
) -> _PrimaryArtifacts:
    fold_plan_by_name = {fold_plan.fold.name: fold_plan for fold_plan in fold_plans}
    daily_strategy_equity = _build_daily_strategy_equity(
        raw_df=raw_df,
        result=result,
        fold_plan_by_name=fold_plan_by_name,
    )
    rebalance_log = _build_rebalance_log(result)
    daily_holdings = _build_daily_holdings(result, rebalance_log)
    trade_turnover_diagnostics = _build_trade_turnover_diagnostics(
        result=result,
        fold_plan_by_name=fold_plan_by_name,
        rebalance_log=rebalance_log,
        daily_holdings=daily_holdings,
    )
    fold_performance = _build_fold_performance(
        result=result,
        fold_plan_by_name=fold_plan_by_name,
        daily_strategy_equity=daily_strategy_equity,
    )
    underperformance_windows = _build_underperformance_windows(fold_performance)
    selected_symbol_summary = _build_selected_symbol_summary(daily_holdings)

    # Keep the variable visible for readers checking the selected cost path.
    _ = transaction_cost_bps
    return _PrimaryArtifacts(
        daily_strategy_equity=daily_strategy_equity,
        daily_holdings=daily_holdings,
        rebalance_log=rebalance_log,
        trade_turnover_diagnostics=trade_turnover_diagnostics,
        fold_performance=fold_performance,
        underperformance_windows=underperformance_windows,
        selected_symbol_summary=selected_symbol_summary,
    )


def _build_daily_strategy_equity(
    *,
    raw_df: pd.DataFrame,
    result: BaselineE2EExperimentResult,
    fold_plan_by_name: dict[str, RichReplayFoldPlan],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for fold_result in result.fold_results:
        fold_plan = fold_plan_by_name[fold_result.fold_name]
        strategy = fold_result.test_backtest.equity_curve.copy()
        if strategy.empty:
            continue
        strategy["date"] = pd.to_datetime(strategy["date"])
        strategy["strategy_drawdown"] = _drawdown_series(strategy["equity"])
        frame = strategy[
            ["date", "equity", "portfolio_return", "strategy_drawdown"]
        ].rename(
            columns={
                "equity": "strategy_equity",
                "portfolio_return": "strategy_daily_return",
            }
        )
        for symbol in DEFAULT_BENCHMARK_SYMBOLS:
            benchmark = _benchmark_curve_for_fold(raw_df, symbol, fold_plan.fold)
            prefix = symbol.lower()
            frame = frame.merge(
                benchmark.rename(
                    columns={
                        "benchmark_equity": f"{prefix}_equity",
                        "benchmark_daily_return": f"{prefix}_daily_return",
                        "benchmark_drawdown": f"{prefix}_drawdown",
                    }
                ),
                on="date",
                how="left",
                validate="one_to_one",
            )
        frame.insert(1, "fold", fold_result.fold_name)
        frame.insert(2, "test_start", fold_plan.fold.test_start)
        frame.insert(3, "test_end", fold_plan.fold.test_end)
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=DAILY_STRATEGY_EQUITY_COLUMNS)
    return (
        pd.concat(frames, ignore_index=True)
        .loc[:, DAILY_STRATEGY_EQUITY_COLUMNS]
        .sort_values(["fold", "date"], kind="mergesort")
        .reset_index(drop=True)
    )


def _benchmark_curve_for_fold(
    raw_df: pd.DataFrame,
    symbol: str,
    fold: WalkForwardFold,
) -> pd.DataFrame:
    benchmark = raw_df.loc[
        (raw_df["symbol"].astype(str) == symbol)
        & (pd.to_datetime(raw_df["date"]) >= fold.test_start)
        & (pd.to_datetime(raw_df["date"]) <= fold.test_end),
        ["date", "close"],
    ].copy()
    if benchmark.empty:
        return pd.DataFrame(
            columns=["date", "benchmark_daily_return", "benchmark_equity", "benchmark_drawdown"]
        )
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    benchmark = (
        benchmark.sort_values("date", kind="mergesort")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    benchmark["benchmark_daily_return"] = benchmark["close"].pct_change().fillna(0.0)
    benchmark["benchmark_equity"] = (1.0 + benchmark["benchmark_daily_return"]).cumprod()
    benchmark["benchmark_drawdown"] = _drawdown_series(benchmark["benchmark_equity"])
    return benchmark[
        ["date", "benchmark_daily_return", "benchmark_equity", "benchmark_drawdown"]
    ]


def _build_daily_holdings(
    result: BaselineE2EExperimentResult,
    rebalance_log: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    rebalance_dates = {
        (row.fold, pd.Timestamp(row.date))
        for row in rebalance_log[["fold", "date"]].itertuples(index=False)
    }
    for fold_result in result.fold_results:
        signals = fold_result.test_signals.copy()
        if signals.empty:
            continue
        signals["selected"] = signals["selected"].fillna(False).astype(bool)
        selected = signals.loc[signals["selected"]].copy()
        if selected.empty:
            continue
        selected["date"] = pd.to_datetime(selected["date"])
        selected["fold"] = fold_result.fold_name
        selected["rebalance_flag"] = selected.apply(
            lambda row: (row["fold"], pd.Timestamp(row["date"])) in rebalance_dates,
            axis=1,
        )
        frames.append(
            selected[
                [
                    "date",
                    "fold",
                    "symbol",
                    "target_weight",
                    "prediction",
                    "signal_rank",
                    "selected",
                    "rebalance_flag",
                ]
            ]
        )
    if not frames:
        return pd.DataFrame(columns=DAILY_HOLDINGS_COLUMNS)
    return (
        pd.concat(frames, ignore_index=True)
        .loc[:, DAILY_HOLDINGS_COLUMNS]
        .sort_values(["fold", "date", "signal_rank", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )


def _build_rebalance_log(result: BaselineE2EExperimentResult) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold_result in result.fold_results:
        signals = fold_result.test_signals.copy()
        equity_curve = fold_result.test_backtest.equity_curve.copy()
        if signals.empty:
            continue
        signals["date"] = pd.to_datetime(signals["date"])
        signals["selected"] = signals["selected"].fillna(False).astype(bool)
        equity_by_date = _equity_cost_turnover_by_date(equity_curve)
        previous_symbols: tuple[str, ...] = tuple()
        for date_value, date_group in signals.groupby("date", sort=True, dropna=False):
            selected_symbols = _selected_symbols_for_date(date_group)
            previous_symbol_set = set(previous_symbols)
            selected_symbol_set = set(selected_symbols)
            added = tuple(symbol for symbol in selected_symbols if symbol not in previous_symbol_set)
            removed = tuple(symbol for symbol in previous_symbols if symbol not in selected_symbol_set)
            turnover, transaction_cost = equity_by_date.get(
                pd.Timestamp(date_value),
                (np.nan, np.nan),
            )
            changed = selected_symbol_set != previous_symbol_set
            if changed or _positive_number(turnover):
                rows.append(
                    {
                        "date": date_value,
                        "fold": fold_result.fold_name,
                        "selected_symbols": ",".join(selected_symbols),
                        "previous_symbols": ",".join(previous_symbols),
                        "added_symbols": ",".join(added),
                        "removed_symbols": ",".join(removed),
                        "turnover": turnover,
                        "transaction_cost": transaction_cost,
                    }
                )
            previous_symbols = selected_symbols
    if not rows:
        return pd.DataFrame(columns=REBALANCE_LOG_COLUMNS)
    return (
        pd.DataFrame(rows, columns=REBALANCE_LOG_COLUMNS)
        .sort_values(["fold", "date"], kind="mergesort")
        .reset_index(drop=True)
    )


def _selected_symbols_for_date(date_group: pd.DataFrame) -> tuple[str, ...]:
    selected = date_group.loc[date_group["selected"]].copy()
    if selected.empty:
        return tuple()
    selected["signal_rank"] = pd.to_numeric(selected["signal_rank"], errors="coerce")
    return tuple(
        selected.sort_values(["signal_rank", "symbol"], kind="mergesort")["symbol"]
        .astype(str)
        .tolist()
    )


def _equity_cost_turnover_by_date(equity_curve: pd.DataFrame) -> dict[pd.Timestamp, tuple[float, float]]:
    if equity_curve.empty:
        return {}
    out = equity_curve.copy()
    out["date"] = pd.to_datetime(out["date"])
    return {
        pd.Timestamp(row.date): (float(row.turnover), float(row.transaction_cost))
        for row in out[["date", "turnover", "transaction_cost"]].itertuples(index=False)
    }


def _build_trade_turnover_diagnostics(
    *,
    result: BaselineE2EExperimentResult,
    fold_plan_by_name: dict[str, RichReplayFoldPlan],
    rebalance_log: pd.DataFrame,
    daily_holdings: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rebalance_counts = (
        rebalance_log.groupby("fold", sort=False).size().to_dict()
        if not rebalance_log.empty
        else {}
    )
    for fold_result in result.fold_results:
        fold_plan = fold_plan_by_name[fold_result.fold_name]
        equity_curve = fold_result.test_backtest.equity_curve
        turnover_values = pd.to_numeric(equity_curve.get("turnover"), errors="coerce")
        transaction_cost_values = pd.to_numeric(
            equity_curve.get("transaction_cost"),
            errors="coerce",
        )
        fold_holdings = daily_holdings.loc[daily_holdings["fold"].eq(fold_result.fold_name)]
        average_holding_days = _average_holding_days(fold_holdings)
        rows.append(
            {
                "fold": fold_result.fold_name,
                "test_start": fold_plan.fold.test_start,
                "test_end": fold_plan.fold.test_end,
                "rebalance_count": int(rebalance_counts.get(fold_result.fold_name, 0)),
                "average_turnover": _mean_or_nan(turnover_values),
                "max_turnover": _max_or_nan(turnover_values),
                "total_transaction_cost": _sum_or_nan(transaction_cost_values),
                "average_holding_days_if_available": average_holding_days,
            }
        )
    return pd.DataFrame(rows, columns=TRADE_TURNOVER_DIAGNOSTICS_COLUMNS)


def _build_fold_performance(
    *,
    result: BaselineE2EExperimentResult,
    fold_plan_by_name: dict[str, RichReplayFoldPlan],
    daily_strategy_equity: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold_result in result.fold_results:
        fold_plan = fold_plan_by_name[fold_result.fold_name]
        fold_daily = daily_strategy_equity.loc[
            daily_strategy_equity["fold"].eq(fold_result.fold_name)
        ].copy()
        if fold_daily.empty:
            continue
        strategy_metrics = _series_metrics(
            fold_daily,
            return_col="strategy_daily_return",
            equity_col="strategy_equity",
            drawdown_col="strategy_drawdown",
        )
        spy_metrics = _series_metrics(
            fold_daily,
            return_col="spy_daily_return",
            equity_col="spy_equity",
            drawdown_col="spy_drawdown",
        )
        qqq_metrics = _series_metrics(
            fold_daily,
            return_col="qqq_daily_return",
            equity_col="qqq_equity",
            drawdown_col="qqq_drawdown",
        )
        equity_curve = fold_result.test_backtest.equity_curve
        rows.append(
            {
                "fold": fold_result.fold_name,
                "train_window_years": fold_plan.train_window_years,
                "validation_window_months": fold_plan.validation_window_months,
                "test_window_months": fold_plan.test_window_months,
                "test_start": fold_plan.fold.test_start,
                "test_end": fold_plan.fold.test_end,
                "strategy_total_return": strategy_metrics["total_return"],
                "spy_total_return": spy_metrics["total_return"],
                "qqq_total_return": qqq_metrics["total_return"],
                "strategy_vs_spy": strategy_metrics["total_return"]
                - spy_metrics["total_return"],
                "strategy_vs_qqq": strategy_metrics["total_return"]
                - qqq_metrics["total_return"],
                "strategy_sharpe_like": strategy_metrics["sharpe_like"],
                "spy_sharpe_like": spy_metrics["sharpe_like"],
                "qqq_sharpe_like": qqq_metrics["sharpe_like"],
                "strategy_max_drawdown": strategy_metrics["max_drawdown"],
                "spy_max_drawdown": spy_metrics["max_drawdown"],
                "qqq_max_drawdown": qqq_metrics["max_drawdown"],
                "total_transaction_cost": _sum_or_nan(equity_curve.get("transaction_cost")),
                "mean_daily_turnover": _mean_or_nan(equity_curve.get("turnover")),
            }
        )
    return pd.DataFrame(rows, columns=FOLD_PERFORMANCE_COLUMNS)


def _build_underperformance_windows(fold_performance: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in fold_performance.iterrows():
        for symbol in DEFAULT_BENCHMARK_SYMBOLS:
            prefix = symbol.lower()
            strategy_return = _as_float(row["strategy_total_return"])
            benchmark_return = _as_float(row[f"{prefix}_total_return"])
            excess = strategy_return - benchmark_return
            rows.append(
                {
                    "fold": row["fold"],
                    "benchmark_symbol": symbol,
                    "test_start": row["test_start"],
                    "test_end": row["test_end"],
                    "strategy_total_return": strategy_return,
                    "benchmark_total_return": benchmark_return,
                    "excess_return_vs_benchmark": excess,
                    "underperformed_benchmark": bool(excess < 0.0),
                    "underperformance_severity": _underperformance_severity(excess),
                }
            )
    return pd.DataFrame(rows, columns=UNDERPERFORMANCE_WINDOWS_COLUMNS)


def _build_selected_symbol_summary(daily_holdings: pd.DataFrame) -> pd.DataFrame:
    if daily_holdings.empty:
        return pd.DataFrame(columns=SELECTED_SYMBOL_SUMMARY_COLUMNS)
    holdings = daily_holdings.copy()
    holdings["date"] = pd.to_datetime(holdings["date"])
    summary = (
        holdings.groupby("symbol", sort=False, dropna=False)
        .agg(
            selected_days=("symbol", "size"),
            selected_fold_count=("fold", "nunique"),
            avg_target_weight=("target_weight", "mean"),
            avg_prediction=("prediction", "mean"),
            avg_signal_rank=("signal_rank", "mean"),
            first_selected_date=("date", "min"),
            last_selected_date=("date", "max"),
        )
        .reset_index()
    )
    return (
        summary.loc[:, SELECTED_SYMBOL_SUMMARY_COLUMNS]
        .sort_values(["selected_days", "symbol"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )


def _build_reference_performance(
    *,
    raw_df: pd.DataFrame,
    folds: Sequence[WalkForwardFold],
    specs: Sequence[RichReplayCandidateSpec],
    transaction_cost_bps: float,
    min_history_rows: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in specs:
        result = _run_candidate_replay(
            raw_df=raw_df,
            folds=folds,
            spec=spec,
            transaction_cost_bps=transaction_cost_bps,
            min_history_rows=min_history_rows,
        )
        fold_rows = [_summary_metrics_row(fold_result.test_backtest.summary_metrics) for fold_result in result.fold_results]
        frame = pd.DataFrame(fold_rows)
        rows.append(
            {
                "reference_role": spec.role,
                "reference_label": spec.label,
                "model_type": spec.model_type,
                "top_k": spec.top_k,
                "min_holding_days": spec.min_holding_days,
                "switch_margin": spec.switch_margin,
                "fold_count": int(len(frame)),
                "average_total_return": _mean_or_nan(frame.get("total_return")),
                "average_sharpe_like": _mean_or_nan(frame.get("sharpe_like")),
                "average_max_drawdown": _mean_or_nan(frame.get("max_drawdown")),
                "worst_max_drawdown": _min_or_nan(frame.get("max_drawdown")),
                "average_total_transaction_cost": _mean_or_nan(
                    frame.get("total_transaction_cost")
                ),
                "average_mean_daily_turnover": _mean_or_nan(frame.get("mean_daily_turnover")),
            }
        )
    return pd.DataFrame(rows, columns=REFERENCE_PERFORMANCE_COLUMNS)


def _build_rich_replay_summary(
    *,
    daily_strategy_equity: pd.DataFrame,
    daily_holdings: pd.DataFrame,
    rebalance_log: pd.DataFrame,
    fold_performance: pd.DataFrame,
    underperformance_windows: pd.DataFrame,
    reference_performance: pd.DataFrame,
    train_window_years: Sequence[int],
    validation_window_months: Sequence[int],
    test_window_months: Sequence[int],
    transaction_cost_bps: float,
    min_history_rows: int,
    min_train_eligible_rows: int,
) -> pd.DataFrame:
    strategy_avg = _mean_or_nan(fold_performance.get("strategy_total_return"))
    spy_avg = _mean_or_nan(fold_performance.get("spy_total_return"))
    qqq_avg = _mean_or_nan(fold_performance.get("qqq_total_return"))
    avg_vs_spy = _mean_or_nan(fold_performance.get("strategy_vs_spy"))
    avg_vs_qqq = _mean_or_nan(fold_performance.get("strategy_vs_qqq"))
    spy_under_rate = _underperformance_rate(underperformance_windows, "SPY")
    qqq_under_rate = _underperformance_rate(underperformance_windows, "QQQ")
    decision = _decision(
        strategy_average_total_return=strategy_avg,
        average_strategy_vs_spy=avg_vs_spy,
        average_strategy_vs_qqq=avg_vs_qqq,
        spy_underperformance_rate=spy_under_rate,
        qqq_underperformance_rate=qqq_under_rate,
    )
    next_action = _next_action(decision, avg_vs_qqq)
    reference_lookup = {
        row.reference_role: row
        for row in reference_performance.itertuples(index=False)
    }
    row = {
        "dataset_source": DATASET_SOURCE_LABEL,
        "model_family": MODEL_FAMILY,
        "model_type": PRIMARY_SPEC.model_type,
        "top_k": PRIMARY_SPEC.top_k,
        "allocation_mode": ALLOCATION_MODE,
        "min_holding_days": PRIMARY_SPEC.min_holding_days,
        "switch_margin": PRIMARY_SPEC.switch_margin,
        "transaction_cost_bps": transaction_cost_bps,
        "train_window_years": _join_values(train_window_years),
        "validation_window_months": _join_values(validation_window_months),
        "test_window_months": _join_values(test_window_months),
        "universe_mode": UNIVERSE_MODE,
        "feature_selection_scope": FEATURE_SELECTION_SCOPE,
        "min_history_rows": min_history_rows,
        "min_train_eligible_rows": min_train_eligible_rows,
        "feature_set_version": FEATURE_SET_VERSION,
        "target_col": TARGET_COL,
        "label_horizon_days": LABEL_HORIZON_DAYS,
        "fold_count": int(fold_performance["fold"].nunique()) if not fold_performance.empty else 0,
        "daily_equity_rows": int(len(daily_strategy_equity)),
        "daily_holding_rows": int(len(daily_holdings)),
        "rebalance_count": int(len(rebalance_log)),
        "selected_symbol_count": int(daily_holdings["symbol"].nunique()) if not daily_holdings.empty else 0,
        "strategy_average_total_return": strategy_avg,
        "spy_average_total_return": spy_avg,
        "qqq_average_total_return": qqq_avg,
        "average_strategy_vs_spy": avg_vs_spy,
        "average_strategy_vs_qqq": avg_vs_qqq,
        "strategy_average_sharpe_like": _mean_or_nan(
            fold_performance.get("strategy_sharpe_like")
        ),
        "strategy_average_max_drawdown": _mean_or_nan(
            fold_performance.get("strategy_max_drawdown")
        ),
        "strategy_worst_max_drawdown": _min_or_nan(
            fold_performance.get("strategy_max_drawdown")
        ),
        "spy_average_max_drawdown": _mean_or_nan(fold_performance.get("spy_max_drawdown")),
        "qqq_average_max_drawdown": _mean_or_nan(fold_performance.get("qqq_max_drawdown")),
        "spy_underperformance_rate": spy_under_rate,
        "qqq_underperformance_rate": qqq_under_rate,
        "historical_champion_average_total_return": _reference_value(
            reference_lookup,
            "historical_champion",
            "average_total_return",
        ),
        "return_reference_average_total_return": _reference_value(
            reference_lookup,
            "return_reference",
            "average_total_return",
        ),
        "drawdown_guardrail_average_total_return": _reference_value(
            reference_lookup,
            "drawdown_guardrail",
            "average_total_return",
        ),
        "decision": decision,
        "next_action": next_action,
        "artifact_caveats": (
            "daily replay generated from research E2E path; benchmark gates are SPY and QQQ; "
            "not production-ready"
        ),
    }
    return pd.DataFrame([row])


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local daily-level replay for the Tiingo selected candidate."
    )
    parser.add_argument("--market-dataset-glob", default=_default_market_dataset_glob())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument(
        "--max-workers",
        type=_parse_max_workers,
        default=_parse_max_workers(os.environ.get("MAX_WORKERS", str(DEFAULT_MAX_WORKERS))),
        help="Accepted for shell-runner symmetry; this rich replay currently runs locally.",
    )
    parser.add_argument("--train-window-years", nargs="+", default=_env_values("TRAIN_WINDOW_YEARS"))
    parser.add_argument(
        "--validation-window-months",
        nargs="+",
        default=_env_values("VALIDATION_WINDOW_MONTHS"),
    )
    parser.add_argument("--test-window-months", nargs="+", default=_env_values("TEST_WINDOW_MONTHS"))
    parser.add_argument("--step-months", default=os.environ.get("STEP_MONTHS"))
    parser.add_argument("--transaction-cost-bps", default=os.environ.get("TRANSACTION_COST_BPS"))
    parser.add_argument(
        "--include-references",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include historical champion, return reference, and drawdown guardrail summaries.",
    )
    parser.add_argument(
        "--min-history-rows",
        type=_parse_positive_int,
        default=DEFAULT_MIN_HISTORY_ROWS,
    )
    parser.add_argument(
        "--min-train-eligible-rows",
        type=_parse_positive_int,
        default=DEFAULT_MIN_TRAIN_ELIGIBLE_ROWS,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        args.step_months = (
            _parse_positive_int(str(args.step_months), "--step-months")
            if args.step_months is not None
            else DEFAULT_STEP_MONTHS
        )
        args.train_window_years = (
            _parse_positive_int_values(args.train_window_years, "--train-window-years")
            if args.train_window_years is not None
            else DEFAULT_TRAIN_WINDOW_YEARS
        )
        args.validation_window_months = (
            _parse_positive_int_values(
                args.validation_window_months,
                "--validation-window-months",
            )
            if args.validation_window_months is not None
            else DEFAULT_VALIDATION_WINDOW_MONTHS
        )
        args.test_window_months = (
            _parse_positive_int_values(args.test_window_months, "--test-window-months")
            if args.test_window_months is not None
            else DEFAULT_TEST_WINDOW_MONTHS
        )
        args.transaction_cost_bps = (
            _parse_non_negative_float_values(
                [str(args.transaction_cost_bps)],
                "--transaction-cost-bps",
            )[0]
            if args.transaction_cost_bps is not None
            else DEFAULT_TRANSACTION_COST_BPS
        )
        _parse_date(args.start_date, "--start-date")
        _parse_date(args.end_date, "--end-date")
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    if pd.Timestamp(args.start_date) > pd.Timestamp(args.end_date):
        parser.error("--start-date must be earlier than or equal to --end-date.")
    if str(args.market_dataset_glob) == OLD_BRIDGED_DATASET_GLOB:
        parser.error(
            "The old Tiingo/Massive bridged dataset is not allowed for this "
            "Tiingo-only rich replay. Use the Tiingo-only adjusted dataset glob."
        )
    return args


def _print_plan(
    *,
    args: argparse.Namespace,
    fold_plans: Sequence[RichReplayFoldPlan],
    dry_run: bool,
) -> None:
    output_dir = Path(args.output_dir)
    output_names = [
        DAILY_STRATEGY_EQUITY_NAME,
        DAILY_HOLDINGS_NAME,
        REBALANCE_LOG_NAME,
        TRADE_TURNOVER_DIAGNOSTICS_NAME,
        FOLD_PERFORMANCE_NAME,
        UNDERPERFORMANCE_WINDOWS_NAME,
        SELECTED_SYMBOL_SUMMARY_NAME,
        RICH_REPLAY_SUMMARY_NAME,
        REFERENCE_PERFORMANCE_NAME,
        REPORT_PATH_NAME,
    ]
    print("US Market ML Tiingo selected-candidate rich replay")
    print("local-only: true")
    print("uses S3/cloud: false")
    print("uses GPU: false")
    print("includes MLP: false")
    print("includes tree models: false")
    print(f"dry_run: {dry_run}")
    print(f"dataset glob: {args.market_dataset_glob}")
    print(f"dataset source label: {DATASET_SOURCE_LABEL}")
    print(f"start date: {args.start_date}")
    print(f"end date: {args.end_date}")
    print(f"generated fold count: {len(fold_plans)}")
    print(f"train_window_years values: {list(args.train_window_years)}")
    print(f"validation_window_months values: {list(args.validation_window_months)}")
    print(f"test_window_months values: {list(args.test_window_months)}")
    print(f"step_months: {args.step_months}")
    print(f"transaction_cost_bps: {args.transaction_cost_bps:g}")
    print(f"max_workers argument accepted: {args.max_workers}")
    print(f"include_references: {args.include_references}")
    print(f"model_family: {MODEL_FAMILY}")
    print(f"model_type: {PRIMARY_SPEC.model_type}")
    print(f"top_k: {PRIMARY_SPEC.top_k}")
    print(f"allocation_mode: {ALLOCATION_MODE}")
    print(f"min_holding_days: {PRIMARY_SPEC.min_holding_days}")
    print(f"switch_margin: {PRIMARY_SPEC.switch_margin:g}")
    print(f"universe_mode: {UNIVERSE_MODE}")
    print(f"feature_selection_scope: {FEATURE_SELECTION_SCOPE}")
    print(f"min_history_rows: {args.min_history_rows}")
    print(f"min_train_eligible_rows: {args.min_train_eligible_rows}")
    print(f"feature_set_version: {FEATURE_SET_VERSION}")
    print(f"target_col: {TARGET_COL}")
    print(f"label_horizon_days: {LABEL_HORIZON_DAYS}")
    print(f"output directory: {output_dir}")
    print("output paths:")
    for output_name in output_names:
        print(f"  {output_name}: {output_dir / output_name}")
    print()
    print("Equivalent Python command:")
    print(_format_command(_equivalent_command(args)))


def _equivalent_command(args: argparse.Namespace) -> list[str]:
    command = [
        "python",
        "-m",
        "us_market_ml.cli.run_tiingo_selected_candidate_rich_replay",
        "--market-dataset-glob",
        str(args.market_dataset_glob),
        "--output-dir",
        str(args.output_dir),
        "--start-date",
        str(args.start_date),
        "--end-date",
        str(args.end_date),
        "--max-workers",
        str(args.max_workers),
        "--step-months",
        str(args.step_months),
        "--transaction-cost-bps",
        str(args.transaction_cost_bps),
        "--min-history-rows",
        str(args.min_history_rows),
        "--min-train-eligible-rows",
        str(args.min_train_eligible_rows),
    ]
    command.extend(["--train-window-years", *[str(value) for value in args.train_window_years]])
    command.extend(
        [
            "--validation-window-months",
            *[str(value) for value in args.validation_window_months],
        ]
    )
    command.extend(["--test-window-months", *[str(value) for value in args.test_window_months]])
    command.append("--include-references" if args.include_references else "--no-include-references")
    return command


def _series_metrics(
    frame: pd.DataFrame,
    *,
    return_col: str,
    equity_col: str,
    drawdown_col: str,
) -> dict[str, float]:
    returns = pd.to_numeric(frame[return_col], errors="coerce").dropna()
    equity = pd.to_numeric(frame[equity_col], errors="coerce").dropna()
    drawdown = pd.to_numeric(frame[drawdown_col], errors="coerce").dropna()
    if equity.empty:
        return {"total_return": np.nan, "sharpe_like": np.nan, "max_drawdown": np.nan}
    mean_daily_return = float(returns.mean()) if not returns.empty else np.nan
    daily_volatility = float(returns.std()) if not returns.empty else np.nan
    sharpe_like = (
        mean_daily_return / daily_volatility
        if daily_volatility > 0.0 and not np.isnan(daily_volatility)
        else np.nan
    )
    return {
        "total_return": float(equity.iloc[-1] - 1.0),
        "sharpe_like": float(sharpe_like),
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else calculate_max_drawdown(equity),
    }


def _summary_metrics_row(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "total_return": metrics.get("total_return", np.nan),
        "sharpe_like": metrics.get("sharpe_like", np.nan),
        "max_drawdown": metrics.get("max_drawdown", np.nan),
        "total_transaction_cost": metrics.get("total_transaction_cost", np.nan),
        "mean_daily_turnover": metrics.get("mean_daily_turnover", np.nan),
    }


def _average_holding_days(fold_holdings: pd.DataFrame) -> float:
    if fold_holdings.empty:
        return np.nan
    holding_periods = build_holding_periods(fold_holdings[["date", "symbol", "target_weight"]])
    if holding_periods.empty:
        return np.nan
    return float(pd.to_numeric(holding_periods["holding_days"], errors="coerce").mean())


def _drawdown_series(equity: pd.Series) -> pd.Series:
    equity_values = pd.to_numeric(equity, errors="coerce")
    return equity_values / equity_values.cummax() - 1.0


def _underperformance_severity(excess_return: float) -> str:
    if np.isnan(excess_return) or excess_return >= 0.0:
        return "none"
    if excess_return > -0.02:
        return "mild"
    if excess_return > -0.05:
        return "moderate"
    return "severe"


def _underperformance_rate(frame: pd.DataFrame, symbol: str) -> float:
    if frame.empty:
        return np.nan
    rows = frame.loc[frame["benchmark_symbol"].eq(symbol)]
    if rows.empty:
        return np.nan
    return float(rows["underperformed_benchmark"].astype(bool).mean())


def _decision(
    *,
    strategy_average_total_return: float,
    average_strategy_vs_spy: float,
    average_strategy_vs_qqq: float,
    spy_underperformance_rate: float,
    qqq_underperformance_rate: float,
) -> str:
    if (
        average_strategy_vs_spy > 0.0
        and average_strategy_vs_qqq > 0.0
        and spy_underperformance_rate < 0.5
        and qqq_underperformance_rate < 0.5
    ):
        return "promote as current research baseline target"
    if (
        strategy_average_total_return < 0.0
        and average_strategy_vs_spy < 0.0
        and average_strategy_vs_qqq < 0.0
    ):
        return "reject/deprioritize"
    return "keep selected candidate but not target"


def _next_action(decision: str, average_strategy_vs_qqq: float) -> str:
    if decision == "promote as current research baseline target":
        return "model-card draft"
    if np.isfinite(average_strategy_vs_qqq) and average_strategy_vs_qqq < 0.0:
        return "feature engineering"
    return "feature engineering"


def _build_warnings(fold_performance: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    if fold_performance.empty:
        warnings.append("No fold performance rows were produced.")
        return warnings
    qqq_avg = _mean_or_nan(fold_performance.get("qqq_total_return"))
    strategy_avg = _mean_or_nan(fold_performance.get("strategy_total_return"))
    if qqq_avg > strategy_avg:
        warnings.append(
            "QQQ buy-and-hold average raw return is stronger than the selected candidate."
        )
    return warnings


def _reference_value(
    reference_lookup: dict[str, object],
    role: str,
    attr: str,
) -> float:
    row = reference_lookup.get(role)
    if row is None:
        return np.nan
    return _as_float(getattr(row, attr, np.nan))


def _join_values(values: Sequence[object]) -> str:
    return " ".join(str(value) for value in values)


def _default_market_dataset_glob() -> str:
    return os.environ.get("MARKET_DATASET_GLOB", DEFAULT_DATASET_GLOB)


def _env_values(name: str) -> list[str] | None:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return None
    return shlex.split(raw_value)


def _as_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number


def _positive_number(value: object) -> bool:
    number = _as_float(value)
    return bool(np.isfinite(number) and number > 0.0)


def _sum_or_nan(series: object) -> float:
    if series is None:
        return np.nan
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.sum())


def _mean_or_nan(series: object) -> float:
    if series is None:
        return np.nan
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.mean())


def _max_or_nan(series: object) -> float:
    if series is None:
        return np.nan
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.max())


def _min_or_nan(series: object) -> float:
    if series is None:
        return np.nan
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.min())


create_tiingo_selected_candidate_rich_replay_from_args = (
    run_tiingo_selected_candidate_rich_replay_from_args
)


if __name__ == "__main__":
    raise SystemExit(run_tiingo_selected_candidate_rich_replay_from_args())
