"""Run the Tiingo-only QQQ defensive timing validation bundle."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.data.local_reader import read_local_market_data
from us_market_ml.qqq_timing import (
    REQUIRED_SYMBOLS,
    SELECTED_QQQ_DEFENSIVE_TIMING_STRATEGY_NAME,
    QQQDefensiveTimingStrategySpec,
    calculate_qqq_defensive_timing_validation_window_results,
    calculate_qqq_defensive_timing_walk_forward_summary,
    default_qqq_defensive_timing_selected_replay_spec,
    default_qqq_defensive_timing_validation_subperiods,
    discover_qqq_spy_price_coverage,
    generate_qqq_defensive_timing_validation_windows,
    prepare_qqq_timing_prices,
    run_qqq_defensive_timing_selected_replay,
    run_qqq_defensive_timing_strategy,
    select_qqq_spy_common_validation_window,
)

DEFAULT_DATASET_GLOB = (
    "/Users/casper/Projects/Research/Market Data/data/research/"
    "etf_daily_prices_tiingo_only/**/*.parquet"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/validation/tiingo_qqq_defensive_timing_validation")
DEFAULT_START_DATE = "auto"
DEFAULT_END_DATE = "2026-05-15"
DEFAULT_TRANSACTION_COST_BPS = 10.0
DEFAULT_STRATEGY_NAME = SELECTED_QQQ_DEFENSIVE_TIMING_STRATEGY_NAME
DEFAULT_SLOW_SMA_DAYS = 190
DEFAULT_BUFFER_PCT = 0.005
DEFAULT_MIN_HOLD_DAYS = 0
DEFAULT_RISK_OFF_ASSET = "SPY"
DEFAULT_COST_SENSITIVITY_BPS: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0)

OUTPUT_PREFIX = "tiingo_qqq_defensive_timing_validation"
COVERAGE_NAME = f"{OUTPUT_PREFIX}_coverage.csv"
FULL_PERIOD_NAME = f"{OUTPUT_PREFIX}_full_period.csv"
WALK_FORWARD_WINDOWS_NAME = f"{OUTPUT_PREFIX}_walk_forward_windows.csv"
WALK_FORWARD_SUMMARY_NAME = f"{OUTPUT_PREFIX}_walk_forward_summary.csv"
SUBPERIODS_NAME = f"{OUTPUT_PREFIX}_subperiods.csv"
COST_SENSITIVITY_NAME = f"{OUTPUT_PREFIX}_cost_sensitivity.csv"
DAILY_EQUITY_NAME = f"{OUTPUT_PREFIX}_daily_equity.csv"
TRADES_NAME = f"{OUTPUT_PREFIX}_trades.csv"
PREFLIGHT_MD_NAME = f"{OUTPUT_PREFIX}_preflight.md"
REPORT_NAME = f"{OUTPUT_PREFIX}_report.md"

DATASET_SOURCE_LABEL = "tiingo_only"
STUDY_STAGE = "qqq_defensive_timing_validation"

COVERAGE_COLUMNS: tuple[str, ...] = (
    "qqq_first_date",
    "qqq_last_date",
    "spy_first_date",
    "spy_last_date",
    "common_start_date",
    "common_end_date",
    "trading_days",
    "selected_start_date",
    "selected_end_date",
    "selected_trading_days",
    "requested_start_date",
    "requested_end_date",
    "longer_history_available",
    "coverage_note",
)

FULL_PERIOD_COLUMNS: tuple[str, ...] = (
    "strategy_name",
    "risk_on_asset",
    "risk_off_asset",
    "signal_name",
    "slow_sma_days",
    "buffer_pct",
    "min_hold_days",
    "transaction_cost_bps",
    "start_date",
    "end_date",
    "trading_days",
    "total_return",
    "annualized_return",
    "daily_volatility",
    "sharpe_like",
    "max_drawdown",
    "worst_1m_return",
    "worst_3m_return",
    "transaction_count",
    "average_turnover",
    "estimated_transaction_cost",
    "exposure_rate_to_qqq",
    "exposure_rate_to_spy",
    "qqq_buy_hold_total_return",
    "spy_buy_hold_total_return",
    "cash_total_return",
    "strategy_vs_qqq",
    "strategy_vs_spy",
    "qqq_buy_hold_max_drawdown",
    "spy_buy_hold_max_drawdown",
    "drawdown_improvement_vs_qqq",
    "return_to_drawdown",
    "qqq_buy_hold_return_to_drawdown",
    "qqq_return_capture",
    "research_status",
    "production_ready",
)

SUBPERIOD_COLUMNS: tuple[str, ...] = (
    "strategy_name",
    "period_name",
    "period_start",
    "period_end",
    "trading_days",
    "strategy_total_return",
    "qqq_buy_hold_total_return",
    "spy_buy_hold_total_return",
    "strategy_vs_qqq",
    "strategy_vs_spy",
    "strategy_max_drawdown",
    "qqq_max_drawdown",
    "spy_max_drawdown",
    "drawdown_improvement_vs_qqq",
    "return_to_drawdown",
    "qqq_return_to_drawdown",
    "transaction_count",
    "exposure_rate_to_qqq",
    "exposure_rate_to_spy",
)

COST_SENSITIVITY_COLUMNS: tuple[str, ...] = (
    "strategy_name",
    "transaction_cost_bps",
    "is_anchor_cost",
    "total_return",
    "qqq_buy_hold_total_return",
    "spy_buy_hold_total_return",
    "strategy_vs_qqq",
    "strategy_vs_spy",
    "max_drawdown",
    "drawdown_improvement_vs_qqq",
    "return_to_drawdown",
    "qqq_buy_hold_return_to_drawdown",
    "qqq_return_capture",
    "transaction_count",
    "average_turnover",
    "estimated_transaction_cost",
    "exposure_rate_to_qqq",
    "exposure_rate_to_spy",
    "decision_at_cost",
    "decision_changed_vs_anchor",
)

DAILY_EQUITY_COLUMNS: tuple[str, ...] = (
    "date",
    "strategy_name",
    "family",
    "risk_on_asset",
    "risk_off_asset",
    "signal_name",
    "slow_sma_days",
    "buffer_pct",
    "min_hold_days",
    "signal_date",
    "raw_signal_risk_on",
    "effective_signal_risk_on",
    "active_asset",
    "gross_return",
    "transaction_cost",
    "turnover",
    "daily_return",
    "equity",
    "drawdown",
)

TRADE_COLUMNS: tuple[str, ...] = (
    "date",
    "signal_date",
    "strategy_name",
    "family",
    "risk_on_asset",
    "risk_off_asset",
    "signal_name",
    "slow_sma_days",
    "buffer_pct",
    "min_hold_days",
    "previous_asset",
    "new_asset",
    "turnover",
    "estimated_transaction_cost",
    "raw_signal_risk_on",
    "effective_signal_risk_on",
    "reason",
)


@dataclass(frozen=True)
class TiingoQQQDefensiveTimingValidationArtifacts:
    """CSV-ready outputs for the QQQ defensive timing validation bundle."""

    coverage: pd.DataFrame
    full_period: pd.DataFrame
    walk_forward_windows: pd.DataFrame
    walk_forward_summary: pd.DataFrame
    subperiods: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    daily_equity: pd.DataFrame
    trades: pd.DataFrame
    warnings: tuple[str, ...]


def run_tiingo_qqq_defensive_timing_validation_from_args(
    argv: list[str] | None = None,
) -> int:
    """Run or dry-run the local Tiingo QQQ defensive timing validation bundle."""

    args = _parse_args(argv)
    selected_spec = _selected_spec_from_args(args)
    market_data = read_local_market_data(args.market_dataset_glob)
    coverage = discover_qqq_spy_price_coverage(market_data)
    window = select_qqq_spy_common_validation_window(
        coverage,
        requested_start_date=args.start_date,
        requested_end_date=args.end_date,
    )
    prices = prepare_qqq_timing_prices(
        market_data,
        start_date=window["start_date"],
        end_date=window["end_date"],
    )
    coverage_table = _build_coverage_table(
        coverage=coverage,
        window=window,
        selected_trading_days=len(prices),
    )
    rolling_windows = generate_qqq_defensive_timing_validation_windows(
        prices.index,
        start_date=window["start_date"],
        end_date=window["end_date"],
    )

    outputs = _output_paths(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_preflight_markdown(
        path=outputs[PREFLIGHT_MD_NAME],
        market_dataset_glob=args.market_dataset_glob,
        output_dir=args.output_dir,
        selected_spec=selected_spec,
        coverage=coverage_table,
        rolling_windows=rolling_windows,
        transaction_cost_bps=args.transaction_cost_bps,
        dry_run=args.dry_run,
    )
    _print_preflight(
        market_dataset_glob=args.market_dataset_glob,
        output_dir=args.output_dir,
        selected_spec=selected_spec,
        coverage=coverage_table,
        rolling_windows=rolling_windows,
        transaction_cost_bps=args.transaction_cost_bps,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        return 0

    artifacts = build_tiingo_qqq_defensive_timing_validation(
        market_data=market_data,
        selected_spec=selected_spec,
        start_date=args.start_date,
        end_date=args.end_date,
        transaction_cost_bps=args.transaction_cost_bps,
        cost_sensitivity_bps=args.cost_sensitivity_bps,
    )
    write_tiingo_qqq_defensive_timing_validation_artifacts(
        artifacts=artifacts,
        output_dir=args.output_dir,
    )

    print()
    print("Tiingo QQQ defensive timing validation artifacts written.")
    for output_name in (
        COVERAGE_NAME,
        FULL_PERIOD_NAME,
        WALK_FORWARD_WINDOWS_NAME,
        WALK_FORWARD_SUMMARY_NAME,
        SUBPERIODS_NAME,
        COST_SENSITIVITY_NAME,
        DAILY_EQUITY_NAME,
        TRADES_NAME,
        PREFLIGHT_MD_NAME,
    ):
        print(f"{output_name}: {outputs[output_name]}")
    if artifacts.warnings:
        print("warnings:")
        for warning in artifacts.warnings:
            print(f"  - {warning}")
    if not artifacts.full_period.empty:
        preview_cols = [
            "strategy_name",
            "start_date",
            "end_date",
            "total_return",
            "qqq_buy_hold_total_return",
            "strategy_vs_qqq",
            "drawdown_improvement_vs_qqq",
            "return_to_drawdown",
        ]
        print()
        print(artifacts.full_period.loc[:, preview_cols].to_string(index=False))
    return 0


def build_tiingo_qqq_defensive_timing_validation(
    *,
    market_data: pd.DataFrame,
    selected_spec: QQQDefensiveTimingStrategySpec | None = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
    cost_sensitivity_bps: Sequence[float] = DEFAULT_COST_SENSITIVITY_BPS,
) -> TiingoQQQDefensiveTimingValidationArtifacts:
    """Build all validation artifact tables from local market data."""

    spec = selected_spec or default_qqq_defensive_timing_selected_replay_spec()
    coverage = discover_qqq_spy_price_coverage(market_data)
    window = select_qqq_spy_common_validation_window(
        coverage,
        requested_start_date=start_date,
        requested_end_date=end_date,
    )
    prices = prepare_qqq_timing_prices(
        market_data,
        start_date=window["start_date"],
        end_date=window["end_date"],
    )
    coverage_table = _build_coverage_table(
        coverage=coverage,
        window=window,
        selected_trading_days=len(prices),
    )
    subperiod_defs = default_qqq_defensive_timing_validation_subperiods(
        start_date=window["start_date"],
        end_date=window["end_date"],
    )
    replay = run_qqq_defensive_timing_selected_replay(
        market_data,
        start_date=window["start_date"],
        end_date=window["end_date"],
        transaction_cost_bps=transaction_cost_bps,
        selected_spec=spec,
        subperiods=subperiod_defs,
    )
    full_period = _build_full_period(
        replay.results,
        selected_strategy_name=spec.strategy_name,
        transaction_cost_bps=transaction_cost_bps,
    )
    daily_equity = _build_daily_equity(replay.daily_equity)
    trades = _build_trades(
        replay.trades,
        daily_equity=daily_equity,
        selected_strategy_name=spec.strategy_name,
    )
    rolling_window_plan = generate_qqq_defensive_timing_validation_windows(
        prices.index,
        start_date=window["start_date"],
        end_date=window["end_date"],
    )
    walk_forward_windows = calculate_qqq_defensive_timing_validation_window_results(
        daily_equity=daily_equity,
        prices=prices,
        windows=rolling_window_plan,
        selected_strategy_name=spec.strategy_name,
    )
    walk_forward_summary = calculate_qqq_defensive_timing_walk_forward_summary(
        walk_forward_windows
    )
    subperiods = _build_subperiod_results(
        daily_equity=daily_equity,
        prices=prices,
        subperiod_defs=subperiod_defs,
        selected_strategy_name=spec.strategy_name,
    )
    cost_sensitivity = _build_cost_sensitivity(
        prices=prices,
        selected_spec=spec,
        transaction_cost_bps_values=cost_sensitivity_bps,
        anchor_transaction_cost_bps=transaction_cost_bps,
    )
    return TiingoQQQDefensiveTimingValidationArtifacts(
        coverage=coverage_table,
        full_period=full_period,
        walk_forward_windows=walk_forward_windows,
        walk_forward_summary=walk_forward_summary,
        subperiods=subperiods,
        cost_sensitivity=cost_sensitivity,
        daily_equity=daily_equity,
        trades=trades,
        warnings=replay.warnings,
    )


def write_tiingo_qqq_defensive_timing_validation_artifacts(
    *,
    artifacts: TiingoQQQDefensiveTimingValidationArtifacts,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write validation CSV artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = _output_paths(output_dir)
    artifacts.coverage.to_csv(outputs[COVERAGE_NAME], index=False)
    artifacts.full_period.to_csv(outputs[FULL_PERIOD_NAME], index=False)
    artifacts.walk_forward_windows.to_csv(outputs[WALK_FORWARD_WINDOWS_NAME], index=False)
    artifacts.walk_forward_summary.to_csv(outputs[WALK_FORWARD_SUMMARY_NAME], index=False)
    artifacts.subperiods.to_csv(outputs[SUBPERIODS_NAME], index=False)
    artifacts.cost_sensitivity.to_csv(outputs[COST_SENSITIVITY_NAME], index=False)
    artifacts.daily_equity.to_csv(outputs[DAILY_EQUITY_NAME], index=False)
    artifacts.trades.to_csv(outputs[TRADES_NAME], index=False)
    return outputs


def _build_coverage_table(
    *,
    coverage: dict[str, object],
    window: dict[str, object],
    selected_trading_days: int,
) -> pd.DataFrame:
    row = {
        **coverage,
        "selected_start_date": window["start_date"],
        "selected_end_date": window["end_date"],
        "selected_trading_days": int(selected_trading_days),
        "requested_start_date": window["requested_start_date"],
        "requested_end_date": window["requested_end_date"],
        "longer_history_available": bool(window["longer_history_available"]),
        "coverage_note": window["coverage_note"],
    }
    return pd.DataFrame([row], columns=COVERAGE_COLUMNS)


def _build_full_period(
    results: pd.DataFrame,
    *,
    selected_strategy_name: str,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=FULL_PERIOD_COLUMNS)

    selected_rows = results.loc[results["strategy_name"].astype(str).eq(selected_strategy_name)]
    if selected_rows.empty:
        return pd.DataFrame(columns=FULL_PERIOD_COLUMNS)

    selected = selected_rows.iloc[0].to_dict()
    cash_rows = results.loc[results["strategy_name"].astype(str).eq("cash")]
    cash_total_return = (
        _as_float(cash_rows.iloc[0].get("total_return")) if not cash_rows.empty else 0.0
    )
    qqq_return = _as_float(selected.get("qqq_buy_hold_total_return"))
    total_return = _as_float(selected.get("total_return"))
    qqq_return_capture = total_return / qqq_return if qqq_return != 0.0 else np.nan
    exposure_rate_to_spy = (
        _as_float(selected.get("exposure_rate_to_risk_off"))
        if str(selected.get("risk_off_asset")) == "SPY"
        else 0.0
    )
    row = {
        "strategy_name": selected.get("strategy_name"),
        "risk_on_asset": selected.get("risk_on_asset"),
        "risk_off_asset": selected.get("risk_off_asset"),
        "signal_name": selected.get("signal_name"),
        "slow_sma_days": selected.get("slow_sma_days"),
        "buffer_pct": selected.get("buffer_pct"),
        "min_hold_days": selected.get("min_hold_days"),
        "transaction_cost_bps": float(transaction_cost_bps),
        "start_date": selected.get("start_date"),
        "end_date": selected.get("end_date"),
        "trading_days": selected.get("trading_days"),
        "total_return": selected.get("total_return"),
        "annualized_return": selected.get("annualized_return"),
        "daily_volatility": selected.get("daily_volatility"),
        "sharpe_like": selected.get("sharpe_like"),
        "max_drawdown": selected.get("max_drawdown"),
        "worst_1m_return": selected.get("worst_1m_return"),
        "worst_3m_return": selected.get("worst_3m_return"),
        "transaction_count": selected.get("transaction_count"),
        "average_turnover": selected.get("average_turnover"),
        "estimated_transaction_cost": selected.get("estimated_transaction_cost"),
        "exposure_rate_to_qqq": selected.get("exposure_rate_to_qqq"),
        "exposure_rate_to_spy": exposure_rate_to_spy,
        "qqq_buy_hold_total_return": selected.get("qqq_buy_hold_total_return"),
        "spy_buy_hold_total_return": selected.get("spy_buy_hold_total_return"),
        "cash_total_return": cash_total_return,
        "strategy_vs_qqq": selected.get("strategy_vs_qqq"),
        "strategy_vs_spy": selected.get("strategy_vs_spy"),
        "qqq_buy_hold_max_drawdown": selected.get("qqq_buy_hold_max_drawdown"),
        "spy_buy_hold_max_drawdown": selected.get("spy_buy_hold_max_drawdown"),
        "drawdown_improvement_vs_qqq": selected.get("drawdown_improvement_vs_qqq"),
        "return_to_drawdown": selected.get("return_to_drawdown"),
        "qqq_buy_hold_return_to_drawdown": selected.get(
            "qqq_buy_hold_return_to_drawdown"
        ),
        "qqq_return_capture": qqq_return_capture,
        "research_status": "research_only_defensive_candidate",
        "production_ready": False,
    }
    return pd.DataFrame([row], columns=FULL_PERIOD_COLUMNS)


def _build_daily_equity(daily_equity: pd.DataFrame) -> pd.DataFrame:
    if daily_equity.empty:
        return pd.DataFrame(columns=DAILY_EQUITY_COLUMNS)

    daily = daily_equity.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["strategy_name", "date"], kind="mergesort")
    daily["drawdown"] = (
        daily.groupby("strategy_name", sort=False)["equity"]
        .transform(lambda values: values / values.cummax() - 1.0)
        .astype(float)
    )
    for column in DAILY_EQUITY_COLUMNS:
        if column not in daily.columns:
            daily[column] = np.nan
    order = {
        DEFAULT_STRATEGY_NAME: 0,
        "qqq_buy_hold": 1,
        "spy_buy_hold": 2,
        "cash": 3,
    }
    daily["_strategy_order"] = daily["strategy_name"].map(order).fillna(99)
    return (
        daily.sort_values(["date", "_strategy_order"], kind="mergesort")
        .loc[:, DAILY_EQUITY_COLUMNS]
        .reset_index(drop=True)
    )


def _build_trades(
    trades: pd.DataFrame,
    *,
    daily_equity: pd.DataFrame,
    selected_strategy_name: str,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)

    selected_trades = trades.loc[
        trades["strategy_name"].astype(str).eq(selected_strategy_name)
    ].copy()
    if selected_trades.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)

    daily = daily_equity.loc[
        daily_equity["strategy_name"].astype(str).eq(selected_strategy_name),
        ["date", "raw_signal_risk_on", "effective_signal_risk_on"],
    ].copy()
    selected_trades["date"] = pd.to_datetime(selected_trades["date"])
    daily["date"] = pd.to_datetime(daily["date"])
    selected_trades = selected_trades.merge(daily, on="date", how="left")
    selected_trades["reason"] = selected_trades["new_asset"].map(
        {
            "QQQ": "risk_on_signal_true_after_next_day_shift",
            "SPY": "risk_on_signal_false_after_next_day_shift",
            "cash": "risk_off_cash_after_next_day_shift",
        }
    )
    selected_trades["reason"] = selected_trades["reason"].fillna("active_asset_changed")
    for column in TRADE_COLUMNS:
        if column not in selected_trades.columns:
            selected_trades[column] = np.nan
    return selected_trades.loc[:, TRADE_COLUMNS].reset_index(drop=True)


def _build_subperiod_results(
    *,
    daily_equity: pd.DataFrame,
    prices: pd.DataFrame,
    subperiod_defs: Sequence[tuple[str, str, str]],
    selected_strategy_name: str,
) -> pd.DataFrame:
    if not subperiod_defs:
        return pd.DataFrame(columns=SUBPERIOD_COLUMNS)

    rows: list[dict[str, object]] = []
    for index, (period_name, period_start, period_end) in enumerate(subperiod_defs, start=1):
        trading_dates = prices.loc[
            prices.index.to_series().between(pd.Timestamp(period_start), pd.Timestamp(period_end))
        ].index
        if len(trading_dates) == 0:
            continue
        windows = pd.DataFrame(
            [
                {
                    "window_label": period_name,
                    "window_length_years": 0,
                    "step_months": 0,
                    "window_index": index,
                    "nominal_start_date": period_start,
                    "nominal_end_date": period_end,
                    "start_date": str(pd.Timestamp(trading_dates.min()).date()),
                    "end_date": str(pd.Timestamp(trading_dates.max()).date()),
                    "trading_days": int(len(trading_dates)),
                }
            ]
        )
        result = calculate_qqq_defensive_timing_validation_window_results(
            daily_equity=daily_equity,
            prices=prices,
            windows=windows,
            selected_strategy_name=selected_strategy_name,
        )
        if result.empty:
            continue
        row = result.iloc[0].to_dict()
        rows.append(
            {
                "strategy_name": row["strategy_name"],
                "period_name": period_name,
                "period_start": period_start,
                "period_end": period_end,
                "trading_days": row["trading_days"],
                "strategy_total_return": row["strategy_total_return"],
                "qqq_buy_hold_total_return": row["qqq_total_return"],
                "spy_buy_hold_total_return": row["spy_total_return"],
                "strategy_vs_qqq": row["strategy_vs_qqq"],
                "strategy_vs_spy": row["strategy_vs_spy"],
                "strategy_max_drawdown": row["strategy_max_drawdown"],
                "qqq_max_drawdown": row["qqq_max_drawdown"],
                "spy_max_drawdown": row["spy_max_drawdown"],
                "drawdown_improvement_vs_qqq": row["drawdown_improvement_vs_qqq"],
                "return_to_drawdown": row["return_to_drawdown"],
                "qqq_return_to_drawdown": row["qqq_return_to_drawdown"],
                "transaction_count": row["transaction_count"],
                "exposure_rate_to_qqq": row["exposure_rate_to_qqq"],
                "exposure_rate_to_spy": row["exposure_rate_to_spy"],
            }
        )
    return pd.DataFrame(rows, columns=SUBPERIOD_COLUMNS)


def _build_cost_sensitivity(
    *,
    prices: pd.DataFrame,
    selected_spec: QQQDefensiveTimingStrategySpec,
    transaction_cost_bps_values: Sequence[float],
    anchor_transaction_cost_bps: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    decisions: dict[float, str] = {}
    for cost_bps in transaction_cost_bps_values:
        metrics, _, _ = run_qqq_defensive_timing_strategy(
            prices,
            selected_spec,
            transaction_cost_bps=float(cost_bps),
        )
        qqq_return = _as_float(metrics.get("qqq_buy_hold_total_return"))
        total_return = _as_float(metrics.get("total_return"))
        qqq_return_capture = total_return / qqq_return if qqq_return != 0.0 else np.nan
        exposure_rate_to_spy = (
            _as_float(metrics.get("exposure_rate_to_risk_off"))
            if selected_spec.risk_off_asset == "SPY"
            else 0.0
        )
        decision = _decision_at_cost(metrics, qqq_return_capture=qqq_return_capture)
        decisions[float(cost_bps)] = decision
        rows.append(
            {
                "strategy_name": selected_spec.strategy_name,
                "transaction_cost_bps": float(cost_bps),
                "is_anchor_cost": float(cost_bps) == float(anchor_transaction_cost_bps),
                "total_return": metrics.get("total_return"),
                "qqq_buy_hold_total_return": metrics.get("qqq_buy_hold_total_return"),
                "spy_buy_hold_total_return": metrics.get("spy_buy_hold_total_return"),
                "strategy_vs_qqq": metrics.get("strategy_vs_qqq"),
                "strategy_vs_spy": metrics.get("strategy_vs_spy"),
                "max_drawdown": metrics.get("max_drawdown"),
                "drawdown_improvement_vs_qqq": metrics.get("drawdown_improvement_vs_qqq"),
                "return_to_drawdown": metrics.get("return_to_drawdown"),
                "qqq_buy_hold_return_to_drawdown": metrics.get(
                    "qqq_buy_hold_return_to_drawdown"
                ),
                "qqq_return_capture": qqq_return_capture,
                "transaction_count": metrics.get("transaction_count"),
                "average_turnover": metrics.get("average_turnover"),
                "estimated_transaction_cost": metrics.get("estimated_transaction_cost"),
                "exposure_rate_to_qqq": metrics.get("exposure_rate_to_qqq"),
                "exposure_rate_to_spy": exposure_rate_to_spy,
                "decision_at_cost": decision,
                "decision_changed_vs_anchor": False,
            }
        )

    anchor_decision = decisions.get(float(anchor_transaction_cost_bps), "")
    out = pd.DataFrame(rows, columns=COST_SENSITIVITY_COLUMNS)
    if anchor_decision:
        out["decision_changed_vs_anchor"] = out["decision_at_cost"].ne(anchor_decision)
    return out


def _decision_at_cost(
    metrics: dict[str, object],
    *,
    qqq_return_capture: float,
) -> str:
    strategy_vs_qqq = _as_float(metrics.get("strategy_vs_qqq"))
    drawdown_improvement = _as_float(metrics.get("drawdown_improvement_vs_qqq"))
    return_to_drawdown = _as_float(metrics.get("return_to_drawdown"))
    qqq_return_to_drawdown = _as_float(metrics.get("qqq_buy_hold_return_to_drawdown"))
    near_qqq = qqq_return_capture >= 0.95 or strategy_vs_qqq >= -0.10
    improves_drawdown = drawdown_improvement > 0.0
    improves_rtd = return_to_drawdown > qqq_return_to_drawdown
    if near_qqq and improves_drawdown and improves_rtd:
        return "keep_as_best_qqq_defensive_timing_candidate"
    if near_qqq and (improves_drawdown or improves_rtd):
        return "keep_research_only"
    return "reject_deprioritize"


def _write_preflight_markdown(
    *,
    path: Path,
    market_dataset_glob: str,
    output_dir: Path,
    selected_spec: QQQDefensiveTimingStrategySpec,
    coverage: pd.DataFrame,
    rolling_windows: pd.DataFrame,
    transaction_cost_bps: float,
    dry_run: bool,
) -> None:
    coverage_row = coverage.iloc[0].to_dict() if not coverage.empty else {}
    window_counts = _window_counts(rolling_windows)
    lines = [
        "# Tiingo QQQ Defensive Timing Validation Preflight",
        "",
        "## Scope",
        "",
        (
            "This validation is local-only, Tiingo-only, research-only, and uses the "
            "already selected QQQ defensive timing candidate without parameter tuning."
        ),
        "",
        f"- Dataset source: `{DATASET_SOURCE_LABEL}`",
        f"- Study stage: `{STUDY_STAGE}`",
        f"- Market dataset glob: `{market_dataset_glob}`",
        f"- Output directory: `{output_dir}`",
        f"- Requested date range: `{coverage_row.get('requested_start_date')}` to "
        f"`{coverage_row.get('requested_end_date')}`",
        f"- Selected validation range: `{coverage_row.get('selected_start_date')}` to "
        f"`{coverage_row.get('selected_end_date')}`",
        f"- Selected trading days: `{coverage_row.get('selected_trading_days')}`",
        f"- Longer-history available locally: "
        f"`{str(bool(coverage_row.get('longer_history_available'))).lower()}`",
        f"- Coverage note: {coverage_row.get('coverage_note')}",
        f"- Transaction cost: `{transaction_cost_bps:.4f}` bps per unit turnover",
        f"- Dry run: `{str(dry_run).lower()}`",
        "",
        "## Coverage",
        "",
        f"- QQQ first date: `{coverage_row.get('qqq_first_date')}`",
        f"- QQQ last date: `{coverage_row.get('qqq_last_date')}`",
        f"- SPY first date: `{coverage_row.get('spy_first_date')}`",
        f"- SPY last date: `{coverage_row.get('spy_last_date')}`",
        f"- Common start date: `{coverage_row.get('common_start_date')}`",
        f"- Common end date: `{coverage_row.get('common_end_date')}`",
        f"- Common trading days: `{coverage_row.get('trading_days')}`",
        "",
        "## Candidate Configuration",
        "",
        f"- Strategy name: `{selected_spec.strategy_name}`",
        f"- Family: `{selected_spec.family}`",
        f"- Risk-on asset: `{selected_spec.risk_on_asset}`",
        f"- Risk-off asset: `{selected_spec.risk_off_asset}`",
        f"- Signal name: `{selected_spec.signal_name}`",
        f"- Slow SMA days: `{selected_spec.slow_sma_days}`",
        f"- Buffer pct: `{selected_spec.buffer_pct}`",
        f"- Min hold days: `{selected_spec.min_hold_days}`",
        "",
        "## Rolling Windows",
        "",
        f"- Total rolling windows: `{len(rolling_windows)}`",
        *[
            f"- `{label}` windows: `{count}`"
            for label, count in sorted(window_counts.items())
        ],
        "",
        "## Timing Convention",
        "",
        (
            "Signals are computed from close data available on date T. The selected "
            "position applies starting the next trading day, and transaction costs "
            "are charged when that next-day active asset changes."
        ),
        "",
        "## Guardrails",
        "",
        "- Local-only Tiingo research input.",
        "- No S3/cloud/GPU.",
        "- No tree_screen or tree models.",
        "- No MLP.",
        "- No benchmark_gap_features_v4.",
        "- No production signal export.",
        "- No production-ready label.",
        "- Do not modify the Market Data repo.",
        "- Generated validation artifacts should remain uncommitted.",
    ]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _print_preflight(
    *,
    market_dataset_glob: str,
    output_dir: Path,
    selected_spec: QQQDefensiveTimingStrategySpec,
    coverage: pd.DataFrame,
    rolling_windows: pd.DataFrame,
    transaction_cost_bps: float,
    dry_run: bool,
) -> None:
    coverage_row = coverage.iloc[0].to_dict() if not coverage.empty else {}
    print("US Market ML Tiingo QQQ defensive timing validation")
    print(f"dataset source: {DATASET_SOURCE_LABEL}")
    print(f"market dataset glob: {market_dataset_glob}")
    print(f"output directory: {output_dir}")
    print(f"required symbols: {', '.join(REQUIRED_SYMBOLS)}")
    print(f"qqq_first_date: {coverage_row.get('qqq_first_date')}")
    print(f"qqq_last_date: {coverage_row.get('qqq_last_date')}")
    print(f"spy_first_date: {coverage_row.get('spy_first_date')}")
    print(f"spy_last_date: {coverage_row.get('spy_last_date')}")
    print(f"common_start_date: {coverage_row.get('common_start_date')}")
    print(f"common_end_date: {coverage_row.get('common_end_date')}")
    print(f"selected_start_date: {coverage_row.get('selected_start_date')}")
    print(f"selected_end_date: {coverage_row.get('selected_end_date')}")
    print(f"selected_trading_days: {coverage_row.get('selected_trading_days')}")
    print(
        "longer_history_available: "
        f"{str(bool(coverage_row.get('longer_history_available'))).lower()}"
    )
    print(f"coverage_note: {coverage_row.get('coverage_note')}")
    print(f"strategy_name: {selected_spec.strategy_name}")
    print(f"risk_on_asset: {selected_spec.risk_on_asset}")
    print(f"risk_off_asset: {selected_spec.risk_off_asset}")
    print(f"signal_name: {selected_spec.signal_name}")
    print(f"slow_sma_days: {selected_spec.slow_sma_days}")
    print(f"buffer_pct: {selected_spec.buffer_pct:g}")
    print(f"min_hold_days: {selected_spec.min_hold_days}")
    print(f"transaction_cost_bps: {transaction_cost_bps:g}")
    print(f"rolling window count: {len(rolling_windows)}")
    for label, count in sorted(_window_counts(rolling_windows).items()):
        print(f"rolling window count {label}: {count}")
    print("benchmarks included: qqq_buy_hold, spy_buy_hold, cash")
    print("timing convention: close T signal applies from the next trading day")
    print("No S3/cloud/GPU, no tree_screen, no tree models, no MLP, no feature v4.")
    print("No production signal export. Production-ready status: false")
    print("Market Data repo mutation: false")
    if dry_run:
        print("DRY_RUN: preflight only; validation CSV outputs were not generated.")


def _window_counts(rolling_windows: pd.DataFrame) -> dict[str, int]:
    if rolling_windows.empty:
        return {}
    counts = rolling_windows["window_length_years"].value_counts().sort_index()
    return {f"{int(years)}y": int(count) for years, count in counts.items()}


def _selected_spec_from_args(args: argparse.Namespace) -> QQQDefensiveTimingStrategySpec:
    return default_qqq_defensive_timing_selected_replay_spec(
        strategy_name=args.strategy_name,
        slow_sma_days=args.slow_sma_days,
        buffer_pct=args.buffer_pct,
        min_hold_days=args.min_hold_days,
        risk_off_asset=args.risk_off_asset,
    )


def _output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        COVERAGE_NAME: output_dir / COVERAGE_NAME,
        FULL_PERIOD_NAME: output_dir / FULL_PERIOD_NAME,
        WALK_FORWARD_WINDOWS_NAME: output_dir / WALK_FORWARD_WINDOWS_NAME,
        WALK_FORWARD_SUMMARY_NAME: output_dir / WALK_FORWARD_SUMMARY_NAME,
        SUBPERIODS_NAME: output_dir / SUBPERIODS_NAME,
        COST_SENSITIVITY_NAME: output_dir / COST_SENSITIVITY_NAME,
        DAILY_EQUITY_NAME: output_dir / DAILY_EQUITY_NAME,
        TRADES_NAME: output_dir / TRADES_NAME,
        PREFLIGHT_MD_NAME: output_dir / PREFLIGHT_MD_NAME,
        REPORT_NAME: output_dir / REPORT_NAME,
    }


def _parse_cost_bps(value: str) -> tuple[float, ...]:
    costs = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not costs:
        raise argparse.ArgumentTypeError("At least one transaction cost value is required.")
    if any(cost < 0.0 for cost in costs):
        raise argparse.ArgumentTypeError("Transaction cost values must be non-negative.")
    return costs


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Tiingo-only QQQ defensive timing validation bundle."
    )
    parser.add_argument(
        "--market-dataset-glob",
        default=os.environ.get("MARKET_DATASET_GLOB", DEFAULT_DATASET_GLOB),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))),
    )
    parser.add_argument("--start-date", default=os.environ.get("START_DATE", DEFAULT_START_DATE))
    parser.add_argument("--end-date", default=os.environ.get("END_DATE", DEFAULT_END_DATE))
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=float(os.environ.get("TRANSACTION_COST_BPS", DEFAULT_TRANSACTION_COST_BPS)),
    )
    parser.add_argument(
        "--strategy-name",
        default=os.environ.get("STRATEGY_NAME", DEFAULT_STRATEGY_NAME),
    )
    parser.add_argument(
        "--slow-sma-days",
        type=int,
        default=int(os.environ.get("SLOW_SMA_DAYS", DEFAULT_SLOW_SMA_DAYS)),
    )
    parser.add_argument(
        "--buffer-pct",
        type=float,
        default=float(os.environ.get("BUFFER_PCT", DEFAULT_BUFFER_PCT)),
    )
    parser.add_argument(
        "--min-hold-days",
        type=int,
        default=int(os.environ.get("MIN_HOLD_DAYS", DEFAULT_MIN_HOLD_DAYS)),
    )
    parser.add_argument(
        "--risk-off-asset",
        default=os.environ.get("RISK_OFF_ASSET", DEFAULT_RISK_OFF_ASSET),
    )
    parser.add_argument(
        "--cost-sensitivity-bps",
        type=_parse_cost_bps,
        default=_parse_cost_bps(
            os.environ.get(
                "COST_SENSITIVITY_BPS",
                ",".join(f"{cost:g}" for cost in DEFAULT_COST_SENSITIVITY_BPS),
            )
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=os.environ.get("DRY_RUN", "0") == "1",
    )
    return parser.parse_args(argv)


run_tiingo_qqq_defensive_validation_from_args = (
    run_tiingo_qqq_defensive_timing_validation_from_args
)


if __name__ == "__main__":
    raise SystemExit(run_tiingo_qqq_defensive_timing_validation_from_args())
