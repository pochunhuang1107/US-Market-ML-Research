"""Run the selected Tiingo-only QQQ defensive timing replay bundle."""

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
    DEFENSIVE_TIMING_SUBPERIODS,
    REQUIRED_SYMBOLS,
    SELECTED_QQQ_DEFENSIVE_TIMING_STRATEGY_NAME,
    QQQDefensiveTimingStrategySpec,
    available_symbol_summary,
    default_qqq_defensive_timing_selected_replay_spec,
    prepare_qqq_timing_prices,
    run_qqq_defensive_timing_selected_replay,
)

DEFAULT_DATASET_GLOB = (
    "/Users/casper/Projects/Research/Market Data/data/research/"
    "etf_daily_prices_tiingo_only/**/*.parquet"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/replays/tiingo_qqq_sma190_defensive_replay")
DEFAULT_START_DATE = "2019-01-01"
DEFAULT_END_DATE = "2026-05-15"
DEFAULT_TRANSACTION_COST_BPS = 10.0
DEFAULT_STRATEGY_NAME = SELECTED_QQQ_DEFENSIVE_TIMING_STRATEGY_NAME
DEFAULT_SLOW_SMA_DAYS = 190
DEFAULT_BUFFER_PCT = 0.005
DEFAULT_MIN_HOLD_DAYS = 0
DEFAULT_RISK_OFF_ASSET = "SPY"

OUTPUT_PREFIX = "tiingo_qqq_sma190_defensive_replay"
SUMMARY_NAME = f"{OUTPUT_PREFIX}_summary.csv"
DAILY_EQUITY_NAME = f"{OUTPUT_PREFIX}_daily_equity.csv"
TRADES_NAME = f"{OUTPUT_PREFIX}_trades.csv"
SUBPERIOD_SUMMARY_NAME = f"{OUTPUT_PREFIX}_subperiod_summary.csv"
DECISION_MATRIX_NAME = f"{OUTPUT_PREFIX}_decision_matrix.csv"
REPORT_NAME = f"{OUTPUT_PREFIX}_report.md"
PREFLIGHT_MD_NAME = f"{OUTPUT_PREFIX}_preflight.md"

DATASET_SOURCE_LABEL = "tiingo_only"
STUDY_STAGE = "qqq_defensive_timing_selected_replay"

SUMMARY_COLUMNS: tuple[str, ...] = (
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
    "exposure_rate_to_qqq",
    "exposure_rate_to_spy",
    "total_return",
    "qqq_buy_hold_total_return",
    "spy_buy_hold_total_return",
    "strategy_vs_qqq",
    "strategy_vs_spy",
    "annualized_return",
    "daily_volatility",
    "sharpe_like",
    "max_drawdown",
    "qqq_buy_hold_max_drawdown",
    "spy_buy_hold_max_drawdown",
    "drawdown_improvement_vs_qqq",
    "return_to_drawdown",
    "qqq_buy_hold_return_to_drawdown",
    "transaction_count",
    "average_turnover",
    "estimated_transaction_cost",
    "qqq_return_capture",
    "beats_qqq_return",
    "beats_qqq_return_to_drawdown",
    "materially_improves_qqq_drawdown",
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

SUBPERIOD_SUMMARY_COLUMNS: tuple[str, ...] = (
    "strategy_name",
    "period_name",
    "period_start",
    "period_end",
    "strategy_total_return",
    "qqq_buy_hold_total_return",
    "spy_buy_hold_total_return",
    "strategy_vs_qqq",
    "strategy_vs_spy",
    "strategy_max_drawdown",
    "qqq_max_drawdown",
    "spy_max_drawdown",
    "exposure_rate_to_qqq",
    "exposure_rate_to_spy",
    "transaction_count",
)

DECISION_MATRIX_COLUMNS: tuple[str, ...] = (
    "strategy_name",
    "decision",
    "research_status",
    "robustness_decision",
    "decision_reason",
    "next_action",
    "production_ready",
    "qqq_return_capture",
    "strategy_vs_qqq",
    "drawdown_improvement_vs_qqq",
    "return_to_drawdown",
    "qqq_buy_hold_return_to_drawdown",
    "beats_qqq_return",
    "beats_qqq_return_to_drawdown",
    "materially_improves_qqq_drawdown",
    "subperiod_count",
    "positive_strategy_vs_qqq_periods",
    "drawdown_improvement_periods",
    "severe_underperformance_periods",
    "min_strategy_vs_qqq",
    "avg_strategy_vs_qqq",
)

NEAR_QQQ_RETURN_CAPTURE = 0.95
NEAR_QQQ_RETURN_GAP = -0.10
SEVERE_SUBPERIOD_UNDERPERFORMANCE = -0.15


@dataclass(frozen=True)
class TiingoQQQDefensiveTimingSelectedReplayArtifacts:
    """CSV-ready outputs for the selected defensive replay bundle."""

    summary: pd.DataFrame
    daily_equity: pd.DataFrame
    trades: pd.DataFrame
    subperiod_summary: pd.DataFrame
    decision_matrix: pd.DataFrame
    warnings: tuple[str, ...]


def run_tiingo_qqq_defensive_timing_selected_replay_from_args(
    argv: list[str] | None = None,
) -> int:
    """Run or dry-run the selected Tiingo-only QQQ defensive timing replay."""

    args = _parse_args(argv)
    selected_spec = _selected_spec_from_args(args)
    market_data = read_local_market_data(args.market_dataset_glob)
    prices = prepare_qqq_timing_prices(
        market_data,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    outputs = _output_paths(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_preflight_markdown(
        path=outputs[PREFLIGHT_MD_NAME],
        market_dataset_glob=args.market_dataset_glob,
        output_dir=args.output_dir,
        prices=prices,
        selected_spec=selected_spec,
        start_date=args.start_date,
        end_date=args.end_date,
        transaction_cost_bps=args.transaction_cost_bps,
        dry_run=args.dry_run,
    )
    _print_preflight(
        market_dataset_glob=args.market_dataset_glob,
        output_dir=args.output_dir,
        prices=prices,
        selected_spec=selected_spec,
        transaction_cost_bps=args.transaction_cost_bps,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        return 0

    artifacts = build_tiingo_qqq_defensive_timing_selected_replay(
        market_data=market_data,
        selected_spec=selected_spec,
        start_date=args.start_date,
        end_date=args.end_date,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    write_tiingo_qqq_defensive_timing_selected_replay_artifacts(
        artifacts=artifacts,
        output_dir=args.output_dir,
    )

    print()
    print("Tiingo QQQ defensive timing selected replay artifacts written.")
    for output_name in (
        SUMMARY_NAME,
        DAILY_EQUITY_NAME,
        TRADES_NAME,
        SUBPERIOD_SUMMARY_NAME,
        DECISION_MATRIX_NAME,
        PREFLIGHT_MD_NAME,
    ):
        print(f"{output_name}: {outputs[output_name]}")
    if artifacts.warnings:
        print("warnings:")
        for warning in artifacts.warnings:
            print(f"  - {warning}")
    if not artifacts.decision_matrix.empty:
        preview_cols = [
            "strategy_name",
            "decision",
            "research_status",
            "robustness_decision",
            "qqq_return_capture",
            "strategy_vs_qqq",
            "drawdown_improvement_vs_qqq",
        ]
        print()
        print(artifacts.decision_matrix.loc[:, preview_cols].to_string(index=False))
    return 0


def build_tiingo_qqq_defensive_timing_selected_replay(
    *,
    market_data: pd.DataFrame,
    selected_spec: QQQDefensiveTimingStrategySpec | None = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
) -> TiingoQQQDefensiveTimingSelectedReplayArtifacts:
    """Build all selected replay artifact tables from local market data."""

    spec = selected_spec or default_qqq_defensive_timing_selected_replay_spec()
    replay = run_qqq_defensive_timing_selected_replay(
        market_data,
        start_date=start_date,
        end_date=end_date,
        transaction_cost_bps=transaction_cost_bps,
        selected_spec=spec,
    )
    summary = _build_summary(
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
    subperiod_summary = _build_subperiod_summary(
        replay.subperiod_results,
        selected_strategy_name=spec.strategy_name,
    )
    decision_matrix = _build_decision_matrix(summary, subperiod_summary)
    return TiingoQQQDefensiveTimingSelectedReplayArtifacts(
        summary=summary,
        daily_equity=daily_equity,
        trades=trades,
        subperiod_summary=subperiod_summary,
        decision_matrix=decision_matrix,
        warnings=replay.warnings,
    )


def write_tiingo_qqq_defensive_timing_selected_replay_artifacts(
    *,
    artifacts: TiingoQQQDefensiveTimingSelectedReplayArtifacts,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write selected replay CSV artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = _output_paths(output_dir)
    artifacts.summary.to_csv(outputs[SUMMARY_NAME], index=False)
    artifacts.daily_equity.to_csv(outputs[DAILY_EQUITY_NAME], index=False)
    artifacts.trades.to_csv(outputs[TRADES_NAME], index=False)
    artifacts.subperiod_summary.to_csv(outputs[SUBPERIOD_SUMMARY_NAME], index=False)
    artifacts.decision_matrix.to_csv(outputs[DECISION_MATRIX_NAME], index=False)
    return outputs


def _build_summary(
    results: pd.DataFrame,
    *,
    selected_strategy_name: str,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    selected_rows = results.loc[results["strategy_name"].astype(str).eq(selected_strategy_name)]
    if selected_rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    selected = selected_rows.iloc[0].to_dict()
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
        "transaction_cost_bps": transaction_cost_bps,
        "start_date": selected.get("start_date"),
        "end_date": selected.get("end_date"),
        "trading_days": selected.get("trading_days"),
        "exposure_rate_to_qqq": selected.get("exposure_rate_to_qqq"),
        "exposure_rate_to_spy": exposure_rate_to_spy,
        "total_return": selected.get("total_return"),
        "qqq_buy_hold_total_return": selected.get("qqq_buy_hold_total_return"),
        "spy_buy_hold_total_return": selected.get("spy_buy_hold_total_return"),
        "strategy_vs_qqq": selected.get("strategy_vs_qqq"),
        "strategy_vs_spy": selected.get("strategy_vs_spy"),
        "annualized_return": selected.get("annualized_return"),
        "daily_volatility": selected.get("daily_volatility"),
        "sharpe_like": selected.get("sharpe_like"),
        "max_drawdown": selected.get("max_drawdown"),
        "qqq_buy_hold_max_drawdown": selected.get("qqq_buy_hold_max_drawdown"),
        "spy_buy_hold_max_drawdown": selected.get("spy_buy_hold_max_drawdown"),
        "drawdown_improvement_vs_qqq": selected.get("drawdown_improvement_vs_qqq"),
        "return_to_drawdown": selected.get("return_to_drawdown"),
        "qqq_buy_hold_return_to_drawdown": selected.get(
            "qqq_buy_hold_return_to_drawdown"
        ),
        "transaction_count": selected.get("transaction_count"),
        "average_turnover": selected.get("average_turnover"),
        "estimated_transaction_cost": selected.get("estimated_transaction_cost"),
        "qqq_return_capture": qqq_return_capture,
        "beats_qqq_return": bool(selected.get("beats_qqq_return", False)),
        "beats_qqq_return_to_drawdown": bool(
            selected.get("beats_qqq_return_to_drawdown", False)
        ),
        "materially_improves_qqq_drawdown": bool(
            selected.get("materially_improves_qqq_drawdown", False)
        ),
    }
    return pd.DataFrame([row], columns=SUMMARY_COLUMNS)


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


def _build_subperiod_summary(
    subperiod_results: pd.DataFrame,
    *,
    selected_strategy_name: str,
) -> pd.DataFrame:
    if subperiod_results.empty:
        return pd.DataFrame(columns=SUBPERIOD_SUMMARY_COLUMNS)

    selected = subperiod_results.loc[
        subperiod_results["strategy_name"].astype(str).eq(selected_strategy_name)
    ].copy()
    if selected.empty:
        return pd.DataFrame(columns=SUBPERIOD_SUMMARY_COLUMNS)

    rename_map = {
        "total_return": "strategy_total_return",
        "max_drawdown": "strategy_max_drawdown",
    }
    selected = selected.rename(columns=rename_map)
    for column in SUBPERIOD_SUMMARY_COLUMNS:
        if column not in selected.columns:
            selected[column] = np.nan
    return selected.loc[:, SUBPERIOD_SUMMARY_COLUMNS].reset_index(drop=True)


def _build_decision_matrix(
    summary: pd.DataFrame,
    subperiod_summary: pd.DataFrame,
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(columns=DECISION_MATRIX_COLUMNS)

    row = summary.iloc[0].to_dict()
    robustness = _subperiod_robustness(subperiod_summary)
    decision, reason = _decision_for_selected_replay(row, robustness)
    next_action = (
        "model-card draft; longer-history validation if data allows; walk-forward "
        "robustness; do not run feature v4, tree_screen, MLP, or production promotion "
        "from this replay"
    )
    decision_row = {
        "strategy_name": row.get("strategy_name"),
        "decision": decision,
        "research_status": "research_only_defensive_candidate",
        "robustness_decision": robustness["robustness_decision"],
        "decision_reason": reason,
        "next_action": next_action,
        "production_ready": False,
        "qqq_return_capture": row.get("qqq_return_capture"),
        "strategy_vs_qqq": row.get("strategy_vs_qqq"),
        "drawdown_improvement_vs_qqq": row.get("drawdown_improvement_vs_qqq"),
        "return_to_drawdown": row.get("return_to_drawdown"),
        "qqq_buy_hold_return_to_drawdown": row.get("qqq_buy_hold_return_to_drawdown"),
        "beats_qqq_return": row.get("beats_qqq_return"),
        "beats_qqq_return_to_drawdown": row.get("beats_qqq_return_to_drawdown"),
        "materially_improves_qqq_drawdown": row.get("materially_improves_qqq_drawdown"),
        **{
            key: robustness[key]
            for key in [
                "subperiod_count",
                "positive_strategy_vs_qqq_periods",
                "drawdown_improvement_periods",
                "severe_underperformance_periods",
                "min_strategy_vs_qqq",
                "avg_strategy_vs_qqq",
            ]
        },
    }
    return pd.DataFrame([decision_row], columns=DECISION_MATRIX_COLUMNS)


def _subperiod_robustness(subperiod_summary: pd.DataFrame) -> dict[str, object]:
    if subperiod_summary.empty:
        return {
            "subperiod_count": 0,
            "positive_strategy_vs_qqq_periods": 0,
            "drawdown_improvement_periods": 0,
            "severe_underperformance_periods": 0,
            "min_strategy_vs_qqq": np.nan,
            "avg_strategy_vs_qqq": np.nan,
            "robustness_decision": "unknown",
        }

    stress = subperiod_summary.loc[subperiod_summary["period_name"].ne("full")].copy()
    if stress.empty:
        stress = subperiod_summary.copy()
    strategy_vs_qqq = pd.to_numeric(stress["strategy_vs_qqq"], errors="coerce")
    drawdown_improvement = pd.to_numeric(
        stress["strategy_max_drawdown"],
        errors="coerce",
    ) - pd.to_numeric(stress["qqq_max_drawdown"], errors="coerce")
    positive_periods = int((strategy_vs_qqq > 0.0).sum())
    drawdown_periods = int((drawdown_improvement > 0.0).sum())
    severe_periods = int((strategy_vs_qqq <= SEVERE_SUBPERIOD_UNDERPERFORMANCE).sum())
    count = int(len(stress))
    if severe_periods >= 2 and positive_periods <= 1:
        robustness_decision = "mixed"
    elif drawdown_periods >= max(3, count - 1):
        robustness_decision = "robust_defensive"
    elif positive_periods >= max(2, count // 2):
        robustness_decision = "robust_return"
    else:
        robustness_decision = "weak"
    return {
        "subperiod_count": count,
        "positive_strategy_vs_qqq_periods": positive_periods,
        "drawdown_improvement_periods": drawdown_periods,
        "severe_underperformance_periods": severe_periods,
        "min_strategy_vs_qqq": float(strategy_vs_qqq.min()),
        "avg_strategy_vs_qqq": float(strategy_vs_qqq.mean()),
        "robustness_decision": robustness_decision,
    }


def _decision_for_selected_replay(
    summary_row: dict[str, object],
    robustness: dict[str, object],
) -> tuple[str, str]:
    if robustness["robustness_decision"] == "mixed":
        return (
            "mixed_research_only",
            "Subperiod replay shows severe hidden weakness; keep research-only.",
        )

    qqq_return_capture = _as_float(summary_row.get("qqq_return_capture"))
    strategy_vs_qqq = _as_float(summary_row.get("strategy_vs_qqq"))
    drawdown_improvement = _as_float(summary_row.get("drawdown_improvement_vs_qqq"))
    return_to_drawdown = _as_float(summary_row.get("return_to_drawdown"))
    qqq_return_to_drawdown = _as_float(
        summary_row.get("qqq_buy_hold_return_to_drawdown")
    )
    near_qqq_return = (
        qqq_return_capture >= NEAR_QQQ_RETURN_CAPTURE
        or strategy_vs_qqq >= NEAR_QQQ_RETURN_GAP
    )
    improves_drawdown = drawdown_improvement > 0.0
    improves_rtd = return_to_drawdown > qqq_return_to_drawdown

    if near_qqq_return and improves_drawdown and improves_rtd:
        return (
            "keep_as_best_qqq_defensive_timing_candidate",
            "Nearly matches QQQ return while improving drawdown and return-to-drawdown.",
        )
    if near_qqq_return and (improves_drawdown or improves_rtd):
        return (
            "keep_as_research_only_defensive_candidate",
            "Close to QQQ with partial defensive improvement; do not promote.",
        )
    return (
        "reject_deprioritize",
        "QQQ buy-and-hold still dominates the replay profile clearly.",
    )


def _write_preflight_markdown(
    *,
    path: Path,
    market_dataset_glob: str,
    output_dir: Path,
    prices: pd.DataFrame,
    selected_spec: QQQDefensiveTimingStrategySpec,
    start_date: str,
    end_date: str,
    transaction_cost_bps: float,
    dry_run: bool,
) -> None:
    required_present, required_missing, optional_present, optional_missing = (
        available_symbol_summary(prices)
    )
    lines = [
        "# Tiingo QQQ Defensive Timing Selected Replay Preflight",
        "",
        "## Scope",
        "",
        (
            "This selected replay is local-only, Tiingo-only, research-only, and "
            "focused on the best QQQ defensive timing candidate from the narrow sweep."
        ),
        "",
        f"- Dataset source: `{DATASET_SOURCE_LABEL}`",
        f"- Study stage: `{STUDY_STAGE}`",
        f"- Market dataset glob: `{market_dataset_glob}`",
        f"- Output directory: `{output_dir}`",
        f"- Requested date range: `{start_date}` to `{end_date}`",
        f"- Actual price range: `{prices.index.min().date()}` to `{prices.index.max().date()}`",
        f"- Trading days: `{len(prices)}`",
        f"- Transaction cost: `{transaction_cost_bps:.4f}` bps per unit turnover",
        f"- Dry run: `{str(dry_run).lower()}`",
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
        "## Symbols",
        "",
        f"- Required symbols: `{', '.join(REQUIRED_SYMBOLS)}`",
        f"- Required present: `{_join_or_none(required_present)}`",
        f"- Required missing: `{_join_or_none(required_missing)}`",
        f"- Optional present: `{_join_or_none(optional_present)}`",
        f"- Optional missing: `{_join_or_none(optional_missing)}`",
        "",
        "## Subperiods",
        "",
        *[
            f"- `{name}`: `{period_start}` to `{period_end}`"
            for name, period_start, period_end in DEFENSIVE_TIMING_SUBPERIODS
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
        "- Generated replay artifacts should remain uncommitted.",
    ]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _print_preflight(
    *,
    market_dataset_glob: str,
    output_dir: Path,
    prices: pd.DataFrame,
    selected_spec: QQQDefensiveTimingStrategySpec,
    transaction_cost_bps: float,
    dry_run: bool,
) -> None:
    required_present, required_missing, optional_present, optional_missing = (
        available_symbol_summary(prices)
    )
    print("US Market ML Tiingo QQQ defensive timing selected replay")
    print(f"dataset source: {DATASET_SOURCE_LABEL}")
    print(f"market dataset glob: {market_dataset_glob}")
    print(f"output directory: {output_dir}")
    print(f"price range: {prices.index.min().date()} to {prices.index.max().date()}")
    print(f"trading days: {len(prices)}")
    print(f"strategy_name: {selected_spec.strategy_name}")
    print(f"family: {selected_spec.family}")
    print(f"risk_on_asset: {selected_spec.risk_on_asset}")
    print(f"risk_off_asset: {selected_spec.risk_off_asset}")
    print(f"signal_name: {selected_spec.signal_name}")
    print(f"slow_sma_days: {selected_spec.slow_sma_days}")
    print(f"buffer_pct: {selected_spec.buffer_pct:g}")
    print(f"min_hold_days: {selected_spec.min_hold_days}")
    print(f"transaction_cost_bps: {transaction_cost_bps:g}")
    print(f"required symbols: {', '.join(REQUIRED_SYMBOLS)}")
    print(f"required symbols present: {_join_or_none(required_present)}")
    print(f"required symbols missing: {_join_or_none(required_missing)}")
    print(f"optional symbols present: {_join_or_none(optional_present)}")
    print(f"missing optional symbols: {_join_or_none(optional_missing)}")
    print("benchmarks included: qqq_buy_hold, spy_buy_hold, cash")
    print("timing convention: close T signal applies from the next trading day")
    print("No S3/cloud/GPU, no tree_screen, no tree models, no MLP, no feature v4.")
    print("No production signal export. Production-ready status: false")
    print("Market Data repo mutation: false")
    if dry_run:
        print("DRY_RUN: preflight only; replay outputs were not generated.")


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
        SUMMARY_NAME: output_dir / SUMMARY_NAME,
        DAILY_EQUITY_NAME: output_dir / DAILY_EQUITY_NAME,
        TRADES_NAME: output_dir / TRADES_NAME,
        SUBPERIOD_SUMMARY_NAME: output_dir / SUBPERIOD_SUMMARY_NAME,
        DECISION_MATRIX_NAME: output_dir / DECISION_MATRIX_NAME,
        REPORT_NAME: output_dir / REPORT_NAME,
        PREFLIGHT_MD_NAME: output_dir / PREFLIGHT_MD_NAME,
    }


def _join_or_none(values: Sequence[str]) -> str:
    return ", ".join(values) if values else "none"


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the selected Tiingo-only QQQ defensive timing replay."
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
        "--dry-run",
        action="store_true",
        default=os.environ.get("DRY_RUN", "0") == "1",
    )
    return parser.parse_args(argv)


run_tiingo_qqq_defensive_selected_replay_from_args = (
    run_tiingo_qqq_defensive_timing_selected_replay_from_args
)


if __name__ == "__main__":
    raise SystemExit(run_tiingo_qqq_defensive_timing_selected_replay_from_args())
