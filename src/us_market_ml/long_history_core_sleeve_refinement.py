"""Refined SPY core plus long-history QQQ candidate sleeve study.

This module is local research infrastructure. It reuses completed daily equity
artifacts to validate whether the prior SPY core-sleeve result survives finer
SPY core weights and more realistic rebalance accounting. It deliberately avoids
broad model sweeps, MLP batches, model cards, production signal exports, and
cloud access.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.backtest.signal_backtester import calculate_max_drawdown
from us_market_ml.long_history_candidate_deep_dive import CONFIG_COLUMNS
from us_market_ml.long_history_shortlist_validation import (
    DATASET_SOURCE_LABEL,
    PRICE_MODE,
    SUBPERIODS,
    generate_rolling_windows,
)
from us_market_ml.long_history_spy_core_sleeve import (
    SOURCE_COLUMNS,
    estimate_portfolio_cost_edge,
    extract_candidate_reference_returns,
    select_source_candidates,
)
from us_market_ml.long_history_training import END_DATE, START_DATE, TRADING_DAYS_PER_YEAR

OUTPUT_PREFIX = "long_history_qqq_core_sleeve_refinement"
DEFAULT_OUTPUT_DIR = Path("artifacts/sweeps/long_history_qqq_core_sleeve_refinement")
DEFAULT_CANDIDATE_IDS: tuple[str, ...] = ("candidate_0001", "candidate_0003")

DAILY_REBALANCED_FIXED_WEIGHTS = "daily_rebalanced_fixed_weights"
MONTHLY_REBALANCED_WEIGHTS = "monthly_rebalanced_weights"
QUARTERLY_REBALANCED_WEIGHTS = "quarterly_rebalanced_weights"
CANDIDATE_REBALANCE_ALIGNED_WEIGHTS = "candidate_rebalance_aligned_weights"
DEFAULT_PORTFOLIO_ACCOUNTING_MODES: tuple[str, ...] = (
    DAILY_REBALANCED_FIXED_WEIGHTS,
    MONTHLY_REBALANCED_WEIGHTS,
    QUARTERLY_REBALANCED_WEIGHTS,
    CANDIDATE_REBALANCE_ALIGNED_WEIGHTS,
)

DEFAULT_REFINEMENT_SPY_CORE_WEIGHTS: dict[str, tuple[float, ...]] = {
    "candidate_0001": (0.00, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 1.00),
    "candidate_0003": (0.00, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 1.00),
}
FALLBACK_REFINEMENT_SPY_CORE_WEIGHTS: tuple[float, ...] = (
    0.00,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    1.00,
)
MATERIAL_DRAWDOWN_IMPROVEMENT = 0.02
ACCEPTABLE_POSITIVE_ROLLING_RATE = 0.50

RESULTS_NAME = f"{OUTPUT_PREFIX}_results.csv"
DAILY_EQUITY_NAME = f"{OUTPUT_PREFIX}_daily_equity.csv"
DAILY_DRAWDOWN_NAME = f"{OUTPUT_PREFIX}_daily_drawdown.csv"
SUBPERIODS_NAME = f"{OUTPUT_PREFIX}_subperiods.csv"
ROLLING_WINDOWS_NAME = f"{OUTPUT_PREFIX}_rolling_windows.csv"
DECISION_MATRIX_NAME = f"{OUTPUT_PREFIX}_decision_matrix.csv"
PREFLIGHT_MD_NAME = f"{OUTPUT_PREFIX}_preflight.md"

OUTPUT_CSV_NAMES = (
    RESULTS_NAME,
    DAILY_EQUITY_NAME,
    DAILY_DRAWDOWN_NAME,
    SUBPERIODS_NAME,
    ROLLING_WINDOWS_NAME,
    DECISION_MATRIX_NAME,
)

REQUIRED_METRIC_COLUMNS = [
    "candidate_id",
    "candidate_label",
    "spy_core_weight",
    "candidate_sleeve_weight",
    "portfolio_accounting_mode",
    "portfolio_name",
    "total_return",
    "annualized_return",
    "daily_volatility",
    "sharpe_like",
    "max_drawdown",
    "return_to_drawdown",
    "qqq_buy_hold_total_return",
    "spy_buy_hold_total_return",
    "standalone_candidate_total_return",
    "strategy_vs_qqq",
    "qqq_return_capture",
    "drawdown_improvement_vs_qqq",
    "drawdown_improvement_vs_candidate",
    "return_to_drawdown_vs_qqq",
    "beats_qqq_return",
    "improves_drawdown_vs_qqq",
    "improves_return_to_drawdown_vs_qqq",
    "captures_90pct_qqq_return",
    "positive_vs_qqq_window_rate",
    "qqq_underperformance_rate",
    "worst_strategy_vs_qqq",
]

RESULT_COLUMNS = [
    "candidate_id",
    "candidate_label",
    *CONFIG_COLUMNS,
    "spy_core_weight",
    "candidate_sleeve_weight",
    "portfolio_accounting_mode",
    "portfolio_name",
    "is_core_sleeve",
    "total_return",
    "annualized_return",
    "daily_volatility",
    "sharpe_like",
    "max_drawdown",
    "return_to_drawdown",
    "qqq_buy_hold_total_return",
    "spy_buy_hold_total_return",
    "standalone_candidate_total_return",
    "strategy_vs_qqq",
    "strategy_vs_spy",
    "qqq_return_capture",
    "spy_return_capture",
    "drawdown_improvement_vs_qqq",
    "drawdown_improvement_vs_candidate",
    "return_to_drawdown_vs_qqq",
    "transaction_count",
    "average_turnover",
    "estimated_transaction_cost",
    "portfolio_rebalance_count",
    "candidate_rebalance_date_count",
    "positive_vs_qqq_window_rate",
    "qqq_underperformance_rate",
    "worst_strategy_vs_qqq",
    "cost_20bps_strategy_vs_qqq",
    "cost_30bps_strategy_vs_qqq",
    "qqq_buy_hold_max_drawdown",
    "spy_buy_hold_max_drawdown",
    "standalone_candidate_max_drawdown",
    "qqq_buy_hold_return_to_drawdown",
    "spy_buy_hold_return_to_drawdown",
    "standalone_candidate_return_to_drawdown",
    "beats_qqq_return",
    "improves_drawdown_vs_qqq",
    "improves_return_to_drawdown_vs_qqq",
    "captures_90pct_qqq_return",
    "acceptable_rolling_behavior",
    "source_strategy_vs_qqq",
    "source_return_to_drawdown",
    "source_positive_vs_qqq_window_rate",
    "source_worst_strategy_vs_qqq",
    "source_top_symbol",
    "source_top_symbol_exposure_pct",
    "source_top3_symbol_exposure_pct",
    "source_decision",
    "source_rationale",
]

DAILY_EQUITY_COLUMNS = [
    "date",
    "candidate_id",
    "candidate_label",
    "spy_core_weight",
    "candidate_sleeve_weight",
    "portfolio_accounting_mode",
    "portfolio_name",
    "strategy_name",
    "daily_return",
    "equity",
    "drawdown",
    "effective_spy_weight",
    "effective_candidate_weight",
    "is_portfolio_rebalance_date",
    "candidate_daily_return",
    "candidate_equity",
    "candidate_drawdown",
    "candidate_turnover",
    "candidate_transaction_cost",
    "qqq_daily_return",
    "qqq_equity",
    "qqq_drawdown",
    "spy_daily_return",
    "spy_equity",
    "spy_drawdown",
]

DAILY_DRAWDOWN_COLUMNS = [
    "date",
    "candidate_id",
    "candidate_label",
    "spy_core_weight",
    "candidate_sleeve_weight",
    "portfolio_accounting_mode",
    "portfolio_name",
    "strategy_name",
    "equity",
    "drawdown",
]

SUBPERIOD_COLUMNS = [
    "candidate_id",
    "candidate_label",
    "spy_core_weight",
    "candidate_sleeve_weight",
    "portfolio_accounting_mode",
    "portfolio_name",
    "period_name",
    "period_start",
    "period_end",
    "total_return",
    "qqq_total_return",
    "spy_total_return",
    "standalone_candidate_total_return",
    "strategy_vs_qqq",
    "strategy_vs_spy",
    "max_drawdown",
    "qqq_max_drawdown",
    "spy_max_drawdown",
    "standalone_candidate_max_drawdown",
    "drawdown_improvement_vs_qqq",
    "drawdown_improvement_vs_candidate",
    "return_to_drawdown",
    "qqq_return_to_drawdown",
    "spy_return_to_drawdown",
    "standalone_candidate_return_to_drawdown",
    "portfolio_rebalance_count",
]

ROLLING_WINDOW_COLUMNS = [
    "candidate_id",
    "candidate_label",
    "spy_core_weight",
    "candidate_sleeve_weight",
    "portfolio_accounting_mode",
    "portfolio_name",
    "window_label",
    "window_years",
    "window_start",
    "window_end",
    "strategy_total_return",
    "qqq_total_return",
    "spy_total_return",
    "standalone_candidate_total_return",
    "strategy_vs_qqq",
    "strategy_max_drawdown",
    "qqq_max_drawdown",
    "spy_max_drawdown",
    "standalone_candidate_max_drawdown",
    "drawdown_improvement_vs_qqq",
    "drawdown_improvement_vs_candidate",
    "underperformed_qqq",
    "materially_improved_drawdown",
]

ROLLING_SUMMARY_COLUMNS = [
    "candidate_id",
    "spy_core_weight",
    "candidate_sleeve_weight",
    "portfolio_accounting_mode",
    "portfolio_name",
    "rolling_window_count",
    "positive_vs_qqq_window_rate",
    "qqq_underperformance_rate",
    "average_strategy_vs_qqq",
    "median_strategy_vs_qqq",
    "worst_strategy_vs_qqq",
    "drawdown_improvement_window_rate",
    "average_drawdown_improvement_vs_qqq",
    "worst_window_label",
]

OBJECTIVE_COLUMNS = [
    "objective_best_total_return_drawdown_controlled",
    "objective_best_return_to_drawdown",
    "objective_best_total_return_drawdown_improvement",
    "objective_best_balanced_candidate",
    "selected_objectives",
]

DECISION_MATRIX_COLUMNS = [
    *RESULT_COLUMNS,
    *OBJECTIVE_COLUMNS,
    "decision",
    "rationale",
    "production_ready",
]


@dataclass(frozen=True)
class LongHistoryCoreSleeveRefinementArtifacts:
    """CSV-ready refined SPY core plus candidate sleeve outputs."""

    results: pd.DataFrame
    daily_equity: pd.DataFrame
    daily_drawdown: pd.DataFrame
    subperiods: pd.DataFrame
    rolling_windows: pd.DataFrame
    decision_matrix: pd.DataFrame


def generate_refinement_weight_grid(
    candidate_ids: tuple[str, ...] = DEFAULT_CANDIDATE_IDS,
    *,
    weight_map: dict[str, tuple[float, ...]] | None = None,
) -> pd.DataFrame:
    """Return the candidate-specific fine SPY core/candidate sleeve grid."""

    resolved_ids = _resolve_candidate_ids(candidate_ids)
    weights_by_candidate = weight_map or DEFAULT_REFINEMENT_SPY_CORE_WEIGHTS
    rows: list[dict[str, object]] = []
    for candidate_id in resolved_ids:
        raw_weights = weights_by_candidate.get(candidate_id, FALLBACK_REFINEMENT_SPY_CORE_WEIGHTS)
        seen: set[float] = set()
        for weight in raw_weights:
            spy_weight = round(float(weight), 10)
            if spy_weight < 0.0 or spy_weight > 1.0:
                raise ValueError("SPY core weights must be between 0 and 1.")
            if spy_weight in seen:
                continue
            seen.add(spy_weight)
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "spy_core_weight": spy_weight,
                    "candidate_sleeve_weight": round(1.0 - spy_weight, 10),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["candidate_id", "spy_core_weight", "candidate_sleeve_weight"],
    )


def resolve_portfolio_accounting_modes(
    modes: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Resolve explicit/default accounting modes while preserving order."""

    ordered = list(modes) if modes else list(DEFAULT_PORTFOLIO_ACCOUNTING_MODES)
    valid = set(DEFAULT_PORTFOLIO_ACCOUNTING_MODES)
    unknown = [mode for mode in ordered if mode not in valid]
    if unknown:
        raise ValueError(f"Unknown portfolio accounting modes: {', '.join(unknown)}")
    return tuple(dict.fromkeys(ordered))


def candidate_rebalance_dates(aligned_returns: pd.DataFrame) -> tuple[pd.Timestamp, ...]:
    """Return candidate rebalance dates inferred from positive candidate turnover."""

    if "candidate_turnover" not in aligned_returns.columns:
        return ()
    source = aligned_returns.copy()
    source["date"] = pd.to_datetime(source["date"])
    turnover = pd.to_numeric(source["candidate_turnover"], errors="coerce").fillna(0.0)
    dates = source.loc[turnover.gt(0.0), "date"].drop_duplicates().sort_values()
    return tuple(pd.Timestamp(date) for date in dates.tolist())


def accounting_mode_availability(
    *,
    daily_equity: pd.DataFrame,
    candidate_ids: tuple[str, ...] = DEFAULT_CANDIDATE_IDS,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    """Return a small table describing candidate rebalance-date availability."""

    rows: list[dict[str, object]] = []
    for candidate_id in _resolve_candidate_ids(candidate_ids):
        aligned = extract_candidate_reference_returns(
            daily_equity,
            candidate_id,
            start_date=start_date,
            end_date=end_date,
        )
        dates = candidate_rebalance_dates(aligned)
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_rebalance_dates_available": bool(dates),
                "candidate_rebalance_date_count": len(dates),
                "first_candidate_rebalance_date": str(dates[0].date()) if dates else "",
                "last_candidate_rebalance_date": str(dates[-1].date()) if dates else "",
                "candidate_rebalance_aligned_mode": "included" if dates else "skipped",
            }
        )
    return pd.DataFrame(rows)


def build_long_history_core_sleeve_refinement_artifacts(
    *,
    daily_equity: pd.DataFrame,
    deep_dive_summary: pd.DataFrame,
    deep_dive_decision_matrix: pd.DataFrame,
    candidate_ids: tuple[str, ...] = DEFAULT_CANDIDATE_IDS,
    portfolio_accounting_modes: tuple[str, ...] = DEFAULT_PORTFOLIO_ACCOUNTING_MODES,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> LongHistoryCoreSleeveRefinementArtifacts:
    """Build refined SPY core plus candidate sleeve artifacts from local CSVs."""

    source_candidates = select_source_candidates(
        deep_dive_summary=deep_dive_summary,
        deep_dive_decision_matrix=deep_dive_decision_matrix,
        candidate_ids=candidate_ids,
    )
    grid = generate_refinement_weight_grid(tuple(source_candidates["candidate_id"].astype(str)))
    requested_modes = resolve_portfolio_accounting_modes(portfolio_accounting_modes)

    result_frames: list[pd.DataFrame] = []
    daily_frames: list[pd.DataFrame] = []
    subperiod_frames: list[pd.DataFrame] = []
    rolling_frames: list[pd.DataFrame] = []

    for _, candidate in source_candidates.iterrows():
        candidate_id = str(candidate["candidate_id"])
        aligned = extract_candidate_reference_returns(
            daily_equity,
            candidate_id,
            start_date=start_date,
            end_date=end_date,
        )
        available_rebalance_dates = candidate_rebalance_dates(aligned)
        candidate_grid = grid.loc[grid["candidate_id"].astype(str).eq(candidate_id)]
        for weight_row in candidate_grid.itertuples(index=False):
            for mode in requested_modes:
                if mode == CANDIDATE_REBALANCE_ALIGNED_WEIGHTS and not available_rebalance_dates:
                    continue
                daily = build_refined_core_sleeve_daily_equity(
                    aligned_returns=aligned,
                    candidate=candidate,
                    spy_core_weight=float(weight_row.spy_core_weight),
                    candidate_sleeve_weight=float(weight_row.candidate_sleeve_weight),
                    portfolio_accounting_mode=mode,
                )
                rolling = build_rolling_window_rows(
                    daily_equity=daily,
                    start_date=start_date,
                    end_date=end_date,
                )
                rolling_summary = summarize_refinement_rolling_windows(rolling)
                result_frames.append(
                    build_result_row(
                        daily_equity=daily,
                        candidate=candidate,
                        rolling_summary=rolling_summary,
                    )
                )
                daily_frames.append(daily)
                subperiod_frames.append(build_subperiod_rows(daily_equity=daily))
                rolling_frames.append(rolling)

    results = _concat_or_empty(result_frames, RESULT_COLUMNS)
    daily_equity_out = _concat_or_empty(daily_frames, DAILY_EQUITY_COLUMNS)
    daily_drawdown = build_daily_drawdown_rows(daily_equity_out)
    subperiods = _concat_or_empty(subperiod_frames, SUBPERIOD_COLUMNS)
    rolling_windows = _concat_or_empty(rolling_frames, ROLLING_WINDOW_COLUMNS)
    decision_matrix = build_refined_core_sleeve_decision_matrix(results)

    return LongHistoryCoreSleeveRefinementArtifacts(
        results=results,
        daily_equity=daily_equity_out,
        daily_drawdown=daily_drawdown,
        subperiods=subperiods,
        rolling_windows=rolling_windows,
        decision_matrix=decision_matrix,
    )


def build_refined_core_sleeve_daily_equity(
    *,
    aligned_returns: pd.DataFrame,
    candidate: pd.Series,
    spy_core_weight: float,
    candidate_sleeve_weight: float | None = None,
    portfolio_accounting_mode: str = DAILY_REBALANCED_FIXED_WEIGHTS,
) -> pd.DataFrame:
    """Build daily portfolio equity under one accounting mode."""

    _validate_columns(
        aligned_returns,
        ["date", "candidate_daily_return", "qqq_daily_return", "spy_daily_return"],
        "aligned returns",
    )
    mode = resolve_portfolio_accounting_modes((portfolio_accounting_mode,))[0]
    spy_weight = float(spy_core_weight)
    candidate_weight = (
        round(1.0 - spy_weight, 10)
        if candidate_sleeve_weight is None
        else float(candidate_sleeve_weight)
    )
    if abs(spy_weight + candidate_weight - 1.0) > 1.0e-9:
        raise ValueError("SPY core weight and candidate sleeve weight must sum to 1.")

    out = aligned_returns.copy()
    out["date"] = pd.to_datetime(out["date"])
    accounting = _portfolio_accounting_returns(
        aligned_returns=out,
        spy_core_weight=spy_weight,
        candidate_sleeve_weight=candidate_weight,
        portfolio_accounting_mode=mode,
    )
    equity = pd.Series(1.0 + accounting["daily_return"]).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    out["candidate_id"] = str(candidate["candidate_id"])
    out["candidate_label"] = str(candidate["candidate_label"])
    out["spy_core_weight"] = spy_weight
    out["candidate_sleeve_weight"] = candidate_weight
    out["portfolio_accounting_mode"] = mode
    out["portfolio_name"] = portfolio_name(candidate["candidate_id"], spy_weight, mode)
    out["strategy_name"] = "SPY core + candidate sleeve"
    out["daily_return"] = accounting["daily_return"]
    out["equity"] = equity.to_numpy()
    out["drawdown"] = drawdown.to_numpy()
    out["effective_spy_weight"] = accounting["effective_spy_weight"]
    out["effective_candidate_weight"] = accounting["effective_candidate_weight"]
    out["is_portfolio_rebalance_date"] = accounting["is_portfolio_rebalance_date"]
    for column in DAILY_EQUITY_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out.loc[:, DAILY_EQUITY_COLUMNS]


def build_result_row(
    *,
    daily_equity: pd.DataFrame,
    candidate: pd.Series,
    rolling_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build one full-period result row for a refined portfolio."""

    metrics = _metrics_from_daily(daily_equity)
    first = daily_equity.iloc[0]
    rolling_row = (
        rolling_summary.iloc[0] if not rolling_summary.empty else pd.Series(dtype=object)
    )
    candidate_weight = float(first["candidate_sleeve_weight"])
    source_transaction_count = _to_float(candidate.get("transaction_count"))
    transaction_count = (
        int(source_transaction_count)
        if candidate_weight > 0.0 and pd.notna(source_transaction_count)
        else 0
    )
    average_turnover = _scale_candidate_metric(candidate.get("average_turnover"), candidate_weight)
    estimated_transaction_cost = _scale_candidate_metric(
        candidate.get("estimated_transaction_cost"),
        candidate_weight,
    )
    positive_rate = rolling_row.get("positive_vs_qqq_window_rate", pd.NA)
    underperformance_rate = rolling_row.get("qqq_underperformance_rate", pd.NA)
    acceptable_rolling = _acceptable_rolling_behavior(positive_rate, underperformance_rate)
    row = {
        "candidate_id": first["candidate_id"],
        "candidate_label": first["candidate_label"],
        "spy_core_weight": float(first["spy_core_weight"]),
        "candidate_sleeve_weight": candidate_weight,
        "portfolio_accounting_mode": first["portfolio_accounting_mode"],
        "portfolio_name": first["portfolio_name"],
        "is_core_sleeve": bool(0.0 < float(first["spy_core_weight"]) < 1.0),
        "transaction_count": transaction_count,
        "average_turnover": average_turnover,
        "estimated_transaction_cost": estimated_transaction_cost,
        "portfolio_rebalance_count": int(
            pd.Series(daily_equity["is_portfolio_rebalance_date"]).fillna(False).astype(bool).sum()
        ),
        "candidate_rebalance_date_count": int(
            pd.to_numeric(daily_equity["candidate_turnover"], errors="coerce")
            .fillna(0.0)
            .gt(0.0)
            .sum()
        ),
        "positive_vs_qqq_window_rate": positive_rate,
        "qqq_underperformance_rate": underperformance_rate,
        "worst_strategy_vs_qqq": rolling_row.get("worst_strategy_vs_qqq", pd.NA),
        "acceptable_rolling_behavior": acceptable_rolling,
        "source_decision": candidate.get("source_decision", pd.NA),
        "source_rationale": candidate.get("source_rationale", pd.NA),
        "source_strategy_vs_qqq": candidate.get("strategy_vs_qqq", pd.NA),
        "source_return_to_drawdown": candidate.get("return_to_drawdown", pd.NA),
        "source_positive_vs_qqq_window_rate": candidate.get(
            "positive_vs_qqq_window_rate",
            pd.NA,
        ),
        "source_worst_strategy_vs_qqq": candidate.get("worst_strategy_vs_qqq", pd.NA),
        "source_top_symbol": candidate.get("top_symbol", pd.NA),
        "source_top_symbol_exposure_pct": candidate.get("top_symbol_exposure_pct", pd.NA),
        "source_top3_symbol_exposure_pct": candidate.get("top3_symbol_exposure_pct", pd.NA),
    }
    row.update(metrics)
    row["cost_20bps_strategy_vs_qqq"] = estimate_portfolio_cost_edge(
        strategy_vs_qqq=metrics["strategy_vs_qqq"],
        standalone_strategy_vs_qqq=metrics["standalone_candidate_total_return"]
        - metrics["qqq_buy_hold_total_return"],
        source_cost_strategy_vs_qqq=candidate.get("source_cost_20bps_strategy_vs_qqq"),
        candidate_sleeve_weight=candidate_weight,
    )
    row["cost_30bps_strategy_vs_qqq"] = estimate_portfolio_cost_edge(
        strategy_vs_qqq=metrics["strategy_vs_qqq"],
        standalone_strategy_vs_qqq=metrics["standalone_candidate_total_return"]
        - metrics["qqq_buy_hold_total_return"],
        source_cost_strategy_vs_qqq=candidate.get("source_cost_30bps_strategy_vs_qqq"),
        candidate_sleeve_weight=candidate_weight,
    )
    for column in CONFIG_COLUMNS:
        row[column] = candidate.get(column, pd.NA)
    for column in RESULT_COLUMNS:
        if column not in row:
            row[column] = pd.NA
    return pd.DataFrame([row], columns=RESULT_COLUMNS)


def build_daily_drawdown_rows(daily_equity: pd.DataFrame) -> pd.DataFrame:
    """Build daily drawdown rows from daily equity output."""

    if daily_equity.empty:
        return pd.DataFrame(columns=DAILY_DRAWDOWN_COLUMNS)
    out = daily_equity.loc[:, DAILY_DRAWDOWN_COLUMNS].copy()
    return out


def build_subperiod_rows(
    *,
    daily_equity: pd.DataFrame,
    subperiods: tuple[tuple[str, str, str], ...] = SUBPERIODS,
) -> pd.DataFrame:
    """Build named subperiod rows for one refined portfolio."""

    if daily_equity.empty:
        return pd.DataFrame(columns=SUBPERIOD_COLUMNS)
    first = daily_equity.iloc[0]
    rows: list[dict[str, object]] = []
    for period_name, period_start, period_end in subperiods:
        period = _slice_daily(daily_equity, period_start, period_end)
        if period.empty:
            continue
        metrics = _metrics_from_daily(period)
        rows.append(
            {
                "candidate_id": first["candidate_id"],
                "candidate_label": first["candidate_label"],
                "spy_core_weight": first["spy_core_weight"],
                "candidate_sleeve_weight": first["candidate_sleeve_weight"],
                "portfolio_accounting_mode": first["portfolio_accounting_mode"],
                "portfolio_name": first["portfolio_name"],
                "period_name": period_name,
                "period_start": period_start,
                "period_end": period_end,
                "total_return": metrics["total_return"],
                "qqq_total_return": metrics["qqq_buy_hold_total_return"],
                "spy_total_return": metrics["spy_buy_hold_total_return"],
                "standalone_candidate_total_return": (
                    metrics["standalone_candidate_total_return"]
                ),
                "strategy_vs_qqq": metrics["strategy_vs_qqq"],
                "strategy_vs_spy": metrics["strategy_vs_spy"],
                "max_drawdown": metrics["max_drawdown"],
                "qqq_max_drawdown": metrics["qqq_buy_hold_max_drawdown"],
                "spy_max_drawdown": metrics["spy_buy_hold_max_drawdown"],
                "standalone_candidate_max_drawdown": (
                    metrics["standalone_candidate_max_drawdown"]
                ),
                "drawdown_improvement_vs_qqq": metrics["drawdown_improvement_vs_qqq"],
                "drawdown_improvement_vs_candidate": (
                    metrics["drawdown_improvement_vs_candidate"]
                ),
                "return_to_drawdown": metrics["return_to_drawdown"],
                "qqq_return_to_drawdown": metrics["qqq_buy_hold_return_to_drawdown"],
                "spy_return_to_drawdown": metrics["spy_buy_hold_return_to_drawdown"],
                "standalone_candidate_return_to_drawdown": (
                    metrics["standalone_candidate_return_to_drawdown"]
                ),
                "portfolio_rebalance_count": int(
                    pd.Series(period["is_portfolio_rebalance_date"])
                    .fillna(False)
                    .astype(bool)
                    .sum()
                ),
            }
        )
    return pd.DataFrame(rows, columns=SUBPERIOD_COLUMNS)


def build_rolling_window_rows(
    *,
    daily_equity: pd.DataFrame,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    """Build 1y/2y/3y rolling-window robustness rows for one portfolio."""

    if daily_equity.empty:
        return pd.DataFrame(columns=ROLLING_WINDOW_COLUMNS)
    first = daily_equity.iloc[0]
    rows: list[dict[str, object]] = []
    for years, step_months in ((1, 3), (2, 3), (3, 6)):
        for window_start, window_end in generate_rolling_windows(
            start_date=start_date,
            end_date=end_date,
            years=years,
            step_months=step_months,
        ):
            period = _slice_daily(
                daily_equity,
                str(window_start.date()),
                str(window_end.date()),
            )
            if period.empty:
                continue
            metrics = _metrics_from_daily(period)
            strategy_vs_qqq = metrics["strategy_vs_qqq"]
            drawdown_improvement = metrics["drawdown_improvement_vs_qqq"]
            rows.append(
                {
                    "candidate_id": first["candidate_id"],
                    "candidate_label": first["candidate_label"],
                    "spy_core_weight": first["spy_core_weight"],
                    "candidate_sleeve_weight": first["candidate_sleeve_weight"],
                    "portfolio_accounting_mode": first["portfolio_accounting_mode"],
                    "portfolio_name": first["portfolio_name"],
                    "window_label": (
                        f"{years}y_{window_start:%Y%m%d}_{window_end:%Y%m%d}"
                    ),
                    "window_years": years,
                    "window_start": str(window_start.date()),
                    "window_end": str(window_end.date()),
                    "strategy_total_return": metrics["total_return"],
                    "qqq_total_return": metrics["qqq_buy_hold_total_return"],
                    "spy_total_return": metrics["spy_buy_hold_total_return"],
                    "standalone_candidate_total_return": (
                        metrics["standalone_candidate_total_return"]
                    ),
                    "strategy_vs_qqq": strategy_vs_qqq,
                    "strategy_max_drawdown": metrics["max_drawdown"],
                    "qqq_max_drawdown": metrics["qqq_buy_hold_max_drawdown"],
                    "spy_max_drawdown": metrics["spy_buy_hold_max_drawdown"],
                    "standalone_candidate_max_drawdown": (
                        metrics["standalone_candidate_max_drawdown"]
                    ),
                    "drawdown_improvement_vs_qqq": drawdown_improvement,
                    "drawdown_improvement_vs_candidate": (
                        metrics["drawdown_improvement_vs_candidate"]
                    ),
                    "underperformed_qqq": bool(strategy_vs_qqq < 0.0),
                    "materially_improved_drawdown": bool(
                        drawdown_improvement > MATERIAL_DRAWDOWN_IMPROVEMENT
                    ),
                }
            )
    return pd.DataFrame(rows, columns=ROLLING_WINDOW_COLUMNS)


def summarize_refinement_rolling_windows(rolling_windows: pd.DataFrame) -> pd.DataFrame:
    """Summarize rolling-window behavior by refined portfolio."""

    if rolling_windows.empty:
        return pd.DataFrame(columns=ROLLING_SUMMARY_COLUMNS)
    rows: list[dict[str, object]] = []
    group_cols = [
        "candidate_id",
        "spy_core_weight",
        "candidate_sleeve_weight",
        "portfolio_accounting_mode",
        "portfolio_name",
    ]
    for keys, group in rolling_windows.groupby(group_cols, sort=False, dropna=False):
        values = pd.to_numeric(group["strategy_vs_qqq"], errors="coerce")
        drawdown = pd.to_numeric(group["drawdown_improvement_vs_qqq"], errors="coerce")
        worst_index = values.idxmin() if values.notna().any() else None
        rows.append(
            {
                "candidate_id": keys[0],
                "spy_core_weight": keys[1],
                "candidate_sleeve_weight": keys[2],
                "portfolio_accounting_mode": keys[3],
                "portfolio_name": keys[4],
                "rolling_window_count": int(len(group)),
                "positive_vs_qqq_window_rate": float(values.gt(0.0).mean()),
                "qqq_underperformance_rate": float(values.lt(0.0).mean()),
                "average_strategy_vs_qqq": float(values.mean()),
                "median_strategy_vs_qqq": float(values.median()),
                "worst_strategy_vs_qqq": float(values.min()),
                "drawdown_improvement_window_rate": float(drawdown.gt(0.0).mean()),
                "average_drawdown_improvement_vs_qqq": float(drawdown.mean()),
                "worst_window_label": (
                    group.loc[worst_index, "window_label"] if worst_index is not None else ""
                ),
            }
        )
    return pd.DataFrame(rows, columns=ROLLING_SUMMARY_COLUMNS)


def build_refined_core_sleeve_decision_matrix(results: pd.DataFrame) -> pd.DataFrame:
    """Build the final research-only refined core/sleeve decision matrix."""

    if results.empty:
        return pd.DataFrame(columns=DECISION_MATRIX_COLUMNS)
    matrix = results.copy()
    for column in OBJECTIVE_COLUMNS:
        if column != "selected_objectives":
            matrix[column] = False
    matrix["selected_objectives"] = ""
    _mark_objective_winners(matrix)

    decisions = matrix.apply(classify_refined_core_sleeve_decision, axis=1)
    matrix["decision"] = [item[0] for item in decisions]
    matrix["rationale"] = [item[1] for item in decisions]
    matrix["production_ready"] = False
    for column in DECISION_MATRIX_COLUMNS:
        if column not in matrix.columns:
            matrix[column] = pd.NA
    return matrix.loc[:, DECISION_MATRIX_COLUMNS]


def classify_refined_core_sleeve_decision(row: pd.Series) -> tuple[str, str]:
    """Classify one refined core/sleeve row under research-only decision rules."""

    beats_qqq = bool(row.get("beats_qqq_return", False))
    captures_90 = bool(row.get("captures_90pct_qqq_return", False))
    drawdown_improved = bool(row.get("improves_drawdown_vs_qqq", False))
    rtd_improved = bool(row.get("improves_return_to_drawdown_vs_qqq", False))
    is_core_sleeve = bool(row.get("is_core_sleeve", False))
    acceptable_rolling = bool(row.get("acceptable_rolling_behavior", False))
    drawdown_improvement = _to_float(row.get("drawdown_improvement_vs_qqq"))

    reasons = []
    if beats_qqq:
        reasons.append("beats QQQ return")
    elif captures_90:
        reasons.append("captures at least 90% of QQQ return")
    else:
        reasons.append("trails QQQ return")
    reasons.append(
        "improves QQQ drawdown" if drawdown_improved else "does not improve QQQ drawdown"
    )
    reasons.append(
        "improves return-to-drawdown"
        if rtd_improved
        else "does not improve return-to-drawdown"
    )
    reasons.append(
        "acceptable rolling behavior"
        if acceptable_rolling
        else "rolling behavior needs caution"
    )

    if is_core_sleeve and beats_qqq and drawdown_improved and rtd_improved:
        return "best_refined_core_sleeve_candidate", "; ".join(reasons)
    if (
        is_core_sleeve
        and captures_90
        and drawdown_improved
        and rtd_improved
        and acceptable_rolling
    ):
        return "balanced_refined_core_sleeve_candidate", "; ".join(reasons)
    if beats_qqq:
        return "return_candidate", "; ".join(reasons)
    if (
        pd.notna(drawdown_improvement)
        and drawdown_improvement > MATERIAL_DRAWDOWN_IMPROVEMENT
        and rtd_improved
    ):
        return "defensive_candidate", "; ".join(reasons)
    return "reject_deprioritize", "; ".join(reasons)


def write_long_history_core_sleeve_refinement_artifacts(
    *,
    artifacts: LongHistoryCoreSleeveRefinementArtifacts,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write all refined core/sleeve CSV artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = output_paths(output_dir)
    artifacts.results.to_csv(outputs[RESULTS_NAME], index=False)
    artifacts.daily_equity.to_csv(outputs[DAILY_EQUITY_NAME], index=False)
    artifacts.daily_drawdown.to_csv(outputs[DAILY_DRAWDOWN_NAME], index=False)
    artifacts.subperiods.to_csv(outputs[SUBPERIODS_NAME], index=False)
    artifacts.rolling_windows.to_csv(outputs[ROLLING_WINDOWS_NAME], index=False)
    artifacts.decision_matrix.to_csv(outputs[DECISION_MATRIX_NAME], index=False)
    return outputs


def output_paths(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    """Return all expected output paths for the refined sleeve study directory."""

    return {
        RESULTS_NAME: output_dir / RESULTS_NAME,
        DAILY_EQUITY_NAME: output_dir / DAILY_EQUITY_NAME,
        DAILY_DRAWDOWN_NAME: output_dir / DAILY_DRAWDOWN_NAME,
        SUBPERIODS_NAME: output_dir / SUBPERIODS_NAME,
        ROLLING_WINDOWS_NAME: output_dir / ROLLING_WINDOWS_NAME,
        DECISION_MATRIX_NAME: output_dir / DECISION_MATRIX_NAME,
        PREFLIGHT_MD_NAME: output_dir / PREFLIGHT_MD_NAME,
    }


def render_preflight_markdown(
    *,
    source_candidates: pd.DataFrame,
    weight_grid: pd.DataFrame,
    accounting_modes: tuple[str, ...],
    accounting_availability: pd.DataFrame,
    spy_core_sleeve_results: Path,
    spy_core_sleeve_daily_equity: Path,
    candidate_deep_dive_daily_equity: Path,
    candidate_deep_dive_summary: Path,
    candidate_deep_dive_decision_matrix: Path,
    output_dir: Path,
    report_output_dir: Path,
    start_date: str,
    end_date: str,
    candidate_ids: tuple[str, ...],
    dry_run: bool,
) -> str:
    """Render research-only preflight markdown for the refinement study."""

    lines = [
        "# Long-History QQQ Core-Sleeve Refinement Preflight",
        "",
        "Focused SPY core plus candidate sleeve refinement using completed daily "
        "returns. The task is portfolio-construction validation only.",
        "",
        "## Controls",
        "",
        "- Local-only: `true`",
        "- S3/cloud access: `false`",
        "- Broad model zoo: `false`",
        "- MLP batch: `false`",
        "- Production signal export: `false`",
        "- Model card: `false`",
        "- Production-ready marking: `false`",
        "- Market Data repo modification: `false`",
        "- Primary benchmark: `QQQ buy-and-hold`",
        "- Secondary references: `SPY buy-and-hold`, `standalone candidate`",
        f"- Dataset source: `{DATASET_SOURCE_LABEL}`",
        f"- Price mode: `{PRICE_MODE}`",
        "",
        "## Inputs",
        "",
        f"- Previous SPY core-sleeve results: `{spy_core_sleeve_results}`",
        f"- Previous SPY core-sleeve daily equity: `{spy_core_sleeve_daily_equity}`",
        f"- Candidate deep-dive daily equity: `{candidate_deep_dive_daily_equity}`",
        f"- Candidate deep-dive summary: `{candidate_deep_dive_summary}`",
        f"- Candidate deep-dive decision matrix: `{candidate_deep_dive_decision_matrix}`",
        f"- Output directory: `{output_dir}`",
        f"- Report output directory: `{report_output_dir}`",
        f"- Start date: `{start_date}`",
        f"- End date: `{end_date}`",
        f"- Requested candidate IDs: `{list(candidate_ids) if candidate_ids else 'defaults'}`",
        f"- Dry run: `{dry_run}`",
        "",
        "## Accounting Modes",
        "",
        _markdown_table(pd.DataFrame({"portfolio_accounting_mode": list(accounting_modes)})),
        "",
        "## Candidate Rebalance-Aligned Availability",
        "",
        _markdown_table(accounting_availability),
        "",
        "## Selected Source Candidates",
        "",
        _markdown_table(
            source_candidates,
            [
                "candidate_id",
                "candidate_label",
                "model_type",
                "training_mode",
                "target_variant",
                "total_return",
                "strategy_vs_qqq",
                "max_drawdown",
                "return_to_drawdown",
                "top_symbol",
                "top_symbol_exposure_pct",
                "top3_symbol_exposure_pct",
                "source_decision",
            ],
        ),
        "",
        "## Fine Core Weight Grid",
        "",
        _markdown_table(weight_grid),
        "",
    ]
    return "\n".join(lines)


def portfolio_name(candidate_id: object, spy_core_weight: float, accounting_mode: str) -> str:
    """Return a stable display name for a refined portfolio row."""

    candidate_weight = 1.0 - float(spy_core_weight)
    if spy_core_weight <= 0.0:
        base = f"{candidate_id} standalone candidate"
    elif spy_core_weight >= 1.0:
        base = f"{candidate_id} SPY standalone reference"
    else:
        base = f"{candidate_id} SPY {spy_core_weight:.0%} / candidate {candidate_weight:.0%}"
    return f"{base} | {accounting_mode}"


def _portfolio_accounting_returns(
    *,
    aligned_returns: pd.DataFrame,
    spy_core_weight: float,
    candidate_sleeve_weight: float,
    portfolio_accounting_mode: str,
) -> dict[str, np.ndarray]:
    spy_returns = (
        pd.to_numeric(aligned_returns["spy_daily_return"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    candidate_returns = (
        pd.to_numeric(aligned_returns["candidate_daily_return"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    if portfolio_accounting_mode == DAILY_REBALANCED_FIXED_WEIGHTS:
        daily_return = spy_core_weight * spy_returns + candidate_sleeve_weight * candidate_returns
        return {
            "daily_return": daily_return,
            "effective_spy_weight": np.repeat(spy_core_weight, len(daily_return)),
            "effective_candidate_weight": np.repeat(candidate_sleeve_weight, len(daily_return)),
            "is_portfolio_rebalance_date": np.repeat(True, len(daily_return)),
        }

    reset_mask = _reset_mask(aligned_returns, portfolio_accounting_mode)
    if portfolio_accounting_mode == CANDIDATE_REBALANCE_ALIGNED_WEIGHTS and not reset_mask.any():
        raise ValueError("Candidate rebalance dates are not available for aligned accounting.")
    if len(reset_mask) > 0:
        reset_mask[0] = True

    daily_returns: list[float] = []
    effective_spy_weights: list[float] = []
    effective_candidate_weights: list[float] = []
    portfolio_value = 1.0
    spy_value = portfolio_value * spy_core_weight
    candidate_value = portfolio_value * candidate_sleeve_weight
    for idx, (spy_return, candidate_return) in enumerate(zip(spy_returns, candidate_returns)):
        if reset_mask[idx]:
            spy_value = portfolio_value * spy_core_weight
            candidate_value = portfolio_value * candidate_sleeve_weight
        denominator = spy_value + candidate_value
        if abs(denominator) <= 1.0e-12:
            effective_spy = spy_core_weight
            effective_candidate = candidate_sleeve_weight
        else:
            effective_spy = spy_value / denominator
            effective_candidate = candidate_value / denominator
        daily_return = effective_spy * spy_return + effective_candidate * candidate_return
        spy_value *= 1.0 + spy_return
        candidate_value *= 1.0 + candidate_return
        portfolio_value = spy_value + candidate_value
        daily_returns.append(float(daily_return))
        effective_spy_weights.append(float(effective_spy))
        effective_candidate_weights.append(float(effective_candidate))

    return {
        "daily_return": np.asarray(daily_returns, dtype=float),
        "effective_spy_weight": np.asarray(effective_spy_weights, dtype=float),
        "effective_candidate_weight": np.asarray(effective_candidate_weights, dtype=float),
        "is_portfolio_rebalance_date": reset_mask,
    }


def _reset_mask(
    aligned_returns: pd.DataFrame,
    portfolio_accounting_mode: str,
) -> np.ndarray:
    dates = pd.to_datetime(aligned_returns["date"])
    if portfolio_accounting_mode == MONTHLY_REBALANCED_WEIGHTS:
        periods = dates.dt.to_period("M")
        return periods.ne(periods.shift(1)).to_numpy(dtype=bool, copy=True)
    if portfolio_accounting_mode == QUARTERLY_REBALANCED_WEIGHTS:
        periods = dates.dt.to_period("Q")
        return periods.ne(periods.shift(1)).to_numpy(dtype=bool, copy=True)
    if portfolio_accounting_mode == CANDIDATE_REBALANCE_ALIGNED_WEIGHTS:
        turnover = pd.to_numeric(
            aligned_returns.get("candidate_turnover", pd.Series(index=aligned_returns.index)),
            errors="coerce",
        ).fillna(0.0)
        return turnover.gt(0.0).to_numpy(dtype=bool, copy=True)
    raise ValueError(f"Unsupported accounting mode: {portfolio_accounting_mode}")


def _metrics_from_daily(daily: pd.DataFrame) -> dict[str, float | bool]:
    returns = pd.to_numeric(daily["daily_return"], errors="coerce").fillna(0.0)
    qqq_returns = pd.to_numeric(daily["qqq_daily_return"], errors="coerce").fillna(0.0)
    spy_returns = pd.to_numeric(daily["spy_daily_return"], errors="coerce").fillna(0.0)
    candidate_returns = pd.to_numeric(
        daily["candidate_daily_return"],
        errors="coerce",
    ).fillna(0.0)

    total_return = _compound_return(returns)
    qqq_total_return = _compound_return(qqq_returns)
    spy_total_return = _compound_return(spy_returns)
    candidate_total_return = _compound_return(candidate_returns)
    max_drawdown = _max_drawdown_from_returns(returns)
    qqq_max_drawdown = _max_drawdown_from_returns(qqq_returns)
    spy_max_drawdown = _max_drawdown_from_returns(spy_returns)
    candidate_max_drawdown = _max_drawdown_from_returns(candidate_returns)
    return_to_drawdown = _return_to_drawdown(total_return, max_drawdown)
    qqq_rtd = _return_to_drawdown(qqq_total_return, qqq_max_drawdown)
    spy_rtd = _return_to_drawdown(spy_total_return, spy_max_drawdown)
    candidate_rtd = _return_to_drawdown(candidate_total_return, candidate_max_drawdown)
    daily_volatility = float(returns.std())
    strategy_vs_qqq = total_return - qqq_total_return
    return_to_drawdown_vs_qqq = return_to_drawdown - qqq_rtd
    drawdown_improvement_vs_qqq = max_drawdown - qqq_max_drawdown
    return {
        "total_return": total_return,
        "annualized_return": _annualized_return(total_return, int(len(returns))),
        "daily_volatility": daily_volatility,
        "sharpe_like": (
            float(returns.mean() / daily_volatility) if daily_volatility > 0.0 else np.nan
        ),
        "max_drawdown": max_drawdown,
        "return_to_drawdown": return_to_drawdown,
        "qqq_buy_hold_total_return": qqq_total_return,
        "spy_buy_hold_total_return": spy_total_return,
        "standalone_candidate_total_return": candidate_total_return,
        "strategy_vs_qqq": strategy_vs_qqq,
        "strategy_vs_spy": total_return - spy_total_return,
        "qqq_return_capture": _safe_ratio(total_return, qqq_total_return),
        "spy_return_capture": _safe_ratio(total_return, spy_total_return),
        "drawdown_improvement_vs_qqq": drawdown_improvement_vs_qqq,
        "drawdown_improvement_vs_candidate": max_drawdown - candidate_max_drawdown,
        "return_to_drawdown_vs_qqq": return_to_drawdown_vs_qqq,
        "qqq_buy_hold_max_drawdown": qqq_max_drawdown,
        "spy_buy_hold_max_drawdown": spy_max_drawdown,
        "standalone_candidate_max_drawdown": candidate_max_drawdown,
        "qqq_buy_hold_return_to_drawdown": qqq_rtd,
        "spy_buy_hold_return_to_drawdown": spy_rtd,
        "standalone_candidate_return_to_drawdown": candidate_rtd,
        "beats_qqq_return": bool(total_return > qqq_total_return),
        "improves_drawdown_vs_qqq": bool(drawdown_improvement_vs_qqq > 0.0),
        "improves_return_to_drawdown_vs_qqq": bool(return_to_drawdown_vs_qqq > 0.0),
        "captures_90pct_qqq_return": bool(_safe_ratio(total_return, qqq_total_return) >= 0.90),
    }


def _mark_objective_winners(matrix: pd.DataFrame) -> None:
    eligible = (
        pd.to_numeric(matrix["candidate_sleeve_weight"], errors="coerce").gt(0.0)
        & pd.to_numeric(matrix["spy_core_weight"], errors="coerce").gt(0.0)
    )
    drawdown_controlled = (
        pd.to_numeric(matrix["drawdown_improvement_vs_qqq"], errors="coerce").ge(0.0)
    )
    drawdown_improved = (
        pd.to_numeric(matrix["drawdown_improvement_vs_qqq"], errors="coerce").gt(0.0)
    )
    rtd_improved = (
        pd.to_numeric(matrix["return_to_drawdown_vs_qqq"], errors="coerce").gt(0.0)
    )
    captures_90 = pd.to_numeric(matrix["qqq_return_capture"], errors="coerce").ge(0.90)
    beats_qqq = _as_bool(matrix["beats_qqq_return"])
    acceptable_rolling = _as_bool(matrix["acceptable_rolling_behavior"])

    _mark_idx(
        matrix,
        eligible & drawdown_controlled,
        "total_return",
        "objective_best_total_return_drawdown_controlled",
        "best_total_return_drawdown_controlled",
    )
    _mark_idx(
        matrix,
        eligible,
        "return_to_drawdown",
        "objective_best_return_to_drawdown",
        "best_return_to_drawdown",
    )
    _mark_idx(
        matrix,
        eligible & drawdown_improved,
        "total_return",
        "objective_best_total_return_drawdown_improvement",
        "best_total_return_drawdown_improvement",
    )
    _mark_idx(
        matrix,
        eligible
        & (beats_qqq | captures_90)
        & drawdown_improved
        & rtd_improved
        & acceptable_rolling,
        "total_return",
        "objective_best_balanced_candidate",
        "best_balanced_candidate",
    )


def _mark_idx(
    matrix: pd.DataFrame,
    mask: pd.Series,
    sort_column: str,
    objective_column: str,
    objective_label: str,
) -> None:
    candidates = matrix.loc[mask].copy()
    if candidates.empty:
        return
    values = pd.to_numeric(candidates[sort_column], errors="coerce")
    if not values.notna().any():
        return
    idx = values.idxmax()
    matrix.loc[idx, objective_column] = True
    existing = str(matrix.loc[idx, "selected_objectives"] or "")
    parts = [part for part in existing.split(";") if part]
    parts.append(objective_label)
    matrix.loc[idx, "selected_objectives"] = ";".join(parts)


def _acceptable_rolling_behavior(positive_rate: object, underperformance_rate: object) -> bool:
    positive = _to_float(positive_rate)
    underperformance = _to_float(underperformance_rate)
    if pd.isna(positive) or pd.isna(underperformance):
        return False
    return bool(
        positive >= ACCEPTABLE_POSITIVE_ROLLING_RATE
        and underperformance <= (1.0 - ACCEPTABLE_POSITIVE_ROLLING_RATE)
    )


def _slice_daily(daily: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    source = daily.copy()
    source["date"] = pd.to_datetime(source["date"])
    return source.loc[
        source["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    ].copy()


def _compound_return(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if clean.empty:
        return np.nan
    return float((1.0 + clean).prod() - 1.0)


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if clean.empty:
        return np.nan
    equity = (1.0 + clean).cumprod()
    return float(calculate_max_drawdown(equity))


def _annualized_return(total_return: float, days: int) -> float:
    if days <= 0 or pd.isna(total_return) or total_return <= -1.0:
        return np.nan
    return float((1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / days) - 1.0)


def _return_to_drawdown(total_return: float, max_drawdown: float) -> float:
    if pd.isna(total_return) or pd.isna(max_drawdown) or abs(max_drawdown) <= 1.0e-12:
        return np.nan
    return float(total_return / abs(max_drawdown))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or abs(denominator) <= 1.0e-12:
        return np.nan
    return float(numerator / denominator)


def _scale_candidate_metric(value: object, candidate_sleeve_weight: float) -> float:
    parsed = _to_float(value)
    if pd.isna(parsed) or candidate_sleeve_weight <= 0.0:
        return 0.0
    return float(parsed * candidate_sleeve_weight)


def _to_float(value: object) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else np.nan


def _as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def _resolve_candidate_ids(candidate_ids: tuple[str, ...] = ()) -> tuple[str, ...]:
    ordered = list(candidate_ids) if candidate_ids else list(DEFAULT_CANDIDATE_IDS)
    return tuple(dict.fromkeys(str(candidate_id) for candidate_id in ordered))


def _validate_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _concat_or_empty(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=columns)
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=columns)
    out = pd.concat(non_empty, ignore_index=True)
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    return out.loc[:, columns]


def _markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    table = frame.copy()
    if columns is None:
        columns = table.columns.tolist()
    for column in columns:
        if column not in table.columns:
            table[column] = ""
    table = table.loc[:, columns].fillna("")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(_markdown_cell(value) for value in row) + " |"
        for row in table.astype(object).to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def _markdown_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


__all__ = [
    "CANDIDATE_REBALANCE_ALIGNED_WEIGHTS",
    "DAILY_DRAWDOWN_COLUMNS",
    "DAILY_EQUITY_COLUMNS",
    "DAILY_REBALANCED_FIXED_WEIGHTS",
    "DECISION_MATRIX_COLUMNS",
    "DEFAULT_CANDIDATE_IDS",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_PORTFOLIO_ACCOUNTING_MODES",
    "MONTHLY_REBALANCED_WEIGHTS",
    "QUARTERLY_REBALANCED_WEIGHTS",
    "REQUIRED_METRIC_COLUMNS",
    "RESULT_COLUMNS",
    "ROLLING_WINDOW_COLUMNS",
    "SUBPERIOD_COLUMNS",
    "LongHistoryCoreSleeveRefinementArtifacts",
    "accounting_mode_availability",
    "build_long_history_core_sleeve_refinement_artifacts",
    "build_refined_core_sleeve_daily_equity",
    "build_refined_core_sleeve_decision_matrix",
    "build_result_row",
    "build_rolling_window_rows",
    "build_subperiod_rows",
    "candidate_rebalance_dates",
    "classify_refined_core_sleeve_decision",
    "generate_refinement_weight_grid",
    "output_paths",
    "render_preflight_markdown",
    "resolve_portfolio_accounting_modes",
    "select_source_candidates",
    "summarize_refinement_rolling_windows",
    "write_long_history_core_sleeve_refinement_artifacts",
]
