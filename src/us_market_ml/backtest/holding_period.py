"""Holding-period and turnover diagnostics for selected strategy holdings."""

from __future__ import annotations

import numpy as np
import pandas as pd

HOLDING_PERIOD_COLUMNS = [
    "symbol",
    "holding_id",
    "start_date",
    "end_date",
    "holding_days",
    "avg_target_weight",
    "max_target_weight",
]
HOLDING_PERIOD_SUMMARY_COLUMNS = [
    "total_holding_periods",
    "unique_symbols_held",
    "avg_holding_days",
    "median_holding_days",
    "max_holding_days",
    "min_holding_days",
    "single_day_holding_periods",
    "single_day_holding_share",
]
HOLDING_PERIOD_BY_SYMBOL_COLUMNS = [
    "symbol",
    "holding_periods",
    "avg_holding_days",
    "median_holding_days",
    "max_holding_days",
    "total_holding_days",
    "single_day_holding_periods",
    "single_day_holding_share",
]
TURNOVER_SUMMARY_COLUMNS = [
    "mean_turnover",
    "median_turnover",
    "max_turnover",
    "p90_turnover",
    "p95_turnover",
    "zero_turnover_days",
    "zero_turnover_share",
]
TOP_TURNOVER_DAYS_COLUMNS = ["date", "turnover"]


def build_holding_periods(
    selected_holdings: pd.DataFrame,
) -> pd.DataFrame:
    """Return consecutive selected-date holding periods by symbol.

    Consecutive dates are adjacent entries in the global strategy-date list
    observed in ``selected_holdings``, not necessarily adjacent calendar days.
    """

    if selected_holdings.empty:
        return pd.DataFrame(columns=HOLDING_PERIOD_COLUMNS)

    _validate_required_columns(
        selected_holdings,
        ["date", "symbol", "target_weight"],
        "selected_holdings",
    )

    holdings = selected_holdings[["date", "symbol", "target_weight"]].copy()
    holdings["date"] = pd.to_datetime(holdings["date"])
    holdings["target_weight"] = pd.to_numeric(holdings["target_weight"], errors="coerce")

    strategy_dates = pd.Series(holdings["date"].dropna().unique()).sort_values().tolist()
    date_positions = {date_value: index for index, date_value in enumerate(strategy_dates)}

    daily_holdings = (
        holdings.groupby(["symbol", "date"], sort=False, dropna=False)
        .agg(
            avg_target_weight=("target_weight", "mean"),
            max_target_weight=("target_weight", "max"),
        )
        .reset_index()
    )
    daily_holdings["strategy_date_index"] = daily_holdings["date"].map(date_positions)
    daily_holdings = daily_holdings.sort_values(
        ["symbol", "date"],
        kind="mergesort",
    ).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for symbol, symbol_holdings in daily_holdings.groupby("symbol", sort=False, dropna=False):
        holding_id = 0
        current_rows: list[pd.Series] = []
        previous_date_index: int | None = None

        for _, holding_row in symbol_holdings.iterrows():
            date_index = int(holding_row["strategy_date_index"])
            starts_new_period = (
                previous_date_index is None or date_index != previous_date_index + 1
            )
            if starts_new_period:
                if current_rows:
                    holding_id += 1
                    rows.append(_build_holding_period_row(symbol, holding_id, current_rows))
                current_rows = []

            current_rows.append(holding_row)
            previous_date_index = date_index

        if current_rows:
            holding_id += 1
            rows.append(_build_holding_period_row(symbol, holding_id, current_rows))

    if not rows:
        return pd.DataFrame(columns=HOLDING_PERIOD_COLUMNS)

    holding_periods = pd.DataFrame(rows, columns=HOLDING_PERIOD_COLUMNS)
    return holding_periods.sort_values(
        ["start_date", "symbol"],
        kind="mergesort",
    ).reset_index(drop=True)


def summarize_holding_periods(
    holding_periods: pd.DataFrame,
) -> pd.DataFrame:
    """Return one overall holding-period summary row."""

    if holding_periods.empty:
        return pd.DataFrame(
            [
                {
                    "total_holding_periods": 0,
                    "unique_symbols_held": 0,
                    "avg_holding_days": np.nan,
                    "median_holding_days": np.nan,
                    "max_holding_days": np.nan,
                    "min_holding_days": np.nan,
                    "single_day_holding_periods": 0,
                    "single_day_holding_share": np.nan,
                }
            ],
            columns=HOLDING_PERIOD_SUMMARY_COLUMNS,
        )

    _validate_required_columns(
        holding_periods,
        ["symbol", "holding_days"],
        "holding_periods",
    )

    holding_days = pd.to_numeric(holding_periods["holding_days"], errors="coerce")
    single_day_holding_periods = int((holding_days == 1).sum())
    total_holding_periods = int(len(holding_periods))

    return pd.DataFrame(
        [
            {
                "total_holding_periods": total_holding_periods,
                "unique_symbols_held": int(holding_periods["symbol"].nunique(dropna=False)),
                "avg_holding_days": float(holding_days.mean()),
                "median_holding_days": float(holding_days.median()),
                "max_holding_days": float(holding_days.max()),
                "min_holding_days": float(holding_days.min()),
                "single_day_holding_periods": single_day_holding_periods,
                "single_day_holding_share": (
                    single_day_holding_periods / total_holding_periods
                    if total_holding_periods
                    else np.nan
                ),
            }
        ],
        columns=HOLDING_PERIOD_SUMMARY_COLUMNS,
    )


def summarize_holding_periods_by_symbol(
    holding_periods: pd.DataFrame,
) -> pd.DataFrame:
    """Return holding-period summary rows by symbol."""

    if holding_periods.empty:
        return pd.DataFrame(columns=HOLDING_PERIOD_BY_SYMBOL_COLUMNS)

    _validate_required_columns(
        holding_periods,
        ["symbol", "holding_days"],
        "holding_periods",
    )

    periods = holding_periods.copy()
    periods["holding_days"] = pd.to_numeric(periods["holding_days"], errors="coerce")
    summary = (
        periods.groupby("symbol", sort=False, dropna=False)
        .agg(
            holding_periods=("holding_days", "size"),
            avg_holding_days=("holding_days", "mean"),
            median_holding_days=("holding_days", "median"),
            max_holding_days=("holding_days", "max"),
            total_holding_days=("holding_days", "sum"),
            single_day_holding_periods=(
                "holding_days",
                lambda values: int((values == 1).sum()),
            ),
        )
        .reset_index()
    )
    summary["single_day_holding_share"] = (
        summary["single_day_holding_periods"] / summary["holding_periods"]
    )

    return (
        summary.loc[:, HOLDING_PERIOD_BY_SYMBOL_COLUMNS]
        .sort_values(
            ["total_holding_days", "symbol"],
            ascending=[False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def summarize_turnover_distribution(
    equity_curve: pd.DataFrame,
    turnover_col: str = "turnover",
    date_col: str = "date",
    top_n: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return turnover distribution summary and highest-turnover dates."""

    _validate_required_columns(equity_curve, [date_col, turnover_col], "equity_curve")

    turnover_frame = equity_curve[[date_col, turnover_col]].copy()
    turnover_frame[date_col] = pd.to_datetime(turnover_frame[date_col])
    turnover_frame[turnover_col] = pd.to_numeric(
        turnover_frame[turnover_col],
        errors="coerce",
    )
    turnover_values = turnover_frame[turnover_col].dropna()
    zero_turnover_days = int((turnover_values == 0.0).sum())
    total_turnover_days = int(len(turnover_values))

    turnover_summary = pd.DataFrame(
        [
            {
                "mean_turnover": _float_or_nan(turnover_values.mean()),
                "median_turnover": _float_or_nan(turnover_values.median()),
                "max_turnover": _float_or_nan(turnover_values.max()),
                "p90_turnover": _float_or_nan(turnover_values.quantile(0.90)),
                "p95_turnover": _float_or_nan(turnover_values.quantile(0.95)),
                "zero_turnover_days": zero_turnover_days,
                "zero_turnover_share": (
                    zero_turnover_days / total_turnover_days
                    if total_turnover_days
                    else np.nan
                ),
            }
        ],
        columns=TURNOVER_SUMMARY_COLUMNS,
    )

    top_turnover_days = (
        turnover_frame.rename(columns={turnover_col: "turnover"})
        .loc[:, TOP_TURNOVER_DAYS_COLUMNS]
        .sort_values(["turnover", "date"], ascending=[False, True], kind="mergesort")
        .head(top_n)
        .reset_index(drop=True)
    )

    return turnover_summary, top_turnover_days


def build_holding_diagnostics_report(
    selected_holdings: pd.DataFrame,
    equity_curve: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build all holding-period and turnover diagnostic tables."""

    holding_periods = build_holding_periods(selected_holdings)
    holding_period_summary = summarize_holding_periods(holding_periods)
    holding_period_by_symbol = summarize_holding_periods_by_symbol(holding_periods)
    turnover_summary, top_turnover_days = summarize_turnover_distribution(equity_curve)

    return {
        "holding_periods": holding_periods,
        "holding_period_summary": holding_period_summary,
        "holding_period_by_symbol": holding_period_by_symbol,
        "turnover_summary": turnover_summary,
        "top_turnover_days": top_turnover_days,
    }


def _build_holding_period_row(
    symbol: object,
    holding_id: int,
    holding_rows: list[pd.Series],
) -> dict[str, object]:
    period = pd.DataFrame(holding_rows)
    return {
        "symbol": symbol,
        "holding_id": holding_id,
        "start_date": period["date"].min(),
        "end_date": period["date"].max(),
        "holding_days": int(len(period)),
        "avg_target_weight": float(period["avg_target_weight"].mean()),
        "max_target_weight": float(period["max_target_weight"].max()),
    }


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    frame_name: str,
) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"{frame_name} is missing required holding diagnostic columns: {missing}")


def _float_or_nan(value: object) -> float:
    if pd.isna(value):
        return np.nan
    return float(value)


__all__ = [
    "build_holding_diagnostics_report",
    "build_holding_periods",
    "summarize_holding_periods",
    "summarize_holding_periods_by_symbol",
    "summarize_turnover_distribution",
]
