"""Simple signal backtesting for ETF rotation research.

This module converts generated target weights and close prices into
position-level returns, daily portfolio returns, and an equity curve. It is a
deterministic research backtester, not an execution simulator: it does not
model order placement, fills, intraday execution, or taxes. It can apply a
simple turnover-based transaction-cost approximation.

Target weights are lagged by one row per symbol before calculating returns so
a signal generated on date T can only affect the close-to-close return from T
to T+1. Generated results are for research only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalBacktestConfig:
    """Configuration for simple signal backtests.

    The configured ``weight_col`` is interpreted as a target weight generated
    at the close of each row's date. The backtester applies the previous
    row/date target weight per symbol to avoid same-day lookahead. Transaction
    costs are a simple daily turnover times fixed basis-points approximation,
    not a full execution or slippage model.
    """

    date_col: str = "date"
    symbol_col: str = "symbol"
    price_col: str = "close"
    weight_col: str = "target_weight"
    selected_col: str = "selected"
    initial_equity: float = 1.0
    transaction_cost_bps: float = 0.0
    turnover_col: str = "turnover"
    transaction_cost_col: str = "transaction_cost"
    gross_portfolio_return_col: str = "gross_portfolio_return"
    net_portfolio_return_col: str = "portfolio_return"
    portfolio_return_col: str | None = None
    equity_col: str = "equity"

    def __post_init__(self) -> None:
        """Validate signal backtest configuration."""

        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive.")
        if self.transaction_cost_bps < 0.0:
            raise ValueError("transaction_cost_bps must be greater than or equal to 0.")
        net_return_col = (
            self.portfolio_return_col
            if self.portfolio_return_col is not None
            else self.net_portfolio_return_col
        )
        object.__setattr__(self, "net_portfolio_return_col", net_return_col)
        object.__setattr__(self, "portfolio_return_col", net_return_col)


@dataclass(frozen=True)
class SignalBacktestResult:
    """Complete output from a simple signal backtest."""

    equity_curve: pd.DataFrame
    position_returns: pd.DataFrame
    summary_metrics: dict[str, float]


def run_signal_backtest(
    signals: pd.DataFrame,
    config: SignalBacktestConfig | None = None,
) -> SignalBacktestResult:
    """Run a deterministic close-to-close backtest on generated signal rows.

    The input frame must contain date, symbol, close price, and target-weight
    columns using the configured names. The returned ``position_returns`` frame
    preserves the original signal rows and columns, then adds ``symbol_return``,
    ``lagged_target_weight``, and ``position_return``.

    Weights are shifted by one row per symbol before being multiplied by
    close-to-close symbol returns. This prevents a target weight generated on
    date T from affecting the return already realized on date T. Transaction
    costs are calculated as daily target-weight turnover times fixed bps divided
    by 10000, then subtracted from gross portfolio return. Results are for
    research only.
    """

    resolved_config = config if config is not None else SignalBacktestConfig()
    _validate_required_columns(
        signals,
        [
            resolved_config.date_col,
            resolved_config.symbol_col,
            resolved_config.price_col,
            resolved_config.weight_col,
        ],
    )

    original_order_col = _temporary_column_name(signals, "__signal_backtest_original_order")
    filled_weight_col = _temporary_column_name(signals, "__signal_backtest_filled_weight")

    position_returns = signals.copy()
    position_returns[original_order_col] = np.arange(len(position_returns))
    position_returns[filled_weight_col] = position_returns[resolved_config.weight_col].fillna(0.0)

    position_returns = position_returns.sort_values(
        by=[resolved_config.symbol_col, resolved_config.date_col],
        kind="mergesort",
    )

    grouped = position_returns.groupby(resolved_config.symbol_col, sort=False, dropna=False)
    previous_close = grouped[resolved_config.price_col].shift(1)
    position_returns["symbol_return"] = (
        position_returns[resolved_config.price_col] / previous_close - 1.0
    )
    position_returns["lagged_target_weight"] = grouped[filled_weight_col].shift(1).fillna(0.0)
    position_returns["position_return"] = (
        position_returns["lagged_target_weight"] * position_returns["symbol_return"].fillna(0.0)
    )

    gross_returns = (
        position_returns.groupby(resolved_config.date_col, sort=True, dropna=False)[
            "position_return"
        ]
        .sum()
        .reset_index(name=resolved_config.gross_portfolio_return_col)
    )
    turnover = calculate_daily_turnover(signals, resolved_config)
    equity_curve = gross_returns.merge(
        turnover,
        on=resolved_config.date_col,
        how="left",
        validate="one_to_one",
    )
    equity_curve[resolved_config.turnover_col] = equity_curve[
        resolved_config.turnover_col
    ].fillna(0.0)
    equity_curve[resolved_config.transaction_cost_col] = (
        equity_curve[resolved_config.turnover_col]
        * resolved_config.transaction_cost_bps
        / 10000.0
    )
    if resolved_config.transaction_cost_bps == 0.0:
        equity_curve[resolved_config.transaction_cost_col] = 0.0
    equity_curve[resolved_config.net_portfolio_return_col] = (
        equity_curve[resolved_config.gross_portfolio_return_col]
        - equity_curve[resolved_config.transaction_cost_col]
    )
    equity_curve[resolved_config.equity_col] = resolved_config.initial_equity * (
        1.0 + equity_curve[resolved_config.net_portfolio_return_col]
    ).cumprod()
    equity_curve = equity_curve[
        [
            resolved_config.date_col,
            resolved_config.gross_portfolio_return_col,
            resolved_config.transaction_cost_col,
            resolved_config.turnover_col,
            resolved_config.net_portfolio_return_col,
            resolved_config.equity_col,
        ]
    ]

    position_returns = (
        position_returns.sort_values(original_order_col, kind="mergesort")
        .drop(columns=[original_order_col, filled_weight_col])
        .copy()
    )

    return SignalBacktestResult(
        equity_curve=equity_curve,
        position_returns=position_returns,
        summary_metrics=summarize_backtest(equity_curve, resolved_config),
    )


def summarize_backtest(
    equity_curve: pd.DataFrame,
    config: SignalBacktestConfig | None = None,
) -> dict[str, float]:
    """Return simple daily summary metrics for a signal backtest.

    The equity curve must contain one row per date with configured portfolio
    return and equity columns. Metrics are daily and intentionally not
    annualized. ``sharpe_like`` is the mean daily return divided by daily
    return volatility when volatility is positive, otherwise ``NaN``.
    """

    resolved_config = config if config is not None else SignalBacktestConfig()
    _validate_required_columns(
        equity_curve,
        [
            resolved_config.date_col,
            resolved_config.portfolio_return_col,
            resolved_config.equity_col,
        ],
    )

    if equity_curve.empty:
        return {
            "total_return": np.nan,
            "mean_daily_return": np.nan,
            "daily_volatility": np.nan,
            "sharpe_like": np.nan,
            "max_drawdown": np.nan,
            "final_equity": np.nan,
            "total_transaction_cost": np.nan,
            "mean_daily_turnover": np.nan,
            "max_daily_turnover": np.nan,
        }

    portfolio_returns = equity_curve[resolved_config.portfolio_return_col]
    final_equity = float(equity_curve[resolved_config.equity_col].iloc[-1])
    mean_daily_return = float(portfolio_returns.mean())
    daily_volatility = float(portfolio_returns.std())
    sharpe_like = (
        mean_daily_return / daily_volatility
        if daily_volatility > 0.0 and not np.isnan(daily_volatility)
        else np.nan
    )

    total_transaction_cost = _sum_or_nan(
        equity_curve[resolved_config.transaction_cost_col]
        if resolved_config.transaction_cost_col in equity_curve.columns
        else None
    )
    mean_daily_turnover = _mean_or_nan(
        equity_curve[resolved_config.turnover_col]
        if resolved_config.turnover_col in equity_curve.columns
        else None
    )
    max_daily_turnover = _max_or_nan(
        equity_curve[resolved_config.turnover_col]
        if resolved_config.turnover_col in equity_curve.columns
        else None
    )

    return {
        "total_return": final_equity / resolved_config.initial_equity - 1.0,
        "mean_daily_return": mean_daily_return,
        "daily_volatility": daily_volatility,
        "sharpe_like": float(sharpe_like),
        "max_drawdown": calculate_max_drawdown(equity_curve[resolved_config.equity_col]),
        "final_equity": final_equity,
        "total_transaction_cost": total_transaction_cost,
        "mean_daily_turnover": mean_daily_turnover,
        "max_daily_turnover": max_daily_turnover,
    }


def calculate_daily_turnover(
    signals: pd.DataFrame,
    config: SignalBacktestConfig | None = None,
) -> pd.DataFrame:
    """Return daily target-weight turnover from signal target weights.

    Missing target weights are treated as zero. The first date's turnover is
    set to zero because the first day return is not investable from prior
    signals in this v1 close-to-close research backtest.
    """

    resolved_config = config if config is not None else SignalBacktestConfig()
    _validate_required_columns(
        signals,
        [
            resolved_config.date_col,
            resolved_config.symbol_col,
            resolved_config.weight_col,
        ],
    )

    if signals.empty:
        return pd.DataFrame(columns=[resolved_config.date_col, resolved_config.turnover_col])

    weights = signals[
        [
            resolved_config.date_col,
            resolved_config.symbol_col,
            resolved_config.weight_col,
        ]
    ].copy()
    weights[resolved_config.weight_col] = weights[resolved_config.weight_col].fillna(0.0)
    weights_by_date_symbol = weights.pivot_table(
        index=resolved_config.date_col,
        columns=resolved_config.symbol_col,
        values=resolved_config.weight_col,
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()
    turnover = weights_by_date_symbol.diff().abs().sum(axis=1)
    if not turnover.empty:
        turnover.iloc[0] = 0.0

    return turnover.rename(resolved_config.turnover_col).reset_index()


def calculate_max_drawdown(equity: pd.Series) -> float:
    """Return the maximum drawdown from an equity series.

    Drawdown is calculated as ``equity / running_max - 1`` and returned as the
    most negative drawdown value. Empty inputs return ``NaN``.
    """

    if equity.empty:
        return np.nan

    running_max = equity.cummax()
    drawdowns = equity / running_max - 1.0
    return float(drawdowns.min())


def _validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Input DataFrame is missing required backtest columns: {missing}")


def _temporary_column_name(df: pd.DataFrame, base_name: str) -> str:
    column_name = base_name
    suffix = 1
    while column_name in df.columns:
        column_name = f"{base_name}_{suffix}"
        suffix += 1
    return column_name


def _sum_or_nan(series: pd.Series | None) -> float:
    if series is None:
        return np.nan
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.sum())


def _mean_or_nan(series: pd.Series | None) -> float:
    if series is None:
        return np.nan
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.mean())


def _max_or_nan(series: pd.Series | None) -> float:
    if series is None:
        return np.nan
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.max())


__all__ = [
    "SignalBacktestConfig",
    "SignalBacktestResult",
    "calculate_daily_turnover",
    "calculate_max_drawdown",
    "run_signal_backtest",
    "summarize_backtest",
]
