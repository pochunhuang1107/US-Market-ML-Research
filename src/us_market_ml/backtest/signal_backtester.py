"""Simple signal backtesting for ETF rotation research.

This module converts generated target weights and close prices into
position-level returns, daily portfolio returns, and an equity curve. It is a
deterministic research backtester, not an execution simulator: it does not
model order placement, fills, transaction costs, or slippage.

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
    costs, slippage, and execution modeling are intentionally future tasks.
    """

    date_col: str = "date"
    symbol_col: str = "symbol"
    price_col: str = "close"
    weight_col: str = "target_weight"
    selected_col: str = "selected"
    initial_equity: float = 1.0
    portfolio_return_col: str = "portfolio_return"
    equity_col: str = "equity"

    def __post_init__(self) -> None:
        """Validate signal backtest configuration."""

        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive.")


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
    date T from affecting the return already realized on date T. This function
    is a simple research backtester only; transaction costs, slippage, and
    advanced execution simulation are intentionally not modeled. Results are
    for research only.
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

    equity_curve = (
        position_returns.groupby(resolved_config.date_col, sort=True, dropna=False)[
            "position_return"
        ]
        .sum()
        .reset_index(name=resolved_config.portfolio_return_col)
    )
    equity_curve[resolved_config.equity_col] = resolved_config.initial_equity * (
        1.0 + equity_curve[resolved_config.portfolio_return_col]
    ).cumprod()

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

    return {
        "total_return": final_equity / resolved_config.initial_equity - 1.0,
        "mean_daily_return": mean_daily_return,
        "daily_volatility": daily_volatility,
        "sharpe_like": float(sharpe_like),
        "max_drawdown": calculate_max_drawdown(equity_curve[resolved_config.equity_col]),
        "final_equity": final_equity,
    }


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


__all__ = [
    "SignalBacktestConfig",
    "SignalBacktestResult",
    "calculate_max_drawdown",
    "run_signal_backtest",
    "summarize_backtest",
]
