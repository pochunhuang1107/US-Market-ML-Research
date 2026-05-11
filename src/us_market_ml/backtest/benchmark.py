"""Benchmark equity curve utilities for strategy comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from us_market_ml.backtest.signal_backtester import calculate_max_drawdown


@dataclass(frozen=True)
class BenchmarkConfig:
    """Column and initial-equity configuration for benchmark curves."""

    date_col: str = "date"
    symbol_col: str = "symbol"
    price_col: str = "close"
    initial_equity: float = 1.0

    def __post_init__(self) -> None:
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive.")


def build_buy_and_hold_benchmark(
    market_data: pd.DataFrame,
    symbol: str,
    config: BenchmarkConfig | None = None,
    output_name: str | None = None,
) -> pd.DataFrame:
    """Return a buy-and-hold benchmark equity curve for one symbol."""

    resolved_config = config if config is not None else BenchmarkConfig()
    _validate_required_columns(
        market_data,
        [resolved_config.date_col, resolved_config.symbol_col, resolved_config.price_col],
    )

    symbol_mask = market_data[resolved_config.symbol_col] == symbol
    if not symbol_mask.any():
        raise ValueError(f"Benchmark symbol is not present in market_data: {symbol}")

    benchmark = (
        market_data.loc[
            symbol_mask,
            [resolved_config.date_col, resolved_config.price_col],
        ]
        .copy()
        .sort_values(resolved_config.date_col, kind="mergesort")
        .reset_index(drop=True)
    )
    benchmark["benchmark_return"] = (
        benchmark[resolved_config.price_col].pct_change().fillna(0.0)
    )
    benchmark["benchmark_equity"] = resolved_config.initial_equity * (
        1.0 + benchmark["benchmark_return"]
    ).cumprod()
    benchmark["benchmark_name"] = output_name if output_name is not None else symbol

    return benchmark[
        [
            resolved_config.date_col,
            "benchmark_name",
            "benchmark_return",
            "benchmark_equity",
        ]
    ].rename(columns={resolved_config.date_col: "date"})


def build_equal_weight_benchmark(
    market_data: pd.DataFrame,
    symbols: list[str] | None = None,
    config: BenchmarkConfig | None = None,
    output_name: str = "equal_weight",
) -> pd.DataFrame:
    """Return an equal-weight benchmark using available symbol returns per date."""

    resolved_config = config if config is not None else BenchmarkConfig()
    _validate_required_columns(
        market_data,
        [resolved_config.date_col, resolved_config.symbol_col, resolved_config.price_col],
    )

    if symbols is None:
        selected_symbols = list(pd.unique(market_data[resolved_config.symbol_col].dropna()))
    else:
        selected_symbols = list(dict.fromkeys(symbols))
    if not selected_symbols:
        raise ValueError("At least one benchmark symbol is required.")

    available_symbols = set(market_data[resolved_config.symbol_col].dropna().unique())
    missing_symbols = [symbol for symbol in selected_symbols if symbol not in available_symbols]
    if missing_symbols and len(missing_symbols) == len(selected_symbols):
        missing = ", ".join(missing_symbols)
        raise ValueError(f"No requested benchmark symbols are present in market_data: {missing}")

    symbol_returns = (
        market_data.loc[
            market_data[resolved_config.symbol_col].isin(selected_symbols),
            [
                resolved_config.date_col,
                resolved_config.symbol_col,
                resolved_config.price_col,
            ],
        ]
        .copy()
        .sort_values(
            by=[resolved_config.symbol_col, resolved_config.date_col],
            kind="mergesort",
        )
    )
    if symbol_returns.empty:
        raise ValueError("No benchmark symbols are available in market_data.")

    symbol_returns["symbol_return"] = (
        symbol_returns.groupby(resolved_config.symbol_col, sort=False)[
            resolved_config.price_col
        ]
        .pct_change()
        .fillna(0.0)
    )

    benchmark = (
        symbol_returns.groupby(resolved_config.date_col, sort=True, dropna=False)[
            "symbol_return"
        ]
        .mean()
        .reset_index(name="benchmark_return")
    )
    benchmark["benchmark_equity"] = resolved_config.initial_equity * (
        1.0 + benchmark["benchmark_return"]
    ).cumprod()
    benchmark["benchmark_name"] = output_name

    return benchmark[
        [
            resolved_config.date_col,
            "benchmark_name",
            "benchmark_return",
            "benchmark_equity",
        ]
    ].rename(columns={resolved_config.date_col: "date"})


def combine_strategy_and_benchmarks(
    strategy_equity_curve: pd.DataFrame,
    benchmark_curves: list[pd.DataFrame],
    strategy_name: str = "strategy",
    date_col: str = "date",
    strategy_return_col: str = "portfolio_return",
    strategy_equity_col: str = "equity",
) -> pd.DataFrame:
    """Combine a strategy curve and benchmark curves into long format."""

    _validate_required_columns(
        strategy_equity_curve,
        [date_col, strategy_return_col, strategy_equity_col],
    )
    frames = [
        strategy_equity_curve[[date_col, strategy_return_col, strategy_equity_col]]
        .copy()
        .rename(
            columns={
                date_col: "date",
                strategy_return_col: "daily_return",
                strategy_equity_col: "equity",
            }
        )
    ]
    frames[0]["series_name"] = strategy_name

    for benchmark_curve in benchmark_curves:
        _validate_required_columns(
            benchmark_curve,
            ["date", "benchmark_name", "benchmark_return", "benchmark_equity"],
        )
        benchmark_long = benchmark_curve[
            ["date", "benchmark_name", "benchmark_return", "benchmark_equity"]
        ].copy()
        benchmark_long = benchmark_long.rename(
            columns={
                "benchmark_name": "series_name",
                "benchmark_return": "daily_return",
                "benchmark_equity": "equity",
            }
        )
        frames.append(benchmark_long)

    combined = pd.concat(frames, ignore_index=True)
    return combined[
        ["date", "series_name", "daily_return", "equity"]
    ].sort_values(
        by=["date", "series_name"],
        kind="mergesort",
    ).reset_index(drop=True)


def stitch_test_equity_curves(
    fold_curves: list[tuple[str, pd.DataFrame]],
    date_col: str = "date",
    return_col: str = "portfolio_return",
    equity_col: str = "equity",
    output_return_col: str = "portfolio_return",
    output_equity_col: str = "equity",
) -> pd.DataFrame:
    """Stitch non-overlapping fold test curves into one OOS equity curve."""

    output_columns = [date_col, output_return_col, output_equity_col, "source_fold"]
    if not fold_curves:
        return pd.DataFrame(columns=output_columns)

    frames: list[pd.DataFrame] = []
    for fold_name, curve in fold_curves:
        _validate_required_columns(curve, [date_col, return_col, equity_col])
        frame = curve[[date_col, return_col]].copy()
        frame[date_col] = pd.to_datetime(frame[date_col])
        frame["source_fold"] = fold_name
        frames.append(frame)

    stitched_returns = (
        pd.concat(frames, ignore_index=True)
        .sort_values(by=[date_col, "source_fold"], kind="mergesort")
        .reset_index(drop=True)
    )
    if stitched_returns.empty:
        return pd.DataFrame(columns=output_columns)

    duplicate_dates = stitched_returns.loc[
        stitched_returns.duplicated(subset=[date_col], keep=False),
        date_col,
    ]
    if not duplicate_dates.empty:
        overlap_dates = ", ".join(
            date.strftime("%Y-%m-%d")
            for date in pd.to_datetime(duplicate_dates.drop_duplicates()).sort_values()
        )
        raise ValueError(f"Fold test equity curves have overlapping dates: {overlap_dates}")

    stitched_returns[output_return_col] = stitched_returns[return_col].astype(float)
    stitched_returns.loc[stitched_returns.index[0], output_return_col] = 0.0
    stitched_returns[output_equity_col] = (1.0 + stitched_returns[output_return_col]).cumprod()

    return stitched_returns[
        [date_col, output_return_col, output_equity_col, "source_fold"]
    ].reset_index(drop=True)


def summarize_equity_series(
    combined_equity: pd.DataFrame,
) -> pd.DataFrame:
    """Return daily summary metrics for each long-format equity series."""

    _validate_required_columns(combined_equity, ["date", "series_name", "daily_return", "equity"])
    metric_columns = [
        "series_name",
        "total_return",
        "mean_daily_return",
        "daily_volatility",
        "sharpe_like",
        "max_drawdown",
        "final_equity",
    ]
    if combined_equity.empty:
        return pd.DataFrame(columns=metric_columns)

    rows: list[dict[str, object]] = []
    sorted_equity = combined_equity.sort_values(
        by=["series_name", "date"],
        kind="mergesort",
    )
    for series_name, series_df in sorted_equity.groupby("series_name", sort=True, dropna=False):
        returns = series_df["daily_return"]
        final_equity = float(series_df["equity"].iloc[-1])
        first_equity = float(series_df["equity"].iloc[0])
        mean_daily_return = float(returns.mean())
        daily_volatility = float(returns.std())
        sharpe_like = (
            mean_daily_return / daily_volatility
            if daily_volatility > 0.0 and not np.isnan(daily_volatility)
            else np.nan
        )
        rows.append(
            {
                "series_name": series_name,
                "total_return": final_equity / first_equity - 1.0,
                "mean_daily_return": mean_daily_return,
                "daily_volatility": daily_volatility,
                "sharpe_like": float(sharpe_like),
                "max_drawdown": calculate_max_drawdown(series_df["equity"]),
                "final_equity": final_equity,
            }
        )

    return pd.DataFrame(rows, columns=metric_columns)


def _validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Input DataFrame is missing required benchmark columns: {missing}")


__all__ = [
    "BenchmarkConfig",
    "build_buy_and_hold_benchmark",
    "build_equal_weight_benchmark",
    "combine_strategy_and_benchmarks",
    "summarize_equity_series",
    "stitch_test_equity_curves",
]
