"""Benchmark-gap feature set for SPY/QQQ relative ETF rotation research.

The features in this module are built from same-row or trailing historical
prices only. Benchmark-level features are calculated on SPY/QQQ histories and
joined back by date without forward-fill or backward-fill. Per-symbol
expanding statistics explicitly shift the trailing 252-day return history so
the current row's return is not part of its own long-run expectation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BENCHMARK_SYMBOLS: tuple[str, ...] = ("SPY", "QQQ")
BENCHMARK_RETURN_LOOKBACKS: tuple[int, ...] = (21, 63, 126, 252)
BENCHMARK_DRAWDOWN_LOOKBACKS: tuple[int, ...] = (63, 126, 252)
BENCHMARK_VOLATILITY_LOOKBACKS: tuple[int, ...] = (21, 63)
RELATIVE_STRENGTH_LOOKBACKS: tuple[int, ...] = (63, 126, 252)
REGIME_LOOKBACKS: tuple[int, ...] = (63, 126, 252)
LONG_RUN_RETURN_LOOKBACK = 252

BENCHMARK_GAP_FEATURE_COLUMNS: tuple[str, ...] = (
    "spy_return_21d",
    "spy_return_63d",
    "spy_return_126d",
    "spy_return_252d",
    "qqq_return_21d",
    "qqq_return_63d",
    "qqq_return_126d",
    "qqq_return_252d",
    "spy_drawdown_63d",
    "spy_drawdown_126d",
    "spy_drawdown_252d",
    "qqq_drawdown_63d",
    "qqq_drawdown_126d",
    "qqq_drawdown_252d",
    "spy_volatility_21d",
    "spy_volatility_63d",
    "qqq_volatility_21d",
    "qqq_volatility_63d",
    "symbol_return_63d_minus_spy_return_63d",
    "symbol_return_126d_minus_spy_return_126d",
    "symbol_return_252d_minus_spy_return_252d",
    "symbol_return_63d_minus_qqq_return_63d",
    "symbol_return_126d_minus_qqq_return_126d",
    "symbol_return_252d_minus_qqq_return_252d",
    "qqq_minus_spy_return_63d",
    "qqq_minus_spy_return_126d",
    "qqq_minus_spy_return_252d",
    "qqq_div_spy_price_ratio_return_63d",
    "qqq_div_spy_price_ratio_return_126d",
    "qqq_div_spy_price_ratio_return_252d",
    "symbol_return_252d",
    "symbol_expanding_mean_return_252d_shifted",
    "symbol_return_252d_minus_expanding_mean",
    "symbol_return_252d_zscore_vs_expanding_history",
)


def add_benchmark_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add SPY/QQQ benchmark-gap features to an ETF feature frame."""

    out = df.copy()
    out["_symbol_upper"] = out["symbol"].astype("string").str.upper()

    benchmark_by_date = _benchmark_features_by_date(out)
    out = out.merge(benchmark_by_date.reset_index(), on="date", how="left")
    out = _add_symbol_relative_strength(out)
    out = _add_symbol_expanding_return_deviation(out)

    return out.drop(columns=["_symbol_upper"])


def _benchmark_features_by_date(df: pd.DataFrame) -> pd.DataFrame:
    prices = df[["date", "_symbol_upper", "close"]].copy()
    price_by_date = prices.pivot_table(
        index="date",
        columns="_symbol_upper",
        values="close",
        aggfunc="first",
    ).sort_index(kind="mergesort")

    features = pd.DataFrame(index=price_by_date.index)
    daily_returns_by_symbol: dict[str, pd.Series] = {}

    for symbol in BENCHMARK_SYMBOLS:
        prefix = symbol.lower()
        if symbol not in price_by_date.columns:
            _add_missing_benchmark_columns(features, prefix)
            continue

        close = pd.to_numeric(price_by_date[symbol], errors="coerce")
        daily_return = close.pct_change()
        daily_returns_by_symbol[symbol] = daily_return

        for lookback in BENCHMARK_RETURN_LOOKBACKS:
            features[f"{prefix}_return_{lookback}d"] = close.pct_change(periods=lookback)
        for lookback in BENCHMARK_DRAWDOWN_LOOKBACKS:
            rolling_peak = close.rolling(window=lookback, min_periods=lookback).max()
            features[f"{prefix}_drawdown_{lookback}d"] = close / rolling_peak - 1.0
        for lookback in BENCHMARK_VOLATILITY_LOOKBACKS:
            features[f"{prefix}_volatility_{lookback}d"] = daily_return.rolling(
                window=lookback,
                min_periods=lookback,
            ).std()

    if {"SPY", "QQQ"}.issubset(price_by_date.columns):
        spy_close = pd.to_numeric(price_by_date["SPY"], errors="coerce")
        qqq_close = pd.to_numeric(price_by_date["QQQ"], errors="coerce")
        for lookback in REGIME_LOOKBACKS:
            features[f"qqq_minus_spy_return_{lookback}d"] = (
                features[f"qqq_return_{lookback}d"] - features[f"spy_return_{lookback}d"]
            )
            ratio = qqq_close / spy_close.replace(0.0, np.nan)
            features[f"qqq_div_spy_price_ratio_return_{lookback}d"] = ratio.pct_change(
                periods=lookback
            )
    else:
        for lookback in REGIME_LOOKBACKS:
            features[f"qqq_minus_spy_return_{lookback}d"] = pd.NA
            features[f"qqq_div_spy_price_ratio_return_{lookback}d"] = pd.NA

    return features


def _add_missing_benchmark_columns(features: pd.DataFrame, prefix: str) -> None:
    for lookback in BENCHMARK_RETURN_LOOKBACKS:
        features[f"{prefix}_return_{lookback}d"] = pd.NA
    for lookback in BENCHMARK_DRAWDOWN_LOOKBACKS:
        features[f"{prefix}_drawdown_{lookback}d"] = pd.NA
    for lookback in BENCHMARK_VOLATILITY_LOOKBACKS:
        features[f"{prefix}_volatility_{lookback}d"] = pd.NA


def _add_symbol_relative_strength(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    grouped_close = out.groupby("symbol", sort=False)["close"]

    symbol_returns: dict[int, pd.Series] = {}
    for lookback in RELATIVE_STRENGTH_LOOKBACKS:
        symbol_returns[lookback] = grouped_close.pct_change(periods=lookback)

    out["symbol_return_252d"] = symbol_returns[LONG_RUN_RETURN_LOOKBACK]
    for benchmark_symbol in BENCHMARK_SYMBOLS:
        prefix = benchmark_symbol.lower()
        for lookback in RELATIVE_STRENGTH_LOOKBACKS:
            out[f"symbol_return_{lookback}d_minus_{prefix}_return_{lookback}d"] = (
                symbol_returns[lookback] - out[f"{prefix}_return_{lookback}d"]
            )
    return out


def _add_symbol_expanding_return_deviation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    grouped_returns = out.groupby("symbol", sort=False)["symbol_return_252d"]

    shifted_returns = grouped_returns.shift(1)
    expanding_mean = shifted_returns.groupby(out["symbol"], sort=False).transform(
        lambda returns: returns.expanding(min_periods=1).mean()
    )
    expanding_std = shifted_returns.groupby(out["symbol"], sort=False).transform(
        lambda returns: returns.expanding(min_periods=2).std()
    )

    out["symbol_expanding_mean_return_252d_shifted"] = expanding_mean
    out["symbol_return_252d_minus_expanding_mean"] = (
        out["symbol_return_252d"] - expanding_mean
    )
    out["symbol_return_252d_zscore_vs_expanding_history"] = (
        out["symbol_return_252d"] - expanding_mean
    ) / expanding_std.replace(0.0, np.nan)
    return out

