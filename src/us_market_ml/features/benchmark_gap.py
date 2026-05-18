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
BENCHMARK_STRENGTH_HISTORY_LOOKBACK = 756
BENCHMARK_DOMINANCE_ZSCORE_LOOKBACK = 252
GROWTH_LIKE_ANCHORS: frozenset[str] = frozenset({"QQQ", "XLK", "VUG", "ARKK", "MTUM"})
DEFENSIVE_OR_BOND_LIKE_SYMBOLS: frozenset[str] = frozenset(
    {"XLU", "XLP", "XLV", "IEF", "TLT", "LQD"}
)
COMMODITY_LIKE_SYMBOLS: frozenset[str] = frozenset({"GLD", "SLV"})

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

BENCHMARK_GAP_V2_FEATURE_COLUMNS: tuple[str, ...] = (
    "qqq_vs_spy_return_21d",
    "qqq_vs_spy_return_63d",
    "qqq_vs_spy_return_126d",
    "qqq_vs_spy_return_252d",
    "qqq_vs_spy_return_zscore_252d",
    "qqq_vs_spy_ratio_drawdown_63d",
    "qqq_vs_spy_ratio_drawdown_126d",
    "qqq_vs_spy_ratio_drawdown_252d",
    "spy_return_63d_zscore_756d",
    "spy_return_126d_zscore_756d",
    "spy_return_252d_zscore_756d",
    "qqq_return_63d_zscore_756d",
    "qqq_return_126d_zscore_756d",
    "qqq_return_252d_zscore_756d",
    "spy_return_63d_percentile_756d",
    "spy_return_126d_percentile_756d",
    "qqq_return_63d_percentile_756d",
    "qqq_return_126d_percentile_756d",
    "symbol_minus_max_spy_qqq_return_21d",
    "symbol_minus_max_spy_qqq_return_63d",
    "symbol_minus_max_spy_qqq_return_126d",
    "symbol_minus_max_spy_qqq_return_252d",
    "symbol_return_63d_rank_vs_spy_qqq",
    "symbol_return_126d_rank_vs_spy_qqq",
    "symbol_return_252d_rank_vs_spy_qqq",
    "is_spy",
    "is_qqq",
    "is_spy_or_qqq",
    "is_growth_like_anchor",
    "is_defensive_or_bond_like",
    "is_commodity_like",
    "is_qqq_x_qqq_return_126d_zscore_756d",
    "is_spy_x_spy_return_126d_zscore_756d",
    "is_growth_like_x_qqq_vs_spy_return_126d",
)

BENCHMARK_GAP_ALL_FEATURE_COLUMNS: tuple[str, ...] = (
    *BENCHMARK_GAP_FEATURE_COLUMNS,
    *BENCHMARK_GAP_V2_FEATURE_COLUMNS,
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


def add_benchmark_gap_v2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add stronger benchmark-retention/regime features on top of v1."""

    out = df.copy()
    if not set(BENCHMARK_GAP_FEATURE_COLUMNS).issubset(out.columns):
        out = add_benchmark_gap_features(out)

    out["_symbol_upper"] = out["symbol"].astype("string").str.upper()
    benchmark_by_date = _benchmark_v2_features_by_date(out)
    out = out.merge(benchmark_by_date.reset_index(), on="date", how="left")
    out = _add_symbol_vs_benchmark_opportunity(out)
    out = _add_benchmark_identity_features(out)
    out = _add_benchmark_interaction_features(out)

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


def _benchmark_v2_features_by_date(df: pd.DataFrame) -> pd.DataFrame:
    prices = df[["date", "_symbol_upper", "close"]].copy()
    price_by_date = prices.pivot_table(
        index="date",
        columns="_symbol_upper",
        values="close",
        aggfunc="first",
    ).sort_index(kind="mergesort")

    features = pd.DataFrame(index=price_by_date.index)
    for symbol in BENCHMARK_SYMBOLS:
        prefix = symbol.lower()
        if symbol not in price_by_date.columns:
            _add_missing_benchmark_v2_strength_columns(features, prefix)
            continue

        close = pd.to_numeric(price_by_date[symbol], errors="coerce")
        for lookback in (63, 126, 252):
            returns = close.pct_change(periods=lookback)
            min_periods = min(BENCHMARK_STRENGTH_HISTORY_LOOKBACK, max(lookback, 126))
            features[f"{prefix}_return_{lookback}d_zscore_756d"] = _rolling_zscore(
                returns,
                window=BENCHMARK_STRENGTH_HISTORY_LOOKBACK,
                min_periods=min_periods,
            )
            if lookback in (63, 126):
                features[f"{prefix}_return_{lookback}d_percentile_756d"] = (
                    _rolling_percentile(
                        returns,
                        window=BENCHMARK_STRENGTH_HISTORY_LOOKBACK,
                        min_periods=min_periods,
                    )
                )

    if {"SPY", "QQQ"}.issubset(price_by_date.columns):
        spy_close = pd.to_numeric(price_by_date["SPY"], errors="coerce")
        qqq_close = pd.to_numeric(price_by_date["QQQ"], errors="coerce")
        ratio = qqq_close / spy_close.replace(0.0, np.nan)
        ratio_return_21d = ratio.pct_change(periods=21)
        for lookback in BENCHMARK_RETURN_LOOKBACKS:
            features[f"qqq_vs_spy_return_{lookback}d"] = ratio.pct_change(
                periods=lookback
            )
        features["qqq_vs_spy_return_zscore_252d"] = _rolling_zscore(
            ratio_return_21d,
            window=BENCHMARK_DOMINANCE_ZSCORE_LOOKBACK,
            min_periods=126,
        )
        for lookback in BENCHMARK_DRAWDOWN_LOOKBACKS:
            rolling_peak = ratio.rolling(window=lookback, min_periods=lookback).max()
            features[f"qqq_vs_spy_ratio_drawdown_{lookback}d"] = (
                ratio / rolling_peak - 1.0
            )
    else:
        for column in _benchmark_dominance_columns():
            features[column] = pd.NA

    return features


def _add_missing_benchmark_columns(features: pd.DataFrame, prefix: str) -> None:
    for lookback in BENCHMARK_RETURN_LOOKBACKS:
        features[f"{prefix}_return_{lookback}d"] = pd.NA
    for lookback in BENCHMARK_DRAWDOWN_LOOKBACKS:
        features[f"{prefix}_drawdown_{lookback}d"] = pd.NA
    for lookback in BENCHMARK_VOLATILITY_LOOKBACKS:
        features[f"{prefix}_volatility_{lookback}d"] = pd.NA


def _add_missing_benchmark_v2_strength_columns(
    features: pd.DataFrame,
    prefix: str,
) -> None:
    for lookback in (63, 126, 252):
        features[f"{prefix}_return_{lookback}d_zscore_756d"] = pd.NA
        if lookback in (63, 126):
            features[f"{prefix}_return_{lookback}d_percentile_756d"] = pd.NA


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


def _add_symbol_vs_benchmark_opportunity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    grouped_close = out.groupby("symbol", sort=False)["close"]

    for lookback in BENCHMARK_RETURN_LOOKBACKS:
        symbol_return = grouped_close.pct_change(periods=lookback)
        spy_return = out[f"spy_return_{lookback}d"]
        qqq_return = out[f"qqq_return_{lookback}d"]
        benchmark_returns = pd.concat([spy_return, qqq_return], axis=1)
        max_benchmark_return = benchmark_returns.max(axis=1, skipna=True)
        out[f"symbol_minus_max_spy_qqq_return_{lookback}d"] = (
            symbol_return - max_benchmark_return
        )
        if lookback in RELATIVE_STRENGTH_LOOKBACKS:
            out[f"symbol_return_{lookback}d_rank_vs_spy_qqq"] = (
                _rank_symbol_return_vs_benchmarks(symbol_return, spy_return, qqq_return)
            )
    return out


def _add_benchmark_identity_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    symbols = out["_symbol_upper"]
    out["is_spy"] = symbols.eq("SPY").astype("int8")
    out["is_qqq"] = symbols.eq("QQQ").astype("int8")
    out["is_spy_or_qqq"] = symbols.isin(BENCHMARK_SYMBOLS).astype("int8")
    out["is_growth_like_anchor"] = symbols.isin(GROWTH_LIKE_ANCHORS).astype("int8")
    out["is_defensive_or_bond_like"] = symbols.isin(
        DEFENSIVE_OR_BOND_LIKE_SYMBOLS
    ).astype("int8")
    out["is_commodity_like"] = symbols.isin(COMMODITY_LIKE_SYMBOLS).astype("int8")
    return out


def _add_benchmark_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_qqq_x_qqq_return_126d_zscore_756d"] = (
        out["is_qqq"] * out["qqq_return_126d_zscore_756d"]
    )
    out["is_spy_x_spy_return_126d_zscore_756d"] = (
        out["is_spy"] * out["spy_return_126d_zscore_756d"]
    )
    out["is_growth_like_x_qqq_vs_spy_return_126d"] = (
        out["is_growth_like_anchor"] * out["qqq_vs_spy_return_126d"]
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


def _benchmark_dominance_columns() -> tuple[str, ...]:
    return tuple(
        column
        for column in BENCHMARK_GAP_V2_FEATURE_COLUMNS
        if column.startswith("qqq_vs_spy_")
    )


def _rolling_zscore(
    values: pd.Series,
    *,
    window: int,
    min_periods: int,
) -> pd.Series:
    rolling = values.rolling(window=window, min_periods=min_periods)
    mean = rolling.mean()
    std = rolling.std()
    return (values - mean) / std.replace(0.0, np.nan)


def _rolling_percentile(
    values: pd.Series,
    *,
    window: int,
    min_periods: int,
) -> pd.Series:
    return values.rolling(window=window, min_periods=min_periods).apply(
        _last_value_percentile,
        raw=True,
    )


def _last_value_percentile(values: np.ndarray) -> float:
    current = values[-1]
    if np.isnan(current):
        return float("nan")
    non_null_values = values[~np.isnan(values)]
    if non_null_values.size == 0:
        return float("nan")
    return float(np.mean(non_null_values <= current))


def _rank_symbol_return_vs_benchmarks(
    symbol_return: pd.Series,
    spy_return: pd.Series,
    qqq_return: pd.Series,
) -> pd.Series:
    rank_frame = pd.concat(
        [symbol_return, spy_return, qqq_return],
        axis=1,
        keys=["symbol", "spy", "qqq"],
    )
    return rank_frame.rank(axis=1, ascending=False, method="min")["symbol"]
