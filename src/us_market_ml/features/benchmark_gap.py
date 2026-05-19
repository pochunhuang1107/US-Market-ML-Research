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
SAFE_DIVISION_EPSILON = 1.0e-12
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

BENCHMARK_GAP_V3_FEATURE_COLUMNS: tuple[str, ...] = (
    "qqq_return_63d_positive_flag",
    "qqq_return_126d_positive_flag",
    "qqq_return_252d_positive_flag",
    "spy_return_63d_positive_flag",
    "spy_return_126d_positive_flag",
    "spy_return_252d_positive_flag",
    "qqq_return_126d_top_quartile_756d",
    "qqq_return_252d_top_quartile_756d",
    "spy_return_126d_top_quartile_756d",
    "spy_return_252d_top_quartile_756d",
    "qqq_strong_regime_score",
    "spy_strong_regime_score",
    "max_spy_qqq_strong_regime_score",
    "symbol_minus_best_benchmark_return_21d",
    "symbol_minus_best_benchmark_return_63d",
    "symbol_minus_best_benchmark_return_126d",
    "symbol_minus_best_benchmark_return_252d",
    "symbol_minus_best_benchmark_return_63d_positive_flag",
    "symbol_minus_best_benchmark_return_126d_positive_flag",
    "symbol_minus_best_benchmark_return_252d_positive_flag",
    "symbol_minus_best_benchmark_return_126d_zscore_756d",
    "symbol_minus_best_benchmark_return_252d_zscore_756d",
    "symbol_return_63d_div_volatility_63d",
    "symbol_return_126d_div_volatility_63d",
    "qqq_return_63d_div_volatility_63d",
    "qqq_return_126d_div_volatility_63d",
    "spy_return_63d_div_volatility_63d",
    "spy_return_126d_div_volatility_63d",
    "symbol_risk_adjusted_return_63d_minus_qqq",
    "symbol_risk_adjusted_return_126d_minus_qqq",
    "symbol_risk_adjusted_return_63d_minus_spy",
    "symbol_risk_adjusted_return_126d_minus_spy",
    "symbol_risk_adjusted_return_126d_minus_best_benchmark",
    "symbol_drawdown_63d_minus_qqq_drawdown_63d",
    "symbol_drawdown_126d_minus_qqq_drawdown_126d",
    "symbol_drawdown_252d_minus_qqq_drawdown_252d",
    "symbol_drawdown_63d_minus_spy_drawdown_63d",
    "symbol_drawdown_126d_minus_spy_drawdown_126d",
    "symbol_drawdown_252d_minus_spy_drawdown_252d",
    "symbol_return_126d_minus_best_benchmark_return_126d_minus_drawdown_penalty",
    "symbol_return_252d_minus_best_benchmark_return_252d_minus_drawdown_penalty",
    "is_non_benchmark",
    "is_qqq_x_qqq_strong_regime_score",
    "is_spy_x_spy_strong_regime_score",
    "is_spy_or_qqq_x_max_spy_qqq_strong_regime_score",
    "is_growth_like_anchor_x_qqq_strong_regime_score",
    "is_non_benchmark_x_max_spy_qqq_strong_regime_score",
    "symbol_minus_best_benchmark_return_126d_x_is_non_benchmark",
    "symbol_risk_adjusted_return_126d_minus_best_benchmark_x_is_non_benchmark",
    "qqq_retention_pressure_126d",
    "qqq_retention_pressure_252d",
)

BENCHMARK_GAP_ALL_FEATURE_COLUMNS: tuple[str, ...] = (
    *BENCHMARK_GAP_FEATURE_COLUMNS,
    *BENCHMARK_GAP_V2_FEATURE_COLUMNS,
    *BENCHMARK_GAP_V3_FEATURE_COLUMNS,
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


def add_benchmark_gap_v3_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add benchmark-retention and risk-adjusted opportunity features on top of v2."""

    out = df.copy()
    if not set(BENCHMARK_GAP_ALL_FEATURE_COLUMNS).difference(
        BENCHMARK_GAP_V3_FEATURE_COLUMNS
    ).issubset(out.columns):
        out = add_benchmark_gap_v2_features(out)

    out["_symbol_upper"] = out["symbol"].astype("string").str.upper()
    benchmark_by_date = _benchmark_v3_features_by_date(out)
    out = out.merge(benchmark_by_date.reset_index(), on="date", how="left")
    out = _add_symbol_vs_best_benchmark_edge(out)
    out = _add_risk_adjusted_relative_strength(out)
    out = _add_drawdown_adjusted_opportunity(out)
    out = _add_benchmark_retention_interaction_features(out)
    out = _add_qqq_retention_pressure_features(out)

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


def _benchmark_v3_features_by_date(df: pd.DataFrame) -> pd.DataFrame:
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
            _add_missing_benchmark_v3_strength_columns(features, prefix)
            continue

        close = pd.to_numeric(price_by_date[symbol], errors="coerce")
        score_columns: list[str] = []
        for lookback in (63, 126, 252):
            returns = close.pct_change(periods=lookback)
            positive_flag_col = f"{prefix}_return_{lookback}d_positive_flag"
            features[positive_flag_col] = _positive_flag(returns)
            score_columns.append(positive_flag_col)
            if lookback in (126, 252):
                min_periods = min(
                    BENCHMARK_STRENGTH_HISTORY_LOOKBACK,
                    max(lookback, 126),
                )
                percentile = _rolling_percentile(
                    returns,
                    window=BENCHMARK_STRENGTH_HISTORY_LOOKBACK,
                    min_periods=min_periods,
                )
                top_quartile_col = f"{prefix}_return_{lookback}d_top_quartile_756d"
                features[top_quartile_col] = _positive_flag(percentile - 0.75)
                score_columns.append(top_quartile_col)

        features[f"{prefix}_strong_regime_score"] = _row_mean_or_nan(
            features[score_columns]
        )

    features["max_spy_qqq_strong_regime_score"] = pd.concat(
        [
            pd.to_numeric(features.get("spy_strong_regime_score"), errors="coerce"),
            pd.to_numeric(features.get("qqq_strong_regime_score"), errors="coerce"),
        ],
        axis=1,
    ).max(axis=1, skipna=True)
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


def _add_missing_benchmark_v3_strength_columns(
    features: pd.DataFrame,
    prefix: str,
) -> None:
    for lookback in (63, 126, 252):
        features[f"{prefix}_return_{lookback}d_positive_flag"] = pd.NA
    for lookback in (126, 252):
        features[f"{prefix}_return_{lookback}d_top_quartile_756d"] = pd.NA
    features[f"{prefix}_strong_regime_score"] = pd.NA


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


def _add_symbol_vs_best_benchmark_edge(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    grouped_close = out.groupby("symbol", sort=False)["close"]

    for lookback in BENCHMARK_RETURN_LOOKBACKS:
        symbol_return = grouped_close.pct_change(periods=lookback)
        best_benchmark_return = _best_benchmark_return(out, lookback)
        edge = symbol_return - best_benchmark_return
        edge_col = f"symbol_minus_best_benchmark_return_{lookback}d"
        out[edge_col] = edge
        if lookback in (63, 126, 252):
            out[f"{edge_col}_positive_flag"] = _positive_flag(edge)
        if lookback in (126, 252):
            out[f"{edge_col}_zscore_756d"] = _rolling_zscore_by_symbol(
                edge,
                out["symbol"],
                lookback=lookback,
            )
    return out


def _add_risk_adjusted_relative_strength(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    grouped_close = out.groupby("symbol", sort=False)["close"]
    daily_return = grouped_close.pct_change()
    symbol_volatility_63d = daily_return.groupby(out["symbol"], sort=False).transform(
        lambda returns: returns.rolling(window=63, min_periods=63).std()
    )

    symbol_risk_adjusted: dict[int, pd.Series] = {}
    for lookback in (63, 126):
        symbol_return = grouped_close.pct_change(periods=lookback)
        col = f"symbol_return_{lookback}d_div_volatility_63d"
        out[col] = _safe_divide(symbol_return, symbol_volatility_63d)
        symbol_risk_adjusted[lookback] = out[col]

    for prefix in ("qqq", "spy"):
        for lookback in (63, 126):
            out[f"{prefix}_return_{lookback}d_div_volatility_63d"] = _safe_divide(
                out[f"{prefix}_return_{lookback}d"],
                out[f"{prefix}_volatility_63d"],
            )

    for benchmark in ("qqq", "spy"):
        for lookback in (63, 126):
            out[f"symbol_risk_adjusted_return_{lookback}d_minus_{benchmark}"] = (
                symbol_risk_adjusted[lookback]
                - out[f"{benchmark}_return_{lookback}d_div_volatility_63d"]
            )

    best_benchmark_risk_adjusted_126d = pd.concat(
        [
            out["qqq_return_126d_div_volatility_63d"],
            out["spy_return_126d_div_volatility_63d"],
        ],
        axis=1,
    ).max(axis=1, skipna=True)
    out["symbol_risk_adjusted_return_126d_minus_best_benchmark"] = (
        symbol_risk_adjusted[126] - best_benchmark_risk_adjusted_126d
    )
    return out


def _add_drawdown_adjusted_opportunity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    grouped_close = out.groupby("symbol", sort=False)["close"]

    symbol_drawdowns: dict[int, pd.Series] = {}
    for lookback in BENCHMARK_DRAWDOWN_LOOKBACKS:
        rolling_peak = grouped_close.transform(
            lambda close, lookback=lookback: close.rolling(
                window=lookback,
                min_periods=lookback,
            ).max()
        )
        symbol_drawdown = pd.to_numeric(out["close"], errors="coerce") / rolling_peak - 1.0
        symbol_drawdowns[lookback] = symbol_drawdown
        for benchmark in ("qqq", "spy"):
            out[f"symbol_drawdown_{lookback}d_minus_{benchmark}_drawdown_{lookback}d"] = (
                symbol_drawdown - out[f"{benchmark}_drawdown_{lookback}d"]
            )

    for lookback in (126, 252):
        best_benchmark_drawdown = _best_benchmark_metric_by_return(
            out,
            lookback=lookback,
            spy_metric_col=f"spy_drawdown_{lookback}d",
            qqq_metric_col=f"qqq_drawdown_{lookback}d",
        )
        drawdown_penalty = (best_benchmark_drawdown - symbol_drawdowns[lookback]).clip(
            lower=0.0
        )
        out[
            "symbol_return_"
            f"{lookback}d_minus_best_benchmark_return_{lookback}d_minus_drawdown_penalty"
        ] = out[f"symbol_minus_best_benchmark_return_{lookback}d"] - drawdown_penalty
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


def _add_benchmark_retention_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_non_benchmark"] = (~out["_symbol_upper"].isin(BENCHMARK_SYMBOLS)).astype(
        "int8"
    )
    out["is_qqq_x_qqq_strong_regime_score"] = (
        out["is_qqq"] * out["qqq_strong_regime_score"]
    )
    out["is_spy_x_spy_strong_regime_score"] = (
        out["is_spy"] * out["spy_strong_regime_score"]
    )
    out["is_spy_or_qqq_x_max_spy_qqq_strong_regime_score"] = (
        out["is_spy_or_qqq"] * out["max_spy_qqq_strong_regime_score"]
    )
    out["is_growth_like_anchor_x_qqq_strong_regime_score"] = (
        out["is_growth_like_anchor"] * out["qqq_strong_regime_score"]
    )
    out["is_non_benchmark_x_max_spy_qqq_strong_regime_score"] = (
        out["is_non_benchmark"] * out["max_spy_qqq_strong_regime_score"]
    )
    out["symbol_minus_best_benchmark_return_126d_x_is_non_benchmark"] = (
        out["symbol_minus_best_benchmark_return_126d"] * out["is_non_benchmark"]
    )
    out[
        "symbol_risk_adjusted_return_126d_minus_best_benchmark_x_is_non_benchmark"
    ] = (
        out["symbol_risk_adjusted_return_126d_minus_best_benchmark"]
        * out["is_non_benchmark"]
    )
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


def _add_qqq_retention_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for lookback in (126, 252):
        symbol_minus_qqq_zscore = _rolling_zscore_by_symbol(
            out[f"symbol_return_{lookback}d_minus_qqq_return_{lookback}d"],
            out["symbol"],
            lookback=lookback,
        )
        out[f"qqq_retention_pressure_{lookback}d"] = (
            out[f"qqq_return_{lookback}d_zscore_756d"] - symbol_minus_qqq_zscore
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


def _best_benchmark_return(df: pd.DataFrame, lookback: int) -> pd.Series:
    return pd.concat(
        [
            pd.to_numeric(df[f"spy_return_{lookback}d"], errors="coerce"),
            pd.to_numeric(df[f"qqq_return_{lookback}d"], errors="coerce"),
        ],
        axis=1,
    ).max(axis=1, skipna=True)


def _best_benchmark_metric_by_return(
    df: pd.DataFrame,
    *,
    lookback: int,
    spy_metric_col: str,
    qqq_metric_col: str,
) -> pd.Series:
    spy_return = pd.to_numeric(df[f"spy_return_{lookback}d"], errors="coerce")
    qqq_return = pd.to_numeric(df[f"qqq_return_{lookback}d"], errors="coerce")
    spy_metric = pd.to_numeric(df[spy_metric_col], errors="coerce")
    qqq_metric = pd.to_numeric(df[qqq_metric_col], errors="coerce")

    choose_qqq = qqq_return >= spy_return
    best_metric = pd.Series(
        np.where(choose_qqq, qqq_metric, spy_metric),
        index=df.index,
        dtype="float64",
    )
    best_metric.loc[spy_return.isna() & qqq_return.notna()] = qqq_metric.loc[
        spy_return.isna() & qqq_return.notna()
    ]
    best_metric.loc[qqq_return.isna() & spy_return.notna()] = spy_metric.loc[
        qqq_return.isna() & spy_return.notna()
    ]
    best_metric.loc[spy_return.isna() & qqq_return.isna()] = np.nan
    return best_metric


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


def _rolling_zscore_by_symbol(
    values: pd.Series,
    symbols: pd.Series,
    *,
    lookback: int,
) -> pd.Series:
    min_periods = min(BENCHMARK_STRENGTH_HISTORY_LOOKBACK, max(lookback, 126))
    return values.groupby(symbols, sort=False).transform(
        lambda group: _rolling_zscore(
            group,
            window=BENCHMARK_STRENGTH_HISTORY_LOOKBACK,
            min_periods=min_periods,
        )
    )


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


def _positive_flag(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(
        np.where(numeric.notna(), numeric.gt(0.0).astype("float64"), np.nan),
        index=values.index,
        dtype="float64",
    )


def _row_mean_or_nan(frame: pd.DataFrame) -> pd.Series:
    non_null_count = frame.notna().sum(axis=1)
    mean = frame.mean(axis=1, skipna=True)
    mean.loc[non_null_count.eq(0)] = np.nan
    return mean


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator_numeric = pd.to_numeric(numerator, errors="coerce")
    denominator_numeric = pd.to_numeric(denominator, errors="coerce")
    safe_denominator = denominator_numeric.where(
        denominator_numeric.abs() > SAFE_DIVISION_EPSILON
    )
    result = numerator_numeric / safe_denominator
    result = result.replace([np.inf, -np.inf], np.nan)
    zero_denominator = (
        numerator_numeric.notna()
        & denominator_numeric.notna()
        & denominator_numeric.abs().le(SAFE_DIVISION_EPSILON)
    )
    result.loc[zero_denominator] = 0.0
    return result


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
