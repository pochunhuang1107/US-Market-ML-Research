"""Long-history regime and QQQ-relative opportunity features.

All features in this module are same-row or trailing historical statistics.
Cross-sectional breadth features use the row's own date and only symbols that
have already accumulated enough history by that date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from us_market_ml.features.benchmark_gap import add_benchmark_gap_v3_features

LONG_HISTORY_REGIME_FEATURE_COLUMNS: tuple[str, ...] = (
    "qqq_return_252d_minus_long_run_mean",
    "qqq_return_756d_minus_long_run_mean",
    "qqq_distance_from_sma200",
    "qqq_distance_from_sma50",
    "qqq_distance_from_252d_high",
    "qqq_drawdown_63d",
    "qqq_drawdown_126d",
    "qqq_drawdown_252d",
    "qqq_recovery_from_252d_low",
    "qqq_rolling_volatility_63d",
    "qqq_volatility_percentile_756d",
    "qqq_vs_spy_return_63d",
    "qqq_vs_spy_return_126d",
    "qqq_vs_spy_return_252d",
    "qqq_vs_spy_volatility_ratio_63d",
    "qqq_vs_spy_drawdown_diff_126d",
    "xlk_vs_spy_return_126d",
    "vug_vs_vtv_return_126d",
    "iwm_vs_spy_return_126d",
    "hyg_vs_lqd_return_126d",
    "tlt_return_126d",
    "ief_return_126d",
    "gld_return_126d",
    "slv_return_126d",
    "pct_symbols_above_sma200",
    "pct_symbols_above_sma50",
    "cross_section_return_dispersion_63d",
    "cross_section_return_dispersion_126d",
    "top_decile_return_63d_minus_bottom_decile_return_63d",
    "median_symbol_return_63d",
    "median_symbol_return_126d",
    "symbol_minus_qqq_return_20d",
    "symbol_minus_qqq_return_63d",
    "symbol_minus_qqq_return_126d",
    "symbol_minus_qqq_return_252d",
    "symbol_return_63d_div_volatility_63d_minus_qqq",
    "symbol_drawdown_126d_minus_qqq_drawdown_126d",
    "symbol_distance_from_252d_high_minus_qqq",
    "symbol_252d_return_minus_symbol_long_run_mean",
    "symbol_756d_return_minus_symbol_long_run_mean",
    "symbol_return_252d_zscore_expanding",
    "symbol_return_756d_zscore_expanding",
    "qqq_return_252d_zscore_expanding",
    "qqq_return_756d_zscore_expanding",
)

LONG_HISTORY_MIN_BREADTH_HISTORY_ROWS = 252
SAFE_DIVISION_EPSILON = 1.0e-12


def add_long_history_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add long-history QQQ-relative regime and opportunity features."""

    out = df.copy()
    if not _has_benchmark_gap_v3_columns(out):
        out = add_benchmark_gap_v3_features(out)

    out["_symbol_upper"] = out["symbol"].astype("string").str.upper()
    out = out.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

    by_date_features = _long_history_features_by_date(out)
    out = out.merge(by_date_features.reset_index(), on="date", how="left")
    out = _add_breadth_features(out)
    out = _add_symbol_qqq_opportunity_features(out)
    out = _add_symbol_cycle_features(out)

    return out.drop(columns=["_symbol_upper"])


def _has_benchmark_gap_v3_columns(df: pd.DataFrame) -> bool:
    return {
        "qqq_return_252d",
        "qqq_drawdown_252d",
        "qqq_vs_spy_return_252d",
        "symbol_drawdown_126d_minus_qqq_drawdown_126d",
    }.issubset(df.columns)


def _long_history_features_by_date(df: pd.DataFrame) -> pd.DataFrame:
    price_by_date = _price_by_date(df)
    features = pd.DataFrame(index=price_by_date.index)

    if "QQQ" in price_by_date.columns:
        qqq_close = pd.to_numeric(price_by_date["QQQ"], errors="coerce")
        qqq_daily_return = qqq_close.pct_change()
        qqq_return_252d = qqq_close.pct_change(periods=252)
        qqq_return_756d = qqq_close.pct_change(periods=756)
        qqq_volatility_63d = qqq_daily_return.rolling(window=63, min_periods=63).std()

        features["qqq_return_252d_minus_long_run_mean"] = (
            qqq_return_252d - qqq_return_252d.expanding(min_periods=1).mean()
        )
        features["qqq_return_756d_minus_long_run_mean"] = (
            qqq_return_756d - qqq_return_756d.expanding(min_periods=1).mean()
        )
        features["qqq_distance_from_sma200"] = qqq_close / qqq_close.rolling(
            window=200,
            min_periods=200,
        ).mean() - 1.0
        features["qqq_distance_from_sma50"] = qqq_close / qqq_close.rolling(
            window=50,
            min_periods=50,
        ).mean() - 1.0
        rolling_high_252 = qqq_close.rolling(window=252, min_periods=252).max()
        rolling_low_252 = qqq_close.rolling(window=252, min_periods=252).min()
        features["qqq_distance_from_252d_high"] = qqq_close / rolling_high_252 - 1.0
        features["qqq_recovery_from_252d_low"] = qqq_close / rolling_low_252 - 1.0
        features["qqq_rolling_volatility_63d"] = qqq_volatility_63d
        features["qqq_volatility_percentile_756d"] = _rolling_percentile(
            qqq_volatility_63d,
            window=756,
            min_periods=252,
        )
        features["qqq_return_252d_zscore_expanding"] = _expanding_zscore(qqq_return_252d)
        features["qqq_return_756d_zscore_expanding"] = _expanding_zscore(qqq_return_756d)
    else:
        for column in (
            "qqq_return_252d_minus_long_run_mean",
            "qqq_return_756d_minus_long_run_mean",
            "qqq_distance_from_sma200",
            "qqq_distance_from_sma50",
            "qqq_distance_from_252d_high",
            "qqq_recovery_from_252d_low",
            "qqq_rolling_volatility_63d",
            "qqq_volatility_percentile_756d",
            "qqq_return_252d_zscore_expanding",
            "qqq_return_756d_zscore_expanding",
        ):
            features[column] = pd.NA

    features["qqq_vs_spy_volatility_ratio_63d"] = _volatility_ratio(
        price_by_date,
        "QQQ",
        "SPY",
        lookback=63,
    )
    features["qqq_vs_spy_drawdown_diff_126d"] = _drawdown(price_by_date, "QQQ", 126) - (
        _drawdown(price_by_date, "SPY", 126)
    )

    spread_specs = {
        "xlk_vs_spy_return_126d": ("XLK", "SPY"),
        "vug_vs_vtv_return_126d": ("VUG", "VTV"),
        "iwm_vs_spy_return_126d": ("IWM", "SPY"),
        "hyg_vs_lqd_return_126d": ("HYG", "LQD"),
    }
    for column, (left, right) in spread_specs.items():
        features[column] = _return_spread(price_by_date, left, right, lookback=126)

    for symbol in ("TLT", "IEF", "GLD", "SLV"):
        features[f"{symbol.lower()}_return_126d"] = _symbol_return(
            price_by_date,
            symbol,
            lookback=126,
        )

    return features


def _add_breadth_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    grouped_close = out.groupby("symbol", sort=False)["close"]
    history_count = grouped_close.cumcount() + 1
    out["__long_history_above_sma200"] = out["close_to_sma_200"].gt(0.0)
    out["__long_history_above_sma50"] = out["close_to_sma_50"].gt(0.0)
    out["__long_history_symbol_return_63d"] = grouped_close.pct_change(periods=63)
    out["__long_history_symbol_return_126d"] = grouped_close.pct_change(periods=126)
    eligible = out.loc[history_count.ge(LONG_HISTORY_MIN_BREADTH_HISTORY_ROWS)].copy()

    if eligible.empty:
        for column in (
            "pct_symbols_above_sma200",
            "pct_symbols_above_sma50",
            "cross_section_return_dispersion_63d",
            "cross_section_return_dispersion_126d",
            "top_decile_return_63d_minus_bottom_decile_return_63d",
            "median_symbol_return_63d",
            "median_symbol_return_126d",
        ):
            out[column] = np.nan
        return out.drop(
            columns=[
                "__long_history_above_sma200",
                "__long_history_above_sma50",
                "__long_history_symbol_return_63d",
                "__long_history_symbol_return_126d",
            ]
        )

    breadth = (
        eligible.groupby("date", sort=False)
        .agg(
            pct_symbols_above_sma200=("__long_history_above_sma200", "mean"),
            pct_symbols_above_sma50=("__long_history_above_sma50", "mean"),
            cross_section_return_dispersion_63d=(
                "__long_history_symbol_return_63d",
                "std",
            ),
            cross_section_return_dispersion_126d=(
                "__long_history_symbol_return_126d",
                "std",
            ),
            median_symbol_return_63d=("__long_history_symbol_return_63d", "median"),
            median_symbol_return_126d=("__long_history_symbol_return_126d", "median"),
            top_decile_return_63d_minus_bottom_decile_return_63d=(
                "__long_history_symbol_return_63d",
                _top_decile_minus_bottom_decile,
            ),
        )
        .reset_index()
    )
    out = out.drop(
        columns=[
            "__long_history_above_sma200",
            "__long_history_above_sma50",
            "__long_history_symbol_return_63d",
            "__long_history_symbol_return_126d",
        ]
    )
    return out.merge(breadth, on="date", how="left")


def _add_symbol_qqq_opportunity_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    grouped_close = out.groupby("symbol", sort=False)["close"]
    symbol_returns = {
        lookback: grouped_close.pct_change(periods=lookback)
        for lookback in (20, 63, 126, 252)
    }
    qqq_return_by_date = _price_by_date(out)["QQQ"].pct_change(periods=20) if _has_symbol(
        out,
        "QQQ",
    ) else pd.Series(index=out["date"].drop_duplicates(), dtype="float64")
    qqq_return_20d = out["date"].map(qqq_return_by_date)
    qqq_returns = {
        20: qqq_return_20d,
        63: out["qqq_return_63d"],
        126: out["qqq_return_126d"],
        252: out["qqq_return_252d"],
    }

    for lookback in (20, 63, 126, 252):
        out[f"symbol_minus_qqq_return_{lookback}d"] = (
            symbol_returns[lookback] - qqq_returns[lookback]
        )

    daily_return = grouped_close.pct_change()
    symbol_volatility_63d = daily_return.groupby(out["symbol"], sort=False).transform(
        lambda returns: returns.rolling(window=63, min_periods=63).std()
    )
    symbol_risk_adjusted_63d = _safe_divide(symbol_returns[63], symbol_volatility_63d)
    qqq_risk_adjusted_63d = _safe_divide(out["qqq_return_63d"], out["qqq_volatility_63d"])
    out["symbol_return_63d_div_volatility_63d_minus_qqq"] = (
        symbol_risk_adjusted_63d - qqq_risk_adjusted_63d
    )

    symbol_high_252 = grouped_close.transform(
        lambda close: close.rolling(window=252, min_periods=252).max()
    )
    out["symbol_distance_from_252d_high_minus_qqq"] = (
        pd.to_numeric(out["close"], errors="coerce") / symbol_high_252 - 1.0
    ) - out["qqq_distance_from_252d_high"]
    return out


def _add_symbol_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    grouped_close = out.groupby("symbol", sort=False)["close"]
    symbol_return_252d = grouped_close.pct_change(periods=252)
    symbol_return_756d = grouped_close.pct_change(periods=756)

    expanding_mean_252d = symbol_return_252d.groupby(out["symbol"], sort=False).transform(
        lambda values: values.expanding(min_periods=1).mean()
    )
    expanding_mean_756d = symbol_return_756d.groupby(out["symbol"], sort=False).transform(
        lambda values: values.expanding(min_periods=1).mean()
    )
    out["symbol_252d_return_minus_symbol_long_run_mean"] = (
        symbol_return_252d - expanding_mean_252d
    )
    out["symbol_756d_return_minus_symbol_long_run_mean"] = (
        symbol_return_756d - expanding_mean_756d
    )
    out["symbol_return_252d_zscore_expanding"] = symbol_return_252d.groupby(
        out["symbol"],
        sort=False,
    ).transform(_expanding_zscore)
    out["symbol_return_756d_zscore_expanding"] = symbol_return_756d.groupby(
        out["symbol"],
        sort=False,
    ).transform(_expanding_zscore)
    return out


def _price_by_date(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[["date", "_symbol_upper", "close"]]
        .pivot_table(index="date", columns="_symbol_upper", values="close", aggfunc="first")
        .sort_index(kind="mergesort")
    )


def _has_symbol(df: pd.DataFrame, symbol: str) -> bool:
    return bool(df["_symbol_upper"].eq(symbol.upper()).any())


def _symbol_return(price_by_date: pd.DataFrame, symbol: str, *, lookback: int) -> pd.Series:
    if symbol not in price_by_date.columns:
        return pd.Series(np.nan, index=price_by_date.index, dtype="float64")
    return pd.to_numeric(price_by_date[symbol], errors="coerce").pct_change(periods=lookback)


def _return_spread(
    price_by_date: pd.DataFrame,
    left: str,
    right: str,
    *,
    lookback: int,
) -> pd.Series:
    return _symbol_return(price_by_date, left, lookback=lookback) - _symbol_return(
        price_by_date,
        right,
        lookback=lookback,
    )


def _volatility_ratio(
    price_by_date: pd.DataFrame,
    left: str,
    right: str,
    *,
    lookback: int,
) -> pd.Series:
    if left not in price_by_date.columns or right not in price_by_date.columns:
        return pd.Series(np.nan, index=price_by_date.index, dtype="float64")
    left_vol = pd.to_numeric(price_by_date[left], errors="coerce").pct_change().rolling(
        window=lookback,
        min_periods=lookback,
    ).std()
    right_vol = pd.to_numeric(price_by_date[right], errors="coerce").pct_change().rolling(
        window=lookback,
        min_periods=lookback,
    ).std()
    return _safe_divide(left_vol, right_vol)


def _drawdown(price_by_date: pd.DataFrame, symbol: str, lookback: int) -> pd.Series:
    if symbol not in price_by_date.columns:
        return pd.Series(np.nan, index=price_by_date.index, dtype="float64")
    close = pd.to_numeric(price_by_date[symbol], errors="coerce")
    peak = close.rolling(window=lookback, min_periods=lookback).max()
    return close / peak - 1.0


def _expanding_zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    mean = numeric.expanding(min_periods=1).mean()
    std = numeric.expanding(min_periods=2).std()
    return (numeric - mean) / std.replace(0.0, np.nan)


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


def _top_decile_minus_bottom_decile(values: pd.Series) -> float:
    clean_values = pd.to_numeric(values, errors="coerce").dropna()
    if clean_values.empty:
        return float("nan")
    top_cutoff = clean_values.quantile(0.9)
    bottom_cutoff = clean_values.quantile(0.1)
    top = clean_values.loc[clean_values.ge(top_cutoff)]
    bottom = clean_values.loc[clean_values.le(bottom_cutoff)]
    if top.empty or bottom.empty:
        return float("nan")
    return float(top.mean() - bottom.mean())


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


__all__ = [
    "LONG_HISTORY_REGIME_FEATURE_COLUMNS",
    "add_long_history_regime_features",
]
