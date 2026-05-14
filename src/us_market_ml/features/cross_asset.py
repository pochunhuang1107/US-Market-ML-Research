"""Cross-asset relative strength features joined at the date level.

The public function expects rows that contain a leakage-safe ``return_20d``
feature, typically produced by ``build_feature_frame`` before this module is
called. Each cross-asset value is calculated from same-date trailing returns
only, then joined back to every row for that date. Missing pair constituents
produce all-NaN feature columns instead of errors.
"""

from __future__ import annotations

import pandas as pd

CROSS_ASSET_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("qqq_spy_return_ratio_20d", "QQQ", "SPY"),
    ("iwm_spy_return_ratio_20d", "IWM", "SPY"),
    ("tlt_spy_return_ratio_20d", "TLT", "SPY"),
    ("gld_spy_return_ratio_20d", "GLD", "SPY"),
    ("hyg_lqd_return_ratio_20d", "HYG", "LQD"),
    ("xlk_xlf_return_ratio_20d", "XLK", "XLF"),
)

CROSS_ASSET_FEATURES: tuple[str, ...] = tuple(name for name, _, _ in CROSS_ASSET_PAIRS)
TECHNICAL_V2_RANK_FEATURES: tuple[str, ...] = (
    "return_20d_rank_pct",
    "return_60d_rank_pct",
    "volatility_20d_rank_pct",
    "close_to_sma_20_rank_pct",
    "dollar_volume_20d_rank_pct",
)
TECHNICAL_V2_RELATIVE_FEATURES: tuple[str, ...] = (
    "relative_strength_vs_spy_20d",
    "relative_strength_vs_qqq_20d",
    "beta_to_spy_60d",
    "correlation_to_spy_60d",
)


def add_cross_asset_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add date-level relative strength features without using future rows.

    Each feature is ``symbol_A return_20d - symbol_B return_20d`` for the same
    date. Because ``return_20d`` is a trailing per-symbol return, the resulting
    date-level values remain leakage-safe as long as the input ``return_20d`` is
    leakage-safe. No generated values are forward-filled or backward-filled.
    """

    out = df.copy()

    if "return_20d" not in out.columns:
        for feature_name in CROSS_ASSET_FEATURES:
            out[feature_name] = pd.NA
        return out

    returns = out[["date", "symbol", "return_20d"]].copy()
    returns["_symbol_upper"] = returns["symbol"].astype("string").str.upper()
    date_symbol_returns = returns.pivot_table(
        index="date",
        columns="_symbol_upper",
        values="return_20d",
        aggfunc="first",
    )

    cross_asset_by_date = pd.DataFrame(index=date_symbol_returns.index)
    for feature_name, symbol_a, symbol_b in CROSS_ASSET_PAIRS:
        if symbol_a in date_symbol_returns.columns and symbol_b in date_symbol_returns.columns:
            cross_asset_by_date[feature_name] = (
                date_symbol_returns[symbol_a] - date_symbol_returns[symbol_b]
            )
        else:
            cross_asset_by_date[feature_name] = pd.NA

    return out.merge(cross_asset_by_date.reset_index(), on="date", how="left")


def add_technical_v2_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add same-date ranks and benchmark-relative features for technical v2."""

    out = df.copy()

    rank_sources = {
        "return_20d_rank_pct": "return_20d",
        "return_60d_rank_pct": "return_60d",
        "volatility_20d_rank_pct": "volatility_20d",
        "close_to_sma_20_rank_pct": "close_to_sma_20",
        "dollar_volume_20d_rank_pct": "dollar_volume_20d",
    }
    for output_column, source_column in rank_sources.items():
        if source_column in out.columns:
            out[output_column] = out.groupby("date", sort=False)[source_column].rank(
                pct=True,
                method="average",
            )
        else:
            out[output_column] = pd.NA

    out = _add_relative_strength_feature(out, benchmark_symbol="SPY")
    out = _add_relative_strength_feature(out, benchmark_symbol="QQQ")
    return _add_spy_beta_and_correlation(out)


def _add_relative_strength_feature(df: pd.DataFrame, benchmark_symbol: str) -> pd.DataFrame:
    out = df.copy()
    output_column = f"relative_strength_vs_{benchmark_symbol.lower()}_20d"
    if "return_20d" not in out.columns:
        out[output_column] = pd.NA
        return out

    benchmark_returns = _benchmark_series(out, benchmark_symbol, "return_20d")
    if benchmark_returns.empty:
        out[output_column] = pd.NA
        return out

    out = out.merge(
        benchmark_returns.rename(columns={"return_20d": "_benchmark_return_20d"}),
        on="date",
        how="left",
    )
    out[output_column] = out["return_20d"] - out["_benchmark_return_20d"]
    return out.drop(columns=["_benchmark_return_20d"])


def _add_spy_beta_and_correlation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "close" not in out.columns:
        out["beta_to_spy_60d"] = pd.NA
        out["correlation_to_spy_60d"] = pd.NA
        return out

    out["_daily_return"] = out.groupby("symbol", sort=False)["close"].pct_change()
    benchmark_returns = _benchmark_series(out, "SPY", "_daily_return")
    if benchmark_returns.empty:
        out["beta_to_spy_60d"] = pd.NA
        out["correlation_to_spy_60d"] = pd.NA
        return out.drop(columns=["_daily_return"])

    out = out.merge(
        benchmark_returns.rename(columns={"_daily_return": "_spy_daily_return"}),
        on="date",
        how="left",
    )
    grouped_returns = out.groupby("symbol", sort=False)["_daily_return"]
    rolling_covariance = grouped_returns.transform(
        lambda returns: returns.rolling(window=60, min_periods=60).cov(
            out.loc[returns.index, "_spy_daily_return"]
        )
    )
    rolling_spy_variance = out.groupby("symbol", sort=False)["_spy_daily_return"].transform(
        lambda returns: returns.rolling(window=60, min_periods=60).var()
    )
    out["beta_to_spy_60d"] = rolling_covariance / rolling_spy_variance.replace(0.0, pd.NA)
    out["correlation_to_spy_60d"] = grouped_returns.transform(
        lambda returns: returns.rolling(window=60, min_periods=60).corr(
            out.loc[returns.index, "_spy_daily_return"]
        )
    )
    return out.drop(columns=["_daily_return", "_spy_daily_return"])


def _benchmark_series(df: pd.DataFrame, symbol: str, value_column: str) -> pd.DataFrame:
    if value_column not in df.columns:
        return pd.DataFrame(columns=["date", value_column])

    benchmark = df.loc[
        df["symbol"].astype("string").str.upper() == symbol.upper(),
        ["date", value_column],
    ].copy()
    if benchmark.empty:
        return pd.DataFrame(columns=["date", value_column])
    return benchmark.drop_duplicates(subset=["date"], keep="first")
