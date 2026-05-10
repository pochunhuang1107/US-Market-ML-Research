"""Per-symbol technical features for ETF daily OHLCV rows.

All functions in this module assume the input has already been sorted by
``symbol`` and ``date``. Rolling, lagged, and exponential calculations are
performed within each symbol group so values for one ETF never use another
ETF's history. The features use the current row and earlier rows only; callers
should shift features later only if they need next-session execution semantics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RETURN_LOOKBACKS: tuple[int, ...] = (1, 5, 10, 20, 60)
SMA_LOOKBACKS: tuple[int, ...] = (20, 50, 200)
VOLATILITY_LOOKBACKS: tuple[int, ...] = (10, 20, 60)
RSI_LOOKBACK = 14
VOLUME_ZSCORE_LOOKBACK = 20


def add_price_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe per-symbol close-to-close return features."""

    out = df.copy()
    grouped_close = out.groupby("symbol", sort=False)["close"]

    for lookback in RETURN_LOOKBACKS:
        out[f"return_{lookback}d"] = grouped_close.pct_change(periods=lookback)

    return out


def add_moving_average_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-symbol close distance to rolling simple moving averages."""

    out = df.copy()
    grouped_close = out.groupby("symbol", sort=False)["close"]

    for lookback in SMA_LOOKBACKS:
        rolling_sma = grouped_close.transform(
            lambda close, window=lookback: close.rolling(
                window=window,
                min_periods=window,
            ).mean()
        )
        out[f"close_to_sma_{lookback}"] = out["close"] / rolling_sma - 1.0

    return out


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-symbol rolling volatility of daily close-to-close returns."""

    out = df.copy()
    out["_daily_return"] = out.groupby("symbol", sort=False)["close"].pct_change()
    grouped_returns = out.groupby("symbol", sort=False)["_daily_return"]

    for lookback in VOLATILITY_LOOKBACKS:
        out[f"realized_vol_{lookback}d"] = grouped_returns.transform(
            lambda returns, window=lookback: returns.rolling(
                window=window,
                min_periods=window,
            ).std()
        )

    return out.drop(columns=["_daily_return"])


def add_rsi_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add a leakage-safe simple 14-day RSI calculated independently per symbol."""

    out = df.copy()
    close_delta = out.groupby("symbol", sort=False)["close"].diff()
    gains = close_delta.clip(lower=0.0)
    losses = -close_delta.clip(upper=0.0)

    avg_gain = gains.groupby(out["symbol"], sort=False).transform(
        lambda gain: gain.rolling(window=RSI_LOOKBACK, min_periods=RSI_LOOKBACK).mean()
    )
    avg_loss = losses.groupby(out["symbol"], sort=False).transform(
        lambda loss: loss.rolling(window=RSI_LOOKBACK, min_periods=RSI_LOOKBACK).mean()
    )

    relative_strength = avg_gain / avg_loss.replace(0.0, np.nan)
    out["rsi_14"] = 100.0 - (100.0 / (1.0 + relative_strength))
    out.loc[(avg_loss == 0.0) & (avg_gain > 0.0), "rsi_14"] = 100.0

    return out


def add_macd_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add MACD features using per-symbol EMAs over current and historical closes."""

    out = df.copy()
    grouped_close = out.groupby("symbol", sort=False)["close"]
    ema_12 = grouped_close.transform(
        lambda close: close.ewm(span=12, adjust=False, min_periods=12).mean()
    )
    ema_26 = grouped_close.transform(
        lambda close: close.ewm(span=26, adjust=False, min_periods=26).mean()
    )

    out["macd_line"] = ema_12 - ema_26
    out["macd_signal"] = out.groupby("symbol", sort=False)["macd_line"].transform(
        lambda macd: macd.ewm(span=9, adjust=False, min_periods=9).mean()
    )
    out["macd_histogram"] = out["macd_line"] - out["macd_signal"]

    return out


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-symbol volume z-score and same-row close/VWAP gap features."""

    out = df.copy()
    grouped_volume = out.groupby("symbol", sort=False)["volume"]
    rolling_mean = grouped_volume.transform(
        lambda volume: volume.rolling(
            window=VOLUME_ZSCORE_LOOKBACK,
            min_periods=VOLUME_ZSCORE_LOOKBACK,
        ).mean()
    )
    rolling_std = grouped_volume.transform(
        lambda volume: volume.rolling(
            window=VOLUME_ZSCORE_LOOKBACK,
            min_periods=VOLUME_ZSCORE_LOOKBACK,
        ).std()
    )

    out["volume_zscore_20d"] = (out["volume"] - rolling_mean) / rolling_std.replace(0.0, np.nan)
    out["vwap_gap"] = out["close"] / out["vwap"] - 1.0

    return out


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all per-symbol technical features without cross-asset joins."""

    out = add_price_return_features(df)
    out = add_moving_average_features(out)
    out = add_volatility_features(out)
    out = add_rsi_feature(out)
    out = add_macd_features(out)
    return add_volume_features(out)
