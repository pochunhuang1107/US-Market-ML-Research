"""Feature pipeline for curated ETF daily OHLCV research rows."""

from __future__ import annotations

import pandas as pd

from us_market_ml.features.cross_asset import add_cross_asset_features
from us_market_ml.features.technical import add_technical_features

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"date", "symbol", "open", "high", "low", "close", "volume", "vwap"}
)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build a leakage-safe ETF feature frame from daily OHLCV rows.

    The input is copied, sorted by ``symbol`` and ``date``, and never filtered.
    Per-symbol features are calculated with pandas groupby operations so rolling
    windows, lagged returns, RSI, MACD, and volume statistics only use rows from
    the same symbol at the current date or earlier dates. Cross-asset features
    are same-date differences between already trailing 20-day returns and are
    joined back by date without forward-fill or backward-fill.
    """

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    out = df.copy()
    out = out.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    out = add_technical_features(out)
    return add_cross_asset_features(out)
