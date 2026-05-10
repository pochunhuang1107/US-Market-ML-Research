"""Forward-return supervised labels for ETF rotation research.

Forward-return labels intentionally look into future rows because they are
supervised learning targets. They must never be included in feature columns or
used by feature generation. Generate these labels only after leakage-safe
features have been built, or in a clearly separated dataset-building step.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def add_forward_returns(
    df: pd.DataFrame,
    horizons: Sequence[int] = (5, 10, 20),
    price_col: str = "close",
) -> pd.DataFrame:
    """Add per-symbol future close-to-close return labels.

    For each horizon, ``forward_return_{horizon}d`` is calculated as the close
    price ``horizon`` rows ahead within the same symbol divided by the current
    close price minus one. End-of-series rows without enough future data remain
    NaN. Generated labels are not forward-filled, backward-filled, or dropped.

    These labels intentionally use future data as model targets and must never
    be used as features. Label generation should happen after feature generation
    or in a clearly separated dataset-building step.
    """

    required_columns = {"date", "symbol", price_col}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    _validate_horizons(horizons)

    out = df.copy()
    out = out.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    grouped_price = out.groupby("symbol", sort=False)[price_col]

    for horizon in horizons:
        future_price = grouped_price.shift(-horizon)
        out[f"forward_return_{horizon}d"] = future_price / out[price_col] - 1.0

    return out


def _validate_horizons(horizons: Sequence[int]) -> None:
    for horizon in horizons:
        if horizon <= 0:
            raise ValueError("Forward return horizons must be positive integers.")
