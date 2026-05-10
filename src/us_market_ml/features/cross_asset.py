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
