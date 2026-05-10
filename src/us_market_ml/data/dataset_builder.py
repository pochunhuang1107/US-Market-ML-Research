"""Supervised dataset construction for ETF rotation research.

This module combines leakage-safe historical features with supervised labels in
one explicit construction step. Feature generation must only use current and
past market rows. Label generation intentionally uses future data because those
columns are model targets, and target columns must never be used as features.

Rows are preserved during dataset construction, including warmup and terminal
NaNs. Any row dropping for model fitting should happen explicitly after the
full dataset has been built.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from us_market_ml.features import build_feature_frame
from us_market_ml.labels import (
    add_cross_sectional_rank_labels,
    add_excess_return_vs_benchmark,
    add_forward_returns,
)

TARGET_COLUMN_PREFIXES: tuple[str, ...] = (
    "forward_return_",
    "excess_return_vs_",
    "rank_",
)


@dataclass(frozen=True)
class SupervisedDatasetConfig:
    """Configuration for supervised ETF dataset construction.

    Forward and excess-return labels intentionally use future returns as model
    targets. The resulting target columns must be excluded from feature columns
    to avoid leakage.
    """

    forward_horizons: tuple[int, ...] = (5, 10, 20)
    benchmark_symbol: str = "SPY"
    excess_return_horizons: tuple[int, ...] = (20,)
    rank_target_col: str | None = "excess_return_vs_spy_20d"
    rank_col: str | None = "rank_excess_return_vs_spy_20d"
    price_col: str = "close"


def build_supervised_dataset(
    df: pd.DataFrame,
    config: SupervisedDatasetConfig | None = None,
) -> pd.DataFrame:
    """Build an ML-ready supervised research frame without dropping rows.

    The input rows are copied, then leakage-safe historical features are built
    before future-looking supervised labels are added. Forward-return, excess
    return, and optional rank labels intentionally use future information and
    must never be used as features. This function preserves rows and does not
    perform splitting, scaling, model training, or implicit NaN filtering.
    """

    resolved_config = config if config is not None else SupervisedDatasetConfig()

    out = build_feature_frame(df.copy())
    out = add_forward_returns(
        out,
        horizons=resolved_config.forward_horizons,
        price_col=resolved_config.price_col,
    )
    out = add_excess_return_vs_benchmark(
        out,
        benchmark_symbol=resolved_config.benchmark_symbol,
        horizons=resolved_config.excess_return_horizons,
    )

    if resolved_config.rank_target_col is not None:
        if resolved_config.rank_target_col not in out.columns:
            raise ValueError(
                "Configured rank_target_col is missing after label generation: "
                f"{resolved_config.rank_target_col}"
            )
        out = add_cross_sectional_rank_labels(
            out,
            target_col=resolved_config.rank_target_col,
            rank_col=resolved_config.rank_col,
        )

    return out


def get_target_columns(df: pd.DataFrame) -> list[str]:
    """Return supervised target columns that must never be used as features."""

    return [
        column
        for column in df.columns
        if any(column.startswith(prefix) for prefix in TARGET_COLUMN_PREFIXES)
    ]


def get_default_non_feature_columns() -> set[str]:
    """Return standard metadata and raw OHLCV columns excluded from features."""

    return {
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "year",
        "month",
        "transactions",
    }


def get_feature_columns(
    df: pd.DataFrame,
    extra_exclude: set[str] | None = None,
) -> list[str]:
    """Return numeric historical feature columns with leakage targets removed.

    Default metadata/OHLCV columns, supervised target columns, and any
    ``extra_exclude`` columns are omitted. Target columns must never be used as
    features because forward, excess, and rank labels intentionally depend on
    future data.
    """

    excluded_columns = get_default_non_feature_columns()
    excluded_columns.update(get_target_columns(df))
    if extra_exclude is not None:
        excluded_columns.update(extra_exclude)

    return [
        column
        for column in df.columns
        if column not in excluded_columns
        and is_numeric_dtype(df[column])
        and not is_bool_dtype(df[column])
    ]


def drop_rows_with_missing_model_values(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> pd.DataFrame:
    """Drop rows with missing selected model inputs and target values.

    Dataset construction intentionally preserves NaNs from rolling-feature
    warmups and unavailable future labels. Row dropping should be an explicit
    modeling step after construction, using only the feature columns and target
    column selected for that model.
    """

    required_columns = [*feature_cols, target_col]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Input DataFrame is missing selected model columns: {missing}")

    return df.dropna(subset=required_columns).copy()
