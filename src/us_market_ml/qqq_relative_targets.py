"""QQQ-relative supervised targets for long-history ETF research.

The columns created here are labels, not model features. They intentionally
look forward by the requested horizon and should always be excluded from
feature selection by the dataset builder.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

QQQ_RELATIVE_REGRESSION_TARGETS: tuple[str, ...] = (
    "excess_return_vs_qqq_20d",
    "excess_return_vs_qqq_63d",
    "excess_return_vs_qqq_126d",
    "return_drawdown_score_vs_qqq_63d",
    "return_drawdown_score_vs_qqq_126d",
)
QQQ_RELATIVE_CLASSIFICATION_TARGETS: tuple[str, ...] = ("qqq_outperformance_class_63d",)
QQQ_RELATIVE_TARGETS: tuple[str, ...] = (
    *QQQ_RELATIVE_REGRESSION_TARGETS,
    *QQQ_RELATIVE_CLASSIFICATION_TARGETS,
)


def add_qqq_relative_targets(
    df: pd.DataFrame,
    *,
    excess_return_horizons: Sequence[int] = (20, 63, 126),
    return_drawdown_horizons: Sequence[int] = (63, 126),
    classification_horizons: Sequence[int] = (),
    benchmark_symbol: str = "QQQ",
    date_col: str = "date",
    symbol_col: str = "symbol",
    price_col: str = "close",
) -> pd.DataFrame:
    """Add QQQ-relative future-return labels to a market feature frame.

    Forward returns are calculated per symbol with ``shift(-horizon)``. The
    benchmark's same-date forward return is then joined back by date. This
    means target rows depend only on future prices for that row's label and do
    not mutate historical feature columns.
    """

    _validate_required_columns(df, [date_col, symbol_col, price_col])
    horizons = _sorted_positive_horizons(
        [*excess_return_horizons, *return_drawdown_horizons, *classification_horizons]
    )
    if not horizons:
        return df.copy()

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    original_order = np.arange(len(out))
    order_col = _temporary_column_name(out, "__qqq_relative_original_order")
    symbol_upper_col = _temporary_column_name(out, "__qqq_relative_symbol_upper")
    out[order_col] = original_order
    out[symbol_upper_col] = out[symbol_col].astype("string").str.upper()
    out = out.sort_values([symbol_col, date_col, order_col], kind="mergesort")

    grouped = out.groupby(symbol_col, sort=False, dropna=False)
    benchmark_upper = benchmark_symbol.upper()

    for horizon in horizons:
        future_price = grouped[price_col].shift(-horizon)
        forward_return_col = f"forward_return_{horizon}d"
        end_date_col = f"forward_return_end_date_{horizon}d"
        out[forward_return_col] = future_price / out[price_col] - 1.0
        out[end_date_col] = grouped[date_col].shift(-horizon)

        benchmark_returns = (
            out.loc[
                out[symbol_upper_col].eq(benchmark_upper),
                [date_col, forward_return_col],
            ]
            .drop_duplicates(subset=[date_col], keep="first")
            .rename(columns={forward_return_col: f"benchmark_forward_return_{horizon}d"})
        )
        out = out.merge(benchmark_returns, on=date_col, how="left", sort=False)
        benchmark_col = f"benchmark_forward_return_{horizon}d"

        if horizon in set(excess_return_horizons):
            out[f"excess_return_vs_qqq_{horizon}d"] = (
                out[forward_return_col] - out[benchmark_col]
            )

        if horizon in set(return_drawdown_horizons):
            score_col = f"return_drawdown_score_{horizon}d"
            benchmark_score_col = f"benchmark_return_drawdown_score_{horizon}d"
            out[score_col] = _forward_return_drawdown_score(
                out,
                horizon=horizon,
                group_col=symbol_col,
                price_col=price_col,
            )
            benchmark_scores = (
                out.loc[out[symbol_upper_col].eq(benchmark_upper), [date_col, score_col]]
                .drop_duplicates(subset=[date_col], keep="first")
                .rename(columns={score_col: benchmark_score_col})
            )
            out = out.merge(benchmark_scores, on=date_col, how="left", sort=False)
            out[f"return_drawdown_score_vs_qqq_{horizon}d"] = (
                out[score_col] - out[benchmark_score_col]
            )
            out = out.drop(columns=[score_col, benchmark_score_col])

        if horizon in set(classification_horizons):
            excess_col = f"excess_return_vs_qqq_{horizon}d"
            if excess_col not in out.columns:
                out[excess_col] = out[forward_return_col] - out[benchmark_col]
            out[f"qqq_outperformance_class_{horizon}d"] = np.where(
                out[excess_col].notna(),
                out[excess_col].gt(0.0).astype("int8"),
                np.nan,
            )

        out = out.drop(columns=[benchmark_col])

    return (
        out.sort_values(order_col, kind="mergesort")
        .drop(columns=[order_col, symbol_upper_col])
        .reset_index(drop=True)
    )


def target_horizon_from_variant(target_variant: str) -> int:
    """Return the horizon encoded in a supported QQQ-relative target name."""

    suffix = target_variant.rsplit("_", maxsplit=1)[-1]
    if not suffix.endswith("d"):
        raise ValueError(f"Cannot parse horizon from target_variant={target_variant!r}.")
    try:
        horizon = int(suffix[:-1])
    except ValueError as exc:
        raise ValueError(
            f"Cannot parse horizon from target_variant={target_variant!r}."
        ) from exc
    if horizon <= 0:
        raise ValueError("target horizon must be positive.")
    return horizon


def _forward_return_drawdown_score(
    df: pd.DataFrame,
    *,
    horizon: int,
    group_col: str,
    price_col: str,
) -> pd.Series:
    future_return = df.groupby(group_col, sort=False, dropna=False)[price_col].shift(
        -horizon
    ) / df[price_col] - 1.0
    future_drawdown = df.groupby(group_col, sort=False, dropna=False)[price_col].transform(
        lambda prices: _future_min_return(prices, horizon)
    )
    return future_return + future_drawdown.fillna(0.0)


def _future_min_return(prices: pd.Series, horizon: int) -> pd.Series:
    values = pd.to_numeric(prices, errors="coerce")
    future_low = (
        values.iloc[::-1]
        .rolling(window=horizon + 1, min_periods=horizon + 1)
        .min()
        .iloc[::-1]
    )
    return future_low / values - 1.0


def _sorted_positive_horizons(values: Sequence[int]) -> tuple[int, ...]:
    horizons = tuple(sorted({int(value) for value in values}))
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("Target horizons must be positive integers.")
    return horizons


def _validate_required_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required target columns: {missing}")


def _temporary_column_name(df: pd.DataFrame, base_name: str) -> str:
    column_name = base_name
    suffix = 1
    while column_name in df.columns:
        column_name = f"{base_name}_{suffix}"
        suffix += 1
    return column_name


__all__ = [
    "QQQ_RELATIVE_CLASSIFICATION_TARGETS",
    "QQQ_RELATIVE_REGRESSION_TARGETS",
    "QQQ_RELATIVE_TARGETS",
    "add_qqq_relative_targets",
    "target_horizon_from_variant",
]
