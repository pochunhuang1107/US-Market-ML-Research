"""Benchmark-relative supervised labels for ETF rotation research.

Excess-return labels intentionally depend on future forward-return labels
because they are supervised learning targets. They must never be included in
feature columns or used by feature generation. Generate these labels only after
leakage-safe features have been built, or in a clearly separated dataset-building
step.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def add_excess_return_vs_benchmark(
    df: pd.DataFrame,
    benchmark_symbol: str = "SPY",
    horizons: Sequence[int] = (20,),
) -> pd.DataFrame:
    """Add excess forward-return labels versus a benchmark symbol by date.

    For each horizon, ``excess_return_vs_{benchmark_symbol.lower()}_{horizon}d``
    equals the row's ``forward_return_{horizon}d`` minus the benchmark's
    ``forward_return_{horizon}d`` on the same date. Benchmark values are joined
    by date, not by row order. Dates without a benchmark row produce NaN excess
    labels.

    These labels intentionally use future data as model targets and must never
    be used as features. Label generation should happen after feature generation
    or in a clearly separated dataset-building step.
    """

    required_columns = {"date", "symbol"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    for horizon in horizons:
        forward_col = f"forward_return_{horizon}d"
        if forward_col not in df.columns:
            raise ValueError(f"Input DataFrame is missing required column: {forward_col}")

    out = df.copy()
    benchmark_mask = (
        out["symbol"].astype("string").str.upper() == benchmark_symbol.upper()
    )

    for horizon in horizons:
        forward_col = f"forward_return_{horizon}d"
        benchmark_col = f"_{benchmark_symbol.lower()}_{horizon}d_benchmark_forward_return"
        excess_col = f"excess_return_vs_{benchmark_symbol.lower()}_{horizon}d"

        benchmark_returns = (
            out.loc[benchmark_mask, ["date", forward_col]]
            .drop_duplicates(subset=["date"], keep="first")
            .rename(columns={forward_col: benchmark_col})
        )
        out = out.merge(benchmark_returns, on="date", how="left")
        out[excess_col] = out[forward_col] - out[benchmark_col]
        out = out.drop(columns=[benchmark_col])

    return out
