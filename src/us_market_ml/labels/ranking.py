"""Cross-sectional rank labels for ETF rotation research.

Rank labels are derived from supervised target columns that may intentionally
use future data. They must never be included in feature columns or used by
feature generation. Generate these labels only after leakage-safe features have
been built, or in a clearly separated dataset-building step.
"""

from __future__ import annotations

import pandas as pd


def add_cross_sectional_rank_labels(
    df: pd.DataFrame,
    target_col: str,
    rank_col: str | None = None,
    ascending: bool = False,
    pct: bool = True,
) -> pd.DataFrame:
    """Rank a target column within each date.

    By default, higher target values are better. With pandas percentile ranks
    (``pct=True``) and ``ascending=False``, the best-performing ETF receives the
    smallest percentile rank, equal to ``1 / number_of_non_null_targets`` for
    that date.

    These labels intentionally depend on supervised targets and must never be
    used as features. Label generation should happen after feature generation
    or in a clearly separated dataset-building step.
    """

    required_columns = {"date", target_col}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    out = df.copy()
    resolved_rank_col = rank_col if rank_col is not None else f"rank_{target_col}"
    out[resolved_rank_col] = out.groupby("date", sort=False)[target_col].rank(
        ascending=ascending,
        pct=pct,
    )

    return out
