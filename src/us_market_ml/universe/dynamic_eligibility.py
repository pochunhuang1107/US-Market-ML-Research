"""Point-in-time dynamic universe eligibility.

The helpers in this module keep symbol membership local to each row date.
Late-start symbols are not permanently excluded: once a symbol has enough
history and complete selected features for a row, it can participate in that
row's training or scoring universe.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DynamicEligibilityConfig:
    """Configuration for point-in-time symbol eligibility checks."""

    min_history_rows: int = 252
    date_col: str = "date"
    symbol_col: str = "symbol"

    def __post_init__(self) -> None:
        if self.min_history_rows < 1:
            raise ValueError("min_history_rows must be greater than or equal to 1.")
        if not self.date_col:
            raise ValueError("date_col must be non-empty.")
        if not self.symbol_col:
            raise ValueError("symbol_col must be non-empty.")


def dynamic_eligibility_mask(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_col: str | None = None,
    require_target: bool = False,
    config: DynamicEligibilityConfig | None = None,
    history_df: pd.DataFrame | None = None,
) -> pd.Series:
    """Return a boolean mask for point-in-time eligible rows.

    A row is eligible when its symbol has at least ``min_history_rows`` rows up
    to and including that row, all selected feature values are present on the
    row, and, when requested, the target value is present. When ``history_df``
    is supplied, history counts are calculated on that full point-in-time
    frame and aligned back to ``df`` by index, allowing validation/test rows to
    use lookback accumulated before their split. The target check is
    intentionally optional so prediction/scoring rows near the end of a dataset
    are not removed merely because future labels are unavailable.
    """

    resolved_config = config if config is not None else DynamicEligibilityConfig()
    feature_cols = list(feature_cols)
    required_columns = [resolved_config.date_col, resolved_config.symbol_col, *feature_cols]
    if require_target:
        if target_col is None:
            raise ValueError("target_col is required when require_target=True.")
        required_columns.append(target_col)
    elif target_col is not None and target_col not in df.columns:
        # A non-required target is allowed to be absent from scoring frames.
        target_col = None

    _validate_columns(df, required_columns)
    history_source = history_df if history_df is not None else df
    _validate_columns(history_source, [resolved_config.date_col, resolved_config.symbol_col])

    if df.empty:
        return pd.Series(False, index=df.index, dtype=bool)

    sorted_df = (
        history_source[[resolved_config.date_col, resolved_config.symbol_col]]
        .assign(__dynamic_universe_original_index=history_source.index)
        .sort_values(
            [
                resolved_config.symbol_col,
                resolved_config.date_col,
                "__dynamic_universe_original_index",
            ],
            kind="mergesort",
        )
    )
    history_counts = (
        sorted_df.groupby(resolved_config.symbol_col, sort=False)
        .cumcount()
        .add(1)
        .set_axis(sorted_df["__dynamic_universe_original_index"])
        .reindex(df.index)
    )
    mask = history_counts.ge(resolved_config.min_history_rows)

    if feature_cols:
        mask &= df[feature_cols].notna().all(axis=1)
    if require_target and target_col is not None:
        mask &= df[target_col].notna()

    return mask.fillna(False).astype(bool)


def apply_dynamic_universe_eligibility(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_col: str | None = None,
    require_target: bool = False,
    config: DynamicEligibilityConfig | None = None,
    history_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return rows that satisfy point-in-time dynamic eligibility."""

    mask = dynamic_eligibility_mask(
        df,
        feature_cols=feature_cols,
        target_col=target_col,
        require_target=require_target,
        config=config,
        history_df=history_df,
    )
    return df.loc[mask].copy()


def select_fold_train_feature_columns(
    train_df: pd.DataFrame,
    candidate_feature_cols: Sequence[str],
    *,
    min_non_null_ratio: float = 0.0,
) -> list[str]:
    """Select fold-local usable features using only training rows.

    The candidate list should already be leakage-safe. This helper only checks
    training-period availability, excluding columns that are all-null or below
    the configured non-null ratio in the training frame. Validation and test
    rows are never inspected.
    """

    if min_non_null_ratio < 0.0 or min_non_null_ratio > 1.0:
        raise ValueError("min_non_null_ratio must satisfy 0 <= value <= 1.")

    selected: list[str] = []
    for column in candidate_feature_cols:
        if column not in train_df.columns or train_df.empty:
            continue
        non_null_ratio = float(train_df[column].notna().mean())
        if non_null_ratio == 0.0:
            continue
        if non_null_ratio < min_non_null_ratio:
            continue
        selected.append(column)
    return selected


def symbol_count(
    df: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
) -> int:
    """Return the number of distinct non-null symbols in ``df``."""

    if symbol_col not in df.columns or df.empty:
        return 0
    return int(df[symbol_col].dropna().nunique())


def min_symbol_count_per_date(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
) -> int:
    """Return the minimum distinct symbol count across dates in ``df``."""

    if df.empty or date_col not in df.columns or symbol_col not in df.columns:
        return 0
    counts = df.groupby(date_col, sort=False)[symbol_col].nunique(dropna=True)
    if counts.empty:
        return 0
    return int(counts.min())


def _validate_columns(df: pd.DataFrame, required_columns: Sequence[str]) -> None:
    missing_columns = [
        column for column in dict.fromkeys(required_columns) if column not in df.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Input DataFrame is missing dynamic eligibility columns: {missing}")
