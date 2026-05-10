"""Top-K ETF rotation signal generation from model prediction scores.

This module converts model-score prediction frames into target portfolio
weights for future backtesting. It is intentionally not a portfolio simulator:
it does not calculate returns, transaction costs, positions, or equity curves.

Rows are preserved so signal frames stay aligned with experiment prediction
outputs. The generated signals select the top-K eligible symbols per date by
descending prediction score and assign equal target weights to selected rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TopKSignalConfig:
    """Configuration for deterministic top-K equal-weight signal generation.

    Prediction scores are converted into top-K ETF selections independently
    for each date. Selected symbols receive equal target portfolio weights for
    future backtesting. This configuration does not define a trading simulator,
    return calculation, transaction-cost model, or equity curve.
    """

    prediction_col: str = "prediction"
    date_col: str = "date"
    symbol_col: str = "symbol"
    top_k: int = 3
    min_score: float | None = None
    allow_negative_scores: bool = True
    weight_col: str = "target_weight"
    rank_col: str = "signal_rank"
    selected_col: str = "selected"

    def __post_init__(self) -> None:
        """Validate top-K signal configuration."""

        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")


def generate_top_k_signals(
    predictions: pd.DataFrame,
    config: TopKSignalConfig | None = None,
) -> pd.DataFrame:
    """Return prediction rows with deterministic top-K target weights.

    The input frame must contain date, symbol, and prediction score columns
    using the configured names. The output is a copy that preserves all input
    rows and columns, then adds selection, rank, and target-weight columns for
    future signal backtesting. It does not simulate a portfolio, calculate
    returns, or apply transaction costs.

    Higher prediction scores are better. Ties are resolved deterministically by
    sorting symbols ascending after prediction descending. Missing prediction
    rows are preserved but remain unselected.
    """

    resolved_config = config if config is not None else TopKSignalConfig()
    _validate_required_columns(
        predictions,
        [
            resolved_config.date_col,
            resolved_config.symbol_col,
            resolved_config.prediction_col,
        ],
    )

    out = predictions.copy()
    out[resolved_config.selected_col] = False
    out[resolved_config.rank_col] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out[resolved_config.weight_col] = 0.0

    positional = out.reset_index(drop=True)
    eligible_mask = positional[resolved_config.prediction_col].notna()
    if not resolved_config.allow_negative_scores:
        eligible_mask &= positional[resolved_config.prediction_col] >= 0
    if resolved_config.min_score is not None:
        eligible_mask &= positional[resolved_config.prediction_col] >= resolved_config.min_score

    selected_column_position = out.columns.get_loc(resolved_config.selected_col)
    rank_column_position = out.columns.get_loc(resolved_config.rank_col)
    weight_column_position = out.columns.get_loc(resolved_config.weight_col)

    eligible_rows = positional.loc[eligible_mask]
    for _, date_group in eligible_rows.groupby(resolved_config.date_col, sort=False, dropna=False):
        ranked_group = date_group.sort_values(
            by=[resolved_config.prediction_col, resolved_config.symbol_col],
            ascending=[False, True],
            kind="mergesort",
        )
        selected_positions = ranked_group.index[: resolved_config.top_k].to_list()
        if not selected_positions:
            continue

        target_weight = 1.0 / len(selected_positions)
        out.iloc[selected_positions, selected_column_position] = True
        out.iloc[selected_positions, weight_column_position] = target_weight
        for rank, position in enumerate(selected_positions, start=1):
            out.iloc[position, rank_column_position] = rank

    return out


def summarize_signals(
    signals: pd.DataFrame,
    config: TopKSignalConfig | None = None,
) -> pd.DataFrame:
    """Summarize generated top-K signal weights by date.

    The summary reports selected counts, selected target-weight totals, and
    prediction score ranges needed to inspect signal generation output. It is
    not a backtest summary and does not calculate portfolio returns or equity
    curves.
    """

    resolved_config = config if config is not None else TopKSignalConfig()
    _validate_required_columns(
        signals,
        [
            resolved_config.date_col,
            resolved_config.prediction_col,
            resolved_config.selected_col,
            resolved_config.weight_col,
        ],
    )

    summary_columns = [
        resolved_config.date_col,
        "selected_count",
        "total_weight",
        "max_score",
        "min_selected_score",
    ]
    rows: list[dict[str, object]] = []
    grouped_signals = signals.groupby(resolved_config.date_col, sort=False, dropna=False)
    for date_value, date_group in grouped_signals:
        selected_mask = date_group[resolved_config.selected_col].fillna(False).astype(bool)
        selected_group = date_group.loc[selected_mask]
        selected_count = int(selected_mask.sum())

        rows.append(
            {
                resolved_config.date_col: date_value,
                "selected_count": selected_count,
                "total_weight": (
                    float(selected_group[resolved_config.weight_col].sum())
                    if selected_count > 0
                    else 0.0
                ),
                "max_score": float(date_group[resolved_config.prediction_col].max()),
                "min_selected_score": (
                    float(selected_group[resolved_config.prediction_col].min())
                    if selected_count > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows, columns=summary_columns)


def validate_signal_weights(
    signals: pd.DataFrame,
    config: TopKSignalConfig | None = None,
    tolerance: float = 1e-9,
) -> bool:
    """Return whether each date has valid selected target weights.

    A valid signal date has total selected weight equal to 1.0 when at least
    one symbol is selected, or 0.0 when no symbols are selected. This check is
    for signal-generation output only; it does not run a portfolio simulator.
    """

    resolved_config = config if config is not None else TopKSignalConfig()
    summary = summarize_signals(signals, resolved_config)

    for _, row in summary.iterrows():
        expected_weight = 1.0 if int(row["selected_count"]) > 0 else 0.0
        if abs(float(row["total_weight"]) - expected_weight) > tolerance:
            return False

    return True


def _validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Input DataFrame is missing required signal columns: {missing}")


__all__ = [
    "TopKSignalConfig",
    "generate_top_k_signals",
    "summarize_signals",
    "validate_signal_weights",
]
