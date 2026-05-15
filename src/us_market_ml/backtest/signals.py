"""Top-K ETF rotation signal generation from model prediction scores.

This module converts model-score prediction frames into target portfolio
weights for future backtesting. It is intentionally not a portfolio simulator:
it does not calculate returns, transaction costs, positions, or equity curves.

Rows are preserved so signal frames stay aligned with experiment prediction
outputs. The generated signals select the top-K eligible symbols per date by
descending prediction score and assign target weights to selected rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ALLOCATION_MODES = {
    "equal_weight",
    "score_weighted",
    "softmax_score_weighted",
    "volatility_adjusted_score_weighted",
    "capped_score_weighted",
}
_ALLOCATION_EPSILON = 1e-12


@dataclass(frozen=True)
class TopKSignalConfig:
    """Configuration for deterministic top-K signal generation.

    Prediction scores are converted into top-K ETF selections independently
    for each date. Selected symbols receive target portfolio weights for future
    backtesting according to ``allocation_mode``. This configuration does not
    define a trading simulator, return calculation, transaction-cost model, or
    equity curve.
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
    allocation_mode: str = "equal_weight"
    score_temperature: float = 1.0
    volatility_col: str = "volatility_20d"
    max_weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate top-K signal configuration."""

        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")
        if self.allocation_mode not in ALLOCATION_MODES:
            modes = ", ".join(sorted(ALLOCATION_MODES))
            raise ValueError(f"allocation_mode must be one of: {modes}.")
        if self.score_temperature <= 0.0:
            raise ValueError("score_temperature must be greater than 0.")
        if not 0.0 < self.max_weight <= 1.0:
            raise ValueError("max_weight must satisfy 0 < max_weight <= 1.")
        if (
            self.allocation_mode == "capped_score_weighted"
            and self.top_k * self.max_weight < 1.0
        ):
            raise ValueError(
                "top_k * max_weight must be greater than or equal to 1 for "
                "capped_score_weighted allocation."
            )


@dataclass(frozen=True)
class SignalSmoothingConfig:
    """Configuration for v1 signal smoothing research logic.

    Smoothing is applied after top-K signal generation. It can enforce a
    minimum strategy-date holding period and require a prediction-score margin
    before replacing an existing holding. This is portfolio construction
    research logic, not an execution model.
    """

    min_holding_days: int = 1
    switch_margin: float = 0.0
    date_col: str = "date"
    symbol_col: str = "symbol"
    prediction_col: str = "prediction"
    selected_col: str = "selected"
    signal_rank_col: str = "signal_rank"
    target_weight_col: str = "target_weight"

    def __post_init__(self) -> None:
        """Validate signal smoothing configuration."""

        if self.min_holding_days < 1:
            raise ValueError("min_holding_days must be greater than or equal to 1.")
        if self.switch_margin < 0.0:
            raise ValueError("switch_margin must be greater than or equal to 0.")


def generate_top_k_signals(
    predictions: pd.DataFrame,
    config: TopKSignalConfig | None = None,
    smoothing_config: SignalSmoothingConfig | None = None,
) -> pd.DataFrame:
    """Return prediction rows with deterministic top-K target weights.

    The input frame must contain date, symbol, and prediction score columns
    using the configured names. The output is a copy that preserves all input
    rows and columns, then adds selection, rank, and target-weight columns for
    future signal backtesting. It does not simulate a portfolio, calculate
    returns, or apply transaction costs.

    Higher prediction scores are better. Ties are resolved deterministically by
    sorting symbols ascending after prediction descending. Missing prediction
    rows are preserved but remain unselected. The default allocation mode is
    equal weight, preserving the original top-K behavior.
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

        selected_rows = ranked_group.iloc[: len(selected_positions)]
        target_weights = _calculate_target_weights(selected_rows, resolved_config)
        out.iloc[selected_positions, selected_column_position] = True
        for rank, (position, target_weight) in enumerate(
            zip(selected_positions, target_weights, strict=True),
            start=1,
        ):
            out.iloc[position, rank_column_position] = rank
            out.iloc[position, weight_column_position] = target_weight

    if smoothing_config is not None:
        return smooth_top_k_signals(
            out,
            smoothing_config,
            allocation_config=resolved_config,
        )

    return out


def smooth_top_k_signals(
    signals: pd.DataFrame,
    config: SignalSmoothingConfig | None = None,
    allocation_config: TopKSignalConfig | None = None,
) -> pd.DataFrame:
    """Return smoothed top-K signals with minimum holding-period support.

    The input is expected to be the output of ``generate_top_k_signals``. All
    original columns and rows are preserved, while selected/rank/target-weight
    columns are recalculated date by date. Allocation defaults to equal weight
    unless a compatible ``TopKSignalConfig`` is supplied. For top-1 signals,
    the currently selected symbol is held until ``min_holding_days`` strategy
    dates have elapsed, then replaced only when the best available candidate
    improves on the current holding's prediction by at least ``switch_margin``.
    If the current holding is absent or has a missing prediction on a date, the
    best available symbol may replace it immediately.

    For top-K where K > 1, v1 smoothing keeps still-available selected symbols
    that have not met the minimum holding period, then fills open slots by
    descending prediction score. Mature holdings can be retained when the best
    replacement does not exceed their score by the configured margin. This is
    intentionally simple research logic, not an execution model.
    """

    resolved_config = config if config is not None else SignalSmoothingConfig()
    resolved_allocation_config = (
        allocation_config
        if allocation_config is not None
        else TopKSignalConfig(
            prediction_col=resolved_config.prediction_col,
            date_col=resolved_config.date_col,
            symbol_col=resolved_config.symbol_col,
            weight_col=resolved_config.target_weight_col,
            rank_col=resolved_config.signal_rank_col,
            selected_col=resolved_config.selected_col,
        )
    )
    _validate_required_columns(
        signals,
        [
            resolved_config.date_col,
            resolved_config.symbol_col,
            resolved_config.prediction_col,
            resolved_config.selected_col,
            resolved_config.signal_rank_col,
            resolved_config.target_weight_col,
        ],
    )

    out = signals.copy()
    out[resolved_config.selected_col] = False
    out[resolved_config.signal_rank_col] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out[resolved_config.target_weight_col] = 0.0

    if out.empty:
        return out

    original_selected_counts = (
        signals.assign(
            __signal_smoothing_selected=signals[resolved_config.selected_col]
            .fillna(False)
            .astype(bool)
        )
        .groupby(resolved_config.date_col, sort=False, dropna=False)[
            "__signal_smoothing_selected"
        ]
        .sum()
        .astype(int)
        .to_dict()
    )
    top_k = max(original_selected_counts.values(), default=0)
    if top_k <= 1:
        return _smooth_top_1_signals(
            out,
            original_selected_counts,
            resolved_config,
            resolved_allocation_config,
        )

    return _smooth_top_k_signals(
        out,
        original_selected_counts,
        top_k,
        resolved_config,
        resolved_allocation_config,
    )


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


def _smooth_top_1_signals(
    out: pd.DataFrame,
    original_selected_counts: dict[object, int],
    config: SignalSmoothingConfig,
    allocation_config: TopKSignalConfig,
) -> pd.DataFrame:
    current_symbol: object | None = None
    current_holding_days = 0

    for date_value, date_group in _iter_date_groups(out, config.date_col):
        target_count = min(int(original_selected_counts.get(date_value, 0)), 1)
        if target_count == 0:
            current_symbol = None
            current_holding_days = 0
            continue

        available = _available_rows(date_group, config)
        if available.empty:
            current_symbol = None
            current_holding_days = 0
            continue

        best_row = available.iloc[0]
        best_symbol = best_row[config.symbol_col]
        selected_symbol = best_symbol
        switched = current_symbol != best_symbol

        current_row = available.loc[available[config.symbol_col] == current_symbol]
        current_available = current_symbol is not None and not current_row.empty
        if current_available:
            current_prediction = float(current_row.iloc[0][config.prediction_col])
            best_prediction = float(best_row[config.prediction_col])
            improvement = best_prediction - current_prediction

            if current_holding_days < config.min_holding_days:
                selected_symbol = current_symbol
                switched = False
            elif best_symbol == current_symbol:
                selected_symbol = current_symbol
                switched = False
            elif best_prediction > current_prediction and improvement >= config.switch_margin:
                selected_symbol = best_symbol
                switched = True
            else:
                selected_symbol = current_symbol
                switched = False

        _assign_selected_symbols(
            out,
            date_group,
            [selected_symbol],
            config,
            allocation_config,
        )
        current_symbol = selected_symbol
        current_holding_days = 1 if switched else current_holding_days + 1

    return out


def _smooth_top_k_signals(
    out: pd.DataFrame,
    original_selected_counts: dict[object, int],
    top_k: int,
    config: SignalSmoothingConfig,
    allocation_config: TopKSignalConfig,
) -> pd.DataFrame:
    holding_days: dict[object, int] = {}

    for date_value, date_group in _iter_date_groups(out, config.date_col):
        target_count = min(int(original_selected_counts.get(date_value, 0)), top_k)
        available = _available_rows(date_group, config)
        available_symbols = set(available[config.symbol_col].tolist())
        holding_days = {
            symbol: days for symbol, days in holding_days.items() if symbol in available_symbols
        }

        if target_count == 0 or available.empty:
            holding_days = {}
            continue

        selected_symbols: list[object] = [
            symbol
            for symbol, days in holding_days.items()
            if days < config.min_holding_days and symbol in available_symbols
        ][:target_count]

        while len(selected_symbols) < target_count:
            candidate = _next_replacement_candidate(available, selected_symbols, config)
            if candidate is None:
                break

            mature_symbol = _best_margin_protected_holding(
                available=available,
                selected_symbols=selected_symbols,
                holding_days=holding_days,
                replacement_prediction=float(candidate[config.prediction_col]),
                config=config,
            )
            selected_symbol = (
                mature_symbol if mature_symbol is not None else candidate[config.symbol_col]
            )
            if selected_symbol in selected_symbols:
                break
            selected_symbols.append(selected_symbol)

        _assign_selected_symbols(
            out,
            date_group,
            selected_symbols,
            config,
            allocation_config,
        )
        holding_days = {
            symbol: holding_days.get(symbol, 0) + 1 for symbol in selected_symbols
        }

    return out


def _iter_date_groups(
    signals: pd.DataFrame,
    date_col: str,
) -> list[tuple[object, pd.DataFrame]]:
    positional = signals.reset_index(drop=True).copy()
    positional["__signal_smoothing_position"] = np.arange(len(positional))
    positional = positional.sort_values(
        by=[date_col, "__signal_smoothing_position"],
        kind="mergesort",
    )
    return list(positional.groupby(date_col, sort=False, dropna=False))


def _available_rows(
    date_group: pd.DataFrame,
    config: SignalSmoothingConfig,
) -> pd.DataFrame:
    return date_group.loc[date_group[config.prediction_col].notna()].sort_values(
        by=[config.prediction_col, config.symbol_col],
        ascending=[False, True],
        kind="mergesort",
    )


def _next_replacement_candidate(
    available: pd.DataFrame,
    selected_symbols: list[object],
    config: SignalSmoothingConfig,
) -> pd.Series | None:
    selected = set(selected_symbols)
    candidates = available.loc[~available[config.symbol_col].isin(selected)]
    if candidates.empty:
        return None
    return candidates.iloc[0]


def _best_margin_protected_holding(
    available: pd.DataFrame,
    selected_symbols: list[object],
    holding_days: dict[object, int],
    replacement_prediction: float,
    config: SignalSmoothingConfig,
) -> object | None:
    selected = set(selected_symbols)
    mature_symbols = [
        symbol
        for symbol, days in holding_days.items()
        if days >= config.min_holding_days and symbol not in selected
    ]
    if not mature_symbols:
        return None

    mature_rows = available.loc[available[config.symbol_col].isin(mature_symbols)].copy()
    if mature_rows.empty:
        return None

    mature_rows = mature_rows.sort_values(
        by=[config.prediction_col, config.symbol_col],
        ascending=[False, True],
        kind="mergesort",
    )
    best_mature = mature_rows.iloc[0]
    mature_prediction = float(best_mature[config.prediction_col])
    if replacement_prediction > mature_prediction and (
        replacement_prediction - mature_prediction
    ) >= config.switch_margin:
        return None
    mature_symbol: object = best_mature[config.symbol_col]
    return mature_symbol


def _assign_selected_symbols(
    out: pd.DataFrame,
    date_group: pd.DataFrame,
    selected_symbols: list[object],
    config: SignalSmoothingConfig,
    allocation_config: TopKSignalConfig,
) -> None:
    if not selected_symbols:
        return

    selected_rows = date_group.loc[date_group[config.symbol_col].isin(selected_symbols)].copy()
    selected_rows = selected_rows.sort_values(
        by=[config.prediction_col, config.symbol_col],
        ascending=[False, True],
        kind="mergesort",
    )
    selected_positions = selected_rows["__signal_smoothing_position"].to_list()
    target_weights = _calculate_target_weights(selected_rows, allocation_config)

    selected_column_position = out.columns.get_loc(config.selected_col)
    rank_column_position = out.columns.get_loc(config.signal_rank_col)
    weight_column_position = out.columns.get_loc(config.target_weight_col)
    for rank, (position, target_weight) in enumerate(
        zip(selected_positions, target_weights, strict=True),
        start=1,
    ):
        out.iloc[position, selected_column_position] = True
        out.iloc[position, rank_column_position] = rank
        out.iloc[position, weight_column_position] = target_weight


def _calculate_target_weights(
    selected_rows: pd.DataFrame,
    config: TopKSignalConfig,
) -> np.ndarray:
    selected_count = len(selected_rows)
    if selected_count == 0:
        return np.array([], dtype=float)
    if config.allocation_mode == "equal_weight":
        return _equal_weights(selected_count)
    if config.allocation_mode == "score_weighted":
        return _score_weighted_weights(selected_rows, config)
    if config.allocation_mode == "softmax_score_weighted":
        return _softmax_score_weights(selected_rows, config)
    if config.allocation_mode == "volatility_adjusted_score_weighted":
        return _volatility_adjusted_score_weights(selected_rows, config)
    if config.allocation_mode == "capped_score_weighted":
        return _capped_score_weights(selected_rows, config)
    raise ValueError(f"Unsupported allocation_mode: {config.allocation_mode!r}")


def _equal_weights(selected_count: int) -> np.ndarray:
    return np.full(selected_count, 1.0 / selected_count, dtype=float)


def _score_weighted_weights(
    selected_rows: pd.DataFrame,
    config: TopKSignalConfig,
) -> np.ndarray:
    scores = _score_values(selected_rows, config)
    shifted_scores = _positive_shifted_scores(scores)
    total = float(shifted_scores.sum())
    if not np.isfinite(total) or total <= _ALLOCATION_EPSILON:
        return _equal_weights(len(selected_rows))
    return shifted_scores / total


def _softmax_score_weights(
    selected_rows: pd.DataFrame,
    config: TopKSignalConfig,
) -> np.ndarray:
    scores = _score_values(selected_rows, config) / config.score_temperature
    stabilized_scores = scores - np.nanmax(scores)
    exp_scores = np.exp(stabilized_scores)
    total = float(exp_scores.sum())
    if not np.isfinite(total) or total <= _ALLOCATION_EPSILON:
        return _equal_weights(len(selected_rows))
    return exp_scores / total


def _volatility_adjusted_score_weights(
    selected_rows: pd.DataFrame,
    config: TopKSignalConfig,
) -> np.ndarray:
    if config.volatility_col not in selected_rows.columns:
        return _score_weighted_weights(selected_rows, config)

    volatility = pd.to_numeric(selected_rows[config.volatility_col], errors="coerce").to_numpy(
        dtype=float
    )
    valid_volatility = np.isfinite(volatility) & (volatility > _ALLOCATION_EPSILON)
    if not bool(valid_volatility.all()):
        return _score_weighted_weights(selected_rows, config)

    shifted_scores = _positive_shifted_scores(_score_values(selected_rows, config))
    adjusted_scores = shifted_scores / np.maximum(volatility, _ALLOCATION_EPSILON)
    total = float(adjusted_scores.sum())
    if not np.isfinite(total) or total <= _ALLOCATION_EPSILON:
        return _score_weighted_weights(selected_rows, config)
    return adjusted_scores / total


def _capped_score_weights(
    selected_rows: pd.DataFrame,
    config: TopKSignalConfig,
) -> np.ndarray:
    base_weights = _score_weighted_weights(selected_rows, config)
    if len(base_weights) * config.max_weight < 1.0:
        total = float(base_weights.sum())
        if total > _ALLOCATION_EPSILON:
            return base_weights / total
        return _equal_weights(len(base_weights))
    return _cap_and_redistribute(base_weights, config.max_weight)


def _score_values(
    selected_rows: pd.DataFrame,
    config: TopKSignalConfig,
) -> np.ndarray:
    return pd.to_numeric(selected_rows[config.prediction_col], errors="coerce").to_numpy(
        dtype=float
    )


def _positive_shifted_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0 or not bool(np.isfinite(scores).all()):
        return np.zeros_like(scores, dtype=float)
    min_score = float(scores.min())
    if min_score <= 0.0:
        return scores - min_score + _ALLOCATION_EPSILON
    return scores.copy()


def _cap_and_redistribute(weights: np.ndarray, max_weight: float) -> np.ndarray:
    capped = np.minimum(weights.astype(float), max_weight)
    excess = max(0.0, float(weights.sum() - capped.sum()))

    for _ in range(len(capped) + 1):
        if excess <= _ALLOCATION_EPSILON:
            break

        capacity = np.maximum(max_weight - capped, 0.0)
        eligible = capacity > _ALLOCATION_EPSILON
        if not bool(eligible.any()):
            break

        preferred = weights[eligible]
        preferred_total = float(preferred.sum())
        if preferred_total <= _ALLOCATION_EPSILON:
            preferred = capacity[eligible]
            preferred_total = float(preferred.sum())
        if preferred_total <= _ALLOCATION_EPSILON:
            break

        additions = excess * preferred / preferred_total
        applied = np.minimum(additions, capacity[eligible])
        capped[eligible] += applied
        new_excess = excess - float(applied.sum())
        if abs(new_excess - excess) <= _ALLOCATION_EPSILON:
            break
        excess = max(0.0, new_excess)

    total = float(capped.sum())
    if total <= _ALLOCATION_EPSILON:
        return _equal_weights(len(capped))
    if abs(total - 1.0) > _ALLOCATION_EPSILON:
        capped = capped / total
    return capped


def _validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Input DataFrame is missing required signal columns: {missing}")


__all__ = [
    "SignalSmoothingConfig",
    "TopKSignalConfig",
    "generate_top_k_signals",
    "smooth_top_k_signals",
    "summarize_signals",
    "validate_signal_weights",
]
