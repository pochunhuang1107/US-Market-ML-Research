"""Research allocation overlays for already-selected ETF rotation signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

OverlayKind = Literal[
    "equal_weight_topk",
    "rank_weighted_topk",
    "capped_score_weighted_topk",
    "static_core_satellite",
    "dynamic_core_satellite",
    "qqq_floor_if_not_selected",
]

_EPSILON = 1.0e-12
_FALLBACK_QQQ_REGIME_COL = "_fallback_qqq_strong_regime_score"
_FALLBACK_BENCHMARK_REGIME_COL = "_fallback_max_spy_qqq_strong_regime_score"


@dataclass(frozen=True)
class AllocationOverlaySpec:
    """One allocation overlay applied after top-K selection/smoothing."""

    name: str
    kind: OverlayKind
    rank_weights: tuple[float, ...] = ()
    max_weight: float | None = None
    core_weights: tuple[tuple[str, float], ...] = ()
    regime_columns: tuple[str, ...] = ()
    satellite_weight: float | None = None


def default_allocation_overlay_specs(
    *,
    include_optional_qqq_floor: bool = False,
) -> tuple[AllocationOverlaySpec, ...]:
    """Return the focused benchmark-gap v3 overlay set."""

    required = (
        AllocationOverlaySpec(
            name="equal_weight_top5_baseline",
            kind="equal_weight_topk",
        ),
        AllocationOverlaySpec(
            name="rank_weighted_top5",
            kind="rank_weighted_topk",
            rank_weights=(0.30, 0.25, 0.20, 0.15, 0.10),
        ),
        AllocationOverlaySpec(
            name="capped_score_weighted_top5_cap0.30",
            kind="capped_score_weighted_topk",
            max_weight=0.30,
        ),
        AllocationOverlaySpec(
            name="capped_score_weighted_top5_cap0.40",
            kind="capped_score_weighted_topk",
            max_weight=0.40,
        ),
        AllocationOverlaySpec(
            name="qqq_core_20_satellite_top5",
            kind="static_core_satellite",
            core_weights=(("QQQ", 0.20),),
            satellite_weight=0.80,
        ),
        AllocationOverlaySpec(
            name="qqq_core_30_satellite_top5",
            kind="static_core_satellite",
            core_weights=(("QQQ", 0.30),),
            satellite_weight=0.70,
        ),
        AllocationOverlaySpec(
            name="spy_qqq_core_40_satellite_top5",
            kind="static_core_satellite",
            core_weights=(("SPY", 0.20), ("QQQ", 0.20)),
            satellite_weight=0.60,
        ),
        AllocationOverlaySpec(
            name="dynamic_qqq_core_when_strong",
            kind="dynamic_core_satellite",
            core_weights=(("QQQ", 0.30),),
            satellite_weight=0.70,
            regime_columns=(
                "qqq_strong_regime_score",
                "qqq_return_126d_top_quartile_756d",
                "qqq_return_252d_top_quartile_756d",
                _FALLBACK_QQQ_REGIME_COL,
            ),
        ),
        AllocationOverlaySpec(
            name="dynamic_spy_qqq_core_when_benchmark_strong",
            kind="dynamic_core_satellite",
            core_weights=(("SPY", 0.20), ("QQQ", 0.20)),
            satellite_weight=0.60,
            regime_columns=(
                "max_spy_qqq_strong_regime_score",
                _FALLBACK_BENCHMARK_REGIME_COL,
            ),
        ),
    )
    if not include_optional_qqq_floor:
        return required
    return (
        *required,
        AllocationOverlaySpec(
            name="qqq_floor_if_not_selected",
            kind="qqq_floor_if_not_selected",
            core_weights=(("QQQ", 0.20),),
            satellite_weight=0.80,
            regime_columns=(
                "qqq_strong_regime_score",
                "qqq_return_126d_top_quartile_756d",
                "qqq_return_252d_top_quartile_756d",
                _FALLBACK_QQQ_REGIME_COL,
            ),
        ),
    )


def apply_allocation_overlay(
    signals: pd.DataFrame,
    spec: AllocationOverlaySpec,
    *,
    top_k: int = 5,
    date_col: str = "date",
    symbol_col: str = "symbol",
    selected_col: str = "selected",
    rank_col: str = "signal_rank",
    prediction_col: str = "prediction",
    weight_col: str = "target_weight",
    price_col: str = "close",
) -> pd.DataFrame:
    """Apply an allocation overlay to a signal frame without changing predictions."""

    _validate_required_columns(
        signals,
        [date_col, symbol_col, selected_col, rank_col, prediction_col, weight_col],
    )
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    base = _ensure_fallback_regime_columns(
        signals.copy(),
        date_col=date_col,
        symbol_col=symbol_col,
        price_col=price_col,
    )
    out = base.copy()
    out[selected_col] = False
    out[rank_col] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out[weight_col] = 0.0

    positioned = base.copy().reset_index(drop=True)
    positioned["__allocation_overlay_position"] = np.arange(len(positioned))
    positioned[date_col] = pd.to_datetime(positioned[date_col])
    out[date_col] = pd.to_datetime(out[date_col])

    selected_col_pos = out.columns.get_loc(selected_col)
    rank_col_pos = out.columns.get_loc(rank_col)
    weight_col_pos = out.columns.get_loc(weight_col)

    for _, date_group in positioned.groupby(date_col, sort=False, dropna=False):
        selected_rows = _selected_rows_for_date(
            date_group,
            top_k=top_k,
            symbol_col=symbol_col,
            selected_col=selected_col,
            rank_col=rank_col,
            prediction_col=prediction_col,
        )
        selected_symbols = selected_rows[symbol_col].astype(str).tolist()
        weights_by_symbol, rank_symbols = _weights_for_date(
            date_group=date_group,
            selected_rows=selected_rows,
            selected_symbols=selected_symbols,
            spec=spec,
            symbol_col=symbol_col,
            prediction_col=prediction_col,
        )
        _assign_weights_for_date(
            out=out,
            date_group=date_group,
            weights_by_symbol=weights_by_symbol,
            rank_symbols=rank_symbols,
            symbol_col=symbol_col,
            selected_col_pos=selected_col_pos,
            rank_col_pos=rank_col_pos,
            weight_col_pos=weight_col_pos,
        )

    return out.drop(
        columns=[
            column
            for column in (_FALLBACK_QQQ_REGIME_COL, _FALLBACK_BENCHMARK_REGIME_COL)
            if column in out.columns and column not in signals.columns
        ],
        errors="ignore",
    )


def validate_overlay_weights(
    signals: pd.DataFrame,
    *,
    date_col: str = "date",
    selected_col: str = "selected",
    weight_col: str = "target_weight",
    tolerance: float = 1.0e-9,
) -> bool:
    """Return whether each selected date is fully invested."""

    _validate_required_columns(signals, [date_col, selected_col, weight_col])
    if signals.empty:
        return True
    frame = signals.copy()
    frame[selected_col] = frame[selected_col].fillna(False).astype(bool)
    for _, date_group in frame.groupby(date_col, sort=False, dropna=False):
        selected = date_group.loc[date_group[selected_col]]
        total = float(pd.to_numeric(selected[weight_col], errors="coerce").fillna(0.0).sum())
        expected = 1.0 if not selected.empty else 0.0
        if abs(total - expected) > tolerance:
            return False
    return True


def _weights_for_date(
    *,
    date_group: pd.DataFrame,
    selected_rows: pd.DataFrame,
    selected_symbols: list[str],
    spec: AllocationOverlaySpec,
    symbol_col: str,
    prediction_col: str,
) -> tuple[dict[str, float], list[str]]:
    if spec.kind == "equal_weight_topk":
        return _topk_weights(selected_symbols, _equal_weights(len(selected_symbols)))
    if spec.kind == "rank_weighted_topk":
        weights = _rank_weights(len(selected_symbols), spec.rank_weights)
        return _topk_weights(selected_symbols, weights)
    if spec.kind == "capped_score_weighted_topk":
        max_weight = spec.max_weight if spec.max_weight is not None else 1.0
        weights = _capped_score_weights(
            selected_rows,
            prediction_col=prediction_col,
            max_weight=max_weight,
        )
        return _topk_weights(selected_symbols, weights)
    if spec.kind == "static_core_satellite":
        return _core_satellite_weights(
            selected_symbols,
            spec,
            available_symbols=set(date_group[symbol_col].astype(str).tolist()),
        )
    if spec.kind == "dynamic_core_satellite":
        if _regime_active(date_group, spec.regime_columns):
            return _core_satellite_weights(
                selected_symbols,
                spec,
                available_symbols=set(date_group[symbol_col].astype(str).tolist()),
            )
        return _topk_weights(selected_symbols, _equal_weights(len(selected_symbols)))
    if spec.kind == "qqq_floor_if_not_selected":
        if not _regime_active(date_group, spec.regime_columns) or "QQQ" in selected_symbols:
            return _topk_weights(selected_symbols, _equal_weights(len(selected_symbols)))
        non_qqq_symbols = [symbol for symbol in selected_symbols if symbol != "QQQ"][:4]
        weights_by_symbol = _satellite_weights(non_qqq_symbols, spec.satellite_weight or 0.80)
        available_symbols = set(date_group[symbol_col].astype(str).tolist())
        rank_symbols = list(non_qqq_symbols)
        if "QQQ" in available_symbols:
            weights_by_symbol["QQQ"] = weights_by_symbol.get("QQQ", 0.0) + 0.20
            rank_symbols.append("QQQ")
        return _normalize_weight_map(weights_by_symbol), rank_symbols
    raise ValueError(f"Unsupported allocation overlay kind: {spec.kind!r}")


def _selected_rows_for_date(
    date_group: pd.DataFrame,
    *,
    top_k: int,
    symbol_col: str,
    selected_col: str,
    rank_col: str,
    prediction_col: str,
) -> pd.DataFrame:
    selected_mask = date_group[selected_col].fillna(False).astype(bool)
    selected = date_group.loc[selected_mask].copy()
    if selected.empty:
        return selected
    selected["__allocation_overlay_rank"] = pd.to_numeric(
        selected[rank_col],
        errors="coerce",
    )
    selected["__allocation_overlay_prediction"] = pd.to_numeric(
        selected[prediction_col],
        errors="coerce",
    )
    return (
        selected.sort_values(
            ["__allocation_overlay_rank", "__allocation_overlay_prediction", symbol_col],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .head(top_k)
        .drop(
            columns=["__allocation_overlay_rank", "__allocation_overlay_prediction"],
            errors="ignore",
        )
    )


def _topk_weights(
    selected_symbols: list[str],
    weights: np.ndarray,
) -> tuple[dict[str, float], list[str]]:
    if not selected_symbols:
        return {}, []
    weights_by_symbol = {
        symbol: float(weight)
        for symbol, weight in zip(selected_symbols, weights, strict=True)
        if float(weight) > _EPSILON
    }
    return _normalize_weight_map(weights_by_symbol), list(selected_symbols)


def _core_satellite_weights(
    selected_symbols: list[str],
    spec: AllocationOverlaySpec,
    *,
    available_symbols: set[str],
) -> tuple[dict[str, float], list[str]]:
    core_weights = {
        symbol: float(weight)
        for symbol, weight in spec.core_weights
        if symbol in available_symbols
    }
    core_total = sum(core_weights.values())
    satellite_weight = (
        float(spec.satellite_weight)
        if spec.satellite_weight is not None
        else max(0.0, 1.0 - core_total)
    )
    weights_by_symbol = _satellite_weights(selected_symbols, satellite_weight)
    for symbol, weight in core_weights.items():
        weights_by_symbol[symbol] = weights_by_symbol.get(symbol, 0.0) + weight

    rank_symbols = list(selected_symbols)
    for symbol in core_weights:
        if symbol not in rank_symbols:
            rank_symbols.append(symbol)
    return _normalize_weight_map(weights_by_symbol), rank_symbols


def _satellite_weights(symbols: list[str], satellite_weight: float) -> dict[str, float]:
    if not symbols:
        return {}
    per_symbol = float(satellite_weight) / len(symbols)
    return {symbol: per_symbol for symbol in symbols}


def _equal_weights(count: int) -> np.ndarray:
    if count <= 0:
        return np.array([], dtype=float)
    return np.full(count, 1.0 / count, dtype=float)


def _rank_weights(count: int, rank_weights: tuple[float, ...]) -> np.ndarray:
    if count <= 0:
        return np.array([], dtype=float)
    weights = np.asarray(rank_weights[:count], dtype=float)
    if len(weights) < count:
        weights = np.append(weights, np.full(count - len(weights), 0.0))
    total = float(weights.sum())
    if total <= _EPSILON:
        return _equal_weights(count)
    return weights / total


def _capped_score_weights(
    selected_rows: pd.DataFrame,
    *,
    prediction_col: str,
    max_weight: float,
) -> np.ndarray:
    if not 0.0 < max_weight <= 1.0:
        raise ValueError("max_weight must satisfy 0 < max_weight <= 1.")
    if len(selected_rows) * max_weight < 1.0:
        raise ValueError("selected_count * max_weight must be at least 1.")
    if selected_rows.empty:
        return np.array([], dtype=float)

    scores = pd.to_numeric(selected_rows[prediction_col], errors="coerce").to_numpy(
        dtype=float
    )
    if not bool(np.isfinite(scores).all()) or float(np.nanmax(scores)) <= 0.0:
        return _equal_weights(len(selected_rows))
    shifted = _positive_shifted_scores(scores)
    total = float(shifted.sum())
    if not np.isfinite(total) or total <= _EPSILON:
        return _equal_weights(len(selected_rows))
    return _cap_and_redistribute(shifted / total, max_weight)


def _positive_shifted_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0 or not bool(np.isfinite(scores).all()):
        return np.zeros_like(scores, dtype=float)
    min_score = float(scores.min())
    if min_score <= 0.0:
        return scores - min_score + _EPSILON
    return scores.copy()


def _cap_and_redistribute(weights: np.ndarray, max_weight: float) -> np.ndarray:
    capped = np.minimum(weights.astype(float), max_weight)
    excess = max(0.0, float(weights.sum() - capped.sum()))

    for _ in range(len(capped) + 1):
        if excess <= _EPSILON:
            break
        capacity = np.maximum(max_weight - capped, 0.0)
        eligible = capacity > _EPSILON
        if not bool(eligible.any()):
            break
        preferred = weights[eligible]
        preferred_total = float(preferred.sum())
        if preferred_total <= _EPSILON:
            preferred = capacity[eligible]
            preferred_total = float(preferred.sum())
        if preferred_total <= _EPSILON:
            break
        additions = excess * preferred / preferred_total
        applied = np.minimum(additions, capacity[eligible])
        capped[eligible] += applied
        new_excess = excess - float(applied.sum())
        if abs(new_excess - excess) <= _EPSILON:
            break
        excess = max(0.0, new_excess)

    total = float(capped.sum())
    if total <= _EPSILON:
        return _equal_weights(len(capped))
    return capped / total


def _normalize_weight_map(weights_by_symbol: dict[str, float]) -> dict[str, float]:
    positive = {
        symbol: float(weight)
        for symbol, weight in weights_by_symbol.items()
        if np.isfinite(weight) and float(weight) > _EPSILON
    }
    total = sum(positive.values())
    if total <= _EPSILON:
        return {}
    return {symbol: weight / total for symbol, weight in positive.items()}


def _assign_weights_for_date(
    *,
    out: pd.DataFrame,
    date_group: pd.DataFrame,
    weights_by_symbol: dict[str, float],
    rank_symbols: list[str],
    symbol_col: str,
    selected_col_pos: int,
    rank_col_pos: int,
    weight_col_pos: int,
) -> None:
    if not weights_by_symbol:
        return

    rank_by_symbol = {symbol: rank for rank, symbol in enumerate(rank_symbols, start=1)}
    next_rank = len(rank_by_symbol) + 1
    for _, row in date_group.iterrows():
        symbol = str(row[symbol_col])
        weight = weights_by_symbol.get(symbol, 0.0)
        if weight <= _EPSILON:
            continue
        position = int(row["__allocation_overlay_position"])
        out.iloc[position, selected_col_pos] = True
        out.iloc[position, weight_col_pos] = weight
        if symbol not in rank_by_symbol:
            rank_by_symbol[symbol] = next_rank
            next_rank += 1
        out.iloc[position, rank_col_pos] = rank_by_symbol[symbol]


def _regime_active(date_group: pd.DataFrame, regime_columns: tuple[str, ...]) -> bool:
    for column in regime_columns:
        if column not in date_group.columns:
            continue
        values = pd.to_numeric(date_group[column], errors="coerce")
        if bool((values.fillna(0.0) > 0.0).any()):
            return True
    return False


def _ensure_fallback_regime_columns(
    out: pd.DataFrame,
    *,
    date_col: str,
    symbol_col: str,
    price_col: str,
) -> pd.DataFrame:
    if _FALLBACK_QQQ_REGIME_COL in out.columns and _FALLBACK_BENCHMARK_REGIME_COL in out.columns:
        return out
    if price_col not in out.columns:
        return out

    out = out.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    price_frame = out[[date_col, symbol_col, price_col]].copy()
    price_frame[date_col] = pd.to_datetime(price_frame[date_col])
    price_frame[symbol_col] = price_frame[symbol_col].astype(str).str.upper()
    prices = price_frame.pivot_table(
        index=date_col,
        columns=symbol_col,
        values=price_col,
        aggfunc="last",
    ).sort_index(kind="mergesort")

    fallback = pd.DataFrame(index=prices.index)
    for symbol in ("SPY", "QQQ"):
        if symbol in prices.columns:
            fallback[f"_{symbol.lower()}_fallback_strong_regime_score"] = (
                _fallback_strong_regime_score(prices[symbol])
            )
        else:
            fallback[f"_{symbol.lower()}_fallback_strong_regime_score"] = np.nan

    fallback[_FALLBACK_QQQ_REGIME_COL] = fallback["_qqq_fallback_strong_regime_score"]
    fallback[_FALLBACK_BENCHMARK_REGIME_COL] = pd.concat(
        [
            fallback["_spy_fallback_strong_regime_score"],
            fallback["_qqq_fallback_strong_regime_score"],
        ],
        axis=1,
    ).max(axis=1, skipna=True)
    fallback = fallback[[_FALLBACK_QQQ_REGIME_COL, _FALLBACK_BENCHMARK_REGIME_COL]]
    return out.merge(
        fallback.reset_index(),
        on=date_col,
        how="left",
        validate="many_to_one",
    )


def _fallback_strong_regime_score(close: pd.Series) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce")
    flags: list[pd.Series] = []
    for lookback in (126, 252):
        returns = close.pct_change(periods=lookback)
        flags.append((returns > 0.0).astype("float64").where(returns.notna(), np.nan))
        percentile = returns.rolling(
            window=756,
            min_periods=min(756, max(lookback, 126)),
        ).apply(_last_value_percentile, raw=True)
        flags.append(
            (percentile > 0.75).astype("float64").where(percentile.notna(), np.nan)
        )
    frame = pd.concat(flags, axis=1)
    non_null_count = frame.notna().sum(axis=1)
    score = frame.mean(axis=1, skipna=True)
    score.loc[non_null_count.eq(0)] = np.nan
    return score


def _last_value_percentile(values: np.ndarray) -> float:
    current = values[-1]
    if np.isnan(current):
        return float("nan")
    non_null_values = values[~np.isnan(values)]
    if non_null_values.size == 0:
        return float("nan")
    return float(np.mean(non_null_values <= current))


def _validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            "Input DataFrame is missing required allocation overlay columns: "
            + ", ".join(missing)
        )


__all__ = [
    "AllocationOverlaySpec",
    "apply_allocation_overlay",
    "default_allocation_overlay_specs",
    "validate_overlay_weights",
]
