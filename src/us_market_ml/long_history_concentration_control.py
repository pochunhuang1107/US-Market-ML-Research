"""Targeted concentration-control sweep for long-history QQQ-relative candidates.

This module is local research infrastructure. It reuses the already-selected
candidate specifications, generates point-in-time portfolio construction
variants from each candidate's rebalance-date scores, and writes compact
diagnostic tables. It deliberately avoids broad model sweeps, MLP batches,
model cards, production signal exports, and cloud access.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.long_history_candidate_deep_dive import CONFIG_COLUMNS
from us_market_ml.long_history_shortlist_validation import (
    DATASET_SOURCE_LABEL,
    PRICE_MODE,
    build_candidate_daily_equity,
    build_cost_sensitivity,
    build_rolling_window_robustness,
    build_subperiod_robustness,
    generate_predictions_and_weights,
    spec_from_candidate,
    summarize_rolling_windows,
)
from us_market_ml.long_history_training import (
    END_DATE,
    START_DATE,
    build_rebalance_weights,
    evaluate_weight_schedule,
)

OUTPUT_PREFIX = "long_history_qqq_concentration_control"
DEFAULT_OUTPUT_DIR = Path("artifacts/sweeps/long_history_qqq_concentration_control_sweep")
DEFAULT_CANDIDATE_IDS: tuple[str, ...] = ("candidate_0001", "candidate_0003")
DEFAULT_COST_SENSITIVITY_BPS: tuple[float, ...] = (10.0, 20.0, 30.0)

RESULTS_NAME = f"{OUTPUT_PREFIX}_results.csv"
CANDIDATES_NAME = f"{OUTPUT_PREFIX}_candidates.csv"
SELECTED_SYMBOLS_NAME = f"{OUTPUT_PREFIX}_selected_symbols.csv"
SYMBOL_EXPOSURE_NAME = f"{OUTPUT_PREFIX}_symbol_exposure.csv"
SUBPERIODS_NAME = f"{OUTPUT_PREFIX}_subperiods.csv"
ROLLING_WINDOWS_NAME = f"{OUTPUT_PREFIX}_rolling_windows.csv"
COST_SENSITIVITY_NAME = f"{OUTPUT_PREFIX}_cost_sensitivity.csv"
PREFLIGHT_MD_NAME = f"{OUTPUT_PREFIX}_preflight.md"

OUTPUT_CSV_NAMES = (
    RESULTS_NAME,
    CANDIDATES_NAME,
    SELECTED_SYMBOLS_NAME,
    SYMBOL_EXPOSURE_NAME,
    SUBPERIODS_NAME,
    ROLLING_WINDOWS_NAME,
    COST_SENSITIVITY_NAME,
)

CONTROL_RULES: tuple[str, ...] = (
    "baseline_clone",
    "top2_equal_weight",
    "top3_equal_weight",
    "top3_rank_weighted",
    "top1_max_consecutive_rebalances",
    "top1_trailing_exposure_cap",
    "top1_score_margin_hold",
    "top1_exclude_top_symbol_sanity",
    "top1_exclude_top3_symbols_sanity",
    "top3_score_weighted_cap050",
)
MAX_CONSECUTIVE_REBALANCE_VALUES: tuple[int, ...] = (4, 6, 8)
TRAILING_EXPOSURE_CAP_VALUES: tuple[float, ...] = (0.35, 0.45, 0.55)
SCORE_MARGIN_VALUES: tuple[float, ...] = (0.0, 0.002, 0.005, 0.01)
EXCLUDED_SYMBOLS_BY_SOURCE: dict[str, dict[str, tuple[str, ...]]] = {
    "candidate_0001": {
        "top_symbol": ("XLE",),
        "top3_symbols": ("XLE", "XLU", "VUG"),
    },
    "candidate_0003": {
        "top_symbol": ("ARKK",),
        "top3_symbols": ("ARKK", "XLK", "MTUM"),
    },
}

SOURCE_METRIC_COLUMNS = [
    "source_total_return",
    "source_strategy_vs_qqq",
    "source_return_to_drawdown",
    "source_positive_vs_qqq_window_rate",
    "source_worst_strategy_vs_qqq",
    "source_top_symbol",
    "source_top_symbol_exposure_pct",
    "source_top3_symbol_exposure_pct",
    "source_unique_symbol_count",
    "source_decision",
]

CANDIDATE_COLUMNS = [
    "candidate_id",
    "candidate_label",
    "candidate_source_id",
    "source_candidate_label",
    "concentration_rule",
    "concentration_rule_family",
    "control_value",
    "max_consecutive_rebalances",
    "trailing_exposure_cap",
    "score_margin",
    "excluded_symbols",
    "diagnostic_control",
    *CONFIG_COLUMNS,
    *SOURCE_METRIC_COLUMNS,
]

RESULT_COLUMNS = [
    "candidate_id",
    "candidate_label",
    "candidate_source_id",
    "concentration_rule",
    "concentration_rule_family",
    "control_value",
    "model_type",
    "training_mode",
    "target_variant",
    "feature_set_version",
    "allocation_rule",
    "rebalance_frequency",
    "min_holding_days",
    "switch_margin",
    "transaction_cost_bps",
    "recency_decay_profile",
    "base_weight",
    "recent_window_years",
    "finetune_window_years",
    "finetune_lr_scale",
    "total_return",
    "annualized_return",
    "max_drawdown",
    "sharpe_like",
    "return_to_drawdown",
    "transaction_count",
    "average_turnover",
    "estimated_transaction_cost",
    "qqq_buy_hold_total_return",
    "spy_buy_hold_total_return",
    "strategy_vs_qqq",
    "strategy_vs_spy",
    "qqq_return_capture",
    "beats_qqq_return",
    "beats_qqq_return_to_drawdown",
    "drawdown_improvement_vs_qqq",
    "positive_vs_qqq_window_rate",
    "qqq_underperformance_rate",
    "worst_strategy_vs_qqq",
    "drawdown_improvement_window_rate",
    "top_symbol",
    "top_symbol_exposure_pct",
    "top3_symbol_exposure_pct",
    "unique_symbol_count",
    "cost_20bps_strategy_vs_qqq",
    "cost_30bps_strategy_vs_qqq",
    *SOURCE_METRIC_COLUMNS,
    "decision",
    "rationale",
    "production_ready",
]

SELECTED_SYMBOL_COLUMNS = [
    "date",
    "candidate_id",
    "candidate_label",
    "candidate_source_id",
    "concentration_rule",
    "symbol",
    "target_weight",
    "predicted_score",
    "rank",
    "allocation_rule",
    "rebalance_frequency",
]

SYMBOL_EXPOSURE_COLUMNS = [
    "candidate_id",
    "candidate_label",
    "candidate_source_id",
    "concentration_rule",
    "symbol",
    "rank_by_exposure",
    "selected_rebalance_count",
    "exposure_days",
    "exposure_pct",
    "total_return_contribution",
    "is_top_symbol",
    "is_top3_symbol",
]

SUBPERIOD_COLUMNS = [
    "candidate_id",
    "candidate_label",
    "candidate_source_id",
    "concentration_rule",
    "period_name",
    "period_start",
    "period_end",
    "total_return",
    "qqq_total_return",
    "spy_total_return",
    "strategy_vs_qqq",
    "max_drawdown",
    "qqq_max_drawdown",
    "drawdown_improvement_vs_qqq",
    "transaction_count",
    "top_symbol",
]

ROLLING_WINDOW_COLUMNS = [
    "candidate_id",
    "candidate_label",
    "candidate_source_id",
    "concentration_rule",
    "window_label",
    "window_start",
    "window_end",
    "strategy_total_return",
    "qqq_total_return",
    "strategy_vs_qqq",
    "strategy_max_drawdown",
    "qqq_max_drawdown",
    "drawdown_improvement_vs_qqq",
    "underperformed_qqq",
    "materially_improved_drawdown",
]

COST_SENSITIVITY_COLUMNS = [
    "candidate_id",
    "candidate_label",
    "candidate_source_id",
    "concentration_rule",
    "transaction_cost_bps",
    "total_return",
    "strategy_vs_qqq",
    "max_drawdown",
    "return_to_drawdown",
    "decision_at_cost",
]


@dataclass(frozen=True)
class ConcentrationControlReplay:
    """Replay artifacts for one concentration-control variant."""

    candidate: pd.Series
    weights: pd.DataFrame
    metrics: object
    daily_equity: pd.DataFrame
    position_returns: pd.DataFrame
    selected_symbols: pd.DataFrame
    symbol_exposure: pd.DataFrame
    subperiods: pd.DataFrame
    rolling_windows: pd.DataFrame
    cost_sensitivity: pd.DataFrame


@dataclass(frozen=True)
class LongHistoryConcentrationControlArtifacts:
    """CSV-ready targeted concentration-control sweep outputs."""

    candidates: pd.DataFrame
    results: pd.DataFrame
    selected_symbols: pd.DataFrame
    symbol_exposure: pd.DataFrame
    subperiods: pd.DataFrame
    rolling_windows: pd.DataFrame
    cost_sensitivity: pd.DataFrame


def select_concentration_control_source_candidates(
    *,
    deep_dive_summary: pd.DataFrame,
    deep_dive_decision_matrix: pd.DataFrame,
    candidate_ids: tuple[str, ...] = DEFAULT_CANDIDATE_IDS,
) -> pd.DataFrame:
    """Select source candidate rows from the completed candidate deep dive."""

    _validate_columns(deep_dive_summary, ["candidate_id", *CONFIG_COLUMNS], "deep-dive summary")
    _validate_columns(
        deep_dive_decision_matrix,
        ["candidate_id", "decision", "rationale"],
        "deep-dive decision matrix",
    )
    resolved_ids = candidate_ids or DEFAULT_CANDIDATE_IDS
    summary = deep_dive_summary.assign(candidate_id=lambda df: df["candidate_id"].astype(str))
    available = set(summary["candidate_id"])
    unknown = [candidate_id for candidate_id in resolved_ids if candidate_id not in available]
    if unknown:
        raise ValueError(f"Unknown candidate_id values: {', '.join(unknown)}")

    selected = (
        summary.set_index("candidate_id", drop=False)
        .loc[list(dict.fromkeys(resolved_ids))]
        .reset_index(drop=True)
        .copy()
    )
    if selected["model_type"].astype(str).str.lower().eq("mlp").any():
        raise ValueError("MLP candidates are intentionally excluded from this targeted sweep.")

    decision_cols = [
        column
        for column in [
            "candidate_id",
            "decision",
            "rationale",
            "cost_20bps_strategy_vs_qqq",
            "cost_30bps_strategy_vs_qqq",
        ]
        if column in deep_dive_decision_matrix.columns
    ]
    decisions = deep_dive_decision_matrix.loc[:, decision_cols].copy()
    decisions["candidate_id"] = decisions["candidate_id"].astype(str)
    decisions = decisions.rename(
        columns={
            "decision": "source_decision",
            "rationale": "source_rationale",
            "cost_20bps_strategy_vs_qqq": "source_cost_20bps_strategy_vs_qqq",
            "cost_30bps_strategy_vs_qqq": "source_cost_30bps_strategy_vs_qqq",
        }
    )
    selected = selected.merge(decisions, on="candidate_id", how="left")
    if "source_decision" not in selected.columns:
        selected["source_decision"] = pd.NA
    if "source_rationale" not in selected.columns:
        selected["source_rationale"] = pd.NA
    return selected


def build_concentration_control_candidates(
    source_candidates: pd.DataFrame,
    *,
    include_score_weighted_cap: bool = True,
) -> pd.DataFrame:
    """Generate the compact concentration-control config set."""

    if source_candidates.empty:
        raise ValueError("At least one source candidate is required.")

    rows: list[dict[str, object]] = []
    ordinal = 1
    for _, source in source_candidates.iterrows():
        source_id = str(source["candidate_id"])
        control_specs: list[dict[str, object]] = [
            {
                "concentration_rule": "baseline_clone",
                "concentration_rule_family": "baseline_clone",
                "allocation_rule": source["allocation_rule"],
            },
            {
                "concentration_rule": "top2_equal_weight",
                "concentration_rule_family": "top2_equal_weight",
                "allocation_rule": "top2_equal_weight",
            },
            {
                "concentration_rule": "top3_equal_weight",
                "concentration_rule_family": "top3_equal_weight",
                "allocation_rule": "top3_equal_weight",
            },
            {
                "concentration_rule": "top3_rank_weighted",
                "concentration_rule_family": "top3_rank_weighted",
                "allocation_rule": "top3_rank_weighted",
            },
        ]
        control_specs.extend(
            {
                "concentration_rule": "top1_max_consecutive_rebalances",
                "concentration_rule_family": "top1_max_consecutive_rebalances",
                "allocation_rule": "top1",
                "max_consecutive_rebalances": value,
                "control_value": str(value),
            }
            for value in MAX_CONSECUTIVE_REBALANCE_VALUES
        )
        control_specs.extend(
            {
                "concentration_rule": "top1_trailing_exposure_cap",
                "concentration_rule_family": "top1_trailing_exposure_cap",
                "allocation_rule": "top1",
                "trailing_exposure_cap": value,
                "control_value": f"{value:.2f}",
            }
            for value in TRAILING_EXPOSURE_CAP_VALUES
        )
        control_specs.extend(
            {
                "concentration_rule": "top1_score_margin_hold",
                "concentration_rule_family": "top1_score_margin_hold",
                "allocation_rule": "top1",
                "score_margin": value,
                "switch_margin": value,
                "control_value": f"{value:g}",
            }
            for value in SCORE_MARGIN_VALUES
        )
        excluded_top = EXCLUDED_SYMBOLS_BY_SOURCE.get(source_id, {}).get("top_symbol", ())
        excluded_top3 = EXCLUDED_SYMBOLS_BY_SOURCE.get(source_id, {}).get("top3_symbols", ())
        control_specs.extend(
            [
                {
                    "concentration_rule": "top1_exclude_top_symbol_sanity",
                    "concentration_rule_family": "top1_exclude_top_symbol_sanity",
                    "allocation_rule": "top1",
                    "excluded_symbols": excluded_top,
                    "control_value": ",".join(excluded_top),
                    "diagnostic_control": True,
                },
                {
                    "concentration_rule": "top1_exclude_top3_symbols_sanity",
                    "concentration_rule_family": "top1_exclude_top3_symbols_sanity",
                    "allocation_rule": "top1",
                    "excluded_symbols": excluded_top3,
                    "control_value": ",".join(excluded_top3),
                    "diagnostic_control": True,
                },
            ]
        )
        if include_score_weighted_cap:
            control_specs.append(
                {
                    "concentration_rule": "top3_score_weighted_cap050",
                    "concentration_rule_family": "top3_score_weighted_cap050",
                    "allocation_rule": "top3_score_weighted_cap050",
                }
            )

        for spec in control_specs:
            candidate_id = f"{source_id}_cc_{ordinal:04d}"
            row = _candidate_config_row(source, spec, candidate_id)
            rows.append(row)
            ordinal += 1

    out = pd.DataFrame(rows)
    for column in CANDIDATE_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out.loc[:, CANDIDATE_COLUMNS]


def build_long_history_concentration_control_artifacts(
    *,
    raw_df: pd.DataFrame,
    source_candidates: pd.DataFrame,
    control_candidates: pd.DataFrame | None = None,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    min_history_rows: int = 252,
    min_train_eligible_rows: int = 1000,
    cost_sensitivity_bps: tuple[float, ...] = DEFAULT_COST_SENSITIVITY_BPS,
) -> LongHistoryConcentrationControlArtifacts:
    """Replay source predictions once, then evaluate all concentration controls."""

    candidates = (
        control_candidates
        if control_candidates is not None
        else build_concentration_control_candidates(source_candidates)
    ).copy()
    if candidates.empty:
        raise ValueError("At least one concentration-control candidate is required.")

    replays: list[ConcentrationControlReplay] = []
    source_by_id = {
        str(row["candidate_id"]): row for _, row in source_candidates.iterrows()
    }
    for source_id, family in candidates.groupby("candidate_source_id", sort=False):
        source = pd.Series(source_by_id[str(source_id)])
        spec = spec_from_candidate(source)
        predictions, baseline_weights, prepared_raw = generate_predictions_and_weights(
            raw_df=raw_df,
            spec=spec,
            start_date=start_date,
            end_date=end_date,
            min_history_rows=min_history_rows,
            min_train_eligible_rows=min_train_eligible_rows,
        )
        eval_dates = _evaluation_dates(prepared_raw, start_date=start_date, end_date=end_date)
        for _, candidate in family.iterrows():
            weights = build_control_rebalance_weights(
                predictions=predictions,
                candidate=candidate,
                baseline_weights=baseline_weights,
                eval_dates=eval_dates,
            )
            replays.append(
                replay_concentration_control_candidate(
                    raw_df=prepared_raw,
                    predictions=predictions,
                    weights=weights,
                    candidate=candidate,
                    start_date=start_date,
                    end_date=end_date,
                    cost_sensitivity_bps=cost_sensitivity_bps,
                )
            )

    results = _concat_or_empty(
        [_result_row_from_replay(replay) for replay in replays],
        RESULT_COLUMNS,
    )
    selected_symbols = _concat_or_empty(
        [replay.selected_symbols for replay in replays],
        SELECTED_SYMBOL_COLUMNS,
    )
    symbol_exposure = _concat_or_empty(
        [replay.symbol_exposure for replay in replays],
        SYMBOL_EXPOSURE_COLUMNS,
    )
    subperiods = _concat_or_empty([replay.subperiods for replay in replays], SUBPERIOD_COLUMNS)
    rolling_windows = _concat_or_empty(
        [replay.rolling_windows for replay in replays],
        ROLLING_WINDOW_COLUMNS,
    )
    cost_sensitivity = _concat_or_empty(
        [replay.cost_sensitivity for replay in replays],
        COST_SENSITIVITY_COLUMNS,
    )
    return LongHistoryConcentrationControlArtifacts(
        candidates=candidates,
        results=results,
        selected_symbols=selected_symbols,
        symbol_exposure=symbol_exposure,
        subperiods=subperiods,
        rolling_windows=rolling_windows,
        cost_sensitivity=cost_sensitivity,
    )


def build_control_rebalance_weights(
    *,
    predictions: pd.DataFrame,
    candidate: pd.Series,
    baseline_weights: pd.DataFrame | None = None,
    eval_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Convert source predictions into concentration-controlled target weights."""

    rule = str(candidate["concentration_rule"])
    if rule == "baseline_clone":
        if baseline_weights is not None:
            return baseline_weights.copy()
        return build_rebalance_weights(
            predictions,
            allocation_rule="top1",
            min_holding_days=int(candidate.get("min_holding_days", 0)),
            switch_margin=float(candidate.get("switch_margin", 0.0)),
        )

    source = predictions.copy()
    if source.empty:
        return pd.DataFrame(columns=["date", "symbol", "target_weight"])
    source["date"] = pd.to_datetime(source["date"])
    eval_index = (
        pd.DatetimeIndex(pd.to_datetime(eval_dates)).sort_values()
        if eval_dates is not None
        else pd.DatetimeIndex(sorted(source["date"].dropna().unique()))
    )

    rows: list[dict[str, object]] = []
    current_symbol: str | None = None
    current_streak = 0
    exposure_days: dict[str, int] = {}
    total_exposure_days = 0
    previous_exposure_symbol: str | None = None
    previous_rebalance_date: pd.Timestamp | None = None

    for date, date_group in source.groupby("date", sort=True):
        date = pd.Timestamp(date)
        ranked = _ranked_predictions(
            date_group,
            excluded_symbols=_excluded_symbols(candidate),
        )
        if previous_rebalance_date is not None and previous_exposure_symbol is not None:
            interval_days = int(
                (
                    (eval_index >= previous_rebalance_date)
                    & (eval_index < date)
                ).sum()
            )
            if interval_days > 0:
                exposure_days[previous_exposure_symbol] = (
                    exposure_days.get(previous_exposure_symbol, 0) + interval_days
                )
                total_exposure_days += interval_days

        selected = _select_for_rule(
            ranked=ranked,
            candidate=candidate,
            current_symbol=current_symbol,
            current_streak=current_streak,
            exposure_days=exposure_days,
            total_exposure_days=total_exposure_days,
        )
        weights = _weights_for_rule(selected, rule)
        selected_symbols = [symbol for symbol, weight in weights.items() if weight > 0.0]
        for symbol, weight in weights.items():
            rows.append({"date": date, "symbol": symbol, "target_weight": float(weight)})

        primary_symbol = selected_symbols[0] if selected_symbols else None
        if rule == "top1_max_consecutive_rebalances":
            if primary_symbol is not None and primary_symbol == current_symbol:
                current_streak += 1
            else:
                current_symbol = primary_symbol
                current_streak = 1 if primary_symbol is not None else 0
        elif rule == "top1_score_margin_hold":
            current_symbol = primary_symbol
            current_streak = current_streak + 1 if primary_symbol else 0
        else:
            current_symbol = primary_symbol
            current_streak = 1 if primary_symbol is not None else 0
        previous_exposure_symbol = primary_symbol
        previous_rebalance_date = date

    return pd.DataFrame(rows, columns=["date", "symbol", "target_weight"])


def replay_concentration_control_candidate(
    *,
    raw_df: pd.DataFrame,
    predictions: pd.DataFrame,
    weights: pd.DataFrame,
    candidate: pd.Series,
    start_date: str,
    end_date: str,
    cost_sensitivity_bps: tuple[float, ...] = DEFAULT_COST_SENSITIVITY_BPS,
) -> ConcentrationControlReplay:
    """Evaluate one concentration-control weight schedule."""

    metrics = evaluate_weight_schedule(
        raw_df,
        weights,
        start_date=start_date,
        end_date=end_date,
        transaction_cost_bps=float(candidate["transaction_cost_bps"]),
    )
    daily_equity, position_returns = build_candidate_daily_equity(
        raw_df=raw_df,
        rebalance_weights=weights,
        candidate=candidate,
        start_date=start_date,
        end_date=end_date,
        transaction_cost_bps=float(candidate["transaction_cost_bps"]),
    )
    selected_symbols = build_control_selected_symbols_log(
        candidate=candidate,
        weights=weights,
        predictions=predictions,
    )
    symbol_summary = build_weighted_symbol_exposure_summary(
        candidate=candidate,
        weights=weights,
        daily_equity=daily_equity,
        position_returns=position_returns,
    )
    symbol_exposure = build_symbol_exposure_rows(symbol_summary)
    subperiods = _with_control_metadata(
        build_subperiod_robustness(
            candidate=candidate,
            daily_equity=daily_equity,
            trades=_trades_from_weights(weights, candidate),
        ),
        candidate,
        SUBPERIOD_COLUMNS,
    )
    rolling_windows = _with_control_metadata(
        build_rolling_window_robustness(
            candidate=candidate,
            daily_equity=daily_equity,
            start_date=start_date,
            end_date=end_date,
        ),
        candidate,
        ROLLING_WINDOW_COLUMNS,
    )
    cost_sensitivity = _with_control_metadata(
        build_cost_sensitivity(
            raw_df=raw_df,
            rebalance_weights=weights,
            candidate=candidate,
            start_date=start_date,
            end_date=end_date,
            cost_sensitivity_bps=cost_sensitivity_bps,
        ),
        candidate,
        COST_SENSITIVITY_COLUMNS,
    )
    return ConcentrationControlReplay(
        candidate=candidate,
        weights=weights,
        metrics=metrics,
        daily_equity=daily_equity,
        position_returns=position_returns,
        selected_symbols=selected_symbols,
        symbol_exposure=symbol_exposure,
        subperiods=subperiods,
        rolling_windows=rolling_windows,
        cost_sensitivity=cost_sensitivity,
    )


def build_control_selected_symbols_log(
    *,
    candidate: pd.Series,
    weights: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Return selected symbols at each rebalance with rank and score."""

    if weights.empty:
        return pd.DataFrame(columns=SELECTED_SYMBOL_COLUMNS)
    out = weights.copy()
    out["date"] = pd.to_datetime(out["date"])
    rankings = _prediction_rankings(predictions)
    out = out.merge(rankings, on=["date", "symbol"], how="left")
    out["candidate_id"] = candidate["candidate_id"]
    out["candidate_label"] = candidate["candidate_label"]
    out["candidate_source_id"] = candidate["candidate_source_id"]
    out["concentration_rule"] = candidate["concentration_rule"]
    out["allocation_rule"] = candidate["allocation_rule"]
    out["rebalance_frequency"] = candidate["rebalance_frequency"]
    out = out.rename(columns={"prediction": "predicted_score"})
    for column in SELECTED_SYMBOL_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out.loc[:, SELECTED_SYMBOL_COLUMNS].sort_values(
        ["date", "rank", "symbol"],
        kind="mergesort",
    )


def build_weighted_symbol_exposure_summary(
    *,
    candidate: pd.Series,
    weights: pd.DataFrame,
    daily_equity: pd.DataFrame,
    position_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Build allocation-weighted per-symbol exposure and contribution rows."""

    if weights.empty:
        return pd.DataFrame(
            columns=[
                "candidate_id",
                "candidate_label",
                "candidate_source_id",
                "concentration_rule",
                "symbol",
                "selected_rebalance_count",
                "exposure_days",
                "exposure_pct",
                "total_return_contribution",
            ]
        )
    selected_counts = (
        weights.loc[pd.to_numeric(weights["target_weight"], errors="coerce").gt(0.0)]
        .groupby("symbol", dropna=False)
        .size()
        .rename("selected_rebalance_count")
    )
    positions = position_returns.copy()
    if positions.empty:
        positions = pd.DataFrame(columns=["symbol", "lagged_target_weight", "position_return"])
    exposure_weight_col = (
        "lagged_target_weight" if "lagged_target_weight" in positions.columns else "target_weight"
    )
    if exposure_weight_col not in positions.columns:
        positions[exposure_weight_col] = 0.0
    positions[exposure_weight_col] = pd.to_numeric(
        positions[exposure_weight_col],
        errors="coerce",
    ).fillna(0.0)
    exposure_days = (
        positions.loc[positions[exposure_weight_col].gt(0.0)]
        .groupby("symbol", dropna=False)
        .size()
    )
    exposure_weight = positions.groupby("symbol", dropna=False)[exposure_weight_col].sum()
    contributions = (
        positions.groupby("symbol", dropna=False)["position_return"].sum()
        if "position_return" in positions.columns
        else pd.Series(dtype="float64")
    )
    symbols = sorted(
        set(selected_counts.index.astype(str))
        .union(exposure_days.index.astype(str))
        .union(exposure_weight.index.astype(str))
        .union(contributions.index.astype(str))
    )
    total_days = max(int(len(daily_equity)), 1)
    rows = []
    for symbol in symbols:
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "candidate_label": candidate["candidate_label"],
                "candidate_source_id": candidate["candidate_source_id"],
                "concentration_rule": candidate["concentration_rule"],
                "symbol": symbol,
                "selected_rebalance_count": int(selected_counts.get(symbol, 0)),
                "exposure_days": int(exposure_days.get(symbol, 0)),
                "exposure_pct": float(exposure_weight.get(symbol, 0.0)) / total_days,
                "total_return_contribution": float(contributions.get(symbol, 0.0)),
            }
        )
    return pd.DataFrame(rows)


def build_symbol_exposure_rows(symbol_summary: pd.DataFrame) -> pd.DataFrame:
    """Add concentration ranks and top-symbol flags to exposure rows."""

    if symbol_summary.empty:
        return pd.DataFrame(columns=SYMBOL_EXPOSURE_COLUMNS)
    out = symbol_summary.copy()
    out = out.sort_values(
        ["candidate_id", "exposure_pct", "selected_rebalance_count", "symbol"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    out["rank_by_exposure"] = out.groupby("candidate_id", sort=False).cumcount() + 1
    out["is_top_symbol"] = out["rank_by_exposure"].eq(1)
    out["is_top3_symbol"] = out["rank_by_exposure"].le(3)
    for column in SYMBOL_EXPOSURE_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out.loc[:, SYMBOL_EXPOSURE_COLUMNS]


def build_concentration_control_decision_matrix(results: pd.DataFrame) -> pd.DataFrame:
    """Build report decision rows from sweep results."""

    if results.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    out = results.copy()
    decisions = out.apply(classify_concentration_control_decision, axis=1)
    out["decision"] = [item[0] for item in decisions]
    out["rationale"] = [item[1] for item in decisions]
    out["production_ready"] = False
    for column in RESULT_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out.loc[:, RESULT_COLUMNS]


def classify_concentration_control_decision(row: pd.Series) -> tuple[str, str]:
    """Classify a concentration-control result using research-only decision rules."""

    strategy_vs_qqq = _float(row.get("strategy_vs_qqq"))
    beats_return = _bool_value(row.get("beats_qqq_return"))
    beats_rtd = _bool_value(row.get("beats_qqq_return_to_drawdown"))
    cost20 = _float(row.get("cost_20bps_strategy_vs_qqq"))
    cost30 = _float(row.get("cost_30bps_strategy_vs_qqq"))
    source_edge = _float(row.get("source_strategy_vs_qqq"))
    source_positive = _float(row.get("source_positive_vs_qqq_window_rate"))
    positive = _float(row.get("positive_vs_qqq_window_rate"))
    source_top = _float(row.get("source_top_symbol_exposure_pct"))
    source_top3 = _float(row.get("source_top3_symbol_exposure_pct"))
    source_unique = _float(row.get("source_unique_symbol_count"))
    top = _float(row.get("top_symbol_exposure_pct"))
    top3 = _float(row.get("top3_symbol_exposure_pct"))
    unique = _float(row.get("unique_symbol_count"))
    rule = str(row.get("concentration_rule", ""))

    concentration_improved = _concentration_improved(
        source_top=source_top,
        source_top3=source_top3,
        source_unique=source_unique,
        top=top,
        top3=top3,
        unique=unique,
    )
    rolling_improved = np.isfinite(positive) and np.isfinite(source_positive) and (
        positive >= source_positive + 0.05
    )
    edge_retention = _safe_ratio(strategy_vs_qqq, source_edge)
    edge_acceptable = strategy_vs_qqq > 0.0 and (
        not np.isfinite(source_edge) or source_edge <= 0.0 or edge_retention >= 0.50
    )
    costs_positive = cost20 > 0.0 and cost30 > 0.0
    exclusion_diagnostic = rule in {
        "top1_exclude_top_symbol_sanity",
        "top1_exclude_top3_symbols_sanity",
    }

    parts = [
        "beats QQQ return" if beats_return else "does not beat QQQ return",
        "beats QQQ return-to-drawdown" if beats_rtd else "does not beat QQQ return-to-drawdown",
    ]
    if np.isfinite(strategy_vs_qqq):
        parts.append(f"QQQ edge {strategy_vs_qqq:.4f}")
    if np.isfinite(edge_retention):
        parts.append(f"source edge retention {edge_retention:.2f}")
    if concentration_improved:
        parts.append("concentration improved")
    if rolling_improved:
        parts.append("rolling QQQ-positive rate improved")
    if np.isfinite(cost20):
        parts.append(f"20bps edge {cost20:.4f}")
    if np.isfinite(cost30):
        parts.append(f"30bps edge {cost30:.4f}")

    rationale = "; ".join(parts)
    if not beats_return or strategy_vs_qqq <= 0.0:
        return "reject_deprioritize", rationale
    if exclusion_diagnostic and edge_retention < 0.25:
        return "reject_deprioritize", rationale + "; exclusion diagnostic exposed dependency"
    if rolling_improved and edge_acceptable and costs_positive:
        return "robust_candidate", rationale
    if concentration_improved and beats_rtd and costs_positive and edge_acceptable:
        return "keep_for_deep_dive", rationale
    if concentration_improved:
        return "concentration_improved_mixed", rationale
    return "reject_deprioritize", rationale


def write_long_history_concentration_control_artifacts(
    *,
    artifacts: LongHistoryConcentrationControlArtifacts,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write all concentration-control sweep CSV artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = output_paths(output_dir)
    artifacts.results.to_csv(outputs[RESULTS_NAME], index=False)
    artifacts.candidates.to_csv(outputs[CANDIDATES_NAME], index=False)
    artifacts.selected_symbols.to_csv(outputs[SELECTED_SYMBOLS_NAME], index=False)
    artifacts.symbol_exposure.to_csv(outputs[SYMBOL_EXPOSURE_NAME], index=False)
    artifacts.subperiods.to_csv(outputs[SUBPERIODS_NAME], index=False)
    artifacts.rolling_windows.to_csv(outputs[ROLLING_WINDOWS_NAME], index=False)
    artifacts.cost_sensitivity.to_csv(outputs[COST_SENSITIVITY_NAME], index=False)
    return outputs


def output_paths(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    """Return all expected output paths for the sweep directory."""

    return {
        RESULTS_NAME: output_dir / RESULTS_NAME,
        CANDIDATES_NAME: output_dir / CANDIDATES_NAME,
        SELECTED_SYMBOLS_NAME: output_dir / SELECTED_SYMBOLS_NAME,
        SYMBOL_EXPOSURE_NAME: output_dir / SYMBOL_EXPOSURE_NAME,
        SUBPERIODS_NAME: output_dir / SUBPERIODS_NAME,
        ROLLING_WINDOWS_NAME: output_dir / ROLLING_WINDOWS_NAME,
        COST_SENSITIVITY_NAME: output_dir / COST_SENSITIVITY_NAME,
        PREFLIGHT_MD_NAME: output_dir / PREFLIGHT_MD_NAME,
    }


def render_preflight_markdown(
    *,
    source_candidates: pd.DataFrame,
    control_candidates: pd.DataFrame,
    market_dataset_glob: str,
    model_zoo_results: Path,
    candidate_deep_dive_summary: Path,
    candidate_deep_dive_decision_matrix: Path,
    candidate_deep_dive_selected_symbols: Path,
    model_zoo_result_count: int,
    output_dir: Path,
    report_output_dir: Path,
    start_date: str,
    end_date: str,
    candidate_ids: tuple[str, ...],
    dry_run: bool,
) -> str:
    """Render research-only preflight markdown for the targeted sweep."""

    lines = [
        "# Long-History QQQ Concentration-Control Sweep Preflight",
        "",
        "Targeted concentration-control sweep around two long-history QQQ-relative "
        "source candidates.",
        "",
        "## Controls",
        "",
        "- Local-only: `true`",
        "- S3/cloud access: `false`",
        "- Broad sweep: `false`",
        "- MLP batch: `false`",
        "- Production signal export: `false`",
        "- Model card: `false`",
        "- Production-ready marking: `false`",
        "- Primary benchmark: `QQQ buy-and-hold`",
        "- Secondary reference: `SPY buy-and-hold`",
        f"- Dataset source: `{DATASET_SOURCE_LABEL}`",
        f"- Price mode: `{PRICE_MODE}`",
        "",
        "## Inputs",
        "",
        f"- Model-zoo results: `{model_zoo_results}`",
        f"- Model-zoo result rows: `{model_zoo_result_count}`",
        f"- Candidate deep-dive summary: `{candidate_deep_dive_summary}`",
        f"- Candidate deep-dive decision matrix: `{candidate_deep_dive_decision_matrix}`",
        f"- Candidate deep-dive selected symbols: `{candidate_deep_dive_selected_symbols}`",
        f"- Market dataset glob: `{market_dataset_glob}`",
        f"- Output directory: `{output_dir}`",
        f"- Report output directory: `{report_output_dir}`",
        f"- Start date: `{start_date}`",
        f"- End date: `{end_date}`",
        f"- Requested candidate IDs: `{list(candidate_ids) if candidate_ids else 'defaults'}`",
        f"- Dry run: `{dry_run}`",
        "",
        "## Sweep Size",
        "",
        f"- Source candidate count: `{len(source_candidates)}`",
        f"- Generated concentration-control configs: `{len(control_candidates)}`",
        f"- Smaller than model zoo: `{len(control_candidates) < model_zoo_result_count}`",
        "",
        "## Selected Source Candidates",
        "",
        _markdown_table(
            source_candidates,
            [
                "candidate_id",
                "candidate_label",
                "model_type",
                "training_mode",
                "target_variant",
                "strategy_vs_qqq",
                "top_symbol",
                "top_symbol_exposure_pct",
                "top3_symbol_exposure_pct",
                "source_decision",
            ],
        ),
        "",
        "## Concentration Rules",
        "",
        _markdown_table(
            control_candidates,
            [
                "candidate_source_id",
                "concentration_rule",
                "control_value",
                "allocation_rule",
                "excluded_symbols",
                "diagnostic_control",
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def _candidate_config_row(
    source: pd.Series,
    spec: dict[str, object],
    candidate_id: str,
) -> dict[str, object]:
    source_id = str(source["candidate_id"])
    rule = str(spec["concentration_rule"])
    control_value = str(spec.get("control_value", ""))
    suffix = f" {control_value}" if control_value else ""
    row: dict[str, object] = {
        "candidate_id": candidate_id,
        "candidate_label": f"{source.get('candidate_label', source_id)} | {rule}{suffix}",
        "candidate_source_id": source_id,
        "source_candidate_label": source.get("candidate_label", ""),
        "concentration_rule": rule,
        "concentration_rule_family": spec.get("concentration_rule_family", rule),
        "control_value": control_value,
        "max_consecutive_rebalances": spec.get("max_consecutive_rebalances", pd.NA),
        "trailing_exposure_cap": spec.get("trailing_exposure_cap", pd.NA),
        "score_margin": spec.get("score_margin", pd.NA),
        "excluded_symbols": ";".join(str(symbol) for symbol in spec.get("excluded_symbols", ())),
        "diagnostic_control": bool(spec.get("diagnostic_control", False)),
    }
    for column in CONFIG_COLUMNS:
        row[column] = source.get(column, pd.NA)
    row["allocation_rule"] = spec.get("allocation_rule", row.get("allocation_rule"))
    if "switch_margin" in spec:
        row["switch_margin"] = spec["switch_margin"]
    for source_column, output_column in [
        ("total_return", "source_total_return"),
        ("strategy_vs_qqq", "source_strategy_vs_qqq"),
        ("return_to_drawdown", "source_return_to_drawdown"),
        ("positive_vs_qqq_window_rate", "source_positive_vs_qqq_window_rate"),
        ("worst_strategy_vs_qqq", "source_worst_strategy_vs_qqq"),
        ("top_symbol", "source_top_symbol"),
        ("top_symbol_exposure_pct", "source_top_symbol_exposure_pct"),
        ("top3_symbol_exposure_pct", "source_top3_symbol_exposure_pct"),
        ("unique_symbol_count", "source_unique_symbol_count"),
        ("source_decision", "source_decision"),
    ]:
        row[output_column] = source.get(source_column, pd.NA)
    return row


def _select_for_rule(
    *,
    ranked: pd.DataFrame,
    candidate: pd.Series,
    current_symbol: str | None,
    current_streak: int,
    exposure_days: dict[str, int],
    total_exposure_days: int,
) -> pd.DataFrame:
    rule = str(candidate["concentration_rule"])
    if ranked.empty:
        return ranked
    if rule in {
        "top2_equal_weight",
        "top3_equal_weight",
        "top3_rank_weighted",
        "top3_score_weighted_cap050",
    }:
        return ranked.head(2 if rule == "top2_equal_weight" else 3).copy()
    if rule == "top1_max_consecutive_rebalances":
        max_consecutive = int(candidate.get("max_consecutive_rebalances", 0))
        top_symbol = str(ranked.iloc[0]["symbol"])
        if top_symbol == current_symbol and current_streak >= max_consecutive:
            alternatives = ranked.loc[~ranked["symbol"].astype(str).eq(top_symbol)]
            if not alternatives.empty:
                return alternatives.head(1).copy()
        return ranked.head(1).copy()
    if rule == "top1_trailing_exposure_cap":
        cap = _float(candidate.get("trailing_exposure_cap"))
        if not np.isfinite(cap):
            return ranked.head(1).copy()
        denominator = max(total_exposure_days, 1)
        for _, row in ranked.iterrows():
            symbol = str(row["symbol"])
            trailing_exposure = exposure_days.get(symbol, 0) / denominator
            if trailing_exposure <= cap:
                return ranked.loc[ranked["symbol"].astype(str).eq(symbol)].head(1).copy()
        return ranked.head(1).copy()
    if rule == "top1_score_margin_hold":
        margin = _float(candidate.get("score_margin"))
        margin = 0.0 if not np.isfinite(margin) else margin
        if current_symbol is None:
            return ranked.head(1).copy()
        ranked_by_symbol = {str(row.symbol): row for row in ranked.itertuples(index=False)}
        if current_symbol not in ranked_by_symbol:
            return ranked.head(1).copy()
        top_symbol = str(ranked.iloc[0]["symbol"])
        if top_symbol == current_symbol:
            return ranked.head(1).copy()
        top_score = _float(ranked.iloc[0]["prediction"])
        current_score = _float(ranked_by_symbol[current_symbol].prediction)
        if top_score > current_score + margin:
            return ranked.head(1).copy()
        return ranked.loc[ranked["symbol"].astype(str).eq(current_symbol)].head(1).copy()
    return ranked.head(1).copy()


def _weights_for_rule(selected: pd.DataFrame, rule: str) -> dict[str, float]:
    if selected.empty:
        return {}
    symbols = selected["symbol"].astype(str).tolist()
    if rule in {
        "baseline_clone",
        "top1_max_consecutive_rebalances",
        "top1_trailing_exposure_cap",
        "top1_score_margin_hold",
        "top1_exclude_top_symbol_sanity",
        "top1_exclude_top3_symbols_sanity",
    }:
        return {symbols[0]: 1.0}
    if rule in {"top2_equal_weight", "top3_equal_weight"}:
        weight = 1.0 / len(symbols)
        return {symbol: weight for symbol in symbols}
    if rule == "top3_rank_weighted":
        raw = np.array([0.50, 0.30, 0.20], dtype="float64")[: len(symbols)]
        weights = raw / raw.sum()
        return dict(zip(symbols, weights, strict=True))
    if rule == "top3_score_weighted_cap050":
        return _score_weighted_with_cap(selected, cap=0.50)
    raise ValueError(f"Unsupported concentration_rule: {rule!r}.")


def _score_weighted_with_cap(selected: pd.DataFrame, *, cap: float) -> dict[str, float]:
    symbols = selected["symbol"].astype(str).tolist()
    scores = pd.to_numeric(selected["prediction"], errors="coerce").to_numpy(dtype="float64")
    if len(symbols) == 0:
        return {}
    if not np.isfinite(scores).all() or np.nanmax(scores) == np.nanmin(scores):
        weight = 1.0 / len(symbols)
        return {symbol: weight for symbol in symbols}
    shifted = scores - np.nanmin(scores)
    raw = shifted + 1.0e-6
    if not np.isfinite(raw).all() or raw.sum() <= 0.0:
        weight = 1.0 / len(symbols)
        return {symbol: weight for symbol in symbols}
    weights = _cap_and_redistribute(raw / raw.sum(), cap=cap)
    return dict(zip(symbols, weights, strict=True))


def _cap_and_redistribute(weights: np.ndarray, *, cap: float) -> np.ndarray:
    capped = np.zeros_like(weights, dtype="float64")
    remaining = np.ones_like(weights, dtype=bool)
    remaining_weight = 1.0
    source = weights.astype("float64").copy()
    while remaining.any():
        denominator = source[remaining].sum()
        if denominator <= 0.0:
            capped[remaining] = remaining_weight / remaining.sum()
            break
        candidate = source[remaining] / denominator * remaining_weight
        over_cap = candidate > cap
        if not over_cap.any():
            capped[remaining] = candidate
            break
        remaining_indices = np.flatnonzero(remaining)
        capped_indices = remaining_indices[over_cap]
        capped[capped_indices] = cap
        remaining[capped_indices] = False
        remaining_weight = 1.0 - capped.sum()
        if remaining_weight <= 0.0:
            break
    if capped.sum() > 0.0:
        return capped / capped.sum()
    return np.repeat(1.0 / len(weights), len(weights))


def _ranked_predictions(
    date_group: pd.DataFrame,
    *,
    excluded_symbols: tuple[str, ...] = (),
) -> pd.DataFrame:
    excluded = {symbol.upper() for symbol in excluded_symbols}
    ranked = date_group.dropna(subset=["prediction"]).copy()
    if excluded:
        ranked = ranked.loc[~ranked["symbol"].astype(str).str.upper().isin(excluded)]
    return ranked.sort_values(
        ["prediction", "symbol"],
        ascending=[False, True],
        kind="mergesort",
    ).copy()


def _excluded_symbols(candidate: pd.Series) -> tuple[str, ...]:
    value = candidate.get("excluded_symbols", "")
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(
            symbol.strip()
            for symbol in value.replace(",", ";").split(";")
            if symbol.strip()
        )
    if isinstance(value, (tuple, list, set)):
        return tuple(str(symbol) for symbol in value)
    if pd.isna(value):
        return ()
    return tuple(str(symbol) for symbol in value)


def _result_row_from_replay(replay: ConcentrationControlReplay) -> pd.DataFrame:
    metrics = replay.metrics
    candidate = replay.candidate
    rolling_summary = summarize_rolling_windows(replay.rolling_windows)
    rolling_row = rolling_summary.iloc[0] if not rolling_summary.empty else pd.Series(dtype=object)
    exposure_row = _candidate_level_exposure(replay.symbol_exposure)
    cost_edges = replay.cost_sensitivity.set_index("transaction_cost_bps")["strategy_vs_qqq"]
    row = {
        "candidate_id": candidate["candidate_id"],
        "candidate_label": candidate["candidate_label"],
        "candidate_source_id": candidate["candidate_source_id"],
        "concentration_rule": candidate["concentration_rule"],
        "concentration_rule_family": candidate["concentration_rule_family"],
        "control_value": candidate.get("control_value", ""),
        "total_return": metrics.total_return,
        "annualized_return": metrics.annualized_return,
        "max_drawdown": metrics.max_drawdown,
        "sharpe_like": metrics.sharpe_like,
        "return_to_drawdown": metrics.return_to_drawdown,
        "transaction_count": metrics.transaction_count,
        "average_turnover": metrics.average_turnover,
        "estimated_transaction_cost": metrics.estimated_transaction_cost,
        "qqq_buy_hold_total_return": metrics.qqq_buy_hold_total_return,
        "spy_buy_hold_total_return": metrics.spy_buy_hold_total_return,
        "strategy_vs_qqq": metrics.strategy_vs_qqq,
        "strategy_vs_spy": metrics.strategy_vs_spy,
        "qqq_return_capture": metrics.qqq_return_capture,
        "beats_qqq_return": metrics.beats_qqq_return,
        "beats_qqq_return_to_drawdown": metrics.beats_qqq_return_to_drawdown,
        "drawdown_improvement_vs_qqq": metrics.drawdown_improvement_vs_qqq,
        "positive_vs_qqq_window_rate": rolling_row.get(
            "positive_vs_qqq_window_rate",
            pd.NA,
        ),
        "qqq_underperformance_rate": rolling_row.get("qqq_underperformance_rate", pd.NA),
        "worst_strategy_vs_qqq": rolling_row.get("worst_strategy_vs_qqq", pd.NA),
        "drawdown_improvement_window_rate": rolling_row.get(
            "drawdown_improvement_window_rate",
            pd.NA,
        ),
        "top_symbol": exposure_row["top_symbol"],
        "top_symbol_exposure_pct": exposure_row["top_symbol_exposure_pct"],
        "top3_symbol_exposure_pct": exposure_row["top3_symbol_exposure_pct"],
        "unique_symbol_count": exposure_row["unique_symbol_count"],
        "cost_20bps_strategy_vs_qqq": _series_get_float(cost_edges, 20.0),
        "cost_30bps_strategy_vs_qqq": _series_get_float(cost_edges, 30.0),
        "decision": "",
        "rationale": "",
        "production_ready": False,
    }
    for column in CONFIG_COLUMNS:
        row[column] = candidate.get(column, pd.NA)
    for column in SOURCE_METRIC_COLUMNS:
        row[column] = candidate.get(column, pd.NA)
    decision, rationale = classify_concentration_control_decision(pd.Series(row))
    row["decision"] = decision
    row["rationale"] = rationale
    return pd.DataFrame([row], columns=RESULT_COLUMNS)


def _candidate_level_exposure(symbol_exposure: pd.DataFrame) -> dict[str, object]:
    if symbol_exposure.empty:
        return {
            "top_symbol": "",
            "top_symbol_exposure_pct": 0.0,
            "top3_symbol_exposure_pct": 0.0,
            "unique_symbol_count": 0,
        }
    ordered = symbol_exposure.sort_values(
        ["rank_by_exposure", "symbol"],
        ascending=[True, True],
        kind="mergesort",
    )
    return {
        "top_symbol": str(ordered["symbol"].iloc[0]),
        "top_symbol_exposure_pct": float(ordered["exposure_pct"].iloc[0]),
        "top3_symbol_exposure_pct": float(ordered["exposure_pct"].head(3).sum()),
        "unique_symbol_count": int(
            pd.to_numeric(ordered["exposure_pct"], errors="coerce").gt(0).sum()
        ),
    }


def _with_control_metadata(
    frame: pd.DataFrame,
    candidate: pd.Series,
    columns: list[str],
) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return pd.DataFrame(columns=columns)
    out["candidate_label"] = candidate["candidate_label"]
    out["candidate_source_id"] = candidate["candidate_source_id"]
    out["concentration_rule"] = candidate["concentration_rule"]
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    return out.loc[:, columns]


def _trades_from_weights(weights: pd.DataFrame, candidate: pd.Series) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame(columns=["date", "candidate_id", "turnover"])
    rows: list[dict[str, object]] = []
    previous_weights: dict[str, float] = {}
    for date, group in weights.groupby("date", sort=True):
        current_weights = {
            str(row.symbol): float(row.target_weight)
            for row in group.itertuples(index=False)
            if float(row.target_weight) > 0.0
        }
        all_symbols = sorted(set(current_weights).union(previous_weights))
        turnover = (
            sum(
                abs(current_weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0))
                for symbol in all_symbols
            )
            if rows
            else 0.0
        )
        rows.append(
            {
                "date": pd.Timestamp(date),
                "candidate_id": candidate["candidate_id"],
                "turnover": turnover,
            }
        )
        previous_weights = current_weights
    return pd.DataFrame(rows)


def _prediction_rankings(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=["date", "symbol", "prediction", "rank"])
    out = predictions.loc[:, ["date", "symbol", "prediction"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["prediction"] = pd.to_numeric(out["prediction"], errors="coerce")
    out = out.sort_values(["date", "prediction", "symbol"], ascending=[True, False, True])
    out["rank"] = out.groupby("date", sort=False).cumcount() + 1
    return out


def _evaluation_dates(
    raw_df: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
) -> pd.DatetimeIndex:
    source = raw_df.copy()
    source["date"] = pd.to_datetime(source["date"])
    dates = source.loc[
        source["symbol"].astype(str).str.upper().eq("QQQ")
        & source["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date)),
        "date",
    ]
    return pd.DatetimeIndex(pd.to_datetime(sorted(dates.dropna().unique())))


def _concentration_improved(
    *,
    source_top: float,
    source_top3: float,
    source_unique: float,
    top: float,
    top3: float,
    unique: float,
) -> bool:
    top_improved = np.isfinite(source_top) and np.isfinite(top) and top <= source_top - 0.05
    top3_improved = np.isfinite(source_top3) and np.isfinite(top3) and top3 <= source_top3 - 0.10
    unique_improved = (
        np.isfinite(source_unique)
        and np.isfinite(unique)
        and unique >= source_unique + 2
    )
    return bool(top_improved or top3_improved or unique_improved)


def _series_get_float(series: pd.Series, key: float) -> float:
    try:
        return _float(series.loc[key])
    except KeyError:
        return np.nan


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0.0:
        return np.nan
    return numerator / denominator


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes"}


def _validate_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _concat_or_empty(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=columns)
    out = pd.concat(non_empty, ignore_index=True)
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    return out.loc[:, columns]


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    table = frame.copy()
    for column in columns:
        if column not in table.columns:
            table[column] = ""
    table = table.loc[:, columns].fillna("")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| "
        + " | ".join(_markdown_cell(value) for value in row)
        + " |"
        for row in table.astype(object).to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def _markdown_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


__all__ = [
    "CANDIDATES_NAME",
    "CANDIDATE_COLUMNS",
    "COST_SENSITIVITY_COLUMNS",
    "COST_SENSITIVITY_NAME",
    "DEFAULT_CANDIDATE_IDS",
    "DEFAULT_COST_SENSITIVITY_BPS",
    "DEFAULT_OUTPUT_DIR",
    "OUTPUT_CSV_NAMES",
    "PREFLIGHT_MD_NAME",
    "RESULTS_NAME",
    "RESULT_COLUMNS",
    "ROLLING_WINDOWS_NAME",
    "ROLLING_WINDOW_COLUMNS",
    "SELECTED_SYMBOLS_NAME",
    "SELECTED_SYMBOL_COLUMNS",
    "SUBPERIODS_NAME",
    "SUBPERIOD_COLUMNS",
    "SYMBOL_EXPOSURE_COLUMNS",
    "SYMBOL_EXPOSURE_NAME",
    "LongHistoryConcentrationControlArtifacts",
    "build_concentration_control_candidates",
    "build_concentration_control_decision_matrix",
    "build_control_rebalance_weights",
    "build_control_selected_symbols_log",
    "build_long_history_concentration_control_artifacts",
    "build_symbol_exposure_rows",
    "build_weighted_symbol_exposure_summary",
    "classify_concentration_control_decision",
    "output_paths",
    "render_preflight_markdown",
    "replay_concentration_control_candidate",
    "select_concentration_control_source_candidates",
    "write_long_history_concentration_control_artifacts",
]
