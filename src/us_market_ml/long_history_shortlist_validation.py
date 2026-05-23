"""Focused validation helpers for long-history QQQ-relative shortlist candidates.

This module is local research infrastructure. It replays a small candidate
shortlist from a completed long-history model-zoo run, writes diagnostic
tables, and deliberately avoids broad sweeps, MLP batches, model cards,
production signal exports, and cloud access.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from us_market_ml.backtest import SignalBacktestConfig, run_signal_backtest
from us_market_ml.backtest.signal_backtester import calculate_max_drawdown
from us_market_ml.cli.create_long_history_qqq_relative_model_zoo_report import (
    classify_long_history_decision,
)
from us_market_ml.long_history_training import (
    BASE_TRAIN_END,
    END_DATE,
    LONG_HISTORY_FEATURE_SET_VERSION,
    START_DATE,
    LongHistoryRunSpec,
    SkippedLongHistoryRun,
    _eligible_prediction_rows,
    _expand_rebalance_weights,
    _fit_model_for_window,
    _predict_model,
    _rebalance_dates,
    build_long_history_dataset,
    build_long_history_run_key,
    build_rebalance_weights,
    build_training_windows,
    dataset_start_for_required_symbols,
    evaluate_weight_schedule,
)
from us_market_ml.models import predict_baseline
from us_market_ml.qqq_relative_targets import target_horizon_from_variant

OUTPUT_PREFIX = "long_history_qqq_shortlist_validation"
DEFAULT_OUTPUT_DIR = Path("artifacts/validation/long_history_qqq_shortlist_validation")

CANDIDATES_NAME = f"{OUTPUT_PREFIX}_candidates.csv"
SUMMARY_NAME = f"{OUTPUT_PREFIX}_summary.csv"
DAILY_EQUITY_NAME = f"{OUTPUT_PREFIX}_daily_equity.csv"
TRADES_NAME = f"{OUTPUT_PREFIX}_trades.csv"
SELECTED_SYMBOLS_NAME = f"{OUTPUT_PREFIX}_selected_symbols.csv"
SYMBOL_EXPOSURE_NAME = f"{OUTPUT_PREFIX}_symbol_exposure.csv"
SUBPERIODS_NAME = f"{OUTPUT_PREFIX}_subperiods.csv"
ROLLING_WINDOWS_NAME = f"{OUTPUT_PREFIX}_rolling_windows.csv"
COST_SENSITIVITY_NAME = f"{OUTPUT_PREFIX}_cost_sensitivity.csv"
DECISION_MATRIX_NAME = f"{OUTPUT_PREFIX}_decision_matrix.csv"
PREFLIGHT_MD_NAME = f"{OUTPUT_PREFIX}_preflight.md"
REPORT_NAME = f"{OUTPUT_PREFIX}_report.md"

OUTPUT_CSV_NAMES = (
    CANDIDATES_NAME,
    SUMMARY_NAME,
    DAILY_EQUITY_NAME,
    TRADES_NAME,
    SELECTED_SYMBOLS_NAME,
    SYMBOL_EXPOSURE_NAME,
    SUBPERIODS_NAME,
    ROLLING_WINDOWS_NAME,
    COST_SENSITIVITY_NAME,
    DECISION_MATRIX_NAME,
)

DEFAULT_COST_SENSITIVITY_BPS: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0)
DATASET_SOURCE_LABEL = "tiingo_only"
PRICE_MODE = "tiingo_adjusted"
ROLLING_DRAWDOWN_IMPROVEMENT_THRESHOLD = 0.02

CONFIG_COLUMNS = [
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
]

SUMMARY_COLUMNS = [
    "candidate_id",
    "candidate_label",
    "model_type",
    "training_mode",
    "target_variant",
    "feature_set_version",
    "allocation_rule",
    "rebalance_frequency",
    "min_holding_days",
    "switch_margin",
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
    "qqq_buy_hold_max_drawdown",
    "spy_buy_hold_max_drawdown",
    "qqq_buy_hold_return_to_drawdown",
    "spy_buy_hold_return_to_drawdown",
]

DAILY_EQUITY_COLUMNS = [
    "date",
    "candidate_id",
    "candidate_label",
    "strategy_name",
    "active_symbol",
    "active_symbols",
    "daily_return",
    "equity",
    "drawdown",
    "qqq_daily_return",
    "qqq_equity",
    "qqq_drawdown",
    "spy_daily_return",
    "spy_equity",
    "spy_drawdown",
    "cash_daily_return",
    "cash_equity",
    "cash_drawdown",
    "gross_return",
    "transaction_cost",
    "turnover",
]

TRADE_COLUMNS = [
    "date",
    "candidate_id",
    "previous_symbols",
    "new_symbols",
    "selected_symbol",
    "rank_score",
    "turnover",
    "estimated_transaction_cost",
]

SELECTED_SYMBOL_COLUMNS = [
    "candidate_id",
    "candidate_label",
    "symbol",
    "selected_rebalance_count",
    "exposure_days",
    "exposure_pct",
    "total_return_contribution",
]

SYMBOL_EXPOSURE_COLUMNS = [
    "candidate_id",
    "candidate_label",
    "top_symbol",
    "top_symbol_exposure_pct",
    "top3_symbol_exposure_pct",
    "unique_symbol_count",
    "qqq_exposure_pct",
    "spy_exposure_pct",
]

SUBPERIOD_COLUMNS = [
    "candidate_id",
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

ROLLING_SUMMARY_COLUMNS = [
    "candidate_id",
    "rolling_window_count",
    "positive_vs_qqq_window_rate",
    "qqq_underperformance_rate",
    "average_strategy_vs_qqq",
    "median_strategy_vs_qqq",
    "worst_strategy_vs_qqq",
    "drawdown_improvement_window_rate",
    "average_drawdown_improvement_vs_qqq",
    "worst_window_label",
]

COST_SENSITIVITY_COLUMNS = [
    "candidate_id",
    "transaction_cost_bps",
    "total_return",
    "strategy_vs_qqq",
    "max_drawdown",
    "return_to_drawdown",
    "decision_at_cost",
]

DECISION_MATRIX_COLUMNS = [
    "candidate_id",
    "candidate_label",
    *CONFIG_COLUMNS,
    "total_return",
    "qqq_buy_hold_total_return",
    "strategy_vs_qqq",
    "max_drawdown",
    "drawdown_improvement_vs_qqq",
    "return_to_drawdown",
    "beats_qqq_return",
    "beats_qqq_return_to_drawdown",
    "rolling_window_count",
    "positive_vs_qqq_window_rate",
    "qqq_underperformance_rate",
    "average_strategy_vs_qqq",
    "worst_strategy_vs_qqq",
    "drawdown_improvement_window_rate",
    "average_drawdown_improvement_vs_qqq",
    "worst_window_label",
    "top_symbol",
    "top_symbol_exposure_pct",
    "top3_symbol_exposure_pct",
    "unique_symbol_count",
    "qqq_exposure_pct",
    "spy_exposure_pct",
    "cost_20bps_strategy_vs_qqq",
    "model_zoo_decision",
    "selection_reason",
    "decision",
    "rationale",
    "production_ready",
]

SUBPERIODS: tuple[tuple[str, str, str], ...] = (
    ("full", "2019-01-01", "2026-05-15"),
    ("covid_crash_recovery", "2020-01-01", "2020-12-31"),
    ("bear_2022", "2022-01-01", "2022-12-31"),
    ("ai_bull_2023_2024", "2023-01-01", "2024-12-31"),
    ("recent_2025_2026", "2025-01-01", "2026-05-15"),
)

IMPORTANT_CANDIDATES: tuple[dict[str, object], ...] = (
    {
        "selection_reason": "important_candidate_primary",
        "model_type": "elasticnet",
        "training_mode": "recency_weighted_expanding",
        "target_variant": "excess_return_vs_qqq_126d",
        "feature_set_version": LONG_HISTORY_FEATURE_SET_VERSION,
        "allocation_rule": "top1",
        "rebalance_frequency": "quarterly",
        "min_holding_days": 20,
        "switch_margins": (0.0, 0.002),
    },
    {
        "selection_reason": "important_candidate_ridge_frozen_63d",
        "model_type": "ridge",
        "training_mode": "frozen_base_1999_2018",
        "target_variant": "excess_return_vs_qqq_63d",
        "feature_set_version": LONG_HISTORY_FEATURE_SET_VERSION,
        "allocation_rule": "top1",
        "rebalance_frequency": "quarterly",
        "min_holding_days": 0,
        "switch_margins": (0.0, 0.002),
    },
    {
        "selection_reason": "important_candidate_recency_no_hold",
        "model_type": "elasticnet",
        "training_mode": "recency_weighted_expanding",
        "target_variant": "excess_return_vs_qqq_126d",
        "feature_set_version": LONG_HISTORY_FEATURE_SET_VERSION,
        "allocation_rule": "top1",
        "rebalance_frequency": "quarterly",
        "min_holding_days": 0,
        "switch_margins": (0.0, 0.002),
    },
    {
        "selection_reason": "important_candidate_elasticnet_frozen_63d",
        "model_type": "elasticnet",
        "training_mode": "frozen_base_1999_2018",
        "target_variant": "excess_return_vs_qqq_63d",
        "feature_set_version": LONG_HISTORY_FEATURE_SET_VERSION,
        "allocation_rule": "top1",
        "rebalance_frequency": "quarterly",
        "min_holding_days": 20,
        "switch_margins": (0.0, 0.002),
    },
    {
        "selection_reason": "important_candidate_anchored_monthly",
        "model_type": "elasticnet",
        "training_mode": "anchored_expanding_yearly",
        "target_variant": "excess_return_vs_qqq_20d",
        "feature_set_version": LONG_HISTORY_FEATURE_SET_VERSION,
        "allocation_rule": "top1",
        "rebalance_frequency": "monthly",
        "min_holding_days": 0,
        "switch_margins": (0.002,),
    },
    {
        "selection_reason": "important_candidate_anchored_quarterly",
        "model_type": "elasticnet",
        "training_mode": "anchored_expanding_yearly",
        "target_variant": "excess_return_vs_qqq_20d",
        "feature_set_version": LONG_HISTORY_FEATURE_SET_VERSION,
        "allocation_rule": "top1",
        "rebalance_frequency": "quarterly",
        "min_holding_days": 0,
        "switch_margins": (0.002,),
    },
    {
        "selection_reason": "important_candidate_drawdown_control",
        "model_type": "ridge",
        "training_mode": "base_recent_blend",
        "target_variant": "return_drawdown_score_vs_qqq_63d",
        "feature_set_version": LONG_HISTORY_FEATURE_SET_VERSION,
        "allocation_rule": "top1",
        "rebalance_frequency": "quarterly",
        "min_holding_days": 20,
    },
)


@dataclass(frozen=True)
class CandidateReplay:
    """Replay artifacts for one selected candidate."""

    candidate: pd.Series
    spec: LongHistoryRunSpec
    predictions: pd.DataFrame
    rebalance_weights: pd.DataFrame
    metrics: Any
    daily_equity: pd.DataFrame
    position_returns: pd.DataFrame
    trades: pd.DataFrame
    selected_symbols: pd.DataFrame
    symbol_exposure: pd.DataFrame
    subperiods: pd.DataFrame
    rolling_windows: pd.DataFrame
    cost_sensitivity: pd.DataFrame


@dataclass(frozen=True)
class LongHistoryShortlistValidationArtifacts:
    """CSV-ready focused validation outputs."""

    candidates: pd.DataFrame
    summary: pd.DataFrame
    daily_equity: pd.DataFrame
    trades: pd.DataFrame
    selected_symbols: pd.DataFrame
    symbol_exposure: pd.DataFrame
    subperiods: pd.DataFrame
    rolling_windows: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    decision_matrix: pd.DataFrame
    warnings: tuple[str, ...] = ()


def select_shortlist_candidates(
    *,
    results: pd.DataFrame,
    decision_matrix: pd.DataFrame,
    top_n: int = 10,
    candidate_ids: tuple[str, ...] = (),
    include_decisions: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Select, deduplicate, and label a focused shortlist from model-zoo outputs."""

    if top_n <= 0:
        raise ValueError("top_n must be positive.")
    _validate_candidate_inputs(results, decision_matrix)
    ranked = _rank_results_like_model_zoo_report(results, decision_matrix)
    ranked = ranked.loc[~ranked["model_type"].astype(str).str.lower().eq("mlp")].copy()
    ranked = ranked.loc[
        ~ranked["training_mode"].astype(str).str.lower().eq("mlp_pretrain_finetune")
    ].copy()

    unknown_ids = sorted(set(candidate_ids).difference(set(ranked["candidate_id"])))
    if unknown_ids:
        raise ValueError(f"Unknown candidate_id values: {', '.join(unknown_ids)}")

    reason_by_id: dict[str, list[str]] = {}
    for candidate_id in candidate_ids:
        reason_by_id.setdefault(candidate_id, []).append("explicit_candidate_id")

    for important in IMPORTANT_CANDIDATES:
        match = _first_important_match(ranked, important)
        if match is not None:
            reason_by_id.setdefault(match, []).append(str(important["selection_reason"]))

    pool = ranked
    if include_decisions:
        include_set = set(include_decisions)
        pool = ranked.loc[ranked["decision"].astype(str).isin(include_set)].copy()

    seen_top_keys: set[tuple[object, ...]] = set()
    for row in pool.itertuples(index=False):
        dedup_key = _candidate_dedup_key(pd.Series(row._asdict()))
        if dedup_key in seen_top_keys:
            continue
        seen_top_keys.add(dedup_key)
        candidate_id = str(row.candidate_id)
        reason_by_id.setdefault(candidate_id, []).append("top_n_deduplicated")
        if sum("top_n_deduplicated" in reasons for reasons in reason_by_id.values()) >= top_n:
            break

    selected_rows = []
    seen_selected_keys: set[tuple[object, ...]] = set()
    for _, row in ranked.iterrows():
        candidate_id = str(row["candidate_id"])
        if candidate_id not in reason_by_id:
            continue
        dedup_key = _candidate_dedup_key(row)
        if dedup_key in seen_selected_keys:
            continue
        seen_selected_keys.add(dedup_key)
        out = row.copy()
        out["selection_reason"] = ";".join(dict.fromkeys(reason_by_id[candidate_id]))
        selected_rows.append(out)

    if not selected_rows:
        raise ValueError("Candidate selection produced no rows.")

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    for column in ["candidate_id", "candidate_label", *CONFIG_COLUMNS, "selection_reason"]:
        if column not in selected.columns:
            selected[column] = pd.NA
    return selected


def build_long_history_shortlist_validation_artifacts(
    *,
    raw_df: pd.DataFrame,
    candidates: pd.DataFrame,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    min_history_rows: int = 252,
    min_train_eligible_rows: int = 1000,
    cost_sensitivity_bps: tuple[float, ...] = DEFAULT_COST_SENSITIVITY_BPS,
) -> LongHistoryShortlistValidationArtifacts:
    """Replay selected candidates and build all validation artifact tables."""

    if candidates.empty:
        raise ValueError("At least one selected candidate is required.")
    replays: list[CandidateReplay] = []
    for _, candidate in candidates.iterrows():
        if str(candidate.get("model_type", "")).lower() == "mlp":
            raise ValueError("MLP candidates are intentionally excluded from shortlist validation.")
        replay = replay_shortlist_candidate(
            raw_df=raw_df,
            candidate=candidate,
            start_date=start_date,
            end_date=end_date,
            min_history_rows=min_history_rows,
            min_train_eligible_rows=min_train_eligible_rows,
            cost_sensitivity_bps=cost_sensitivity_bps,
        )
        replays.append(replay)

    summary = _concat_or_empty(
        [_summary_row_from_replay(replay) for replay in replays],
        SUMMARY_COLUMNS,
    )
    daily_equity = _concat_or_empty(
        [replay.daily_equity for replay in replays],
        DAILY_EQUITY_COLUMNS,
    )
    trades = _concat_or_empty([replay.trades for replay in replays], TRADE_COLUMNS)
    selected_symbols = _concat_or_empty(
        [replay.selected_symbols for replay in replays],
        SELECTED_SYMBOL_COLUMNS,
    )
    symbol_exposure = _concat_or_empty(
        [replay.symbol_exposure for replay in replays],
        SYMBOL_EXPOSURE_COLUMNS,
    )
    subperiods = _concat_or_empty(
        [replay.subperiods for replay in replays],
        SUBPERIOD_COLUMNS,
    )
    rolling_windows = _concat_or_empty(
        [replay.rolling_windows for replay in replays],
        ROLLING_WINDOW_COLUMNS,
    )
    cost_sensitivity = _concat_or_empty(
        [replay.cost_sensitivity for replay in replays],
        COST_SENSITIVITY_COLUMNS,
    )
    rolling_summary = summarize_rolling_windows(rolling_windows)
    decision_matrix = build_validation_decision_matrix(
        candidates=candidates,
        summary=summary,
        symbol_exposure=symbol_exposure,
        rolling_summary=rolling_summary,
        cost_sensitivity=cost_sensitivity,
    )

    return LongHistoryShortlistValidationArtifacts(
        candidates=candidates,
        summary=summary,
        daily_equity=daily_equity,
        trades=trades,
        selected_symbols=selected_symbols,
        symbol_exposure=symbol_exposure,
        subperiods=subperiods,
        rolling_windows=rolling_windows,
        cost_sensitivity=cost_sensitivity,
        decision_matrix=decision_matrix,
    )


def replay_shortlist_candidate(
    *,
    raw_df: pd.DataFrame,
    candidate: pd.Series,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    min_history_rows: int = 252,
    min_train_eligible_rows: int = 1000,
    cost_sensitivity_bps: tuple[float, ...] = DEFAULT_COST_SENSITIVITY_BPS,
) -> CandidateReplay:
    """Replay one candidate and keep diagnostics that the model zoo discarded."""

    spec = spec_from_candidate(candidate)
    predictions, weights, prepared_raw = generate_predictions_and_weights(
        raw_df=raw_df,
        spec=spec,
        start_date=start_date,
        end_date=end_date,
        min_history_rows=min_history_rows,
        min_train_eligible_rows=min_train_eligible_rows,
    )
    metrics = evaluate_weight_schedule(
        prepared_raw,
        weights,
        start_date=start_date,
        end_date=end_date,
        transaction_cost_bps=spec.transaction_cost_bps,
    )
    daily_equity, position_returns = build_candidate_daily_equity(
        raw_df=prepared_raw,
        rebalance_weights=weights,
        candidate=candidate,
        start_date=start_date,
        end_date=end_date,
        transaction_cost_bps=spec.transaction_cost_bps,
    )
    trades = build_candidate_trades(
        rebalance_weights=weights,
        predictions=predictions,
        candidate=candidate,
        transaction_cost_bps=spec.transaction_cost_bps,
    )
    selected_symbols = build_selected_symbol_summary(
        candidate=candidate,
        rebalance_weights=weights,
        daily_equity=daily_equity,
        position_returns=position_returns,
    )
    symbol_exposure = build_symbol_exposure_summary(
        candidate=candidate,
        selected_symbols=selected_symbols,
    )
    subperiods = build_subperiod_robustness(
        candidate=candidate,
        daily_equity=daily_equity,
        trades=trades,
    )
    rolling_windows = build_rolling_window_robustness(
        candidate=candidate,
        daily_equity=daily_equity,
    )
    cost_sensitivity = build_cost_sensitivity(
        raw_df=prepared_raw,
        rebalance_weights=weights,
        candidate=candidate,
        start_date=start_date,
        end_date=end_date,
        cost_sensitivity_bps=cost_sensitivity_bps,
    )
    return CandidateReplay(
        candidate=candidate,
        spec=spec,
        predictions=predictions,
        rebalance_weights=weights,
        metrics=metrics,
        daily_equity=daily_equity,
        position_returns=position_returns,
        trades=trades,
        selected_symbols=selected_symbols,
        symbol_exposure=symbol_exposure,
        subperiods=subperiods,
        rolling_windows=rolling_windows,
        cost_sensitivity=cost_sensitivity,
    )


def spec_from_candidate(candidate: pd.Series) -> LongHistoryRunSpec:
    """Build a runnable long-history spec from a selected candidate row."""

    payload = {
        "model_type": str(candidate["model_type"]),
        "training_mode": str(candidate["training_mode"]),
        "target_variant": str(candidate["target_variant"]),
        "feature_set_version": str(candidate["feature_set_version"]),
        "allocation_rule": str(candidate["allocation_rule"]),
        "rebalance_frequency": str(candidate["rebalance_frequency"]),
        "min_holding_days": int(candidate["min_holding_days"]),
        "switch_margin": float(candidate["switch_margin"]),
        "transaction_cost_bps": float(candidate.get("transaction_cost_bps", 10.0)),
        "recency_decay_profile": _none_if_missing(candidate.get("recency_decay_profile")),
        "base_weight": _optional_float(candidate.get("base_weight")),
        "recent_window_years": _optional_int(candidate.get("recent_window_years")),
        "finetune_window_years": _optional_int(candidate.get("finetune_window_years")),
        "finetune_lr_scale": _none_if_missing(candidate.get("finetune_lr_scale")),
        "random_state": int(candidate.get("random_state", 42)),
    }
    run_key = str(candidate.get("run_key") or build_long_history_run_key(payload))
    return LongHistoryRunSpec(run_key=run_key, **payload)


def generate_predictions_and_weights(
    *,
    raw_df: pd.DataFrame,
    spec: LongHistoryRunSpec,
    start_date: str,
    end_date: str,
    min_history_rows: int,
    min_train_eligible_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit the requested spec and return replay predictions plus rebalance weights."""

    if spec.training_mode == "mlp_pretrain_finetune" or spec.model_type == "mlp":
        raise SkippedLongHistoryRun("MLP candidates are excluded from shortlist validation.")

    raw = raw_df.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    dataset_start = dataset_start_for_required_symbols(raw)
    raw = raw.loc[
        raw["date"].between(dataset_start, pd.Timestamp(end_date), inclusive="both")
    ].copy()
    dataset = build_long_history_dataset(raw, feature_set_version=spec.feature_set_version)
    feature_cols = _feature_columns(dataset, spec.feature_set_version)
    if not feature_cols:
        raise SkippedLongHistoryRun("No usable feature columns were available.")
    if spec.target_variant not in dataset.columns:
        raise SkippedLongHistoryRun(f"Target variant is missing: {spec.target_variant}")

    windows = build_training_windows(
        spec.training_mode,
        dataset_start=dataset_start,
        start_date=start_date,
        end_date=end_date,
    )
    rebalance_dates = _rebalance_dates(
        raw,
        start_date=start_date,
        end_date=end_date,
        frequency=spec.rebalance_frequency,
    )
    if not rebalance_dates:
        raise SkippedLongHistoryRun("No rebalance dates were available.")

    prediction_frames: list[pd.DataFrame] = []
    base_model: Any | None = None
    base_features: list[str] | None = None
    if spec.training_mode == "base_recent_blend":
        base_model, base_features = _fit_model_for_window(
            dataset=dataset,
            feature_cols=feature_cols,
            target_col=spec.target_variant,
            model_type=spec.model_type,
            train_start=dataset_start,
            train_end=pd.Timestamp(BASE_TRAIN_END),
            target_horizon=target_horizon_from_variant(spec.target_variant),
            min_history_rows=min_history_rows,
            min_train_eligible_rows=min_train_eligible_rows,
            random_state=spec.random_state,
            sample_weights=None,
        )

    for window in windows:
        window_rebalance_dates = [
            date for date in rebalance_dates if window.predict_start <= date <= window.predict_end
        ]
        if not window_rebalance_dates:
            continue

        if spec.training_mode == "base_recent_blend":
            if base_model is None or base_features is None:
                raise SkippedLongHistoryRun("Base model was not fitted.")
            recent_start = window.train_end - pd.DateOffset(
                years=spec.recent_window_years or 3
            ) + pd.Timedelta(days=1)
            recent_model, recent_features = _fit_model_for_window(
                dataset=dataset,
                feature_cols=feature_cols,
                target_col=spec.target_variant,
                model_type=spec.model_type,
                train_start=recent_start,
                train_end=window.train_end,
                target_horizon=target_horizon_from_variant(spec.target_variant),
                min_history_rows=min_history_rows,
                min_train_eligible_rows=min_train_eligible_rows,
                random_state=spec.random_state,
                sample_weights=None,
            )
            prediction_features = sorted(set(base_features).union(recent_features))
            scored = _eligible_prediction_rows(
                dataset=dataset,
                feature_cols=prediction_features,
                target_col=spec.target_variant,
                predict_dates=window_rebalance_dates,
                min_history_rows=min_history_rows,
            )
            if scored.empty:
                continue
            base_predictions = predict_baseline(
                base_model,
                scored,
                base_features,
                prediction_col="base_prediction",
            )
            recent_predictions = predict_baseline(
                recent_model,
                base_predictions,
                recent_features,
                prediction_col="recent_prediction",
            )
            base_weight = spec.base_weight if spec.base_weight is not None else 0.5
            recent_predictions["prediction"] = (
                base_weight * recent_predictions["base_prediction"]
                + (1.0 - base_weight) * recent_predictions["recent_prediction"]
            )
            prediction_frames.append(recent_predictions)
            continue

        sample_weight_col = None
        if spec.training_mode == "recency_weighted_expanding":
            sample_weight_col = "__long_history_sample_weight"

        model, selected_features = _fit_model_for_window(
            dataset=dataset,
            feature_cols=feature_cols,
            target_col=spec.target_variant,
            model_type=spec.model_type,
            train_start=window.train_start,
            train_end=window.train_end,
            target_horizon=target_horizon_from_variant(spec.target_variant),
            min_history_rows=min_history_rows,
            min_train_eligible_rows=min_train_eligible_rows,
            random_state=spec.random_state,
            sample_weights=sample_weight_col,
            recency_decay_profile=spec.recency_decay_profile,
        )
        scored = _eligible_prediction_rows(
            dataset=dataset,
            feature_cols=selected_features,
            target_col=spec.target_variant,
            predict_dates=window_rebalance_dates,
            min_history_rows=min_history_rows,
        )
        if scored.empty:
            continue
        prediction_frames.append(_predict_model(model, scored, selected_features, spec.model_type))

    if not prediction_frames:
        raise SkippedLongHistoryRun("No prediction rows were generated.")

    predictions = pd.concat(prediction_frames, ignore_index=True)
    weights = build_rebalance_weights(
        predictions,
        allocation_rule=spec.allocation_rule,
        min_holding_days=spec.min_holding_days,
        switch_margin=spec.switch_margin,
    )
    return predictions, weights, raw


def build_candidate_daily_equity(
    *,
    raw_df: pd.DataFrame,
    rebalance_weights: pd.DataFrame,
    candidate: pd.Series,
    start_date: str,
    end_date: str,
    transaction_cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build daily candidate equity with QQQ, SPY, and cash references."""

    signals = _expand_rebalance_weights(raw_df, rebalance_weights, start_date, end_date)
    backtest = run_signal_backtest(
        signals,
        SignalBacktestConfig(transaction_cost_bps=transaction_cost_bps),
    )
    equity = backtest.equity_curve.copy()
    equity["date"] = pd.to_datetime(equity["date"])
    active = _active_symbols_by_date(signals)
    equity = equity.merge(active, on="date", how="left")
    qqq = _benchmark_daily(raw_df, "QQQ", equity["date"])
    spy = _benchmark_daily(raw_df, "SPY", equity["date"])
    out = (
        equity.merge(qqq.add_prefix("qqq_").rename(columns={"qqq_date": "date"}), on="date")
        .merge(spy.add_prefix("spy_").rename(columns={"spy_date": "date"}), on="date")
        .copy()
    )
    out["candidate_id"] = candidate["candidate_id"]
    out["candidate_label"] = candidate["candidate_label"]
    out["strategy_name"] = candidate["candidate_label"]
    out["drawdown"] = out["equity"] / out["equity"].cummax() - 1.0
    out["cash_daily_return"] = 0.0
    out["cash_equity"] = 1.0
    out["cash_drawdown"] = 0.0
    out = out.rename(
        columns={
            "gross_portfolio_return": "gross_return",
            "portfolio_return": "daily_return",
        }
    )
    for column in DAILY_EQUITY_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out.loc[:, DAILY_EQUITY_COLUMNS], backtest.position_returns


def build_candidate_trades(
    *,
    rebalance_weights: pd.DataFrame,
    predictions: pd.DataFrame,
    candidate: pd.Series,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    """Build rebalance/change rows with selected symbols and available rank scores."""

    if rebalance_weights.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    weights = rebalance_weights.copy()
    weights["date"] = pd.to_datetime(weights["date"])
    scores = predictions.loc[:, ["date", "symbol", "prediction"]].copy()
    scores["date"] = pd.to_datetime(scores["date"])
    rows: list[dict[str, object]] = []
    previous_weights: dict[str, float] = {}
    for date, group in weights.groupby("date", sort=True):
        current_weights = {
            str(row.symbol): float(row.target_weight)
            for row in group.itertuples(index=False)
            if float(row.target_weight) > 0.0
        }
        current_symbols = sorted(current_weights)
        previous_symbols = sorted(
            symbol for symbol, weight in previous_weights.items() if weight > 0
        )
        turnover = 0.0
        if rows:
            all_symbols = sorted(set(current_weights).union(previous_weights))
            turnover = sum(
                abs(current_weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0))
                for symbol in all_symbols
            )
        selected_symbol = (
            current_symbols[0]
            if len(current_symbols) == 1
            else ";".join(current_symbols)
        )
        selected_scores = scores.loc[
            scores["date"].eq(date) & scores["symbol"].astype(str).isin(current_symbols),
            "prediction",
        ]
        rank_score = (
            float(pd.to_numeric(selected_scores, errors="coerce").max())
            if not selected_scores.empty
            else np.nan
        )
        rows.append(
            {
                "date": date,
                "candidate_id": candidate["candidate_id"],
                "previous_symbols": ";".join(previous_symbols),
                "new_symbols": ";".join(current_symbols),
                "selected_symbol": selected_symbol,
                "rank_score": rank_score,
                "turnover": turnover,
                "estimated_transaction_cost": turnover * transaction_cost_bps / 10000.0,
            }
        )
        previous_weights = current_weights
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


def build_selected_symbol_summary(
    *,
    candidate: pd.Series,
    rebalance_weights: pd.DataFrame,
    daily_equity: pd.DataFrame,
    position_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Build per-symbol selection counts, exposure days, and contribution proxy."""

    selected_counts = (
        rebalance_weights.loc[rebalance_weights["target_weight"].gt(0.0)]
        .groupby("symbol", dropna=False)
        .size()
        .rename("selected_rebalance_count")
    )
    exposure_counts = _exposure_counts(daily_equity)
    contributions = (
        position_returns.groupby("symbol", dropna=False)["position_return"].sum()
        if "position_return" in position_returns.columns
        else pd.Series(dtype="float64")
    )
    symbols = sorted(
        set(selected_counts.index.astype(str))
        .union(exposure_counts.index.astype(str))
        .union(contributions.index.astype(str))
    )
    total_days = max(int(len(daily_equity)), 1)
    rows = []
    for symbol in symbols:
        exposure_days = int(exposure_counts.get(symbol, 0))
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "candidate_label": candidate["candidate_label"],
                "symbol": symbol,
                "selected_rebalance_count": int(selected_counts.get(symbol, 0)),
                "exposure_days": exposure_days,
                "exposure_pct": exposure_days / total_days,
                "total_return_contribution": float(contributions.get(symbol, 0.0)),
            }
        )
    return pd.DataFrame(rows, columns=SELECTED_SYMBOL_COLUMNS)


def build_symbol_exposure_summary(
    *,
    candidate: pd.Series,
    selected_symbols: pd.DataFrame,
) -> pd.DataFrame:
    """Build candidate-level concentration statistics."""

    if selected_symbols.empty:
        row = {
            "candidate_id": candidate["candidate_id"],
            "candidate_label": candidate["candidate_label"],
            "top_symbol": "",
            "top_symbol_exposure_pct": 0.0,
            "top3_symbol_exposure_pct": 0.0,
            "unique_symbol_count": 0,
            "qqq_exposure_pct": 0.0,
            "spy_exposure_pct": 0.0,
        }
        return pd.DataFrame([row], columns=SYMBOL_EXPOSURE_COLUMNS)
    ordered = selected_symbols.sort_values(
        ["exposure_pct", "selected_rebalance_count", "symbol"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    row = {
        "candidate_id": candidate["candidate_id"],
        "candidate_label": candidate["candidate_label"],
        "top_symbol": ordered["symbol"].iloc[0],
        "top_symbol_exposure_pct": float(ordered["exposure_pct"].iloc[0]),
        "top3_symbol_exposure_pct": float(ordered["exposure_pct"].head(3).sum()),
        "unique_symbol_count": int((selected_symbols["exposure_days"] > 0).sum()),
        "qqq_exposure_pct": float(
            selected_symbols.loc[
                selected_symbols["symbol"].astype(str).str.upper().eq("QQQ"),
                "exposure_pct",
            ].sum()
        ),
        "spy_exposure_pct": float(
            selected_symbols.loc[
                selected_symbols["symbol"].astype(str).str.upper().eq("SPY"),
                "exposure_pct",
            ].sum()
        ),
    }
    return pd.DataFrame([row], columns=SYMBOL_EXPOSURE_COLUMNS)


def build_subperiod_robustness(
    *,
    candidate: pd.Series,
    daily_equity: pd.DataFrame,
    trades: pd.DataFrame,
    subperiods: tuple[tuple[str, str, str], ...] = SUBPERIODS,
) -> pd.DataFrame:
    """Build named subperiod robustness rows."""

    rows: list[dict[str, object]] = []
    for period_name, period_start, period_end in subperiods:
        period = _slice_daily(daily_equity, period_start, period_end)
        if period.empty:
            continue
        strategy_return = _compound_return(period["daily_return"])
        qqq_return = _compound_return(period["qqq_daily_return"])
        spy_return = _compound_return(period["spy_daily_return"])
        max_drawdown = _max_drawdown_from_returns(period["daily_return"])
        qqq_max_drawdown = _max_drawdown_from_returns(period["qqq_daily_return"])
        trade_count = int(
            trades.loc[
                pd.to_datetime(trades["date"]).between(
                    pd.Timestamp(period_start),
                    pd.Timestamp(period_end),
                )
                & pd.to_numeric(trades["turnover"], errors="coerce").gt(0.0)
            ].shape[0]
        )
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "period_name": period_name,
                "period_start": period_start,
                "period_end": period_end,
                "total_return": strategy_return,
                "qqq_total_return": qqq_return,
                "spy_total_return": spy_return,
                "strategy_vs_qqq": strategy_return - qqq_return,
                "max_drawdown": max_drawdown,
                "qqq_max_drawdown": qqq_max_drawdown,
                "drawdown_improvement_vs_qqq": max_drawdown - qqq_max_drawdown,
                "transaction_count": trade_count,
                "top_symbol": _top_symbol_for_period(period),
            }
        )
    return pd.DataFrame(rows, columns=SUBPERIOD_COLUMNS)


def build_rolling_window_robustness(
    *,
    candidate: pd.Series,
    daily_equity: pd.DataFrame,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    """Build 1y/2y/3y rolling-window robustness rows."""

    rows: list[dict[str, object]] = []
    for years, step_months in ((1, 3), (2, 3), (3, 6)):
        for window_start, window_end in generate_rolling_windows(
            start_date=start_date,
            end_date=end_date,
            years=years,
            step_months=step_months,
        ):
            period = _slice_daily(daily_equity, str(window_start.date()), str(window_end.date()))
            if period.empty:
                continue
            strategy_return = _compound_return(period["daily_return"])
            qqq_return = _compound_return(period["qqq_daily_return"])
            strategy_drawdown = _max_drawdown_from_returns(period["daily_return"])
            qqq_drawdown = _max_drawdown_from_returns(period["qqq_daily_return"])
            strategy_vs_qqq = strategy_return - qqq_return
            drawdown_improvement = strategy_drawdown - qqq_drawdown
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "window_label": (
                        f"{years}y_{window_start:%Y%m%d}_{window_end:%Y%m%d}"
                    ),
                    "window_start": str(window_start.date()),
                    "window_end": str(window_end.date()),
                    "strategy_total_return": strategy_return,
                    "qqq_total_return": qqq_return,
                    "strategy_vs_qqq": strategy_vs_qqq,
                    "strategy_max_drawdown": strategy_drawdown,
                    "qqq_max_drawdown": qqq_drawdown,
                    "drawdown_improvement_vs_qqq": drawdown_improvement,
                    "underperformed_qqq": bool(strategy_vs_qqq < 0.0),
                    "materially_improved_drawdown": bool(
                        drawdown_improvement > ROLLING_DRAWDOWN_IMPROVEMENT_THRESHOLD
                    ),
                }
            )
    return pd.DataFrame(rows, columns=ROLLING_WINDOW_COLUMNS)


def summarize_rolling_windows(rolling_windows: pd.DataFrame) -> pd.DataFrame:
    """Summarize rolling-window behavior by candidate."""

    if rolling_windows.empty:
        return pd.DataFrame(columns=ROLLING_SUMMARY_COLUMNS)
    rows: list[dict[str, object]] = []
    for candidate_id, group in rolling_windows.groupby("candidate_id", sort=False):
        values = pd.to_numeric(group["strategy_vs_qqq"], errors="coerce")
        drawdown = pd.to_numeric(group["drawdown_improvement_vs_qqq"], errors="coerce")
        worst_index = values.idxmin() if values.notna().any() else None
        rows.append(
            {
                "candidate_id": candidate_id,
                "rolling_window_count": int(len(group)),
                "positive_vs_qqq_window_rate": float(values.gt(0.0).mean()),
                "qqq_underperformance_rate": float(values.lt(0.0).mean()),
                "average_strategy_vs_qqq": float(values.mean()),
                "median_strategy_vs_qqq": float(values.median()),
                "worst_strategy_vs_qqq": float(values.min()),
                "drawdown_improvement_window_rate": float(drawdown.gt(0.0).mean()),
                "average_drawdown_improvement_vs_qqq": float(drawdown.mean()),
                "worst_window_label": (
                    group.loc[worst_index, "window_label"] if worst_index is not None else ""
                ),
            }
        )
    return pd.DataFrame(rows, columns=ROLLING_SUMMARY_COLUMNS)


def build_cost_sensitivity(
    *,
    raw_df: pd.DataFrame,
    rebalance_weights: pd.DataFrame,
    candidate: pd.Series,
    start_date: str,
    end_date: str,
    cost_sensitivity_bps: tuple[float, ...],
) -> pd.DataFrame:
    """Re-evaluate the same weight schedule under selected transaction costs."""

    rows: list[dict[str, object]] = []
    for cost_bps in cost_sensitivity_bps:
        metrics = evaluate_weight_schedule(
            raw_df,
            rebalance_weights,
            start_date=start_date,
            end_date=end_date,
            transaction_cost_bps=float(cost_bps),
        )
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "transaction_cost_bps": float(cost_bps),
                "total_return": metrics.total_return,
                "strategy_vs_qqq": metrics.strategy_vs_qqq,
                "max_drawdown": metrics.max_drawdown,
                "return_to_drawdown": metrics.return_to_drawdown,
                "decision_at_cost": _decision_at_cost(metrics),
            }
        )
    return pd.DataFrame(rows, columns=COST_SENSITIVITY_COLUMNS)


def build_validation_decision_matrix(
    *,
    candidates: pd.DataFrame,
    summary: pd.DataFrame,
    symbol_exposure: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    """Build the final candidate-level validation decision matrix."""

    base = candidates.copy()
    base = base.rename(columns={"decision": "model_zoo_decision"})
    keep_cols = [
        "candidate_id",
        "candidate_label",
        *CONFIG_COLUMNS,
        "model_zoo_decision",
        "selection_reason",
    ]
    for column in keep_cols:
        if column not in base.columns:
            base[column] = pd.NA
    summary_cols = [
        column
        for column in summary.columns
        if column in {"candidate_id", "candidate_label"} or column not in CONFIG_COLUMNS
    ]
    matrix = base.loc[:, keep_cols].merge(
        summary.loc[:, summary_cols],
        on=["candidate_id", "candidate_label"],
        how="left",
    )
    matrix = matrix.merge(rolling_summary, on="candidate_id", how="left")
    matrix = matrix.merge(symbol_exposure, on=["candidate_id", "candidate_label"], how="left")
    cost20 = (
        cost_sensitivity.loc[
            pd.to_numeric(cost_sensitivity["transaction_cost_bps"], errors="coerce").eq(20.0),
            ["candidate_id", "strategy_vs_qqq"],
        ]
        .rename(columns={"strategy_vs_qqq": "cost_20bps_strategy_vs_qqq"})
    )
    matrix = matrix.merge(cost20, on="candidate_id", how="left")
    decisions = matrix.apply(_validation_decision_and_rationale, axis=1)
    matrix["decision"] = [item[0] for item in decisions]
    matrix["rationale"] = [item[1] for item in decisions]
    matrix["production_ready"] = False
    for column in DECISION_MATRIX_COLUMNS:
        if column not in matrix.columns:
            matrix[column] = pd.NA
    return matrix.loc[:, DECISION_MATRIX_COLUMNS]


def generate_rolling_windows(
    *,
    start_date: str,
    end_date: str,
    years: int,
    step_months: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Generate calendar rolling windows with a fixed month step."""

    start = pd.Timestamp(start_date)
    final_end = pd.Timestamp(end_date)
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    window_start = start
    while True:
        window_end = window_start + pd.DateOffset(years=years) - pd.Timedelta(days=1)
        if window_end > final_end:
            break
        windows.append((window_start, window_end))
        window_start = window_start + pd.DateOffset(months=step_months)
    return windows


def write_long_history_shortlist_validation_artifacts(
    *,
    artifacts: LongHistoryShortlistValidationArtifacts,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write all validation CSV artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = output_paths(output_dir)
    artifacts.candidates.to_csv(outputs[CANDIDATES_NAME], index=False)
    artifacts.summary.to_csv(outputs[SUMMARY_NAME], index=False)
    artifacts.daily_equity.to_csv(outputs[DAILY_EQUITY_NAME], index=False)
    artifacts.trades.to_csv(outputs[TRADES_NAME], index=False)
    artifacts.selected_symbols.to_csv(outputs[SELECTED_SYMBOLS_NAME], index=False)
    artifacts.symbol_exposure.to_csv(outputs[SYMBOL_EXPOSURE_NAME], index=False)
    artifacts.subperiods.to_csv(outputs[SUBPERIODS_NAME], index=False)
    artifacts.rolling_windows.to_csv(outputs[ROLLING_WINDOWS_NAME], index=False)
    artifacts.cost_sensitivity.to_csv(outputs[COST_SENSITIVITY_NAME], index=False)
    artifacts.decision_matrix.to_csv(outputs[DECISION_MATRIX_NAME], index=False)
    return outputs


def output_paths(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    """Return all expected output paths for a validation directory."""

    return {
        CANDIDATES_NAME: output_dir / CANDIDATES_NAME,
        SUMMARY_NAME: output_dir / SUMMARY_NAME,
        DAILY_EQUITY_NAME: output_dir / DAILY_EQUITY_NAME,
        TRADES_NAME: output_dir / TRADES_NAME,
        SELECTED_SYMBOLS_NAME: output_dir / SELECTED_SYMBOLS_NAME,
        SYMBOL_EXPOSURE_NAME: output_dir / SYMBOL_EXPOSURE_NAME,
        SUBPERIODS_NAME: output_dir / SUBPERIODS_NAME,
        ROLLING_WINDOWS_NAME: output_dir / ROLLING_WINDOWS_NAME,
        COST_SENSITIVITY_NAME: output_dir / COST_SENSITIVITY_NAME,
        DECISION_MATRIX_NAME: output_dir / DECISION_MATRIX_NAME,
        PREFLIGHT_MD_NAME: output_dir / PREFLIGHT_MD_NAME,
        REPORT_NAME: output_dir / REPORT_NAME,
    }


def render_preflight_markdown(
    *,
    candidates: pd.DataFrame,
    market_dataset_glob: str,
    model_zoo_results: Path,
    model_zoo_decision_matrix: Path,
    output_dir: Path,
    start_date: str,
    end_date: str,
    top_n: int,
    include_decisions: tuple[str, ...],
    candidate_ids: tuple[str, ...],
    dry_run: bool,
) -> str:
    """Render research-only preflight markdown for the shortlist validation."""

    lines = [
        "# Long-History QQQ Shortlist Validation Preflight",
        "",
        "Focused validation/deep-dive bundle for a deduplicated model-zoo shortlist.",
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
        "- Secondary references: `SPY buy-and-hold`, `cash`",
        f"- Dataset source: `{DATASET_SOURCE_LABEL}`",
        f"- Price mode: `{PRICE_MODE}`",
        "",
        "## Inputs",
        "",
        f"- Model-zoo results: `{model_zoo_results}`",
        f"- Model-zoo decision matrix: `{model_zoo_decision_matrix}`",
        f"- Market dataset glob: `{market_dataset_glob}`",
        f"- Output directory: `{output_dir}`",
        f"- Start date: `{start_date}`",
        f"- End date: `{end_date}`",
        f"- Top N: `{top_n}`",
        f"- Include decisions: `{list(include_decisions) if include_decisions else 'all'}`",
        f"- Explicit candidate IDs: `{list(candidate_ids) if candidate_ids else 'none'}`",
        f"- Dry run: `{dry_run}`",
        "",
        "## Selected Candidates",
        "",
        _markdown_table(
            candidates,
            [
                "candidate_id",
                "candidate_label",
                "switch_margin",
                "total_return",
                "strategy_vs_qqq",
                "return_to_drawdown",
                "decision",
                "selection_reason",
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def _summary_row_from_replay(replay: CandidateReplay) -> pd.DataFrame:
    metrics = replay.metrics
    candidate = replay.candidate
    row = {
        "candidate_id": candidate["candidate_id"],
        "candidate_label": candidate["candidate_label"],
        "model_type": candidate["model_type"],
        "training_mode": candidate["training_mode"],
        "target_variant": candidate["target_variant"],
        "feature_set_version": candidate["feature_set_version"],
        "allocation_rule": candidate["allocation_rule"],
        "rebalance_frequency": candidate["rebalance_frequency"],
        "min_holding_days": candidate["min_holding_days"],
        "switch_margin": candidate["switch_margin"],
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
        "qqq_buy_hold_max_drawdown": metrics.qqq_buy_hold_max_drawdown,
        "spy_buy_hold_max_drawdown": metrics.spy_buy_hold_max_drawdown,
        "qqq_buy_hold_return_to_drawdown": metrics.qqq_buy_hold_return_to_drawdown,
        "spy_buy_hold_return_to_drawdown": metrics.spy_buy_hold_return_to_drawdown,
    }
    return pd.DataFrame([row], columns=SUMMARY_COLUMNS)


def _rank_results_like_model_zoo_report(
    results: pd.DataFrame,
    decision_matrix: pd.DataFrame,
) -> pd.DataFrame:
    ranked = results.copy()
    ranked["beats_qqq_return"] = _as_bool(ranked["beats_qqq_return"])
    ranked["beats_qqq_return_to_drawdown"] = _as_bool(
        ranked["beats_qqq_return_to_drawdown"]
    )
    ranked["decision"] = ranked.apply(classify_long_history_decision, axis=1)
    decision_rank = {
        "best_candidate": 0,
        "risk_adjusted_candidate": 1,
        "mixed": 2,
        "reject": 3,
    }
    ranked["_decision_rank"] = ranked["decision"].map(decision_rank).fillna(9)
    ranked = (
        ranked.sort_values(
            [
                "_decision_rank",
                "strategy_vs_qqq",
                "return_to_drawdown",
                "drawdown_improvement_vs_qqq",
            ],
            ascending=[True, False, False, False],
            kind="mergesort",
        )
        .drop(columns=["_decision_rank"])
        .reset_index(drop=True)
    )
    ranked["candidate_id"] = [f"candidate_{index + 1:04d}" for index in range(len(ranked))]
    ranked["candidate_label"] = ranked.apply(_candidate_label, axis=1)
    if not decision_matrix.empty:
        labels = decision_matrix.set_index("candidate_id")["candidate_label"].to_dict()
        ranked["candidate_label"] = ranked.apply(
            lambda row: labels.get(row["candidate_id"], row["candidate_label"]),
            axis=1,
        )
    return ranked


def _candidate_label(row: pd.Series) -> str:
    return (
        f"{row['model_type']} | {row['training_mode']} | {row['target_variant']} | "
        f"{row['allocation_rule']} | {row['rebalance_frequency']}"
    )


def _validate_candidate_inputs(results: pd.DataFrame, decision_matrix: pd.DataFrame) -> None:
    result_required = [
        "model_type",
        "training_mode",
        "target_variant",
        "feature_set_version",
        "allocation_rule",
        "rebalance_frequency",
        "min_holding_days",
        "switch_margin",
        "transaction_cost_bps",
        "total_return",
        "max_drawdown",
        "return_to_drawdown",
        "strategy_vs_qqq",
        "beats_qqq_return",
        "beats_qqq_return_to_drawdown",
        "drawdown_improvement_vs_qqq",
    ]
    decision_required = ["candidate_id", "candidate_label", "decision"]
    _validate_columns(results, result_required, "model-zoo results")
    _validate_columns(decision_matrix, decision_required, "model-zoo decision matrix")


def _validate_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _first_important_match(ranked: pd.DataFrame, important: dict[str, object]) -> str | None:
    mask = pd.Series(True, index=ranked.index)
    for column in [
        "model_type",
        "training_mode",
        "target_variant",
        "feature_set_version",
        "allocation_rule",
        "rebalance_frequency",
        "min_holding_days",
    ]:
        mask &= ranked[column].astype(str).eq(str(important[column]))
    switch_margins = important.get("switch_margins")
    if switch_margins is not None:
        allowed = {round(float(value), 9) for value in switch_margins}  # type: ignore[arg-type]
        mask &= ranked["switch_margin"].astype(float).round(9).isin(allowed)
    matches = ranked.loc[mask]
    if matches.empty:
        return None
    return str(matches.iloc[0]["candidate_id"])


def _candidate_dedup_key(row: pd.Series) -> tuple[object, ...]:
    config_columns = [column for column in CONFIG_COLUMNS if column != "switch_margin"]
    metric_columns = [
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
    ]
    return tuple(_key_value(row.get(column)) for column in [*config_columns, *metric_columns])


def _key_value(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (float, np.floating)):
        return round(float(value), 12)
    return value


def _feature_columns(dataset: pd.DataFrame, feature_set_version: str) -> list[str]:
    from us_market_ml.data import get_feature_columns

    return get_feature_columns(dataset, feature_set_version=feature_set_version)


def _benchmark_daily(raw_df: pd.DataFrame, symbol: str, dates: pd.Series) -> pd.DataFrame:
    date_index = pd.DatetimeIndex(pd.to_datetime(dates)).drop_duplicates().sort_values()
    source = raw_df.copy()
    source["date"] = pd.to_datetime(source["date"])
    prices = (
        source.loc[source["symbol"].astype(str).str.upper().eq(symbol.upper()), ["date", "close"]]
        .dropna(subset=["close"])
        .drop_duplicates(subset=["date"], keep="last")
        .set_index("date")
        .sort_index()["close"]
        .reindex(date_index)
        .ffill()
    )
    returns = prices.pct_change().fillna(0.0)
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return pd.DataFrame(
        {
            "date": date_index,
            "daily_return": returns.to_numpy(dtype="float64"),
            "equity": equity.to_numpy(dtype="float64"),
            "drawdown": drawdown.to_numpy(dtype="float64"),
        }
    )


def _active_symbols_by_date(signals: pd.DataFrame) -> pd.DataFrame:
    source = signals.copy()
    source["date"] = pd.to_datetime(source["date"])
    active = source.loc[pd.to_numeric(source["target_weight"], errors="coerce").gt(0.0)]
    rows = []
    for date, group in active.groupby("date", sort=True):
        symbols = sorted(group["symbol"].astype(str).unique())
        rows.append(
            {
                "date": date,
                "active_symbol": symbols[0] if len(symbols) == 1 else "",
                "active_symbols": ";".join(symbols),
            }
        )
    return pd.DataFrame(rows, columns=["date", "active_symbol", "active_symbols"])


def _exposure_counts(daily_equity: pd.DataFrame) -> pd.Series:
    symbols: list[str] = []
    for value in daily_equity["active_symbols"].fillna("").astype(str):
        symbols.extend([symbol for symbol in value.split(";") if symbol])
    if not symbols:
        return pd.Series(dtype="int64")
    return pd.Series(symbols).value_counts()


def _slice_daily(daily_equity: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    source = daily_equity.copy()
    source["date"] = pd.to_datetime(source["date"])
    return source.loc[
        source["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    ].copy()


def _compound_return(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if values.empty:
        return np.nan
    return float((1.0 + values).prod() - 1.0)


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if values.empty:
        return np.nan
    equity = (1.0 + values).cumprod()
    return calculate_max_drawdown(equity)


def _top_symbol_for_period(period: pd.DataFrame) -> str:
    counts = _exposure_counts(period)
    if counts.empty:
        return ""
    return str(counts.index[0])


def _decision_at_cost(metrics: Any) -> str:
    if metrics.beats_qqq_return and metrics.beats_qqq_return_to_drawdown:
        return "keep_for_deep_dive"
    if metrics.beats_qqq_return:
        return "mixed_candidate"
    return "reject_deprioritize"


def _validation_decision_and_rationale(row: pd.Series) -> tuple[str, str]:
    beats_return = _bool_value(row.get("beats_qqq_return"))
    beats_rtd = _bool_value(row.get("beats_qqq_return_to_drawdown"))
    positive_rate = _float(row.get("positive_vs_qqq_window_rate"))
    top_exposure = _float(row.get("top_symbol_exposure_pct"))
    unique_count = _float(row.get("unique_symbol_count"))
    cost_20bps_edge = _float(row.get("cost_20bps_strategy_vs_qqq"))

    severe_concentration = bool(
        np.isfinite(top_exposure) and (top_exposure >= 0.90 or unique_count <= 1)
    )
    concentration_watch = bool(np.isfinite(top_exposure) and top_exposure >= 0.75)
    rolling_good = bool(np.isfinite(positive_rate) and positive_rate >= 0.60)
    rolling_acceptable = bool(np.isfinite(positive_rate) and positive_rate >= 0.45)
    cost_still_positive = bool(np.isfinite(cost_20bps_edge) and cost_20bps_edge > 0.0)

    parts = []
    parts.append("beats QQQ return" if beats_return else "does not beat QQQ return")
    parts.append(
        "beats QQQ return-to-drawdown"
        if beats_rtd
        else "does not beat QQQ return-to-drawdown"
    )
    if np.isfinite(positive_rate):
        parts.append(f"rolling positive rate {positive_rate:.2f}")
    if severe_concentration:
        parts.append("severe concentration")
    elif concentration_watch:
        parts.append("concentration watch")
    if cost_still_positive:
        parts.append("20bps cost case remains ahead of QQQ")

    if not beats_return:
        return "reject_deprioritize", "; ".join(parts)
    if severe_concentration:
        return "reject_deprioritize", "; ".join(parts)
    if beats_rtd and rolling_good and cost_still_positive and not concentration_watch:
        return "robust_candidate", "; ".join(parts)
    if beats_rtd and rolling_acceptable:
        return "keep_for_deep_dive", "; ".join(parts)
    return "mixed_candidate", "; ".join(parts)


def _concat_or_empty(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=columns)
    out = pd.concat(non_empty, ignore_index=True)
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    return out.loc[:, columns]


def _none_if_missing(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value)
    if text.lower() in {"", "nan", "none", "<na>"}:
        return None
    return text


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes"}


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


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
    "COST_SENSITIVITY_NAME",
    "DAILY_EQUITY_NAME",
    "DECISION_MATRIX_NAME",
    "DEFAULT_COST_SENSITIVITY_BPS",
    "DEFAULT_OUTPUT_DIR",
    "OUTPUT_CSV_NAMES",
    "PREFLIGHT_MD_NAME",
    "REPORT_NAME",
    "ROLLING_WINDOWS_NAME",
    "SELECTED_SYMBOLS_NAME",
    "SUBPERIODS_NAME",
    "SUMMARY_NAME",
    "SYMBOL_EXPOSURE_NAME",
    "TRADES_NAME",
    "LongHistoryShortlistValidationArtifacts",
    "build_cost_sensitivity",
    "build_long_history_shortlist_validation_artifacts",
    "build_rolling_window_robustness",
    "build_selected_symbol_summary",
    "build_subperiod_robustness",
    "build_symbol_exposure_summary",
    "build_validation_decision_matrix",
    "generate_rolling_windows",
    "output_paths",
    "render_preflight_markdown",
    "replay_shortlist_candidate",
    "select_shortlist_candidates",
    "spec_from_candidate",
    "summarize_rolling_windows",
    "write_long_history_shortlist_validation_artifacts",
]
