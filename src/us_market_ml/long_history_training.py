"""Long-history QQQ-relative model-zoo training helpers.

This module is intentionally local research infrastructure. It builds
deterministic configuration grids, training windows, QQQ-relative datasets,
and simple full-period strategy metrics. It does not export production
signals, write model cards, create rich replay artifacts, or access cloud
storage.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from us_market_ml.backtest import SignalBacktestConfig, run_signal_backtest
from us_market_ml.backtest.signal_backtester import calculate_max_drawdown
from us_market_ml.data import (
    SupervisedDatasetConfig,
    build_supervised_dataset,
    get_feature_columns,
)
from us_market_ml.features import LONG_HISTORY_REGIME_FEATURE_SET_VERSION
from us_market_ml.models import (
    BaselineModelConfig,
    MLPRegressorConfig,
    create_baseline_model,
    fit_mlp_regressor,
    predict_baseline,
    predict_mlp,
)
from us_market_ml.qqq_relative_targets import target_horizon_from_variant
from us_market_ml.universe import (
    DynamicEligibilityConfig,
    apply_dynamic_universe_eligibility,
    min_symbol_count_per_date,
    select_fold_train_feature_columns,
    symbol_count,
)

START_DATE = "2019-01-01"
END_DATE = "2026-05-15"
BASE_TRAIN_END = "2018-12-31"
TRADING_DAYS_PER_YEAR = 252

LONG_HISTORY_FEATURE_SET_VERSION = LONG_HISTORY_REGIME_FEATURE_SET_VERSION
TARGET_VARIANTS: tuple[str, ...] = (
    "excess_return_vs_qqq_20d",
    "excess_return_vs_qqq_63d",
    "excess_return_vs_qqq_126d",
    "return_drawdown_score_vs_qqq_63d",
    "return_drawdown_score_vs_qqq_126d",
)
ALLOCATION_RULES: tuple[str, ...] = (
    "top1",
    "top3_equal_weight",
    "top5_equal_weight",
    "top5_rank_weighted_mild",
    "top5_score_weighted_cap035",
)
REBALANCE_FREQUENCIES: tuple[str, ...] = ("monthly", "quarterly")
MIN_HOLDING_DAYS_VALUES: tuple[int, ...] = (0, 10, 20)
SWITCH_MARGIN_VALUES: tuple[float, ...] = (0.0, 0.002, 0.005)
RECENCY_DECAY_PROFILES: dict[str, float] = {
    "mild_decay": 756.0,
    "medium_decay": 504.0,
    "strong_decay": 252.0,
}
BASE_RECENT_BASE_WEIGHTS: tuple[float, ...] = (0.75, 0.50, 0.25)
BASE_RECENT_WINDOWS_YEARS: tuple[int, ...] = (2, 3, 5)
MLP_FINETUNE_WINDOWS_YEARS: tuple[int, ...] = (2, 3, 5)
MLP_FINETUNE_LR_SCALES: tuple[str, ...] = ("low", "medium")
MODEL_TYPES: tuple[str, ...] = (
    "linear_regression",
    "ridge",
    "elasticnet",
    "huber_regression",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
    "mlp",
)
TRAINING_MODES: tuple[str, ...] = (
    "frozen_base_1999_2018",
    "anchored_expanding_yearly",
    "anchored_expanding_quarterly",
    "recency_weighted_expanding",
    "base_recent_blend",
    "mlp_pretrain_finetune",
)


@dataclass(frozen=True)
class LongHistoryTrainingOption:
    """One model/training-mode option before target/allocation expansion."""

    model_type: str
    training_mode: str
    recency_decay_profile: str | None = None
    base_weight: float | None = None
    recent_window_years: int | None = None
    finetune_window_years: int | None = None
    finetune_lr_scale: str | None = None


@dataclass(frozen=True)
class LongHistoryRunSpec:
    """One executable long-history QQQ-relative model-zoo configuration."""

    run_key: str
    model_type: str
    training_mode: str
    target_variant: str
    feature_set_version: str
    allocation_rule: str
    rebalance_frequency: str
    min_holding_days: int
    switch_margin: float
    transaction_cost_bps: float
    recency_decay_profile: str | None = None
    base_weight: float | None = None
    recent_window_years: int | None = None
    finetune_window_years: int | None = None
    finetune_lr_scale: str | None = None
    random_state: int = 42


@dataclass(frozen=True)
class TrainingWindow:
    """A train-through date and the forward period predicted by that model."""

    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    predict_start: pd.Timestamp
    predict_end: pd.Timestamp


@dataclass(frozen=True)
class LongHistoryProfile:
    """Named sweep grid profile."""

    name: str
    purpose: str
    training_options: tuple[LongHistoryTrainingOption, ...]
    target_variants: tuple[str, ...]
    allocation_rules: tuple[str, ...]
    rebalance_frequencies: tuple[str, ...]
    min_holding_days_values: tuple[int, ...]
    switch_margin_values: tuple[float, ...]
    transaction_cost_bps_values: tuple[float, ...] = (10.0,)


@dataclass(frozen=True)
class StrategyMetrics:
    """Full-period strategy and benchmark metrics."""

    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_like: float
    return_to_drawdown: float
    transaction_count: int
    average_turnover: float
    estimated_transaction_cost: float
    qqq_buy_hold_total_return: float
    spy_buy_hold_total_return: float
    strategy_vs_qqq: float
    strategy_vs_spy: float
    qqq_return_capture: float
    beats_qqq_return: bool
    beats_qqq_return_to_drawdown: bool
    drawdown_improvement_vs_qqq: float
    qqq_buy_hold_max_drawdown: float
    spy_buy_hold_max_drawdown: float
    qqq_buy_hold_return_to_drawdown: float
    spy_buy_hold_return_to_drawdown: float


class SkippedLongHistoryRun(RuntimeError):
    """Raised when a config should be checkpointed as skipped."""


SWEEP_PROFILES: dict[str, LongHistoryProfile] = {
    "smoke": LongHistoryProfile(
        name="smoke",
        purpose="Very small plumbing check across all training-mode families.",
        training_options=(
            LongHistoryTrainingOption("ridge", "frozen_base_1999_2018"),
            LongHistoryTrainingOption("ridge", "anchored_expanding_yearly"),
            LongHistoryTrainingOption("ridge", "anchored_expanding_quarterly"),
            LongHistoryTrainingOption(
                "ridge",
                "recency_weighted_expanding",
                recency_decay_profile="mild_decay",
            ),
            LongHistoryTrainingOption(
                "ridge",
                "base_recent_blend",
                base_weight=0.50,
                recent_window_years=3,
            ),
            LongHistoryTrainingOption(
                "mlp",
                "mlp_pretrain_finetune",
                finetune_window_years=3,
                finetune_lr_scale="medium",
            ),
        ),
        target_variants=("excess_return_vs_qqq_20d", "return_drawdown_score_vs_qqq_63d"),
        allocation_rules=("top1", "top3_equal_weight"),
        rebalance_frequencies=("monthly",),
        min_holding_days_values=(0,),
        switch_margin_values=(0.0,),
    ),
    "core": LongHistoryProfile(
        name="core",
        purpose="Main long-run screen across practical classical models and adaptive modes.",
        training_options=(
            LongHistoryTrainingOption("linear_regression", "frozen_base_1999_2018"),
            LongHistoryTrainingOption("ridge", "frozen_base_1999_2018"),
            LongHistoryTrainingOption("elasticnet", "frozen_base_1999_2018"),
            LongHistoryTrainingOption("ridge", "anchored_expanding_yearly"),
            LongHistoryTrainingOption("elasticnet", "anchored_expanding_yearly"),
            LongHistoryTrainingOption("extra_trees", "anchored_expanding_yearly"),
            LongHistoryTrainingOption("hist_gradient_boosting", "anchored_expanding_yearly"),
            LongHistoryTrainingOption("ridge", "anchored_expanding_quarterly"),
            LongHistoryTrainingOption(
                "ridge",
                "recency_weighted_expanding",
                recency_decay_profile="mild_decay",
            ),
            LongHistoryTrainingOption(
                "ridge",
                "recency_weighted_expanding",
                recency_decay_profile="medium_decay",
            ),
            LongHistoryTrainingOption(
                "elasticnet",
                "recency_weighted_expanding",
                recency_decay_profile="medium_decay",
            ),
            LongHistoryTrainingOption(
                "ridge",
                "base_recent_blend",
                base_weight=0.75,
                recent_window_years=3,
            ),
            LongHistoryTrainingOption(
                "ridge",
                "base_recent_blend",
                base_weight=0.50,
                recent_window_years=3,
            ),
            LongHistoryTrainingOption(
                "ridge",
                "base_recent_blend",
                base_weight=0.25,
                recent_window_years=5,
            ),
            LongHistoryTrainingOption(
                "mlp",
                "mlp_pretrain_finetune",
                finetune_window_years=3,
                finetune_lr_scale="medium",
            ),
        ),
        target_variants=TARGET_VARIANTS,
        allocation_rules=("top1", "top3_equal_weight", "top5_equal_weight"),
        rebalance_frequencies=REBALANCE_FREQUENCIES,
        min_holding_days_values=(0, 20),
        switch_margin_values=(0.0, 0.002),
    ),
    "wide": LongHistoryProfile(
        name="wide",
        purpose="Larger overnight grid with all requested classical models and controls.",
        training_options=(
            *tuple(
                LongHistoryTrainingOption(model_type, "frozen_base_1999_2018")
                for model_type in MODEL_TYPES
                if model_type != "mlp"
            ),
            *tuple(
                LongHistoryTrainingOption(model_type, "anchored_expanding_yearly")
                for model_type in MODEL_TYPES
                if model_type != "mlp"
            ),
            *tuple(
                LongHistoryTrainingOption(model_type, "anchored_expanding_quarterly")
                for model_type in ("ridge", "elasticnet", "extra_trees", "hist_gradient_boosting")
            ),
            *tuple(
                LongHistoryTrainingOption(
                    model_type,
                    "recency_weighted_expanding",
                    recency_decay_profile=profile,
                )
                for model_type in ("ridge", "elasticnet", "extra_trees")
                for profile in RECENCY_DECAY_PROFILES
            ),
            *tuple(
                LongHistoryTrainingOption(
                    "ridge",
                    "base_recent_blend",
                    base_weight=base_weight,
                    recent_window_years=recent_window_years,
                )
                for base_weight in BASE_RECENT_BASE_WEIGHTS
                for recent_window_years in BASE_RECENT_WINDOWS_YEARS
            ),
            *tuple(
                LongHistoryTrainingOption(
                    "mlp",
                    "mlp_pretrain_finetune",
                    finetune_window_years=window_years,
                    finetune_lr_scale=lr_scale,
                )
                for window_years in MLP_FINETUNE_WINDOWS_YEARS
                for lr_scale in MLP_FINETUNE_LR_SCALES
            ),
        ),
        target_variants=TARGET_VARIANTS,
        allocation_rules=ALLOCATION_RULES,
        rebalance_frequencies=REBALANCE_FREQUENCIES,
        min_holding_days_values=MIN_HOLDING_DAYS_VALUES,
        switch_margin_values=SWITCH_MARGIN_VALUES,
    ),
}


def build_long_history_run_specs(
    *,
    sweep_profile: str,
    feature_set_versions: tuple[str, ...] = (LONG_HISTORY_FEATURE_SET_VERSION,),
    random_state: int = 42,
) -> list[LongHistoryRunSpec]:
    """Build deterministic run specs for a named long-history profile."""

    profile = get_sweep_profile(sweep_profile)
    specs: list[LongHistoryRunSpec] = []
    for option in profile.training_options:
        for target_variant in profile.target_variants:
            for feature_set_version in feature_set_versions:
                for allocation_rule in profile.allocation_rules:
                    for rebalance_frequency in profile.rebalance_frequencies:
                        for min_holding_days in profile.min_holding_days_values:
                            for switch_margin in profile.switch_margin_values:
                                for transaction_cost_bps in (
                                    profile.transaction_cost_bps_values
                                ):
                                    payload = {
                                        **asdict(option),
                                        "target_variant": target_variant,
                                        "feature_set_version": feature_set_version,
                                        "allocation_rule": allocation_rule,
                                        "rebalance_frequency": rebalance_frequency,
                                        "min_holding_days": min_holding_days,
                                        "switch_margin": switch_margin,
                                        "transaction_cost_bps": transaction_cost_bps,
                                        "random_state": random_state,
                                    }
                                    run_key = build_long_history_run_key(payload)
                                    specs.append(
                                        LongHistoryRunSpec(
                                            run_key=run_key,
                                            model_type=option.model_type,
                                            training_mode=option.training_mode,
                                            target_variant=target_variant,
                                            feature_set_version=feature_set_version,
                                            allocation_rule=allocation_rule,
                                            rebalance_frequency=rebalance_frequency,
                                            min_holding_days=min_holding_days,
                                            switch_margin=switch_margin,
                                            transaction_cost_bps=transaction_cost_bps,
                                            recency_decay_profile=(
                                                option.recency_decay_profile
                                            ),
                                            base_weight=option.base_weight,
                                            recent_window_years=option.recent_window_years,
                                            finetune_window_years=(
                                                option.finetune_window_years
                                            ),
                                            finetune_lr_scale=option.finetune_lr_scale,
                                            random_state=random_state,
                                        )
                                    )
    return specs


def build_long_history_run_key(payload: dict[str, Any]) -> str:
    """Return a stable readable key for one long-history config."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]
    key_parts = [
        f"model-{_slug(str(payload['model_type']))}",
        f"mode-{_slug(str(payload['training_mode']))}",
        f"target-{_slug(str(payload['target_variant']))}",
        f"features-{_slug(str(payload['feature_set_version']))}",
        f"alloc-{_slug(str(payload['allocation_rule']))}",
        f"rebalance-{_slug(str(payload['rebalance_frequency']))}",
        f"hold-{payload['min_holding_days']}",
        f"margin-{payload['switch_margin']:g}",
        f"cost-{payload['transaction_cost_bps']:g}bps",
    ]
    for optional_name in (
        "recency_decay_profile",
        "base_weight",
        "recent_window_years",
        "finetune_window_years",
        "finetune_lr_scale",
    ):
        value = payload.get(optional_name)
        if value is not None:
            key_parts.append(f"{_slug(optional_name)}-{_slug(str(value))}")
    return "__".join([*key_parts, digest])


def build_training_windows(
    training_mode: str,
    *,
    dataset_start: pd.Timestamp,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> list[TrainingWindow]:
    """Build train/predict windows for one requested training mode."""

    eval_start = pd.Timestamp(start_date)
    eval_end = pd.Timestamp(end_date)
    if training_mode == "frozen_base_1999_2018":
        return [
            TrainingWindow(
                name="frozen_base_1999_2018",
                train_start=dataset_start,
                train_end=pd.Timestamp(BASE_TRAIN_END),
                predict_start=eval_start,
                predict_end=eval_end,
            )
        ]
    if training_mode in {
        "anchored_expanding_yearly",
        "recency_weighted_expanding",
        "base_recent_blend",
        "mlp_pretrain_finetune",
    }:
        return _yearly_windows(dataset_start=dataset_start, eval_start=eval_start, eval_end=eval_end)
    if training_mode == "anchored_expanding_quarterly":
        return _quarterly_windows(
            dataset_start=dataset_start,
            eval_start=eval_start,
            eval_end=eval_end,
        )
    raise ValueError(f"Unsupported training_mode: {training_mode!r}.")


def build_long_history_dataset(
    raw_df: pd.DataFrame,
    *,
    feature_set_version: str = LONG_HISTORY_FEATURE_SET_VERSION,
    include_classification_target: bool = False,
) -> pd.DataFrame:
    """Build the long-history feature/target frame used by the model zoo."""

    classification_horizons = (63,) if include_classification_target else ()
    return build_supervised_dataset(
        raw_df,
        SupervisedDatasetConfig(
            forward_horizons=(20, 63, 126),
            benchmark_symbol="QQQ",
            excess_return_horizons=(20, 63, 126),
            rank_target_col=None,
            feature_set_version=feature_set_version,
            qqq_relative_return_drawdown_horizons=(63, 126),
            qqq_relative_classification_horizons=classification_horizons,
        ),
    )


def dataset_start_for_required_symbols(
    raw_df: pd.DataFrame,
    *,
    required_symbols: tuple[str, ...] = ("QQQ", "SPY"),
) -> pd.Timestamp:
    """Return earliest common usable date for required benchmark symbols."""

    required = {symbol.upper() for symbol in required_symbols}
    source = raw_df.copy()
    source["date"] = pd.to_datetime(source["date"])
    source["symbol_upper"] = source["symbol"].astype("string").str.upper()
    missing = sorted(required.difference(set(source["symbol_upper"].dropna().unique())))
    if missing:
        raise ValueError(f"Required benchmark symbols are missing from dataset: {missing}")
    first_dates = (
        source.loc[source["symbol_upper"].isin(required)]
        .groupby("symbol_upper", sort=True)["date"]
        .min()
    )
    return pd.Timestamp(first_dates.max())


def run_long_history_model_zoo_spec(
    raw_df: pd.DataFrame,
    spec: LongHistoryRunSpec,
    *,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    min_history_rows: int = 252,
    min_train_eligible_rows: int = 1,
) -> dict[str, object]:
    """Run one long-history spec and return one full-period result row."""

    if spec.training_mode == "mlp_pretrain_finetune":
        raise SkippedLongHistoryRun(
            "mlp_pretrain_finetune is intentionally limited in this first model-zoo "
            "batch; checkpointed as skipped rather than marked production-ready."
        )
    if spec.model_type == "mlp" and spec.training_mode == "recency_weighted_expanding":
        raise SkippedLongHistoryRun("MLP sample_weight support is unavailable.")

    raw = raw_df.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    dataset_start = dataset_start_for_required_symbols(raw)
    raw = raw.loc[
        raw["date"].between(dataset_start, pd.Timestamp(end_date), inclusive="both")
    ].copy()
    dataset = build_long_history_dataset(raw, feature_set_version=spec.feature_set_version)
    feature_cols = get_feature_columns(dataset, feature_set_version=spec.feature_set_version)
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
    rebalance_dates = _rebalance_dates(raw, start_date=start_date, end_date=end_date, frequency=spec.rebalance_frequency)
    if not rebalance_dates:
        raise SkippedLongHistoryRun("No rebalance dates were available in the forward period.")

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
            date
            for date in rebalance_dates
            if window.predict_start <= date <= window.predict_end
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

        sample_weights = None
        if spec.training_mode == "recency_weighted_expanding":
            model = create_baseline_model(
                _baseline_model_config(spec.model_type, random_state=spec.random_state)
            )
            if not _supports_sample_weight(model):
                raise SkippedLongHistoryRun(
                    f"Model {spec.model_type!r} does not support sample_weight."
                )
            sample_weights = "__long_history_sample_weight"

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
            sample_weights=sample_weights,
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
    metrics = evaluate_weight_schedule(
        raw,
        weights,
        start_date=start_date,
        end_date=end_date,
        transaction_cost_bps=spec.transaction_cost_bps,
    )
    return {
        "run_key": spec.run_key,
        "model_type": spec.model_type,
        "training_mode": spec.training_mode,
        "target_variant": spec.target_variant,
        "feature_set_version": spec.feature_set_version,
        "allocation_rule": spec.allocation_rule,
        "rebalance_frequency": spec.rebalance_frequency,
        "min_holding_days": spec.min_holding_days,
        "switch_margin": spec.switch_margin,
        "transaction_cost_bps": spec.transaction_cost_bps,
        "recency_decay_profile": spec.recency_decay_profile,
        "base_weight": spec.base_weight,
        "recent_window_years": spec.recent_window_years,
        "finetune_window_years": spec.finetune_window_years,
        "finetune_lr_scale": spec.finetune_lr_scale,
        "start_date": start_date,
        "end_date": end_date,
        "historical_training_start": str(dataset_start.date()),
        **asdict(metrics),
        "decision": "unclassified",
    }


def build_rebalance_weights(
    predictions: pd.DataFrame,
    *,
    allocation_rule: str,
    min_holding_days: int = 0,
    switch_margin: float = 0.0,
) -> pd.DataFrame:
    """Convert rebalance-date predictions into target weights."""

    if predictions.empty:
        return pd.DataFrame(columns=["date", "symbol", "target_weight"])
    _validate_allocation_rule(allocation_rule)
    top_k = _top_k_for_allocation_rule(allocation_rule)
    rows: list[dict[str, object]] = []
    current_holdings: dict[str, int] = {}
    current_scores: dict[str, float] = {}

    for date, date_group in predictions.groupby("date", sort=True):
        ranked = (
            date_group.dropna(subset=["prediction"])
            .sort_values(["prediction", "symbol"], ascending=[False, True], kind="mergesort")
            .head(max(top_k * 2, top_k))
        )
        if ranked.empty:
            selected = pd.DataFrame(columns=ranked.columns)
        else:
            selected = _smooth_selection(
                ranked,
                top_k=top_k,
                current_holdings=current_holdings,
                current_scores=current_scores,
                min_holding_days=min_holding_days,
                switch_margin=switch_margin,
            )
        weights = _allocation_weights(selected, allocation_rule)
        current_holdings = {symbol: age + 1 for symbol, age in current_holdings.items()}
        current_scores = {}
        for symbol, weight in weights.items():
            current_holdings[symbol] = 1
            score = selected.loc[selected["symbol"].eq(symbol), "prediction"].iloc[0]
            current_scores[symbol] = float(score)
            rows.append({"date": date, "symbol": symbol, "target_weight": weight})

    return pd.DataFrame(rows, columns=["date", "symbol", "target_weight"])


def evaluate_weight_schedule(
    raw_df: pd.DataFrame,
    rebalance_weights: pd.DataFrame,
    *,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    transaction_cost_bps: float = 10.0,
) -> StrategyMetrics:
    """Evaluate a rebalance schedule against QQQ/SPY buy-and-hold."""

    signals = _expand_rebalance_weights(raw_df, rebalance_weights, start_date, end_date)
    backtest = run_signal_backtest(
        signals,
        SignalBacktestConfig(transaction_cost_bps=transaction_cost_bps),
    )
    summary = backtest.summary_metrics
    qqq = buy_hold_metrics(raw_df, "QQQ", start_date=start_date, end_date=end_date)
    spy = buy_hold_metrics(raw_df, "SPY", start_date=start_date, end_date=end_date)
    days = max(int(len(backtest.equity_curve)), 1)
    total_return = float(summary["total_return"])
    max_drawdown = float(summary["max_drawdown"])
    return_to_drawdown = _return_to_drawdown(total_return, max_drawdown)
    qqq_rtd = _return_to_drawdown(qqq["total_return"], qqq["max_drawdown"])
    spy_rtd = _return_to_drawdown(spy["total_return"], spy["max_drawdown"])
    return StrategyMetrics(
        total_return=total_return,
        annualized_return=_annualized_return(total_return, days),
        max_drawdown=max_drawdown,
        sharpe_like=float(summary["sharpe_like"]),
        return_to_drawdown=return_to_drawdown,
        transaction_count=int(
            (backtest.equity_curve.get("turnover", pd.Series(dtype=float)) > 0.0).sum()
        ),
        average_turnover=float(summary["mean_daily_turnover"]),
        estimated_transaction_cost=float(summary["total_transaction_cost"]),
        qqq_buy_hold_total_return=qqq["total_return"],
        spy_buy_hold_total_return=spy["total_return"],
        strategy_vs_qqq=total_return - qqq["total_return"],
        strategy_vs_spy=total_return - spy["total_return"],
        qqq_return_capture=_safe_ratio(total_return, qqq["total_return"]),
        beats_qqq_return=bool(total_return > qqq["total_return"]),
        beats_qqq_return_to_drawdown=bool(return_to_drawdown > qqq_rtd),
        drawdown_improvement_vs_qqq=max_drawdown - qqq["max_drawdown"],
        qqq_buy_hold_max_drawdown=qqq["max_drawdown"],
        spy_buy_hold_max_drawdown=spy["max_drawdown"],
        qqq_buy_hold_return_to_drawdown=qqq_rtd,
        spy_buy_hold_return_to_drawdown=spy_rtd,
    )


def buy_hold_metrics(
    raw_df: pd.DataFrame,
    symbol: str,
    *,
    start_date: str,
    end_date: str,
) -> dict[str, float]:
    """Return daily buy-and-hold metrics for one symbol."""

    source = raw_df.copy()
    source["date"] = pd.to_datetime(source["date"])
    symbol_frame = (
        source.loc[
            source["symbol"].astype("string").str.upper().eq(symbol.upper())
            & source["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date)),
            ["date", "close"],
        ]
        .dropna(subset=["close"])
        .sort_values("date", kind="mergesort")
    )
    if symbol_frame.empty:
        return {"total_return": np.nan, "max_drawdown": np.nan, "sharpe_like": np.nan}
    returns = symbol_frame["close"].pct_change().fillna(0.0)
    equity = (1.0 + returns).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    volatility = float(returns.std())
    return {
        "total_return": total_return,
        "max_drawdown": calculate_max_drawdown(equity),
        "sharpe_like": float(returns.mean() / volatility) if volatility > 0.0 else np.nan,
    }


def get_sweep_profile(name: str) -> LongHistoryProfile:
    """Return a named sweep profile."""

    try:
        return SWEEP_PROFILES[name]
    except KeyError as exc:
        supported = ", ".join(sorted(SWEEP_PROFILES))
        raise ValueError(f"Unsupported SWEEP_PROFILE={name!r}; expected one of: {supported}") from exc


def run_spec_is_structurally_valid(spec: LongHistoryRunSpec) -> tuple[bool, str]:
    """Return whether a spec is structurally runnable before data preflight."""

    reasons: list[str] = []
    if spec.training_mode == "mlp_pretrain_finetune":
        reasons.append("mlp_pretrain_finetune_intentionally_limited")
    if spec.training_mode == "recency_weighted_expanding" and spec.model_type == "mlp":
        reasons.append("model_does_not_support_sample_weight")
    if spec.model_type == "mlp" and spec.training_mode != "mlp_pretrain_finetune":
        reasons.append("mlp_only_supported_for_mlp_pretrain_finetune_mode")
    return (not reasons, ";".join(reasons))


def _fit_model_for_window(
    *,
    dataset: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_type: str,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    target_horizon: int,
    min_history_rows: int,
    min_train_eligible_rows: int,
    random_state: int,
    sample_weights: str | None,
    recency_decay_profile: str | None = None,
) -> tuple[Any, list[str]]:
    train_df = _training_rows(
        dataset=dataset,
        target_col=target_col,
        train_start=train_start,
        train_end=train_end,
        target_horizon=target_horizon,
    )
    selected_features = select_fold_train_feature_columns(train_df, feature_cols)
    if not selected_features:
        raise SkippedLongHistoryRun("No fold-local features were available.")
    train_eligible = apply_dynamic_universe_eligibility(
        train_df,
        feature_cols=selected_features,
        target_col=target_col,
        require_target=True,
        config=DynamicEligibilityConfig(min_history_rows=min_history_rows),
        history_df=dataset,
    )
    if len(train_eligible) < min_train_eligible_rows:
        raise SkippedLongHistoryRun(
            "train_eligible_row_count below min_train_eligible_rows "
            f"({len(train_eligible)} < {min_train_eligible_rows})"
        )
    if model_type == "mlp":
        model, _ = fit_mlp_regressor(
            train_eligible,
            target_col,
            selected_features,
            config=MLPRegressorConfig(
                hidden_units=(64, 32),
                max_epochs=20,
                patience=5,
                random_state=random_state,
                scale_target=True,
            ),
        )
        return model, selected_features

    model_config = _baseline_model_config(model_type, random_state=random_state)
    model = create_baseline_model(model_config)
    fit_df = train_eligible.dropna(subset=[*selected_features, target_col]).copy()
    if fit_df.empty:
        raise SkippedLongHistoryRun("No usable training rows after selected value drop.")
    fit_kwargs: dict[str, Any] = {}
    if sample_weights is not None:
        weights = _recency_sample_weights(
            fit_df["date"],
            train_end=train_end,
            decay_profile=recency_decay_profile or "medium_decay",
        )
        fit_kwargs["sample_weight"] = weights
    model.fit(fit_df[selected_features], fit_df[target_col], **fit_kwargs)
    return model, selected_features


def _predict_model(
    model: Any,
    scored: pd.DataFrame,
    selected_features: list[str],
    model_type: str,
) -> pd.DataFrame:
    if model_type == "mlp":
        return predict_mlp(model, scored, selected_features)
    return predict_baseline(model, scored, selected_features)


def _training_rows(
    *,
    dataset: pd.DataFrame,
    target_col: str,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    target_horizon: int,
) -> pd.DataFrame:
    end_date_col = f"forward_return_end_date_{target_horizon}d"
    mask = dataset["date"].between(train_start, train_end, inclusive="both")
    if end_date_col in dataset.columns:
        mask &= pd.to_datetime(dataset[end_date_col]).le(train_end)
    return dataset.loc[mask].copy()


def _eligible_prediction_rows(
    *,
    dataset: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    predict_dates: list[pd.Timestamp],
    min_history_rows: int,
) -> pd.DataFrame:
    predict_df = dataset.loc[dataset["date"].isin(predict_dates)].copy()
    if predict_df.empty:
        return predict_df
    return apply_dynamic_universe_eligibility(
        predict_df,
        feature_cols=feature_cols,
        target_col=target_col,
        require_target=False,
        config=DynamicEligibilityConfig(min_history_rows=min_history_rows),
        history_df=dataset,
    )


def _baseline_model_config(model_type: str, *, random_state: int) -> BaselineModelConfig:
    tree_estimators = 50 if model_type in {"random_forest", "extra_trees"} else 100
    return BaselineModelConfig(
        model_type=model_type,
        random_state=random_state,
        n_estimators=tree_estimators,
        max_depth=8 if model_type in {"random_forest", "extra_trees"} else None,
        min_samples_leaf=5 if model_type in {"random_forest", "extra_trees"} else 1,
    )


def _supports_sample_weight(model: Any) -> bool:
    try:
        return "sample_weight" in inspect.signature(model.fit).parameters
    except (TypeError, ValueError):
        return False


def _recency_sample_weights(
    dates: pd.Series,
    *,
    train_end: pd.Timestamp,
    decay_profile: str,
) -> np.ndarray:
    half_life_days = RECENCY_DECAY_PROFILES[decay_profile]
    age_days = (train_end - pd.to_datetime(dates)).dt.days.clip(lower=0)
    weights = np.exp(-np.log(2.0) * age_days / half_life_days)
    return weights.to_numpy(dtype="float64")


def _rebalance_dates(
    raw_df: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
    frequency: str,
) -> list[pd.Timestamp]:
    if frequency not in REBALANCE_FREQUENCIES:
        raise ValueError(f"Unsupported rebalance_frequency: {frequency!r}.")
    source = raw_df.copy()
    source["date"] = pd.to_datetime(source["date"])
    dates = sorted(
        source.loc[
            source["symbol"].astype("string").str.upper().eq("QQQ")
            & source["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date)),
            "date",
        ].unique()
    )
    if not dates:
        return []
    date_frame = pd.DataFrame({"date": pd.to_datetime(dates)})
    if frequency == "monthly":
        grouped = date_frame.groupby(date_frame["date"].dt.to_period("M"), sort=True)
    else:
        grouped = date_frame.groupby(date_frame["date"].dt.to_period("Q"), sort=True)
    return [pd.Timestamp(group["date"].min()) for _, group in grouped]


def _expand_rebalance_weights(
    raw_df: pd.DataFrame,
    rebalance_weights: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    source = raw_df.copy()
    source["date"] = pd.to_datetime(source["date"])
    eval_prices = source.loc[
        source["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date)),
        ["date", "symbol", "close"],
    ].copy()
    eval_dates = sorted(eval_prices["date"].dropna().unique())
    if rebalance_weights.empty:
        first_symbol = "QQQ"
        return eval_prices.loc[eval_prices["symbol"].astype(str).str.upper().eq(first_symbol)].assign(
            target_weight=0.0,
            selected=False,
        )

    changes = rebalance_weights.copy()
    changes["date"] = pd.to_datetime(changes["date"])
    selected_symbols = sorted(changes["symbol"].dropna().astype(str).unique())
    weight_matrix = (
        changes.pivot_table(
            index="date",
            columns="symbol",
            values="target_weight",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(eval_dates)
        .ffill()
        .fillna(0.0)
    )
    for symbol in selected_symbols:
        if symbol not in weight_matrix.columns:
            weight_matrix[symbol] = 0.0
    weights_long = (
        weight_matrix[selected_symbols]
        .reset_index()
        .rename(columns={"index": "date"})
        .melt(id_vars="date", var_name="symbol", value_name="target_weight")
    )
    signals = eval_prices.merge(weights_long, on=["date", "symbol"], how="inner")
    signals["target_weight"] = signals["target_weight"].fillna(0.0)
    signals["selected"] = signals["target_weight"].gt(0.0)
    return signals.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)


def _smooth_selection(
    ranked: pd.DataFrame,
    *,
    top_k: int,
    current_holdings: dict[str, int],
    current_scores: dict[str, float],
    min_holding_days: int,
    switch_margin: float,
) -> pd.DataFrame:
    selected_symbols: list[str] = []
    ranked_by_symbol = {str(row.symbol): row for row in ranked.itertuples(index=False)}
    for symbol, age in current_holdings.items():
        if symbol not in ranked_by_symbol:
            continue
        if age < min_holding_days:
            selected_symbols.append(symbol)
        elif selected_symbols:
            continue
        elif len(selected_symbols) < top_k:
            current_score = current_scores.get(symbol, float("-inf"))
            best_score = float(ranked.iloc[0]["prediction"])
            if best_score <= current_score + switch_margin:
                selected_symbols.append(symbol)

    for symbol in ranked["symbol"].astype(str):
        if len(selected_symbols) >= top_k:
            break
        if symbol not in selected_symbols:
            selected_symbols.append(symbol)
    return ranked.loc[ranked["symbol"].astype(str).isin(selected_symbols)].head(top_k).copy()


def _allocation_weights(selected: pd.DataFrame, allocation_rule: str) -> dict[str, float]:
    if selected.empty:
        return {}
    symbols = selected["symbol"].astype(str).tolist()
    if allocation_rule in {"top1", "top3_equal_weight", "top5_equal_weight"}:
        weight = 1.0 / len(symbols)
        return {symbol: weight for symbol in symbols}
    if allocation_rule == "top5_rank_weighted_mild":
        raw = np.arange(len(symbols), 0, -1, dtype="float64") ** 0.75
        weights = raw / raw.sum()
        return dict(zip(symbols, weights, strict=True))
    if allocation_rule == "top5_score_weighted_cap035":
        scores = pd.to_numeric(selected["prediction"], errors="coerce").to_numpy(dtype="float64")
        shifted = scores - np.nanmin(scores)
        raw = shifted + 1.0e-6
        if not np.isfinite(raw).all() or raw.sum() <= 0.0:
            raw = np.ones_like(scores)
        weights = _cap_and_redistribute(raw / raw.sum(), cap=0.35)
        return dict(zip(symbols, weights, strict=True))
    raise ValueError(f"Unsupported allocation_rule: {allocation_rule!r}.")


def _cap_and_redistribute(weights: np.ndarray, *, cap: float) -> np.ndarray:
    capped = np.zeros_like(weights, dtype="float64")
    remaining = np.ones_like(weights, dtype=bool)
    remaining_weight = 1.0
    source = weights.astype("float64").copy()
    while remaining.any():
        candidate = source[remaining] / source[remaining].sum() * remaining_weight
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


def _top_k_for_allocation_rule(allocation_rule: str) -> int:
    if allocation_rule == "top1":
        return 1
    if allocation_rule.startswith("top3"):
        return 3
    if allocation_rule.startswith("top5"):
        return 5
    raise ValueError(f"Unsupported allocation_rule: {allocation_rule!r}.")


def _validate_allocation_rule(allocation_rule: str) -> None:
    if allocation_rule not in ALLOCATION_RULES:
        supported = ", ".join(ALLOCATION_RULES)
        raise ValueError(f"Unsupported allocation_rule {allocation_rule!r}: {supported}.")


def _yearly_windows(
    *,
    dataset_start: pd.Timestamp,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
) -> list[TrainingWindow]:
    windows: list[TrainingWindow] = []
    for year in range(eval_start.year, eval_end.year + 1):
        predict_start = max(pd.Timestamp(year=year, month=1, day=1), eval_start)
        predict_end = min(pd.Timestamp(year=year, month=12, day=31), eval_end)
        if predict_start > predict_end:
            continue
        windows.append(
            TrainingWindow(
                name=f"year_{year}",
                train_start=dataset_start,
                train_end=predict_start - pd.Timedelta(days=1),
                predict_start=predict_start,
                predict_end=predict_end,
            )
        )
    return windows


def _quarterly_windows(
    *,
    dataset_start: pd.Timestamp,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
) -> list[TrainingWindow]:
    quarter_starts = pd.date_range(eval_start, eval_end, freq="QS")
    if len(quarter_starts) == 0 or quarter_starts[0] != eval_start:
        quarter_starts = pd.DatetimeIndex([eval_start, *quarter_starts])
    windows: list[TrainingWindow] = []
    for index, predict_start in enumerate(quarter_starts):
        next_start = (
            quarter_starts[index + 1]
            if index + 1 < len(quarter_starts)
            else eval_end + pd.Timedelta(days=1)
        )
        predict_end = min(next_start - pd.Timedelta(days=1), eval_end)
        if predict_start > eval_end:
            continue
        windows.append(
            TrainingWindow(
                name=f"quarter_{predict_start.year}_q{((predict_start.month - 1) // 3) + 1}",
                train_start=dataset_start,
                train_end=predict_start - pd.Timedelta(days=1),
                predict_start=pd.Timestamp(predict_start),
                predict_end=pd.Timestamp(predict_end),
            )
        )
    return windows


def preflight_counts_for_spec(
    dataset: pd.DataFrame,
    spec: LongHistoryRunSpec,
    *,
    min_history_rows: int,
    min_train_eligible_rows: int,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> dict[str, object]:
    """Return lightweight fold-local counts for preflight diagnostics."""

    structurally_valid, reason = run_spec_is_structurally_valid(spec)
    feature_cols = get_feature_columns(dataset, feature_set_version=spec.feature_set_version)
    windows = build_training_windows(
        spec.training_mode,
        dataset_start=dataset["date"].min(),
        start_date=start_date,
        end_date=end_date,
    )
    first_window = windows[0]
    train_df = _training_rows(
        dataset=dataset,
        target_col=spec.target_variant,
        train_start=first_window.train_start,
        train_end=first_window.train_end,
        target_horizon=target_horizon_from_variant(spec.target_variant),
    )
    selected_features = select_fold_train_feature_columns(train_df, feature_cols)
    if selected_features:
        train_eligible = apply_dynamic_universe_eligibility(
            train_df,
            feature_cols=selected_features,
            target_col=spec.target_variant,
            require_target=True,
            config=DynamicEligibilityConfig(min_history_rows=min_history_rows),
            history_df=dataset,
        )
    else:
        train_eligible = train_df.iloc[0:0].copy()
    prediction_probe = _eligible_prediction_rows(
        dataset=dataset,
        feature_cols=selected_features,
        target_col=spec.target_variant,
        predict_dates=[
            date
            for date in sorted(dataset["date"].dropna().unique())
            if pd.Timestamp(start_date) <= pd.Timestamp(date) <= pd.Timestamp(end_date)
        ][:10],
        min_history_rows=min_history_rows,
    ) if selected_features else dataset.iloc[0:0].copy()
    data_reasons: list[str] = []
    if len(selected_features) < 1:
        data_reasons.append("no_selected_features")
    if len(train_eligible) < min_train_eligible_rows:
        data_reasons.append("train_eligible_row_count below min_train_eligible_rows")
    if prediction_probe.empty:
        data_reasons.append("no_forward_prediction_probe_rows")
    invalid_reason = ";".join(reason_part for reason_part in [reason, *data_reasons] if reason_part)
    return {
        "selected_feature_count": len(selected_features),
        "train_eligible_row_count": len(train_eligible),
        "train_symbol_count": symbol_count(train_eligible),
        "prediction_probe_row_count": len(prediction_probe),
        "prediction_probe_symbol_count": symbol_count(prediction_probe),
        "min_symbols_per_prediction_probe_date": min_symbol_count_per_date(prediction_probe),
        "valid_config": structurally_valid and not data_reasons,
        "invalid_reason": invalid_reason,
    }


def _annualized_return(total_return: float, days: int) -> float:
    if days <= 0 or pd.isna(total_return) or total_return <= -1.0:
        return np.nan
    return float((1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / days) - 1.0)


def _return_to_drawdown(total_return: float, max_drawdown: float) -> float:
    if pd.isna(total_return) or pd.isna(max_drawdown) or abs(max_drawdown) <= 1.0e-12:
        return np.nan
    return float(total_return / abs(max_drawdown))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or abs(denominator) <= 1.0e-12:
        return np.nan
    return float(numerator / denominator)


def _slug(value: str) -> str:
    safe = [character.lower() if character.isalnum() else "-" for character in value]
    return "-".join("".join(safe).split("-")) or "unnamed"


__all__ = [
    "ALLOCATION_RULES",
    "END_DATE",
    "LONG_HISTORY_FEATURE_SET_VERSION",
    "LongHistoryProfile",
    "LongHistoryRunSpec",
    "LongHistoryTrainingOption",
    "MODEL_TYPES",
    "REBALANCE_FREQUENCIES",
    "START_DATE",
    "SWEEP_PROFILES",
    "SkippedLongHistoryRun",
    "TARGET_VARIANTS",
    "TRAINING_MODES",
    "TrainingWindow",
    "build_long_history_dataset",
    "build_long_history_run_key",
    "build_long_history_run_specs",
    "build_rebalance_weights",
    "build_training_windows",
    "buy_hold_metrics",
    "dataset_start_for_required_symbols",
    "evaluate_weight_schedule",
    "get_sweep_profile",
    "preflight_counts_for_spec",
    "run_long_history_model_zoo_spec",
    "run_spec_is_structurally_valid",
]
