"""Baseline supervised experiment runner for ETF research.

This module connects supervised dataset construction, walk-forward splitting,
classical baseline regression fitting, and prediction output. It is a
supervised model experiment runner, not a trading strategy: generated
predictions are model scores only, not buy/sell signals. Signal generation,
portfolio construction, and backtesting belong to separate research layers.

Walk-forward folds are required so training, validation, and test windows stay
chronological and avoid leakage from future market rows into earlier model
selection or fitting steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from us_market_ml.data import (
    DEFAULT_FEATURE_SET_VERSION,
    SupervisedDatasetConfig,
    build_supervised_dataset,
    get_feature_columns,
)
from us_market_ml.models import (
    BaselineFitResult,
    BaselineModelConfig,
    fit_baseline_regressor,
    predict_baseline,
)
from us_market_ml.splits import PurgeConfig, WalkForwardFold, purge_fold_boundaries
from us_market_ml.universe import (
    DynamicEligibilityConfig,
    apply_dynamic_universe_eligibility,
    select_fold_train_feature_columns,
)


@dataclass(frozen=True)
class BaselineExperimentConfig:
    """Configuration for baseline supervised regression experiments.

    The selected target is a supervised label, and the resulting predictions
    are model scores only. They should not be interpreted as executable
    buy/sell signals without an explicit signal-generation and backtesting
    layer.
    """

    target_col: str = "excess_return_vs_spy_20d"
    model_config: BaselineModelConfig = field(default_factory=BaselineModelConfig)
    dataset_config: SupervisedDatasetConfig = field(default_factory=SupervisedDatasetConfig)
    feature_set_version: str = DEFAULT_FEATURE_SET_VERSION
    prediction_col: str = "prediction"
    extra_exclude_feature_cols: set[str] | None = None
    purge_config: PurgeConfig | None = None
    universe_mode: str = "static"
    min_history_rows: int = 252
    feature_selection_scope: str = "global"
    min_feature_non_null_ratio: float = 0.0


@dataclass(frozen=True)
class BaselineFoldResult:
    """Model fit and aligned prediction frames for one walk-forward fold."""

    fold_name: str
    fit_result: BaselineFitResult
    train_predictions: pd.DataFrame
    validation_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    feature_cols: list[str] | None = None


@dataclass(frozen=True)
class BaselineExperimentResult:
    """Complete output from a baseline supervised walk-forward experiment."""

    config: BaselineExperimentConfig
    dataset: pd.DataFrame
    feature_cols: list[str]
    fold_results: list[BaselineFoldResult]


def run_baseline_experiment(
    raw_df: pd.DataFrame,
    folds: list[WalkForwardFold],
    config: BaselineExperimentConfig | None = None,
) -> BaselineExperimentResult:
    """Run a supervised baseline regressor across walk-forward folds.

    The runner builds a supervised dataset, derives leakage-safe feature
    columns, fits a classical baseline model on each fold's training rows, and
    returns train, validation, and test prediction frames. Predictions are
    model scores only, not trading signals. Signal generation, portfolio
    construction, and backtesting are intentionally outside this runner.

    Walk-forward folds are required to keep model fitting and validation
    chronological, which avoids leakage that random splits would introduce for
    time-ordered market data and future-looking supervised labels.
    """

    if not folds:
        raise ValueError("At least one walk-forward fold is required.")

    resolved_config = config if config is not None else BaselineExperimentConfig()
    _validate_experiment_config(resolved_config)

    feature_set_version = _resolve_feature_set_version(resolved_config)
    dataset_config = replace(
        resolved_config.dataset_config,
        feature_set_version=feature_set_version,
    )
    dataset = build_supervised_dataset(raw_df, dataset_config)
    candidate_feature_cols = get_feature_columns(
        dataset,
        feature_set_version=feature_set_version,
        extra_exclude=resolved_config.extra_exclude_feature_cols,
    )

    if resolved_config.target_col not in dataset.columns:
        raise ValueError(
            "Configured target_col is missing after supervised dataset construction: "
            f"{resolved_config.target_col}"
        )
    if not candidate_feature_cols:
        raise ValueError("No usable feature columns were found for baseline model fitting.")

    fold_results: list[BaselineFoldResult] = []
    selected_feature_cols_by_fold: list[list[str]] = []
    for fold in folds:
        if resolved_config.purge_config is None:
            train_df, validation_df, test_df = fold.split(dataset)
        else:
            train_df, validation_df, test_df = purge_fold_boundaries(
                dataset,
                fold,
                resolved_config.purge_config,
            )
        feature_cols = _feature_columns_for_fold(
            train_df=train_df,
            candidate_feature_cols=candidate_feature_cols,
            config=resolved_config,
        )
        if not feature_cols:
            raise ValueError("No usable feature columns were found for baseline model fitting.")

        train_df, validation_df, test_df = _apply_universe_mode(
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
            eligibility_history_df=dataset,
            feature_cols=feature_cols,
            config=resolved_config,
        )
        model, fit_result = fit_baseline_regressor(
            train_df=train_df,
            target_col=resolved_config.target_col,
            feature_cols=feature_cols,
            config=resolved_config.model_config,
            validation_df=validation_df,
        )
        selected_feature_cols_by_fold.append(feature_cols)

        fold_results.append(
            BaselineFoldResult(
                fold_name=fold.name,
                fit_result=fit_result,
                train_predictions=predict_baseline(
                    model,
                    train_df,
                    feature_cols,
                    prediction_col=resolved_config.prediction_col,
                ),
                validation_predictions=predict_baseline(
                    model,
                    validation_df,
                    feature_cols,
                    prediction_col=resolved_config.prediction_col,
                ),
                test_predictions=predict_baseline(
                    model,
                    test_df,
                    feature_cols,
                    prediction_col=resolved_config.prediction_col,
                ),
                feature_cols=list(feature_cols),
            )
        )

    return BaselineExperimentResult(
        config=resolved_config,
        dataset=dataset,
        feature_cols=_result_feature_columns(
            candidate_feature_cols=candidate_feature_cols,
            selected_feature_cols_by_fold=selected_feature_cols_by_fold,
            config=resolved_config,
        ),
        fold_results=fold_results,
    )


def summarize_baseline_experiment(result: BaselineExperimentResult) -> pd.DataFrame:
    """Return fold-level supervised regression fit and validation metrics.

    The summary describes model fitting quality only. It does not contain
    trading signals, portfolio returns, or backtest statistics; those belong to
    separate layers built on top of model-score predictions.
    """

    rows: list[dict[str, object]] = []
    for fold_result in result.fold_results:
        fit_result = fold_result.fit_result
        validation_metrics = fit_result.validation_metrics
        rows.append(
            {
                "fold_name": fold_result.fold_name,
                "model_type": fit_result.model_type,
                "target_col": fit_result.target_col,
                "train_rows": fit_result.train_rows,
                "validation_rows": fit_result.validation_rows,
                "train_mae": fit_result.train_metrics["mae"],
                "train_rmse": fit_result.train_metrics["rmse"],
                "train_r2": fit_result.train_metrics["r2"],
                "validation_mae": (
                    validation_metrics["mae"] if validation_metrics is not None else np.nan
                ),
                "validation_rmse": (
                    validation_metrics["rmse"] if validation_metrics is not None else np.nan
                ),
                "validation_r2": (
                    validation_metrics["r2"] if validation_metrics is not None else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def _resolve_feature_set_version(config: BaselineExperimentConfig) -> str:
    if config.feature_set_version != DEFAULT_FEATURE_SET_VERSION:
        return config.feature_set_version
    return config.dataset_config.feature_set_version


def _validate_experiment_config(config: BaselineExperimentConfig) -> None:
    if config.universe_mode not in {"static", "dynamic"}:
        raise ValueError("universe_mode must be 'static' or 'dynamic'.")
    if config.min_history_rows < 1:
        raise ValueError("min_history_rows must be greater than or equal to 1.")
    if config.feature_selection_scope not in {"global", "fold_train"}:
        raise ValueError("feature_selection_scope must be 'global' or 'fold_train'.")
    if config.min_feature_non_null_ratio < 0.0 or config.min_feature_non_null_ratio > 1.0:
        raise ValueError("min_feature_non_null_ratio must satisfy 0 <= value <= 1.")


def _feature_columns_for_fold(
    *,
    train_df: pd.DataFrame,
    candidate_feature_cols: list[str],
    config: BaselineExperimentConfig,
) -> list[str]:
    if config.feature_selection_scope == "global":
        return list(candidate_feature_cols)
    return select_fold_train_feature_columns(
        train_df,
        candidate_feature_cols,
        min_non_null_ratio=config.min_feature_non_null_ratio,
    )


def _apply_universe_mode(
    *,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    eligibility_history_df: pd.DataFrame,
    feature_cols: list[str],
    config: BaselineExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if config.universe_mode == "static":
        return train_df, validation_df, test_df

    eligibility_config = DynamicEligibilityConfig(min_history_rows=config.min_history_rows)
    return (
        apply_dynamic_universe_eligibility(
            train_df,
            feature_cols=feature_cols,
            target_col=config.target_col,
            require_target=True,
            config=eligibility_config,
            history_df=eligibility_history_df,
        ),
        apply_dynamic_universe_eligibility(
            validation_df,
            feature_cols=feature_cols,
            target_col=config.target_col,
            require_target=False,
            config=eligibility_config,
            history_df=eligibility_history_df,
        ),
        apply_dynamic_universe_eligibility(
            test_df,
            feature_cols=feature_cols,
            target_col=config.target_col,
            require_target=False,
            config=eligibility_config,
            history_df=eligibility_history_df,
        ),
    )


def _result_feature_columns(
    *,
    candidate_feature_cols: list[str],
    selected_feature_cols_by_fold: list[list[str]],
    config: BaselineExperimentConfig,
) -> list[str]:
    if config.feature_selection_scope == "global":
        return list(candidate_feature_cols)
    selected = {
        column for feature_cols in selected_feature_cols_by_fold for column in feature_cols
    }
    return [column for column in candidate_feature_cols if column in selected]


__all__ = [
    "BaselineExperimentConfig",
    "BaselineExperimentResult",
    "BaselineFoldResult",
    "run_baseline_experiment",
    "summarize_baseline_experiment",
]
