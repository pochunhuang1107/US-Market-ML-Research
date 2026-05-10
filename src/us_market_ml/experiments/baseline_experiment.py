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

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from us_market_ml.data import (
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
from us_market_ml.splits import WalkForwardFold


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
    prediction_col: str = "prediction"
    extra_exclude_feature_cols: set[str] | None = None


@dataclass(frozen=True)
class BaselineFoldResult:
    """Model fit and aligned prediction frames for one walk-forward fold."""

    fold_name: str
    fit_result: BaselineFitResult
    train_predictions: pd.DataFrame
    validation_predictions: pd.DataFrame
    test_predictions: pd.DataFrame


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

    dataset = build_supervised_dataset(raw_df, resolved_config.dataset_config)
    feature_cols = get_feature_columns(
        dataset,
        extra_exclude=resolved_config.extra_exclude_feature_cols,
    )

    if resolved_config.target_col not in dataset.columns:
        raise ValueError(
            "Configured target_col is missing after supervised dataset construction: "
            f"{resolved_config.target_col}"
        )
    if not feature_cols:
        raise ValueError("No usable feature columns were found for baseline model fitting.")

    fold_results: list[BaselineFoldResult] = []
    for fold in folds:
        train_df, validation_df, test_df = fold.split(dataset)
        model, fit_result = fit_baseline_regressor(
            train_df=train_df,
            target_col=resolved_config.target_col,
            feature_cols=feature_cols,
            config=resolved_config.model_config,
            validation_df=validation_df,
        )

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
            )
        )

    return BaselineExperimentResult(
        config=resolved_config,
        dataset=dataset,
        feature_cols=feature_cols,
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


__all__ = [
    "BaselineExperimentConfig",
    "BaselineExperimentResult",
    "BaselineFoldResult",
    "run_baseline_experiment",
    "summarize_baseline_experiment",
]
