"""MLP supervised experiment runner for ETF research.

This module connects supervised dataset construction, walk-forward splitting,
tabular MLP regression fitting, and prediction output. It is a supervised model
experiment runner, not a trading strategy: generated predictions are model
scores only, not buy/sell signals. MLP signal generation, portfolio
construction, and backtesting integration are intentionally left for a future
E2E research layer.

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
    MLPFitResult,
    MLPRegressorConfig,
    fit_mlp_regressor,
    predict_mlp,
)
from us_market_ml.splits import PurgeConfig, WalkForwardFold, purge_fold_boundaries


@dataclass(frozen=True)
class MLPExperimentConfig:
    """Configuration for supervised tabular MLP regression experiments.

    The selected target is a supervised label, and the resulting predictions
    are model scores only. They should not be interpreted as executable
    buy/sell signals without an explicit signal-generation and backtesting
    layer. MLP E2E signal/backtest integration is a future task.
    """

    target_col: str = "excess_return_vs_spy_20d"
    model_config: MLPRegressorConfig = field(default_factory=MLPRegressorConfig)
    dataset_config: SupervisedDatasetConfig = field(default_factory=SupervisedDatasetConfig)
    prediction_col: str = "prediction"
    extra_exclude_feature_cols: set[str] | None = None
    purge_config: PurgeConfig | None = None


@dataclass(frozen=True)
class MLPFoldResult:
    """MLP fit and aligned prediction frames for one walk-forward fold."""

    fold_name: str
    fit_result: MLPFitResult
    train_predictions: pd.DataFrame
    validation_predictions: pd.DataFrame
    test_predictions: pd.DataFrame


@dataclass(frozen=True)
class MLPExperimentResult:
    """Complete output from an MLP supervised walk-forward experiment."""

    config: MLPExperimentConfig
    dataset: pd.DataFrame
    feature_cols: list[str]
    fold_results: list[MLPFoldResult]


def run_mlp_experiment(
    raw_df: pd.DataFrame,
    folds: list[WalkForwardFold],
    config: MLPExperimentConfig | None = None,
) -> MLPExperimentResult:
    """Run a supervised tabular MLP regressor across walk-forward folds.

    The runner builds a supervised dataset, derives leakage-safe feature
    columns, fits an MLP model on each fold's training rows, uses validation
    rows only for model selection/early stopping, and returns train,
    validation, and test prediction frames. Predictions are model scores only,
    not trading signals. Signal generation, portfolio construction, and
    backtesting are intentionally outside this runner and will be integrated in
    a future MLP E2E task.

    Walk-forward folds are required to keep model fitting and validation
    chronological, which avoids leakage that random splits would introduce for
    time-ordered market data and future-looking supervised labels.
    """

    if not folds:
        raise ValueError("At least one walk-forward fold is required.")

    resolved_config = config if config is not None else MLPExperimentConfig()

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
        raise ValueError("No usable feature columns were found for MLP model fitting.")

    fold_results: list[MLPFoldResult] = []
    for fold in folds:
        if resolved_config.purge_config is None:
            train_df, validation_df, test_df = fold.split(dataset)
        else:
            train_df, validation_df, test_df = purge_fold_boundaries(
                dataset,
                fold,
                resolved_config.purge_config,
            )
        model, fit_result = fit_mlp_regressor(
            train_df=train_df,
            target_col=resolved_config.target_col,
            feature_cols=feature_cols,
            config=resolved_config.model_config,
            validation_df=validation_df,
        )

        fold_results.append(
            MLPFoldResult(
                fold_name=fold.name,
                fit_result=fit_result,
                train_predictions=predict_mlp(
                    model,
                    train_df,
                    feature_cols,
                    prediction_col=resolved_config.prediction_col,
                ),
                validation_predictions=predict_mlp(
                    model,
                    validation_df,
                    feature_cols,
                    prediction_col=resolved_config.prediction_col,
                ),
                test_predictions=predict_mlp(
                    model,
                    test_df,
                    feature_cols,
                    prediction_col=resolved_config.prediction_col,
                ),
            )
        )

    return MLPExperimentResult(
        config=resolved_config,
        dataset=dataset,
        feature_cols=feature_cols,
        fold_results=fold_results,
    )


def summarize_mlp_experiment(result: MLPExperimentResult) -> pd.DataFrame:
    """Return fold-level supervised MLP fit and validation metrics.

    The summary describes model fitting quality only. It does not contain
    trading signals, portfolio returns, or backtest statistics; those belong to
    a future MLP E2E signal/backtest integration layer.
    """

    rows: list[dict[str, object]] = []
    for fold_result in result.fold_results:
        fit_result = fold_result.fit_result
        validation_metrics = fit_result.validation_metrics
        rows.append(
            {
                "fold_name": fold_result.fold_name,
                "target_col": fit_result.target_col,
                "train_rows": fit_result.train_rows,
                "validation_rows": fit_result.validation_rows,
                "epochs_trained": fit_result.epochs_trained,
                "best_validation_loss": (
                    fit_result.best_validation_loss
                    if fit_result.best_validation_loss is not None
                    else np.nan
                ),
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
    "MLPExperimentConfig",
    "MLPExperimentResult",
    "MLPFoldResult",
    "run_mlp_experiment",
    "summarize_mlp_experiment",
]
