"""Classical supervised regression baselines for ETF rotation research.

These baseline models are not trading strategies and do not generate ETF
rotation signals or backtest results. They train simple supervised regressors
on explicit feature columns and one selected target column so future neural
network experiments have a grounded comparison.

Callers should build ``feature_cols`` with dataset-builder helpers such as
``get_feature_columns`` to avoid target leakage. Walk-forward split utilities
should be applied before calling these fitting helpers so training,
validation, and test rows remain chronological.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class BaselineModelConfig:
    """Configuration for classical supervised baseline regressors."""

    model_type: str = "ridge"
    random_state: int = 42
    alpha: float = 1.0
    n_estimators: int = 100
    max_depth: int | None = None
    learning_rate: float = 0.05
    subsample: float = 1.0
    min_samples_leaf: int = 1
    max_features: str | float | None = None

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0.")
        if not 0 < self.subsample <= 1:
            raise ValueError("subsample must be greater than 0 and less than or equal to 1.")
        if self.min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be greater than 0.")


@dataclass(frozen=True)
class BaselineFitResult:
    """Summary of a baseline supervised regression fit.

    The result describes model fitting only. It is not a trading signal,
    portfolio construction result, or backtest artifact.
    """

    model_type: str
    target_col: str
    feature_cols: list[str]
    train_rows: int
    validation_rows: int | None
    train_metrics: dict[str, float]
    validation_metrics: dict[str, float] | None


def create_baseline_model(config: BaselineModelConfig) -> Any:
    """Create an unfitted classical baseline regressor from ``config``."""

    if config.model_type == "ridge":
        return Ridge(alpha=config.alpha, random_state=config.random_state)
    if config.model_type == "linear_regression":
        return LinearRegression()
    if config.model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            random_state=config.random_state,
        )
    if config.model_type == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth if config.max_depth is not None else 3,
            subsample=config.subsample,
            min_samples_leaf=config.min_samples_leaf,
            random_state=config.random_state,
        )
    if config.model_type == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            max_features=config.max_features,
            random_state=config.random_state,
            n_jobs=-1,
        )

    supported = "ridge, linear_regression, random_forest, gradient_boosting, extra_trees"
    raise ValueError(
        f"Unsupported baseline model_type {config.model_type!r}; expected one of: {supported}."
    )


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return standard regression metrics for supervised baseline evaluation."""

    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def fit_baseline_regressor(
    train_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    config: BaselineModelConfig | None = None,
    validation_df: pd.DataFrame | None = None,
) -> tuple[Any, BaselineFitResult]:
    """Fit a supervised baseline regressor on explicit model columns.

    Rows with missing selected feature or target values are dropped explicitly
    before fitting. Callers should pass walk-forward train and validation
    frames that were split chronologically before this function is called.
    ``feature_cols`` should come from dataset-builder helpers to avoid target
    leakage from forward-return, excess-return, or rank target columns.
    """

    resolved_config = config if config is not None else BaselineModelConfig()
    _validate_model_columns(train_df, feature_cols, target_col)

    train_model_df = _drop_missing_model_rows(train_df, feature_cols, target_col)
    if train_model_df.empty:
        raise ValueError("No usable training rows remain after dropping missing model values.")

    model = create_baseline_model(resolved_config)
    model.fit(train_model_df[feature_cols], train_model_df[target_col])

    train_predictions = model.predict(train_model_df[feature_cols])
    train_metrics = regression_metrics(train_model_df[target_col], train_predictions)

    validation_rows: int | None = None
    validation_metrics: dict[str, float] | None = None
    if validation_df is not None:
        _validate_model_columns(validation_df, feature_cols, target_col)
        validation_model_df = _drop_missing_model_rows(validation_df, feature_cols, target_col)
        validation_rows = len(validation_model_df)
        if validation_rows > 0:
            validation_predictions = model.predict(validation_model_df[feature_cols])
            validation_metrics = regression_metrics(
                validation_model_df[target_col],
                validation_predictions,
            )

    result = BaselineFitResult(
        model_type=resolved_config.model_type,
        target_col=target_col,
        feature_cols=list(feature_cols),
        train_rows=len(train_model_df),
        validation_rows=validation_rows,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
    )
    return model, result


def predict_baseline(
    model: Any,
    df: pd.DataFrame,
    feature_cols: list[str],
    prediction_col: str = "prediction",
) -> pd.DataFrame:
    """Return ``df`` with model predictions for rows with complete features.

    Missing-feature rows are retained and receive ``NaN`` in
    ``prediction_col`` so the caller's row alignment is preserved for later
    signal or backtest work.
    """

    _validate_feature_columns(df, feature_cols)

    out = df.copy()
    out[prediction_col] = np.nan
    usable_mask = out[feature_cols].notna().all(axis=1)
    if usable_mask.any():
        out.loc[usable_mask, prediction_col] = model.predict(out.loc[usable_mask, feature_cols])
    return out


def _drop_missing_model_rows(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> pd.DataFrame:
    return df.dropna(subset=[*feature_cols, target_col]).copy()


def _validate_model_columns(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> None:
    _validate_feature_columns(df, feature_cols)
    missing_columns = [target_col] if target_col not in df.columns else []
    if missing_columns:
        raise ValueError(f"Input DataFrame is missing selected model columns: {target_col}")


def _validate_feature_columns(df: pd.DataFrame, feature_cols: list[str]) -> None:
    if not feature_cols:
        raise ValueError("At least one feature column must be selected.")

    missing_columns = [column for column in feature_cols if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Input DataFrame is missing selected model columns: {missing}")


__all__ = [
    "BaselineFitResult",
    "BaselineModelConfig",
    "create_baseline_model",
    "fit_baseline_regressor",
    "predict_baseline",
    "regression_metrics",
]
