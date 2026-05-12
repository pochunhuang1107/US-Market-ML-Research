from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from us_market_ml.models import (
    BaselineFitResult,
    BaselineModelConfig,
    create_baseline_model,
    fit_baseline_regressor,
    predict_baseline,
    regression_metrics,
)


def _clean_regression_frame() -> pd.DataFrame:
    feature_a = np.arange(1.0, 9.0)
    feature_b = feature_a * 0.5
    return pd.DataFrame(
        {
            "feature_a": feature_a,
            "feature_b": feature_b,
            "target": 2.0 * feature_a - feature_b + 1.0,
            "metadata": [f"row_{index}" for index in range(len(feature_a))],
        }
    )


def test_create_baseline_model_supports_all_model_types() -> None:
    assert isinstance(create_baseline_model(BaselineModelConfig(model_type="ridge")), Ridge)
    assert isinstance(
        create_baseline_model(BaselineModelConfig(model_type="linear_regression")),
        LinearRegression,
    )
    assert isinstance(
        create_baseline_model(BaselineModelConfig(model_type="random_forest")),
        RandomForestRegressor,
    )
    assert isinstance(
        create_baseline_model(BaselineModelConfig(model_type="gradient_boosting")),
        GradientBoostingRegressor,
    )
    assert isinstance(
        create_baseline_model(BaselineModelConfig(model_type="extra_trees")),
        ExtraTreesRegressor,
    )


def test_create_baseline_model_unsupported_type_raises_value_error() -> None:
    with pytest.raises(ValueError, match="gradient_boosting, extra_trees"):
        create_baseline_model(BaselineModelConfig(model_type="xgboost"))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"learning_rate": -0.1}, "learning_rate"),
        ({"subsample": 0.0}, "subsample"),
        ({"subsample": 1.1}, "subsample"),
        ({"min_samples_leaf": 0}, "min_samples_leaf"),
        ({"min_samples_leaf": -1}, "min_samples_leaf"),
    ],
)
def test_baseline_model_config_rejects_invalid_tree_hyperparameters(
    kwargs: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        BaselineModelConfig(**kwargs)


def test_regression_metrics_returns_mae_rmse_and_r2() -> None:
    metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.5, 2.5])

    assert set(metrics) == {"mae", "rmse", "r2"}
    assert metrics["mae"] == pytest.approx(1.0 / 3.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt((0.0**2 + 0.5**2 + 0.5**2) / 3.0))
    assert metrics["r2"] == pytest.approx(0.75)


def test_fit_baseline_regressor_fits_ridge_on_clean_synthetic_data() -> None:
    train_df = _clean_regression_frame()

    model, result = fit_baseline_regressor(
        train_df=train_df,
        target_col="target",
        feature_cols=["feature_a", "feature_b"],
        config=BaselineModelConfig(model_type="ridge", alpha=0.1),
    )

    assert isinstance(model, Ridge)
    assert isinstance(result, BaselineFitResult)
    assert result.model_type == "ridge"
    assert result.target_col == "target"
    assert result.feature_cols == ["feature_a", "feature_b"]
    assert result.train_rows == len(train_df)
    assert result.validation_rows is None
    assert result.validation_metrics is None
    assert set(result.train_metrics) == {"mae", "rmse", "r2"}


@pytest.mark.parametrize(
    ("model_type", "expected_type"),
    [
        ("gradient_boosting", GradientBoostingRegressor),
        ("extra_trees", ExtraTreesRegressor),
    ],
)
def test_fit_baseline_regressor_fits_tree_baselines_on_clean_synthetic_data(
    model_type: str,
    expected_type: type[GradientBoostingRegressor | ExtraTreesRegressor],
) -> None:
    train_df = _clean_regression_frame()

    model, result = fit_baseline_regressor(
        train_df=train_df,
        target_col="target",
        feature_cols=["feature_a", "feature_b"],
        config=BaselineModelConfig(model_type=model_type, n_estimators=10),
    )

    assert isinstance(model, expected_type)
    assert result.model_type == model_type
    assert result.train_rows == len(train_df)
    assert set(result.train_metrics) == {"mae", "rmse", "r2"}


def test_fit_baseline_regressor_drops_missing_model_rows_explicitly() -> None:
    train_df = _clean_regression_frame()
    train_df.loc[0, "feature_a"] = np.nan
    train_df.loc[1, "target"] = np.nan
    train_df.loc[2, "metadata"] = np.nan
    original = train_df.copy(deep=True)

    _, result = fit_baseline_regressor(
        train_df=train_df,
        target_col="target",
        feature_cols=["feature_a", "feature_b"],
    )

    assert result.train_rows == len(train_df) - 2
    pd.testing.assert_frame_equal(train_df, original)


def test_fit_baseline_regressor_returns_validation_metrics() -> None:
    train_df = _clean_regression_frame().iloc[:6].copy()
    validation_df = _clean_regression_frame().iloc[6:].copy()

    _, result = fit_baseline_regressor(
        train_df=train_df,
        target_col="target",
        feature_cols=["feature_a", "feature_b"],
        validation_df=validation_df,
    )

    assert result.validation_rows == len(validation_df)
    assert result.validation_metrics is not None
    assert set(result.validation_metrics) == {"mae", "rmse", "r2"}


def test_fit_baseline_regressor_validation_metrics_none_when_no_usable_validation_rows() -> None:
    train_df = _clean_regression_frame()
    validation_df = pd.DataFrame(
        {
            "feature_a": [np.nan, 2.0],
            "feature_b": [1.0, np.nan],
            "target": [3.0, 4.0],
        }
    )

    _, result = fit_baseline_regressor(
        train_df=train_df,
        target_col="target",
        feature_cols=["feature_a", "feature_b"],
        validation_df=validation_df,
    )

    assert result.validation_rows == 0
    assert result.validation_metrics is None


def test_fit_baseline_regressor_raises_when_no_training_rows_remain() -> None:
    train_df = pd.DataFrame(
        {
            "feature_a": [np.nan, 1.0],
            "feature_b": [1.0, np.nan],
            "target": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="No usable training rows"):
        fit_baseline_regressor(train_df, "target", ["feature_a", "feature_b"])


def test_predict_baseline_preserves_row_count_and_adds_prediction_column() -> None:
    train_df = _clean_regression_frame()
    model, _ = fit_baseline_regressor(train_df, "target", ["feature_a", "feature_b"])
    predict_df = train_df[["feature_a", "feature_b", "metadata"]].copy()

    predictions = predict_baseline(model, predict_df, ["feature_a", "feature_b"])

    assert len(predictions) == len(predict_df)
    assert list(predictions.columns) == ["feature_a", "feature_b", "metadata", "prediction"]
    assert predictions["prediction"].notna().all()
    assert "prediction" not in predict_df.columns


@pytest.mark.parametrize("model_type", ["gradient_boosting", "extra_trees"])
def test_predict_baseline_works_with_tree_baselines(model_type: str) -> None:
    train_df = _clean_regression_frame()
    model, _ = fit_baseline_regressor(
        train_df,
        "target",
        ["feature_a", "feature_b"],
        BaselineModelConfig(model_type=model_type, n_estimators=10),
    )
    predict_df = train_df[["feature_a", "feature_b"]].copy()

    predictions = predict_baseline(model, predict_df, ["feature_a", "feature_b"])

    assert predictions["prediction"].notna().all()


def test_predict_baseline_leaves_missing_feature_rows_with_nan_predictions() -> None:
    train_df = _clean_regression_frame()
    model, _ = fit_baseline_regressor(train_df, "target", ["feature_a", "feature_b"])
    predict_df = train_df[["feature_a", "feature_b"]].copy()
    predict_df.loc[1, "feature_b"] = np.nan

    predictions = predict_baseline(model, predict_df, ["feature_a", "feature_b"])

    assert pd.isna(predictions.loc[1, "prediction"])
    assert predictions.loc[[0, 2], "prediction"].notna().all()


def test_selected_missing_columns_raise_value_error() -> None:
    train_df = _clean_regression_frame()

    with pytest.raises(ValueError, match="missing_feature"):
        fit_baseline_regressor(train_df, "target", ["feature_a", "missing_feature"])

    with pytest.raises(ValueError, match="missing_target"):
        fit_baseline_regressor(train_df, "missing_target", ["feature_a", "feature_b"])

    model, _ = fit_baseline_regressor(train_df, "target", ["feature_a", "feature_b"])
    with pytest.raises(ValueError, match="missing_feature"):
        predict_baseline(model, train_df, ["missing_feature"])
