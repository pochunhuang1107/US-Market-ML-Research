from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from us_market_ml.models import (  # noqa: E402
    MLPFitResult,
    MLPRegressorConfig,
    MLPRegressorModel,
    fit_mlp_regressor,
    predict_mlp,
    regression_metrics,
)


def _synthetic_regression_frame(row_count: int = 48) -> pd.DataFrame:
    feature_a = np.linspace(-1.0, 1.0, row_count)
    feature_b = np.cos(np.linspace(0.0, 2.0, row_count))
    target = 1.5 * feature_a - 0.25 * feature_b + 0.1
    return pd.DataFrame(
        {
            "feature_a": feature_a,
            "feature_b": feature_b,
            "target": target,
            "metadata": [f"row_{index}" for index in range(row_count)],
        }
    )


def _fast_config(**overrides: object) -> MLPRegressorConfig:
    params: dict[str, object] = {
        "hidden_units": (8,),
        "dropout": 0.0,
        "learning_rate": 0.01,
        "batch_size": 16,
        "max_epochs": 8,
        "patience": 3,
        "random_state": 7,
    }
    params.update(overrides)
    return MLPRegressorConfig(**params)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"hidden_units": ()}, "hidden_units"),
        ({"hidden_units": (8, 0)}, "hidden_units"),
        ({"dropout": -0.1}, "dropout"),
        ({"dropout": 1.0}, "dropout"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"batch_size": 0}, "batch_size"),
        ({"max_epochs": 0}, "max_epochs"),
        ({"patience": 0}, "patience"),
        ({"validation_fraction": 0.0}, "validation_fraction"),
        ({"validation_fraction": 1.0}, "validation_fraction"),
    ],
)
def test_mlp_config_validation(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _fast_config(**kwargs)


def test_fit_and_predict_mlp_do_not_warn_on_non_writable_numpy_inputs() -> None:
    values = np.linspace(-1.0, 1.0, 24)
    values.setflags(write=False)
    train_df = pd.DataFrame(
        {
            "feature_a": values,
            "feature_b": values * 0.5,
            "target": values * 1.25 + 0.2,
        }
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        model, _ = fit_mlp_regressor(
            train_df=train_df,
            target_col="target",
            feature_cols=["feature_a", "feature_b"],
            config=_fast_config(max_epochs=2),
        )
        predict_mlp(model, train_df, ["feature_a", "feature_b"])

    messages = [str(warning.message) for warning in caught_warnings]
    assert not any("NumPy array is not writable" in message for message in messages)


def test_fit_mlp_regressor_on_small_synthetic_data() -> None:
    train_df = _synthetic_regression_frame()

    model, result = fit_mlp_regressor(
        train_df=train_df,
        target_col="target",
        feature_cols=["feature_a", "feature_b"],
        config=_fast_config(),
    )

    assert model.feature_cols == ["feature_a", "feature_b"]
    assert model.scale_target is False
    assert model.target_mean is None
    assert model.target_std is None
    assert isinstance(result, MLPFitResult)
    assert result.target_col == "target"
    assert result.feature_cols == ["feature_a", "feature_b"]
    assert result.train_rows == len(train_df)
    assert result.validation_rows is None
    assert result.validation_metrics is None
    assert result.epochs_trained > 0
    assert result.best_validation_loss is None
    assert set(result.train_metrics) == {"mae", "rmse", "r2"}


def test_fit_mlp_regressor_returns_validation_metrics() -> None:
    df = _synthetic_regression_frame()
    train_df = df.iloc[:36].copy()
    validation_df = df.iloc[36:].copy()

    _, result = fit_mlp_regressor(
        train_df=train_df,
        target_col="target",
        feature_cols=["feature_a", "feature_b"],
        config=_fast_config(max_epochs=10),
        validation_df=validation_df,
    )

    assert result.train_rows == len(train_df)
    assert result.validation_rows == len(validation_df)
    assert result.validation_metrics is not None
    assert result.best_validation_loss is not None
    assert set(result.validation_metrics) == {"mae", "rmse", "r2"}


def test_fit_mlp_regressor_allows_no_usable_validation_rows() -> None:
    train_df = _synthetic_regression_frame()
    validation_df = pd.DataFrame(
        {
            "feature_a": [np.nan, 2.0],
            "feature_b": [1.0, np.nan],
            "target": [3.0, 4.0],
        }
    )

    _, result = fit_mlp_regressor(
        train_df=train_df,
        target_col="target",
        feature_cols=["feature_a", "feature_b"],
        config=_fast_config(),
        validation_df=validation_df,
    )

    assert result.validation_rows == 0
    assert result.validation_metrics is None
    assert result.best_validation_loss is None


def test_predict_mlp_preserves_row_count_and_adds_prediction_column() -> None:
    train_df = _synthetic_regression_frame()
    model, _ = fit_mlp_regressor(
        train_df,
        "target",
        ["feature_a", "feature_b"],
        config=_fast_config(),
    )
    predict_df = train_df[["feature_a", "feature_b", "metadata"]].copy()

    predictions = predict_mlp(model, predict_df, ["feature_a", "feature_b"])

    assert len(predictions) == len(predict_df)
    assert list(predictions.columns) == ["feature_a", "feature_b", "metadata", "prediction"]
    assert predictions["prediction"].notna().all()
    assert "prediction" not in predict_df.columns


def test_predict_mlp_leaves_missing_feature_rows_with_nan_predictions() -> None:
    train_df = _synthetic_regression_frame()
    model, _ = fit_mlp_regressor(
        train_df,
        "target",
        ["feature_a", "feature_b"],
        config=_fast_config(),
    )
    predict_df = train_df[["feature_a", "feature_b"]].copy()
    predict_df.loc[1, "feature_b"] = np.nan

    predictions = predict_mlp(model, predict_df, ["feature_a", "feature_b"])

    assert pd.isna(predictions.loc[1, "prediction"])
    assert predictions.loc[[0, 2], "prediction"].notna().all()


def test_target_scaling_stores_train_only_stats() -> None:
    train_df = pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, 2.0, 3.0],
            "feature_b": [1.0, 1.5, 2.0, 2.5],
            "target": [10.0, 12.0, 14.0, 16.0],
        }
    )
    validation_df = pd.DataFrame(
        {
            "feature_a": [4.0, 5.0],
            "feature_b": [3.0, 3.5],
            "target": [100.0, 200.0],
        }
    )

    model, _ = fit_mlp_regressor(
        train_df=train_df,
        target_col="target",
        feature_cols=["feature_a", "feature_b"],
        config=_fast_config(max_epochs=1, scale_target=True),
        validation_df=validation_df,
    )

    assert model.scale_target is True
    assert model.target_mean == pytest.approx(train_df["target"].mean())
    assert model.target_std == pytest.approx(train_df["target"].std(ddof=0))


def test_predict_mlp_inverse_transforms_scaled_predictions_to_original_units() -> None:
    class ConstantScaledModel(torch.nn.Module):  # type: ignore[misc]
        def forward(self, features: Any) -> Any:
            return torch.full((features.shape[0], 1), 2.0, device=features.device)

    model = MLPRegressorModel(
        torch_model=ConstantScaledModel(),
        feature_means=pd.Series({"feature_a": 0.0, "feature_b": 0.0}),
        feature_stds=pd.Series({"feature_a": 1.0, "feature_b": 1.0}),
        feature_cols=["feature_a", "feature_b"],
        config=_fast_config(scale_target=True),
        target_mean=10.0,
        target_std=5.0,
        scale_target=True,
    )
    predict_df = pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, 2.0],
            "feature_b": [0.0, np.nan, 2.0],
        }
    )

    predictions = predict_mlp(model, predict_df, ["feature_a", "feature_b"])

    assert predictions.loc[[0, 2], "prediction"].tolist() == pytest.approx([20.0, 20.0])
    assert pd.isna(predictions.loc[1, "prediction"])


def test_target_scaled_fit_metrics_are_in_original_target_units() -> None:
    train_df = _synthetic_regression_frame()

    model, result = fit_mlp_regressor(
        train_df=train_df,
        target_col="target",
        feature_cols=["feature_a", "feature_b"],
        config=_fast_config(max_epochs=3, scale_target=True),
    )

    predictions = predict_mlp(model, train_df, ["feature_a", "feature_b"])["prediction"]
    expected_metrics = regression_metrics(train_df["target"], predictions)
    assert result.train_metrics == pytest.approx(expected_metrics)


def test_constant_target_scaling_uses_unit_std() -> None:
    train_df = _synthetic_regression_frame()
    train_df["target"] = 0.25

    model, result = fit_mlp_regressor(
        train_df=train_df,
        target_col="target",
        feature_cols=["feature_a", "feature_b"],
        config=_fast_config(max_epochs=2, scale_target=True),
    )

    assert model.target_mean == pytest.approx(0.25)
    assert model.target_std == pytest.approx(1.0)
    assert set(result.train_metrics) == {"mae", "rmse", "r2"}


def test_fit_mlp_regressor_raises_for_missing_selected_columns() -> None:
    train_df = _synthetic_regression_frame()

    with pytest.raises(ValueError, match="missing_feature"):
        fit_mlp_regressor(train_df, "target", ["feature_a", "missing_feature"])

    with pytest.raises(ValueError, match="missing_target"):
        fit_mlp_regressor(train_df, "missing_target", ["feature_a", "feature_b"])


def test_predict_mlp_raises_for_missing_selected_columns() -> None:
    train_df = _synthetic_regression_frame()
    model, _ = fit_mlp_regressor(
        train_df,
        "target",
        ["feature_a", "feature_b"],
        config=_fast_config(),
    )

    with pytest.raises(ValueError, match="missing_feature"):
        predict_mlp(model, train_df, ["missing_feature"])


def test_fit_mlp_regressor_raises_when_no_training_rows_remain() -> None:
    train_df = pd.DataFrame(
        {
            "feature_a": [np.nan, 1.0],
            "feature_b": [1.0, np.nan],
            "target": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="No usable training rows"):
        fit_mlp_regressor(train_df, "target", ["feature_a", "feature_b"], config=_fast_config())
