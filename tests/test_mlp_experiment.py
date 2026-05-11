from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("torch")

from us_market_ml.data import SupervisedDatasetConfig  # noqa: E402
from us_market_ml.experiments import (  # noqa: E402
    MLPExperimentConfig,
    MLPExperimentResult,
    MLPFoldResult,
    run_mlp_experiment,
    summarize_mlp_experiment,
)
from us_market_ml.models import MLPFitResult, MLPRegressorConfig  # noqa: E402
from us_market_ml.splits import WalkForwardFold  # noqa: E402


def _synthetic_etf_ohlcv() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2020-01-01", periods=620, freq="D")
    symbols = ("SPY", "QQQ", "TLT")

    for symbol_index, symbol in enumerate(symbols):
        base_price = 100.0 + symbol_index * 25.0
        drift = 0.09 + symbol_index * 0.02
        cycle = symbol_index + 2
        for day_index, date in enumerate(dates):
            close = base_price + day_index * drift + (day_index % 17) * cycle * 0.03
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close - 0.25,
                    "high": close + 0.75,
                    "low": close - 0.8,
                    "close": close,
                    "volume": 1_000_000 + symbol_index * 80_000 + day_index * 1_500,
                    "vwap": close - 0.05,
                    "year": date.year,
                    "month": date.month,
                    "transactions": 10_000 + day_index,
                }
            )

    return pd.DataFrame(rows)


def _folds() -> list[WalkForwardFold]:
    return [
        WalkForwardFold(
            name="fold_1",
            train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2020-12-31"),
            validation_start=pd.Timestamp("2021-01-01"),
            validation_end=pd.Timestamp("2021-03-31"),
            test_start=pd.Timestamp("2021-04-01"),
            test_end=pd.Timestamp("2021-06-30"),
        ),
        WalkForwardFold(
            name="fold_2",
            train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2021-03-31"),
            validation_start=pd.Timestamp("2021-04-01"),
            validation_end=pd.Timestamp("2021-06-30"),
            test_start=pd.Timestamp("2021-07-01"),
            test_end=pd.Timestamp("2021-09-11"),
        ),
    ]


def _config(
    target_col: str = "excess_return_vs_spy_20d",
    prediction_col: str = "prediction",
    extra_exclude_feature_cols: set[str] | None = None,
) -> MLPExperimentConfig:
    return MLPExperimentConfig(
        target_col=target_col,
        model_config=MLPRegressorConfig(
            hidden_units=(16,),
            dropout=0.0,
            max_epochs=5,
            patience=2,
            batch_size=16,
            random_state=42,
        ),
        dataset_config=SupervisedDatasetConfig(),
        prediction_col=prediction_col,
        extra_exclude_feature_cols=extra_exclude_feature_cols,
    )


def test_run_mlp_experiment_returns_dataset_features_and_fold_results() -> None:
    result = run_mlp_experiment(_synthetic_etf_ohlcv(), _folds(), _config())

    assert isinstance(result, MLPExperimentResult)
    assert len(result.dataset) == 620 * 3
    assert result.feature_cols
    assert len(result.fold_results) == 2
    assert all(isinstance(fold_result, MLPFoldResult) for fold_result in result.fold_results)


def test_fold_result_contains_train_validation_and_test_predictions() -> None:
    result = run_mlp_experiment(_synthetic_etf_ohlcv(), _folds()[:1], _config())
    fold_result = result.fold_results[0]

    assert fold_result.fold_name == "fold_1"
    assert not fold_result.train_predictions.empty
    assert not fold_result.validation_predictions.empty
    assert not fold_result.test_predictions.empty


def test_prediction_outputs_preserve_split_row_counts() -> None:
    raw_df = _synthetic_etf_ohlcv()
    folds = _folds()
    result = run_mlp_experiment(raw_df, folds, _config())
    dataset = result.dataset

    for fold, fold_result in zip(folds, result.fold_results, strict=True):
        train_df, validation_df, test_df = fold.split(dataset)
        assert len(fold_result.train_predictions) == len(train_df)
        assert len(fold_result.validation_predictions) == len(validation_df)
        assert len(fold_result.test_predictions) == len(test_df)


def test_prediction_columns_are_present_on_all_split_outputs() -> None:
    config = _config(prediction_col="mlp_score")

    result = run_mlp_experiment(_synthetic_etf_ohlcv(), _folds()[:1], config)
    fold_result = result.fold_results[0]

    assert "mlp_score" in fold_result.train_predictions.columns
    assert "mlp_score" in fold_result.validation_predictions.columns
    assert "mlp_score" in fold_result.test_predictions.columns


def test_summary_output_has_one_row_per_fold() -> None:
    result = run_mlp_experiment(_synthetic_etf_ohlcv(), _folds(), _config())

    summary = summarize_mlp_experiment(result)

    assert len(summary) == len(result.fold_results)
    assert summary["fold_name"].tolist() == ["fold_1", "fold_2"]
    assert {
        "target_col",
        "train_rows",
        "validation_rows",
        "epochs_trained",
        "best_validation_loss",
        "train_mae",
        "train_rmse",
        "train_r2",
        "validation_mae",
        "validation_rmse",
        "validation_r2",
    }.issubset(summary.columns)


def test_summary_uses_nan_when_validation_metrics_are_missing() -> None:
    result = MLPExperimentResult(
        config=_config(),
        dataset=pd.DataFrame(),
        feature_cols=["feature_a"],
        fold_results=[
            MLPFoldResult(
                fold_name="fold_without_validation_metrics",
                fit_result=MLPFitResult(
                    target_col="target",
                    feature_cols=["feature_a"],
                    train_rows=10,
                    validation_rows=0,
                    train_metrics={"mae": 0.1, "rmse": 0.2, "r2": 0.3},
                    validation_metrics=None,
                    epochs_trained=3,
                    best_validation_loss=None,
                ),
                train_predictions=pd.DataFrame(),
                validation_predictions=pd.DataFrame(),
                test_predictions=pd.DataFrame(),
            )
        ],
    )

    summary = summarize_mlp_experiment(result)

    assert pd.isna(summary.loc[0, "best_validation_loss"])
    assert pd.isna(summary.loc[0, "validation_mae"])
    assert pd.isna(summary.loc[0, "validation_rmse"])
    assert pd.isna(summary.loc[0, "validation_r2"])


def test_empty_folds_raise_value_error() -> None:
    with pytest.raises(ValueError, match="At least one walk-forward fold"):
        run_mlp_experiment(_synthetic_etf_ohlcv(), [], _config())


def test_missing_target_col_raises_value_error() -> None:
    config = _config(target_col="missing_target")

    with pytest.raises(ValueError, match="missing_target"):
        run_mlp_experiment(_synthetic_etf_ohlcv(), _folds()[:1], config)


def test_raw_df_is_not_mutated() -> None:
    raw_df = _synthetic_etf_ohlcv()
    original = raw_df.copy(deep=True)

    run_mlp_experiment(raw_df, _folds()[:1], _config())

    pd.testing.assert_frame_equal(raw_df, original)
