from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("torch")

from us_market_ml.backtest import SignalBacktestConfig, TopKSignalConfig  # noqa: E402
from us_market_ml.data import SupervisedDatasetConfig  # noqa: E402
from us_market_ml.experiments import (  # noqa: E402
    MLPE2EExperimentConfig,
    MLPE2EExperimentResult,
    MLPE2EFoldResult,
    MLPExperimentConfig,
    run_mlp_e2e_experiment,
    summarize_mlp_e2e_experiment,
)
from us_market_ml.models import MLPRegressorConfig  # noqa: E402
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
    mlp_prediction_col: str = "prediction",
    signal_prediction_col: str = "prediction",
    top_k: int = 2,
) -> MLPE2EExperimentConfig:
    return MLPE2EExperimentConfig(
        mlp_config=MLPExperimentConfig(
            target_col="excess_return_vs_spy_20d",
            model_config=MLPRegressorConfig(
                hidden_units=(16,),
                dropout=0.0,
                max_epochs=5,
                patience=2,
                batch_size=16,
                random_state=42,
            ),
            dataset_config=SupervisedDatasetConfig(),
            prediction_col=mlp_prediction_col,
        ),
        signal_config=TopKSignalConfig(
            prediction_col=signal_prediction_col,
            top_k=top_k,
        ),
        backtest_config=SignalBacktestConfig(initial_equity=100.0),
    )


def test_run_mlp_e2e_experiment_returns_mlp_and_fold_results() -> None:
    result = run_mlp_e2e_experiment(_synthetic_etf_ohlcv(), _folds(), _config())

    assert isinstance(result, MLPE2EExperimentResult)
    assert result.mlp_result.fold_results
    assert len(result.fold_results) == len(result.mlp_result.fold_results) == 2
    assert all(isinstance(fold_result, MLPE2EFoldResult) for fold_result in result.fold_results)


def test_each_fold_has_split_signals_and_backtests() -> None:
    result = run_mlp_e2e_experiment(_synthetic_etf_ohlcv(), _folds(), _config())

    for fold_result in result.fold_results:
        for signals in (
            fold_result.train_signals,
            fold_result.validation_signals,
            fold_result.test_signals,
        ):
            assert not signals.empty
            assert {"selected", "signal_rank", "target_weight"}.issubset(signals.columns)

        for backtest in (
            fold_result.train_backtest,
            fold_result.validation_backtest,
            fold_result.test_backtest,
        ):
            assert not backtest.equity_curve.empty
            assert set(backtest.summary_metrics) == {
                "total_return",
                "mean_daily_return",
                "daily_volatility",
                "sharpe_like",
                "max_drawdown",
                "final_equity",
            }


def test_summary_has_one_row_per_fold_per_split() -> None:
    result = run_mlp_e2e_experiment(_synthetic_etf_ohlcv(), _folds(), _config(top_k=2))

    summary = summarize_mlp_e2e_experiment(result)

    assert len(summary) == len(result.fold_results) * 3
    assert summary["split"].tolist() == [
        "train",
        "validation",
        "test",
        "train",
        "validation",
        "test",
    ]
    assert set(summary["fold_name"]) == {"fold_1", "fold_2"}
    assert summary["model_type"].tolist() == ["mlp"] * 6
    assert summary["selected_top_k"].tolist() == [2] * 6
    assert {
        "target_col",
        "epochs_trained",
        "total_return",
        "mean_daily_return",
        "daily_volatility",
        "sharpe_like",
        "max_drawdown",
        "final_equity",
    }.issubset(summary.columns)


def test_prediction_col_mismatch_raises_value_error() -> None:
    config = _config(
        mlp_prediction_col="mlp_score",
        signal_prediction_col="prediction",
    )

    with pytest.raises(ValueError, match="prediction_col must match"):
        run_mlp_e2e_experiment(_synthetic_etf_ohlcv(), _folds()[:1], config)


def test_raw_df_is_not_mutated() -> None:
    raw_df = _synthetic_etf_ohlcv()
    original = raw_df.copy(deep=True)

    run_mlp_e2e_experiment(raw_df, _folds()[:1], _config())

    pd.testing.assert_frame_equal(raw_df, original)


def test_at_least_one_test_split_has_non_empty_equity_curve() -> None:
    result = run_mlp_e2e_experiment(_synthetic_etf_ohlcv(), _folds(), _config())

    assert any(
        not fold_result.test_backtest.equity_curve.empty for fold_result in result.fold_results
    )
