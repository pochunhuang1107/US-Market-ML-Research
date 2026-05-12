from __future__ import annotations

import pandas as pd
import pytest

from us_market_ml.backtest import SignalBacktestConfig, SignalSmoothingConfig, TopKSignalConfig
from us_market_ml.data import SupervisedDatasetConfig
from us_market_ml.experiments import (
    BaselineE2EExperimentConfig,
    BaselineE2EExperimentResult,
    BaselineE2EFoldResult,
    BaselineExperimentConfig,
    BaselineExperimentResult,
    BaselineFoldResult,
    run_baseline_e2e_experiment,
    summarize_baseline_e2e_experiment,
)
from us_market_ml.models import BaselineFitResult, BaselineModelConfig
from us_market_ml.splits import WalkForwardFold


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
    baseline_prediction_col: str = "prediction",
    signal_prediction_col: str = "prediction",
    top_k: int = 2,
) -> BaselineE2EExperimentConfig:
    return BaselineE2EExperimentConfig(
        baseline_config=BaselineExperimentConfig(
            target_col="excess_return_vs_spy_20d",
            model_config=BaselineModelConfig(model_type="ridge", alpha=0.1),
            dataset_config=SupervisedDatasetConfig(),
            prediction_col=baseline_prediction_col,
        ),
        signal_config=TopKSignalConfig(
            prediction_col=signal_prediction_col,
            top_k=top_k,
        ),
        backtest_config=SignalBacktestConfig(initial_equity=100.0),
    )


def test_run_baseline_e2e_experiment_returns_baseline_and_fold_results() -> None:
    result = run_baseline_e2e_experiment(_synthetic_etf_ohlcv(), _folds(), _config())

    assert isinstance(result, BaselineE2EExperimentResult)
    assert result.baseline_result.fold_results
    assert len(result.fold_results) == len(result.baseline_result.fold_results) == 2
    assert all(
        isinstance(fold_result, BaselineE2EFoldResult) for fold_result in result.fold_results
    )


def test_each_fold_has_split_signals_and_backtests() -> None:
    result = run_baseline_e2e_experiment(_synthetic_etf_ohlcv(), _folds(), _config())

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
                "total_transaction_cost",
                "mean_daily_turnover",
                "max_daily_turnover",
            }


def test_summary_has_one_row_per_fold_per_split() -> None:
    result = run_baseline_e2e_experiment(_synthetic_etf_ohlcv(), _folds(), _config(top_k=2))

    summary = summarize_baseline_e2e_experiment(result)

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
    assert summary["selected_top_k"].tolist() == [2] * 6
    assert {
        "model_type",
        "target_col",
        "total_return",
        "mean_daily_return",
        "daily_volatility",
        "sharpe_like",
        "max_drawdown",
        "final_equity",
        "total_transaction_cost",
        "mean_daily_turnover",
    }.issubset(summary.columns)


def test_prediction_col_mismatch_raises_value_error() -> None:
    config = _config(
        baseline_prediction_col="baseline_score",
        signal_prediction_col="prediction",
    )

    with pytest.raises(ValueError, match="prediction_col must match"):
        run_baseline_e2e_experiment(_synthetic_etf_ohlcv(), _folds()[:1], config)


def test_raw_df_is_not_mutated() -> None:
    raw_df = _synthetic_etf_ohlcv()
    original = raw_df.copy(deep=True)

    run_baseline_e2e_experiment(raw_df, _folds()[:1], _config())

    pd.testing.assert_frame_equal(raw_df, original)


def test_at_least_one_test_split_has_non_empty_equity_curve() -> None:
    result = run_baseline_e2e_experiment(_synthetic_etf_ohlcv(), _folds(), _config())

    assert any(
        not fold_result.test_backtest.equity_curve.empty for fold_result in result.fold_results
    )


def test_e2e_smoothing_config_reduces_switches_in_synthetic_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions = pd.DataFrame(
        [
            _prediction_row("2021-01-01", "SPY", 0.9, 100.0),
            _prediction_row("2021-01-01", "QQQ", 0.1, 100.0),
            _prediction_row("2021-01-02", "SPY", 0.1, 101.0),
            _prediction_row("2021-01-02", "QQQ", 0.9, 101.0),
            _prediction_row("2021-01-03", "SPY", 0.9, 102.0),
            _prediction_row("2021-01-03", "QQQ", 0.1, 102.0),
            _prediction_row("2021-01-04", "SPY", 0.1, 103.0),
            _prediction_row("2021-01-04", "QQQ", 0.9, 103.0),
        ]
    )

    fit_result = BaselineFitResult(
        model_type="ridge",
        target_col="target",
        feature_cols=["feature"],
        train_rows=len(predictions),
        validation_rows=len(predictions),
        train_metrics={},
        validation_metrics={},
    )
    baseline_result = BaselineExperimentResult(
        config=BaselineExperimentConfig(target_col="target"),
        dataset=predictions,
        feature_cols=["feature"],
        fold_results=[
            BaselineFoldResult(
                fold_name="fold_1",
                fit_result=fit_result,
                train_predictions=predictions,
                validation_predictions=predictions,
                test_predictions=predictions,
            )
        ],
    )

    def fake_run_baseline_experiment(*args: object, **kwargs: object) -> BaselineExperimentResult:
        return baseline_result

    monkeypatch.setattr(
        "us_market_ml.experiments.baseline_e2e.run_baseline_experiment",
        fake_run_baseline_experiment,
    )

    fold = _folds()[:1]
    unsmoothed = run_baseline_e2e_experiment(
        predictions,
        fold,
        _config(top_k=1),
    )
    smoothed = run_baseline_e2e_experiment(
        predictions,
        fold,
        BaselineE2EExperimentConfig(
            baseline_config=BaselineExperimentConfig(target_col="target"),
            signal_config=TopKSignalConfig(top_k=1),
            backtest_config=SignalBacktestConfig(initial_equity=100.0),
            smoothing_config=SignalSmoothingConfig(min_holding_days=2),
        ),
    )

    smoothed_switches = _count_selected_switches(smoothed.fold_results[0].test_signals)
    unsmoothed_switches = _count_selected_switches(unsmoothed.fold_results[0].test_signals)
    assert smoothed_switches < unsmoothed_switches


def _prediction_row(
    date: str,
    symbol: str,
    prediction: float,
    close: float,
) -> dict[str, object]:
    return {
        "date": pd.Timestamp(date),
        "symbol": symbol,
        "prediction": prediction,
        "close": close,
    }


def _count_selected_switches(signals: pd.DataFrame) -> int:
    selected_symbols = (
        signals.loc[signals["selected"]]
        .sort_values(["date", "signal_rank"], kind="mergesort")["symbol"]
        .tolist()
    )
    return sum(
        previous_symbol != current_symbol
        for previous_symbol, current_symbol in zip(
            selected_symbols,
            selected_symbols[1:],
            strict=False,
        )
    )
