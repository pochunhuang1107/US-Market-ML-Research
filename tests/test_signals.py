from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_market_ml.backtest import (
    TopKSignalConfig,
    generate_top_k_signals,
    summarize_signals,
    validate_signal_weights,
)


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": pd.Timestamp("2021-01-01"), "symbol": "SPY", "prediction": 0.10},
            {"date": pd.Timestamp("2021-01-01"), "symbol": "QQQ", "prediction": 0.30},
            {"date": pd.Timestamp("2021-01-01"), "symbol": "TLT", "prediction": 0.20},
            {"date": pd.Timestamp("2021-01-01"), "symbol": "IWM", "prediction": np.nan},
            {"date": pd.Timestamp("2021-01-02"), "symbol": "SPY", "prediction": -0.10},
            {"date": pd.Timestamp("2021-01-02"), "symbol": "QQQ", "prediction": 0.50},
            {"date": pd.Timestamp("2021-01-02"), "symbol": "TLT", "prediction": 0.40},
            {"date": pd.Timestamp("2021-01-02"), "symbol": "IWM", "prediction": 0.20},
        ]
    )


def test_generate_top_k_signals_selects_top_k_by_date() -> None:
    signals = generate_top_k_signals(_prediction_frame(), TopKSignalConfig(top_k=2))

    selected = signals.loc[signals["selected"], ["date", "symbol", "signal_rank"]]

    assert selected["symbol"].tolist() == ["QQQ", "TLT", "QQQ", "TLT"]
    assert selected["signal_rank"].tolist() == [1, 2, 1, 2]


def test_generate_top_k_signals_preserves_rows_and_original_columns() -> None:
    predictions = _prediction_frame()
    predictions["fold_name"] = "fold_1"

    signals = generate_top_k_signals(predictions, TopKSignalConfig(top_k=2))

    assert len(signals) == len(predictions)
    assert {"date", "symbol", "prediction", "fold_name"}.issubset(signals.columns)
    pd.testing.assert_frame_equal(predictions, signals[predictions.columns])


def test_equal_weights_sum_to_one_for_dates_with_selected_symbols() -> None:
    signals = generate_top_k_signals(_prediction_frame(), TopKSignalConfig(top_k=2))

    selected_weight_by_date = (
        signals.loc[signals["selected"]].groupby("date")["target_weight"].sum()
    )

    assert selected_weight_by_date.tolist() == [1.0, 1.0]
    assert validate_signal_weights(signals)


def test_dates_with_fewer_eligible_symbols_than_top_k_select_all_eligible() -> None:
    predictions = pd.DataFrame(
        [
            {"date": pd.Timestamp("2021-01-01"), "symbol": "SPY", "prediction": np.nan},
            {"date": pd.Timestamp("2021-01-01"), "symbol": "QQQ", "prediction": 0.25},
        ]
    )

    signals = generate_top_k_signals(predictions, TopKSignalConfig(top_k=3))

    assert signals["selected"].tolist() == [False, True]
    assert signals["target_weight"].tolist() == [0.0, 1.0]
    assert signals["signal_rank"].tolist() == [pd.NA, 1]


def test_missing_predictions_are_not_selected() -> None:
    signals = generate_top_k_signals(_prediction_frame(), TopKSignalConfig(top_k=4))

    missing_prediction_row = signals.loc[signals["prediction"].isna()].iloc[0]

    assert not bool(missing_prediction_row["selected"])
    assert missing_prediction_row["target_weight"] == 0.0
    assert pd.isna(missing_prediction_row["signal_rank"])


def test_allow_negative_scores_false_filters_negative_predictions() -> None:
    predictions = pd.DataFrame(
        [
            {"date": pd.Timestamp("2021-01-01"), "symbol": "SPY", "prediction": -0.10},
            {"date": pd.Timestamp("2021-01-01"), "symbol": "QQQ", "prediction": -0.20},
            {"date": pd.Timestamp("2021-01-01"), "symbol": "TLT", "prediction": 0.05},
        ]
    )

    signals = generate_top_k_signals(
        predictions,
        TopKSignalConfig(top_k=3, allow_negative_scores=False),
    )

    assert signals["selected"].tolist() == [False, False, True]
    assert signals["target_weight"].tolist() == [0.0, 0.0, 1.0]


def test_min_score_filters_predictions_below_threshold() -> None:
    signals = generate_top_k_signals(
        _prediction_frame(),
        TopKSignalConfig(top_k=3, min_score=0.30),
    )

    selected_symbols_by_date = {
        date: group.loc[group["selected"], "symbol"].tolist()
        for date, group in signals.groupby("date", sort=False)
    }

    assert selected_symbols_by_date[pd.Timestamp("2021-01-01")] == ["QQQ"]
    assert selected_symbols_by_date[pd.Timestamp("2021-01-02")] == ["QQQ", "TLT"]


def test_filters_allow_negative_scores_and_min_score_both_apply() -> None:
    predictions = pd.DataFrame(
        [
            {"date": pd.Timestamp("2021-01-01"), "symbol": "SPY", "prediction": -0.10},
            {"date": pd.Timestamp("2021-01-01"), "symbol": "QQQ", "prediction": 0.10},
            {"date": pd.Timestamp("2021-01-01"), "symbol": "TLT", "prediction": 0.25},
        ]
    )

    signals = generate_top_k_signals(
        predictions,
        TopKSignalConfig(top_k=3, min_score=0.20, allow_negative_scores=False),
    )

    assert signals["selected"].tolist() == [False, False, True]


def test_deterministic_tie_breaking_uses_symbol_ascending() -> None:
    predictions = pd.DataFrame(
        [
            {"date": pd.Timestamp("2021-01-01"), "symbol": "TLT", "prediction": 0.50},
            {"date": pd.Timestamp("2021-01-01"), "symbol": "SPY", "prediction": 0.50},
            {"date": pd.Timestamp("2021-01-01"), "symbol": "QQQ", "prediction": 0.50},
        ]
    )

    signals = generate_top_k_signals(predictions, TopKSignalConfig(top_k=2))

    selected = signals.loc[signals["selected"]].sort_values("signal_rank")
    assert selected["symbol"].tolist() == ["QQQ", "SPY"]


def test_summarize_signals_returns_expected_counts_and_weights() -> None:
    predictions = pd.DataFrame(
        [
            {"date": pd.Timestamp("2021-01-01"), "symbol": "SPY", "prediction": 0.10},
            {"date": pd.Timestamp("2021-01-01"), "symbol": "QQQ", "prediction": 0.20},
            {"date": pd.Timestamp("2021-01-02"), "symbol": "SPY", "prediction": -0.10},
            {"date": pd.Timestamp("2021-01-02"), "symbol": "QQQ", "prediction": np.nan},
        ]
    )
    config = TopKSignalConfig(top_k=2, allow_negative_scores=False)
    signals = generate_top_k_signals(predictions, config)

    summary = summarize_signals(signals, config)

    assert summary["selected_count"].tolist() == [2, 0]
    assert summary["total_weight"].tolist() == [1.0, 0.0]
    assert summary["max_score"].tolist() == [0.20, -0.10]
    assert summary.loc[0, "min_selected_score"] == 0.10
    assert pd.isna(summary.loc[1, "min_selected_score"])


def test_validate_signal_weights_returns_false_for_invalid_weights() -> None:
    signals = generate_top_k_signals(_prediction_frame(), TopKSignalConfig(top_k=2))
    signals.loc[signals["selected"], "target_weight"] = 0.25

    assert not validate_signal_weights(signals)


def test_missing_required_prediction_columns_raise_value_error() -> None:
    predictions = _prediction_frame().drop(columns=["prediction"])

    with pytest.raises(ValueError, match="prediction"):
        generate_top_k_signals(predictions)


def test_missing_required_signal_columns_raise_value_error() -> None:
    signals = generate_top_k_signals(_prediction_frame(), TopKSignalConfig(top_k=2))

    with pytest.raises(ValueError, match="selected"):
        summarize_signals(signals.drop(columns=["selected"]))

    with pytest.raises(ValueError, match="target_weight"):
        validate_signal_weights(signals.drop(columns=["target_weight"]))


def test_top_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="top_k must be positive"):
        TopKSignalConfig(top_k=0)
