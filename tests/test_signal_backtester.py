from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_market_ml.backtest import (
    SignalBacktestConfig,
    SignalBacktestResult,
    calculate_max_drawdown,
    run_signal_backtest,
    summarize_backtest,
)


def _signal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2021-01-02"),
                "symbol": "BBB",
                "close": 45.0,
                "target_weight": 1.0,
                "selected": True,
            },
            {
                "date": pd.Timestamp("2021-01-01"),
                "symbol": "AAA",
                "close": 100.0,
                "target_weight": 1.0,
                "selected": True,
            },
            {
                "date": pd.Timestamp("2021-01-03"),
                "symbol": "AAA",
                "close": 121.0,
                "target_weight": 1.0,
                "selected": True,
            },
            {
                "date": pd.Timestamp("2021-01-01"),
                "symbol": "BBB",
                "close": 50.0,
                "target_weight": 0.0,
                "selected": False,
            },
            {
                "date": pd.Timestamp("2021-01-02"),
                "symbol": "AAA",
                "close": 110.0,
                "target_weight": 0.0,
                "selected": False,
            },
            {
                "date": pd.Timestamp("2021-01-03"),
                "symbol": "BBB",
                "close": 49.5,
                "target_weight": 0.0,
                "selected": False,
            },
        ]
    )


def test_run_signal_backtest_returns_result_with_expected_frames() -> None:
    result = run_signal_backtest(_signal_frame())

    assert isinstance(result, SignalBacktestResult)
    assert len(result.position_returns) == len(_signal_frame())
    assert result.equity_curve["date"].tolist() == [
        pd.Timestamp("2021-01-01"),
        pd.Timestamp("2021-01-02"),
        pd.Timestamp("2021-01-03"),
    ]


def test_position_returns_preserve_original_rows_and_columns() -> None:
    signals = _signal_frame()

    result = run_signal_backtest(signals)

    pd.testing.assert_frame_equal(result.position_returns[signals.columns], signals)
    assert {"symbol_return", "lagged_target_weight", "position_return"}.issubset(
        result.position_returns.columns
    )


def test_first_portfolio_return_is_zero() -> None:
    result = run_signal_backtest(_signal_frame())

    assert result.equity_curve.loc[0, "portfolio_return"] == 0.0


def test_weights_are_lagged_and_same_day_weight_is_not_used() -> None:
    result = run_signal_backtest(_signal_frame())
    date_two_bbb = result.position_returns.loc[
        (result.position_returns["date"] == pd.Timestamp("2021-01-02"))
        & (result.position_returns["symbol"] == "BBB")
    ].iloc[0]

    assert date_two_bbb["target_weight"] == 1.0
    assert date_two_bbb["symbol_return"] == pytest.approx(-0.10)
    assert date_two_bbb["lagged_target_weight"] == 0.0
    assert date_two_bbb["position_return"] == 0.0


def test_position_return_math_uses_lagged_weight_times_symbol_return() -> None:
    result = run_signal_backtest(_signal_frame())
    date_three_bbb = result.position_returns.loc[
        (result.position_returns["date"] == pd.Timestamp("2021-01-03"))
        & (result.position_returns["symbol"] == "BBB")
    ].iloc[0]

    assert date_three_bbb["symbol_return"] == pytest.approx(0.10)
    assert date_three_bbb["lagged_target_weight"] == 1.0
    assert date_three_bbb["position_return"] == pytest.approx(0.10)


def test_first_symbol_rows_have_zero_position_return() -> None:
    result = run_signal_backtest(_signal_frame())
    first_symbol_rows = result.position_returns.loc[
        result.position_returns["date"] == pd.Timestamp("2021-01-01")
    ]

    assert first_symbol_rows["symbol_return"].isna().all()
    assert first_symbol_rows["position_return"].tolist() == [0.0, 0.0]


def test_portfolio_return_aggregation_by_date() -> None:
    result = run_signal_backtest(_signal_frame())

    assert result.equity_curve["portfolio_return"].tolist() == pytest.approx([0.0, 0.10, 0.10])


def test_equity_curve_cumulative_math() -> None:
    result = run_signal_backtest(_signal_frame(), SignalBacktestConfig(initial_equity=100.0))

    assert result.equity_curve["equity"].tolist() == pytest.approx([100.0, 110.0, 121.0])


def test_calculate_max_drawdown_math() -> None:
    equity = pd.Series([100.0, 120.0, 90.0, 96.0, 80.0, 130.0])

    assert calculate_max_drawdown(equity) == pytest.approx(80.0 / 120.0 - 1.0)


def test_summary_metrics_include_expected_keys() -> None:
    result = run_signal_backtest(_signal_frame())

    assert set(result.summary_metrics) == {
        "total_return",
        "mean_daily_return",
        "daily_volatility",
        "sharpe_like",
        "max_drawdown",
        "final_equity",
    }
    assert result.summary_metrics["final_equity"] == pytest.approx(1.21)
    assert result.summary_metrics["total_return"] == pytest.approx(0.21)


def test_summarize_backtest_uses_configured_column_names() -> None:
    config = SignalBacktestConfig(
        initial_equity=10.0,
        portfolio_return_col="daily_return",
        equity_col="nav",
    )
    equity_curve = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2021-01-02"]),
            "daily_return": [0.0, 0.05],
            "nav": [10.0, 10.5],
        }
    )

    summary = summarize_backtest(equity_curve, config)

    assert summary["total_return"] == pytest.approx(0.05)
    assert summary["mean_daily_return"] == pytest.approx(0.025)


def test_missing_target_weights_are_treated_as_zero() -> None:
    signals = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2021-01-01"),
                "symbol": "AAA",
                "close": 100.0,
                "target_weight": np.nan,
            },
            {
                "date": pd.Timestamp("2021-01-02"),
                "symbol": "AAA",
                "close": 110.0,
                "target_weight": 1.0,
            },
        ]
    )

    result = run_signal_backtest(signals)

    assert result.position_returns.loc[1, "lagged_target_weight"] == 0.0
    assert result.position_returns.loc[1, "position_return"] == 0.0
    assert result.equity_curve["portfolio_return"].tolist() == [0.0, 0.0]


def test_missing_required_columns_raise_value_error() -> None:
    with pytest.raises(ValueError, match="target_weight"):
        run_signal_backtest(_signal_frame().drop(columns=["target_weight"]))

    with pytest.raises(ValueError, match="equity"):
        summarize_backtest(pd.DataFrame({"date": [], "portfolio_return": []}))


def test_initial_equity_must_be_positive() -> None:
    with pytest.raises(ValueError, match="initial_equity must be positive"):
        SignalBacktestConfig(initial_equity=0.0)
