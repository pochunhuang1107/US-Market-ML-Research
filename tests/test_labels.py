from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_market_ml.labels import (
    add_cross_sectional_rank_labels,
    add_excess_return_vs_benchmark,
    add_forward_returns,
)


def test_forward_return_2d_math_and_preserves_columns() -> None:
    df = pd.DataFrame(
        [
            {"date": "2024-01-03", "symbol": "AAA", "close": 120.0, "source": "keep"},
            {"date": "2024-01-01", "symbol": "AAA", "close": 100.0, "source": "keep"},
            {"date": "2024-01-02", "symbol": "AAA", "close": 110.0, "source": "keep"},
        ]
    )

    labels = add_forward_returns(df, horizons=(2,))

    first_return = labels.loc[
        (labels["symbol"] == "AAA") & (labels["date"] == "2024-01-01"),
        "forward_return_2d",
    ].item()
    assert first_return == pytest.approx(0.20)
    assert "source" in labels.columns
    assert len(labels) == len(df)


def test_forward_returns_do_not_leak_across_symbols() -> None:
    df = pd.DataFrame(
        [
            {"date": "2024-01-01", "symbol": "AAA", "close": 100.0},
            {"date": "2024-01-01", "symbol": "BBB", "close": 200.0},
            {"date": "2024-01-02", "symbol": "BBB", "close": 220.0},
        ]
    )

    labels = add_forward_returns(df, horizons=(1,))

    aaa_return = labels.loc[labels["symbol"] == "AAA", "forward_return_1d"].item()
    bbb_first_return = labels.loc[
        (labels["symbol"] == "BBB") & (labels["date"] == "2024-01-01"),
        "forward_return_1d",
    ].item()
    assert np.isnan(aaa_return)
    assert bbb_first_return == pytest.approx(0.10)


def test_forward_returns_end_of_series_rows_are_nan() -> None:
    df = pd.DataFrame(
        [
            {"date": "2024-01-01", "symbol": "AAA", "close": 100.0},
            {"date": "2024-01-02", "symbol": "AAA", "close": 110.0},
            {"date": "2024-01-03", "symbol": "AAA", "close": 120.0},
            {"date": "2024-01-01", "symbol": "BBB", "close": 200.0},
            {"date": "2024-01-02", "symbol": "BBB", "close": 210.0},
        ]
    )

    labels = add_forward_returns(df, horizons=(2,))

    assert labels.groupby("symbol", sort=False).tail(2)["forward_return_2d"].isna().all()


def test_forward_returns_missing_required_columns_raise_value_error() -> None:
    df = pd.DataFrame([{"date": "2024-01-01", "symbol": "AAA", "adj_close": 100.0}])

    with pytest.raises(ValueError, match="close"):
        add_forward_returns(df)


def test_excess_return_vs_spy_20d_math_by_date() -> None:
    df = pd.DataFrame(
        [
            {"date": "2024-01-02", "symbol": "QQQ", "forward_return_20d": 0.01},
            {"date": "2024-01-01", "symbol": "SPY", "forward_return_20d": 0.05},
            {"date": "2024-01-01", "symbol": "QQQ", "forward_return_20d": 0.08},
            {"date": "2024-01-02", "symbol": "SPY", "forward_return_20d": -0.02},
        ]
    )

    labels = add_excess_return_vs_benchmark(df)

    qqq_day_one = labels.loc[
        (labels["symbol"] == "QQQ") & (labels["date"] == "2024-01-01"),
        "excess_return_vs_spy_20d",
    ].item()
    qqq_day_two = labels.loc[
        (labels["symbol"] == "QQQ") & (labels["date"] == "2024-01-02"),
        "excess_return_vs_spy_20d",
    ].item()
    spy_day_one = labels.loc[
        (labels["symbol"] == "SPY") & (labels["date"] == "2024-01-01"),
        "excess_return_vs_spy_20d",
    ].item()
    assert qqq_day_one == pytest.approx(0.03)
    assert qqq_day_two == pytest.approx(0.03)
    assert spy_day_one == pytest.approx(0.0)


def test_excess_return_missing_forward_column_raises_value_error() -> None:
    df = pd.DataFrame([{"date": "2024-01-01", "symbol": "QQQ"}])

    with pytest.raises(ValueError, match="forward_return_20d"):
        add_excess_return_vs_benchmark(df)


def test_excess_return_missing_benchmark_rows_produce_nan() -> None:
    df = pd.DataFrame(
        [
            {"date": "2024-01-01", "symbol": "SPY", "forward_return_20d": 0.05},
            {"date": "2024-01-01", "symbol": "QQQ", "forward_return_20d": 0.08},
            {"date": "2024-01-02", "symbol": "QQQ", "forward_return_20d": 0.10},
        ]
    )

    labels = add_excess_return_vs_benchmark(df)

    missing_benchmark_excess = labels.loc[
        (labels["symbol"] == "QQQ") & (labels["date"] == "2024-01-02"),
        "excess_return_vs_spy_20d",
    ].item()
    assert np.isnan(missing_benchmark_excess)


def test_rank_labels_are_computed_within_each_date() -> None:
    df = pd.DataFrame(
        [
            {"date": "2024-01-01", "symbol": "AAA", "target": 0.30},
            {"date": "2024-01-01", "symbol": "BBB", "target": 0.10},
            {"date": "2024-01-02", "symbol": "AAA", "target": 0.20},
            {"date": "2024-01-02", "symbol": "BBB", "target": 0.00},
        ]
    )

    labels = add_cross_sectional_rank_labels(df, target_col="target")

    assert labels.loc[
        (labels["date"] == "2024-01-01") & (labels["symbol"] == "AAA"),
        "rank_target",
    ].item() == pytest.approx(0.5)
    assert labels.loc[
        (labels["date"] == "2024-01-01") & (labels["symbol"] == "BBB"),
        "rank_target",
    ].item() == pytest.approx(1.0)
    assert labels.loc[
        (labels["date"] == "2024-01-02") & (labels["symbol"] == "AAA"),
        "rank_target",
    ].item() == pytest.approx(0.5)
    assert labels.loc[
        (labels["date"] == "2024-01-02") & (labels["symbol"] == "BBB"),
        "rank_target",
    ].item() == pytest.approx(1.0)
    assert len(labels) == len(df)


def test_rank_labels_missing_target_column_raises_value_error() -> None:
    df = pd.DataFrame([{"date": "2024-01-01", "symbol": "AAA"}])

    with pytest.raises(ValueError, match="target"):
        add_cross_sectional_rank_labels(df, target_col="target")
