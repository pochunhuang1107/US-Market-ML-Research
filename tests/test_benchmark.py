from __future__ import annotations

import pandas as pd
import pytest

from us_market_ml.backtest import (
    BenchmarkConfig,
    build_buy_and_hold_benchmark,
    build_equal_weight_benchmark,
    combine_strategy_and_benchmarks,
    stitch_test_equity_curves,
    summarize_equity_series,
)


def _market_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": pd.Timestamp("2021-01-02"), "symbol": "AAA", "close": 110.0},
            {"date": pd.Timestamp("2021-01-01"), "symbol": "AAA", "close": 100.0},
            {"date": pd.Timestamp("2021-01-03"), "symbol": "AAA", "close": 121.0},
            {"date": pd.Timestamp("2021-01-01"), "symbol": "BBB", "close": 50.0},
            {"date": pd.Timestamp("2021-01-02"), "symbol": "BBB", "close": 45.0},
            {"date": pd.Timestamp("2021-01-03"), "symbol": "BBB", "close": 49.5},
        ]
    )


def test_build_buy_and_hold_benchmark_math_and_preserves_input() -> None:
    market_data = _market_data()
    original = market_data.copy(deep=True)

    benchmark = build_buy_and_hold_benchmark(
        market_data,
        "AAA",
        BenchmarkConfig(initial_equity=100.0),
    )

    pd.testing.assert_frame_equal(market_data, original)
    assert benchmark["date"].tolist() == pd.to_datetime(
        ["2021-01-01", "2021-01-02", "2021-01-03"]
    ).tolist()
    assert benchmark["benchmark_name"].tolist() == ["AAA", "AAA", "AAA"]
    assert benchmark["benchmark_return"].tolist() == pytest.approx([0.0, 0.10, 0.10])
    assert benchmark["benchmark_equity"].tolist() == pytest.approx([100.0, 110.0, 121.0])


def test_build_buy_and_hold_benchmark_raises_for_missing_symbol() -> None:
    with pytest.raises(ValueError, match="CCC"):
        build_buy_and_hold_benchmark(_market_data(), "CCC")


def test_build_equal_weight_benchmark_math_and_preserves_input() -> None:
    market_data = _market_data()
    original = market_data.copy(deep=True)

    benchmark = build_equal_weight_benchmark(
        market_data,
        symbols=["AAA", "BBB"],
        config=BenchmarkConfig(initial_equity=100.0),
    )

    pd.testing.assert_frame_equal(market_data, original)
    assert benchmark["benchmark_name"].tolist() == [
        "equal_weight",
        "equal_weight",
        "equal_weight",
    ]
    assert benchmark["benchmark_return"].tolist() == pytest.approx([0.0, 0.0, 0.10])
    assert benchmark["benchmark_equity"].tolist() == pytest.approx([100.0, 100.0, 110.0])


def test_build_equal_weight_benchmark_raises_when_no_symbols_available() -> None:
    with pytest.raises(ValueError, match="At least one"):
        build_equal_weight_benchmark(_market_data(), symbols=[])


def test_combine_strategy_and_benchmarks_returns_long_format() -> None:
    strategy = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2021-01-02"]),
            "portfolio_return": [0.0, 0.05],
            "equity": [1.0, 1.05],
        }
    )
    benchmark = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2021-01-02"]),
            "benchmark_name": ["SPY", "SPY"],
            "benchmark_return": [0.0, 0.02],
            "benchmark_equity": [1.0, 1.02],
        }
    )

    combined = combine_strategy_and_benchmarks(strategy, [benchmark], strategy_name="MLP")

    assert combined.columns.tolist() == ["date", "series_name", "daily_return", "equity"]
    assert combined["series_name"].tolist() == ["MLP", "SPY", "MLP", "SPY"]
    assert combined["daily_return"].tolist() == pytest.approx([0.0, 0.0, 0.05, 0.02])


def test_combine_strategy_and_benchmarks_raises_for_missing_columns() -> None:
    with pytest.raises(ValueError, match="portfolio_return"):
        combine_strategy_and_benchmarks(pd.DataFrame({"date": []}), [])


def test_summarize_equity_series_metrics() -> None:
    combined = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
            "series_name": ["strategy", "strategy", "strategy"],
            "daily_return": [0.0, 0.10, -0.05],
            "equity": [1.0, 1.10, 1.045],
        }
    )

    summary = summarize_equity_series(combined)
    row = summary.iloc[0]

    assert row["series_name"] == "strategy"
    assert row["total_return"] == pytest.approx(0.045)
    assert row["mean_daily_return"] == pytest.approx((0.0 + 0.10 - 0.05) / 3)
    assert row["daily_volatility"] == pytest.approx(pd.Series([0.0, 0.10, -0.05]).std())
    assert row["sharpe_like"] == pytest.approx(
        row["mean_daily_return"] / row["daily_volatility"]
    )
    assert row["max_drawdown"] == pytest.approx(1.045 / 1.10 - 1.0)
    assert row["final_equity"] == pytest.approx(1.045)


def test_summarize_equity_series_returns_empty_metric_frame() -> None:
    summary = summarize_equity_series(
        pd.DataFrame(columns=["date", "series_name", "daily_return", "equity"])
    )

    assert summary.empty
    assert summary.columns.tolist() == [
        "series_name",
        "total_return",
        "mean_daily_return",
        "daily_volatility",
        "sharpe_like",
        "max_drawdown",
        "final_equity",
    ]


def test_benchmark_initial_equity_must_be_positive() -> None:
    with pytest.raises(ValueError, match="initial_equity must be positive"):
        BenchmarkConfig(initial_equity=0.0)


def test_stitch_test_equity_curves_non_overlapping_folds() -> None:
    fold_a = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2021-01-02"]),
            "portfolio_return": [0.0, 0.10],
            "equity": [1.0, 1.10],
        }
    )
    fold_b = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-04", "2021-01-03"]),
            "portfolio_return": [0.05, -0.02],
            "equity": [1.05, 0.98],
        }
    )

    stitched = stitch_test_equity_curves([("fold_a", fold_a), ("fold_b", fold_b)])

    assert stitched["date"].tolist() == pd.to_datetime(
        ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"]
    ).tolist()
    assert stitched["source_fold"].tolist() == ["fold_a", "fold_a", "fold_b", "fold_b"]
    assert stitched["portfolio_return"].tolist() == pytest.approx([0.0, 0.10, -0.02, 0.05])


def test_stitch_test_equity_curves_raises_for_overlapping_test_dates() -> None:
    fold_a = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2021-01-02"]),
            "portfolio_return": [0.0, 0.10],
            "equity": [1.0, 1.10],
        }
    )
    fold_b = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-02", "2021-01-03"]),
            "portfolio_return": [0.0, 0.05],
            "equity": [1.0, 1.05],
        }
    )

    with pytest.raises(ValueError, match="overlapping dates"):
        stitch_test_equity_curves([("fold_a", fold_a), ("fold_b", fold_b)])


def test_stitch_test_equity_curves_cumulative_math() -> None:
    stitched = stitch_test_equity_curves(
        [
            (
                "fold_a",
                pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2021-01-01", "2021-01-02"]),
                        "portfolio_return": [0.25, 0.10],
                        "equity": [1.25, 1.375],
                    }
                ),
            ),
            (
                "fold_b",
                pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2021-01-03", "2021-01-04"]),
                        "portfolio_return": [-0.20, 0.05],
                        "equity": [0.80, 0.84],
                    }
                ),
            ),
        ]
    )

    assert stitched["portfolio_return"].tolist() == pytest.approx([0.0, 0.10, -0.20, 0.05])
    assert stitched["equity"].tolist() == pytest.approx([1.0, 1.10, 0.88, 0.924])
