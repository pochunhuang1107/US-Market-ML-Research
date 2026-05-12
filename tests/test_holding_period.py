from __future__ import annotations

import pandas as pd
import pytest

from us_market_ml.backtest import (
    build_holding_diagnostics_report,
    build_holding_periods,
    summarize_holding_periods,
    summarize_holding_periods_by_symbol,
    summarize_turnover_distribution,
)


def test_build_holding_periods_single_symbol_continuous_selection() -> None:
    selected_holdings = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2021-01-05", "2021-01-12"]),
            "symbol": ["AAA", "AAA", "AAA"],
            "target_weight": [0.4, 0.5, 0.6],
        }
    )

    holding_periods = build_holding_periods(selected_holdings)

    assert holding_periods["symbol"].tolist() == ["AAA"]
    assert holding_periods.loc[0, "holding_id"] == 1
    assert holding_periods.loc[0, "start_date"] == pd.Timestamp("2021-01-01")
    assert holding_periods.loc[0, "end_date"] == pd.Timestamp("2021-01-12")
    assert holding_periods.loc[0, "holding_days"] == 3
    assert holding_periods.loc[0, "avg_target_weight"] == pytest.approx(0.5)
    assert holding_periods.loc[0, "max_target_weight"] == pytest.approx(0.6)


def test_build_holding_periods_symbol_disappearing_and_reappearing() -> None:
    selected_holdings = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"]
            ),
            "symbol": ["AAA", "BBB", "AAA", "AAA"],
            "target_weight": [1.0, 1.0, 0.5, 0.5],
        }
    )

    holding_periods = build_holding_periods(selected_holdings)
    aaa_periods = holding_periods.loc[holding_periods["symbol"] == "AAA"].reset_index(
        drop=True
    )

    assert aaa_periods["holding_id"].tolist() == [1, 2]
    assert aaa_periods["start_date"].tolist() == [
        pd.Timestamp("2021-01-01"),
        pd.Timestamp("2021-01-03"),
    ]
    assert aaa_periods["end_date"].tolist() == [
        pd.Timestamp("2021-01-01"),
        pd.Timestamp("2021-01-04"),
    ]
    assert aaa_periods["holding_days"].tolist() == [1, 2]


def test_build_holding_periods_top_k_multi_symbol_selection() -> None:
    selected_holdings = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2021-01-01",
                    "2021-01-01",
                    "2021-01-03",
                    "2021-01-03",
                    "2021-01-07",
                    "2021-01-07",
                ]
            ),
            "symbol": ["AAA", "BBB", "AAA", "CCC", "AAA", "BBB"],
            "target_weight": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        }
    )

    holding_periods = build_holding_periods(selected_holdings)

    assert holding_periods["symbol"].tolist() == ["AAA", "BBB", "CCC", "BBB"]
    assert holding_periods["holding_days"].tolist() == [3, 1, 1, 1]
    assert holding_periods.loc[0, "start_date"] == pd.Timestamp("2021-01-01")
    assert holding_periods.loc[0, "end_date"] == pd.Timestamp("2021-01-07")


def test_summarize_holding_periods() -> None:
    holding_periods = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "AAA"],
            "holding_days": [3, 1, 2],
        }
    )

    summary = summarize_holding_periods(holding_periods)

    assert summary.loc[0, "total_holding_periods"] == 3
    assert summary.loc[0, "unique_symbols_held"] == 2
    assert summary.loc[0, "avg_holding_days"] == pytest.approx(2.0)
    assert summary.loc[0, "median_holding_days"] == pytest.approx(2.0)
    assert summary.loc[0, "max_holding_days"] == pytest.approx(3.0)
    assert summary.loc[0, "min_holding_days"] == pytest.approx(1.0)
    assert summary.loc[0, "single_day_holding_periods"] == 1
    assert summary.loc[0, "single_day_holding_share"] == pytest.approx(1 / 3)


def test_summarize_holding_periods_by_symbol() -> None:
    holding_periods = pd.DataFrame(
        {
            "symbol": ["BBB", "AAA", "AAA", "CCC"],
            "holding_days": [1, 3, 2, 1],
        }
    )

    summary = summarize_holding_periods_by_symbol(holding_periods)

    assert summary["symbol"].tolist() == ["AAA", "BBB", "CCC"]
    assert summary["holding_periods"].tolist() == [2, 1, 1]
    assert summary["total_holding_days"].tolist() == [5, 1, 1]
    assert summary.loc[0, "avg_holding_days"] == pytest.approx(2.5)
    assert summary.loc[0, "median_holding_days"] == pytest.approx(2.5)
    assert summary.loc[0, "max_holding_days"] == 3
    assert summary.loc[1, "single_day_holding_periods"] == 1
    assert summary.loc[1, "single_day_holding_share"] == pytest.approx(1.0)


def test_summarize_turnover_distribution() -> None:
    equity_curve = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=5, freq="D"),
            "turnover": [0.0, 0.5, 1.0, 0.0, 2.0],
        }
    )

    summary, top_turnover_days = summarize_turnover_distribution(equity_curve, top_n=2)

    assert summary.loc[0, "mean_turnover"] == pytest.approx(0.7)
    assert summary.loc[0, "median_turnover"] == pytest.approx(0.5)
    assert summary.loc[0, "max_turnover"] == pytest.approx(2.0)
    assert summary.loc[0, "p90_turnover"] == pytest.approx(1.6)
    assert summary.loc[0, "p95_turnover"] == pytest.approx(1.8)
    assert summary.loc[0, "zero_turnover_days"] == 2
    assert summary.loc[0, "zero_turnover_share"] == pytest.approx(0.4)
    assert top_turnover_days["date"].tolist() == [
        pd.Timestamp("2021-01-05"),
        pd.Timestamp("2021-01-03"),
    ]
    assert top_turnover_days["turnover"].tolist() == [2.0, 1.0]


def test_summarize_turnover_distribution_missing_turnover_raises() -> None:
    equity_curve = pd.DataFrame({"date": pd.date_range("2021-01-01", periods=2)})

    with pytest.raises(ValueError, match="turnover"):
        summarize_turnover_distribution(equity_curve)


def test_build_holding_diagnostics_report() -> None:
    selected_holdings = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
            "symbol": ["AAA", "AAA", "BBB"],
            "target_weight": [1.0, 1.0, 1.0],
        }
    )
    equity_curve = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
            "turnover": [0.0, 0.0, 2.0],
        }
    )

    report = build_holding_diagnostics_report(selected_holdings, equity_curve)

    assert set(report) == {
        "holding_periods",
        "holding_period_summary",
        "holding_period_by_symbol",
        "turnover_summary",
        "top_turnover_days",
    }
    assert report["holding_periods"]["holding_days"].tolist() == [2, 1]
    assert report["holding_period_summary"].loc[0, "total_holding_periods"] == 2
    assert report["turnover_summary"].loc[0, "max_turnover"] == pytest.approx(2.0)
