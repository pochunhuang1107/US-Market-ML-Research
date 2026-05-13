from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli.run_paper_tracking_export import (
    _build_switch_log,
    run_paper_tracking_export_from_args,
)


def _synthetic_etf_ohlcv() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2020-01-01", periods=520, freq="D")
    symbols = ("SPY", "QQQ", "TLT")

    for symbol_index, symbol in enumerate(symbols):
        base_price = 100.0 + symbol_index * 25.0
        drift = 0.06 + symbol_index * 0.03
        cycle = symbol_index + 2
        for day_index, date in enumerate(dates):
            regime_bump = ((day_index // 45 + symbol_index) % 3) * 0.08
            close = base_price + day_index * drift + (day_index % 19) * cycle * regime_bump
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


def test_run_paper_tracking_export_from_args_writes_research_csvs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_path = tmp_path / "etf_daily.csv"
    output_dir = tmp_path / "paper_tracking"
    _synthetic_etf_ohlcv().to_csv(data_path, index=False)

    return_code = run_paper_tracking_export_from_args(
        [
            "--data-path",
            str(data_path),
            "--output-dir",
            str(output_dir),
            "--export-name",
            "candidate_tracking",
            "--candidate",
            "baseline:linear_smooth10_margin002:linear_regression:1:42:10:0.002",
            "--train-start",
            "2020-08-01",
            "--train-end",
            "2020-12-31",
            "--validation-start",
            "2021-01-01",
            "--validation-end",
            "2021-02-28",
            "--tracking-start",
            "2021-03-01",
            "--tracking-end",
            "2021-03-31",
            "--transaction-cost-bps",
            "10",
        ]
    )

    captured = capsys.readouterr()
    daily_targets_path = output_dir / "candidate_tracking_daily_targets.csv"
    daily_summary_path = output_dir / "candidate_tracking_daily_summary.csv"
    switch_log_path = output_dir / "candidate_tracking_switch_log.csv"
    metrics_path = output_dir / "candidate_tracking_metrics.csv"

    daily_targets = pd.read_csv(daily_targets_path)
    daily_summary = pd.read_csv(daily_summary_path)
    switch_log = pd.read_csv(switch_log_path)
    metrics = pd.read_csv(metrics_path)

    assert return_code == 0
    assert "paper-tracking export" in captured.out
    assert "research-only" in captured.out
    assert daily_targets_path.exists()
    assert daily_summary_path.exists()
    assert switch_log_path.exists()
    assert metrics_path.exists()
    assert daily_targets.columns.tolist() == [
        "date",
        "symbol",
        "prediction",
        "selected",
        "signal_rank",
        "target_weight",
        "close",
    ]
    assert "trade_required" in daily_summary.columns
    assert daily_summary["trade_required"].dtype == bool
    assert {
        "candidate_name",
        "candidate_family",
        "model_type",
        "tracking_start",
        "tracking_end",
        "total_return",
        "sharpe_like",
        "max_drawdown",
        "final_equity",
        "total_transaction_cost",
        "mean_daily_turnover",
        "max_daily_turnover",
    }.issubset(metrics.columns)
    assert metrics.loc[0, "candidate_name"] == "linear_smooth10_margin002"
    assert metrics.loc[0, "candidate_family"] == "baseline"
    assert metrics.loc[0, "model_type"] == "linear_regression"
    assert metrics.loc[0, "tracking_start"] == "2021-03-01"
    assert metrics.loc[0, "tracking_end"] == "2021-03-31"
    assert set(switch_log["reason"].dropna()).issubset({"selection_changed"})


def test_build_switch_log_only_includes_changed_selections() -> None:
    daily_summary = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2021-01-01"),
                "selected_symbols": "SPY",
                "previous_selected_symbols": "",
                "turnover": 0.0,
            },
            {
                "date": pd.Timestamp("2021-01-02"),
                "selected_symbols": "SPY",
                "previous_selected_symbols": "SPY",
                "turnover": 0.0,
            },
            {
                "date": pd.Timestamp("2021-01-03"),
                "selected_symbols": "QQQ",
                "previous_selected_symbols": "SPY",
                "turnover": 2.0,
            },
            {
                "date": pd.Timestamp("2021-01-04"),
                "selected_symbols": "QQQ",
                "previous_selected_symbols": "QQQ",
                "turnover": 0.0,
            },
        ]
    )

    switch_log = _build_switch_log(daily_summary)

    assert switch_log["date"].tolist() == [pd.Timestamp("2021-01-03")]
    assert switch_log["previous_selected_symbols"].tolist() == ["SPY"]
    assert switch_log["selected_symbols"].tolist() == ["QQQ"]
    assert switch_log["reason"].tolist() == ["selection_changed"]
