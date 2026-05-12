from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli.run_cost_sensitivity_report import (
    parse_float_list,
    run_cost_sensitivity_report_from_args,
)


def _synthetic_etf_ohlcv() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2020-01-01", periods=760, freq="D")
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


def test_parse_float_list() -> None:
    assert parse_float_list("0,1,2.5,10") == [0.0, 1.0, 2.5, 10.0]


@pytest.mark.parametrize("value", ["", "0,,10", "0,bad,10"])
def test_parse_float_list_validates(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_float_list(value)


def test_run_cost_sensitivity_report_from_args_writes_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("torch")
    import matplotlib

    matplotlib.use("Agg")
    data_path = tmp_path / "etf_daily.csv"
    output_dir = tmp_path / "reports"
    _synthetic_etf_ohlcv().to_csv(data_path, index=False)

    return_code = run_cost_sensitivity_report_from_args(
        [
            "--data-path",
            str(data_path),
            "--output-dir",
            str(output_dir),
            "--report-name",
            "cost_grid",
            "--fold",
            "fold_1:2020-01-01:2020-09-30:2020-10-01:2020-12-31:2021-01-01:2021-02-28",
            "--candidate",
            "baseline:linear_top1:linear_regression:1:42",
            "--candidate",
            "mlp:mlp_top1:1:16:0.001:0.0:43",
            "--cost-bps",
            "0,10",
            "--max-epochs",
            "5",
            "--patience",
            "2",
            "--batch-size",
            "16",
        ]
    )

    captured = capsys.readouterr()
    sensitivity_path = output_dir / "cost_grid_cost_sensitivity.csv"
    pivot_path = output_dir / "cost_grid_cost_sensitivity_pivot.csv"
    plot_path = output_dir / "cost_grid_cost_sensitivity.png"
    sensitivity = pd.read_csv(sensitivity_path)
    pivot = pd.read_csv(pivot_path)

    assert return_code == 0
    assert "cost sensitivity report" in captured.out
    assert sensitivity_path.exists()
    assert pivot_path.exists()
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
    assert set(sensitivity["candidate_name"]) == {"linear_top1", "mlp_top1"}
    assert set(sensitivity["candidate_family"]) == {"baseline", "mlp"}
    assert set(sensitivity["cost_bps"]) == {0.0, 10.0}
    assert len(sensitivity) == 4
    assert {
        "candidate_name",
        "candidate_family",
        "model_type",
        "top_k",
        "cost_bps",
        "total_return",
        "sharpe_like",
        "max_drawdown",
        "final_equity",
        "total_transaction_cost",
        "mean_daily_turnover",
        "max_daily_turnover",
        "return_decay_vs_0bps",
        "final_equity_decay_vs_0bps",
    }.issubset(sensitivity.columns)
    assert set(pivot["candidate_name"]) == {"linear_top1", "mlp_top1"}
    assert {
        "total_return_0bps",
        "total_return_10bps",
        "final_equity_0bps",
        "final_equity_10bps",
        "sharpe_like_0bps",
        "sharpe_like_10bps",
    }.issubset(pivot.columns)
