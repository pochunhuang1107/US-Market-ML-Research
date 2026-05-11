from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli.run_benchmark_report import (
    parse_comma_separated_symbols,
    parse_hidden_units,
    run_benchmark_report_from_args,
)


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


def test_run_benchmark_report_from_args_writes_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("torch")
    import matplotlib

    matplotlib.use("Agg")
    data_path = tmp_path / "etf_daily.csv"
    output_dir = tmp_path / "reports"
    _synthetic_etf_ohlcv().to_csv(data_path, index=False)

    return_code = run_benchmark_report_from_args(
        [
            "--data-path",
            str(data_path),
            "--output-dir",
            str(output_dir),
            "--report-name",
            "cli",
            "--train-start",
            "2020-01-01",
            "--train-end",
            "2020-12-31",
            "--validation-start",
            "2021-01-01",
            "--validation-end",
            "2021-03-31",
            "--test-start",
            "2021-04-01",
            "--test-end",
            "2021-06-30",
            "--top-k",
            "2",
            "--hidden-units",
            "16",
            "--max-epochs",
            "5",
            "--patience",
            "2",
            "--batch-size",
            "16",
        ]
    )

    captured = capsys.readouterr()
    combined_path = output_dir / "cli_combined_equity.csv"
    metrics_path = output_dir / "cli_metrics.csv"
    plot_path = output_dir / "cli_equity.png"
    combined = pd.read_csv(combined_path)
    metrics = pd.read_csv(metrics_path)

    assert return_code == 0
    assert "benchmark report" in captured.out
    assert combined_path.exists()
    assert metrics_path.exists()
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
    assert set(combined["series_name"]) == {"mlp_strategy", "SPY", "QQQ", "equal_weight"}
    assert set(metrics["series_name"]) == {"mlp_strategy", "SPY", "QQQ", "equal_weight"}
    assert {"total_return", "max_drawdown", "final_equity"}.issubset(metrics.columns)


def test_benchmark_report_parser_helpers() -> None:
    assert parse_comma_separated_symbols("spy, qqq") == ["SPY", "QQQ"]
    assert parse_hidden_units("16, 8") == (16, 8)


@pytest.mark.parametrize("value", ["", "0", "16,bad"])
def test_parse_hidden_units_rejects_invalid_values(value: str) -> None:
    with pytest.raises(Exception):  # noqa: B017 - parser helpers expose argparse errors.
        parse_hidden_units(value)
