from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli import run_mlp_smoke as mlp_smoke_cli
from us_market_ml.cli.run_mlp_smoke import (
    parse_comma_separated_ints,
    run_mlp_smoke_from_args,
)
from us_market_ml.data import S3MarketDataConfig


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


def _write_synthetic_csv(path: Path) -> None:
    _synthetic_etf_ohlcv().to_csv(path, index=False)


def _walk_forward_args() -> list[str]:
    return [
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
        "--initial-equity",
        "1.0",
        "--hidden-units",
        "16",
        "--max-epochs",
        "5",
        "--patience",
        "2",
        "--batch-size",
        "16",
    ]


def _smoke_args(path: Path) -> list[str]:
    return [
        "--data-path",
        str(path),
        *_walk_forward_args(),
    ]


def _s3_smoke_args() -> list[str]:
    return [
        "--s3-bucket",
        "example-bucket",
        "--symbols",
        "spy, qqq, tlt",
        "--years",
        "2020,2021",
        *_walk_forward_args(),
    ]


def test_run_mlp_smoke_from_args_prints_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("torch")
    data_path = tmp_path / "etf_daily.csv"
    _write_synthetic_csv(data_path)

    return_code = run_mlp_smoke_from_args(_smoke_args(data_path))

    captured = capsys.readouterr()
    assert return_code == 0
    assert "MLP smoke test" in captured.out
    assert "input source: local" in captured.out
    assert "hidden units: 16" in captured.out
    assert "max epochs: 5" in captured.out
    assert "patience: 2" in captured.out
    assert "feature count:" in captured.out
    assert "mlp_smoke_fold" in captured.out


def test_run_mlp_smoke_s3_mode_prints_source_and_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("torch")

    def fake_read_s3_curated_etf_data(
        symbols: list[str],
        years: tuple[int, ...],
        config: object,
    ) -> pd.DataFrame:
        assert symbols == ["SPY", "QQQ", "TLT"]
        assert years == (2020, 2021)
        assert config == S3MarketDataConfig(
            bucket="example-bucket",
            curated_prefix="curated/etf_daily_prices",
            region_name="us-west-2",
        )
        return _synthetic_etf_ohlcv()

    monkeypatch.setattr(
        mlp_smoke_cli,
        "read_s3_curated_etf_data",
        fake_read_s3_curated_etf_data,
    )

    return_code = run_mlp_smoke_from_args(_s3_smoke_args())

    captured = capsys.readouterr()
    assert return_code == 0
    assert "s3 bucket: example-bucket" in captured.out
    assert "symbols: SPY, QQQ, TLT" in captured.out
    assert "years: 2020, 2021" in captured.out
    assert "mlp_smoke_fold" in captured.out


def test_run_mlp_smoke_with_local_and_s3_inputs_fails(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        run_mlp_smoke_from_args(
            [
                "--data-path",
                str(tmp_path / "etf_daily.csv"),
                "--s3-bucket",
                "example-bucket",
                "--symbols",
                "SPY",
                "--years",
                "2020",
                *_walk_forward_args(),
            ]
        )


def test_run_mlp_smoke_without_local_or_s3_input_fails() -> None:
    with pytest.raises(SystemExit):
        run_mlp_smoke_from_args(_walk_forward_args())


def test_parse_comma_separated_ints_converts_values() -> None:
    assert parse_comma_separated_ints("128, 64") == (128, 64)
