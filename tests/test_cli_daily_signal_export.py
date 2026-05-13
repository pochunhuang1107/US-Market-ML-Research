from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli.run_daily_signal_export import run_daily_signal_export_from_args
from us_market_ml.exports import validate_model_signal_artifact


def _synthetic_etf_ohlcv() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2020-01-01", periods=560, freq="D")
    symbols = ("SPY", "QQQ", "TLT")

    for symbol_index, symbol in enumerate(symbols):
        base_price = 100.0 + symbol_index * 25.0
        drift = 0.07 + symbol_index * 0.025
        cycle = symbol_index + 2
        for day_index, date in enumerate(dates):
            close = base_price + day_index * drift + (day_index % 17) * cycle * 0.04
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


def _daily_export_args(data_path: Path, output_dir: Path, *extra_args: str) -> list[str]:
    return [
        "--data-path",
        str(data_path),
        "--output-dir",
        str(output_dir),
        "--candidate",
        "baseline:linear_smooth10_margin002:linear_regression:1:42:10:0.002",
        "--model-version",
        "linear_smooth10_margin002_v1",
        "--feature-set-version",
        "default_features_v1",
        "--train-start",
        "2020-08-01",
        "--train-end",
        "2020-12-31",
        "--validation-start",
        "2021-01-01",
        "--validation-end",
        "2021-03-31",
        "--latest-date",
        "2021-05-01",
        "--label-horizon-days",
        "20",
        "--format",
        "csv",
        *extra_args,
    ]


def test_run_daily_signal_export_from_args_writes_csv_and_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_path = tmp_path / "etf_daily.csv"
    output_dir = tmp_path / "model_signals"
    _synthetic_etf_ohlcv().to_csv(data_path, index=False)

    return_code = run_daily_signal_export_from_args(_daily_export_args(data_path, output_dir))

    captured = capsys.readouterr()
    version_dir = output_dir / "model_version=linear_smooth10_margin002_v1"
    signals_path = version_dir / "signals.csv"
    manifest_path = version_dir / "model_signal_manifest.json"
    signals = pd.read_csv(signals_path, parse_dates=["date"])
    manifest = json.loads(manifest_path.read_text())

    assert return_code == 0
    assert "daily model signal export" in captured.out
    assert signals_path.exists()
    assert manifest_path.exists()
    assert manifest["model_version"] == "linear_smooth10_margin002_v1"
    assert manifest["feature_set_version"] == "default_features_v1"
    assert manifest["latest_signal_date"] == "2021-05-01"
    assert manifest["row_count"] == len(signals)
    assert manifest["symbol_count"] == signals["symbol"].nunique()
    assert manifest["candidate_id"] == (
        "baseline:linear_smooth10_margin002:linear_regression:1:42:10:0.002"
    )
    assert manifest["tracking_window"] == {"start": "2021-05-01", "end": "2021-05-01"}
    assert manifest["status"] == "ready"


def test_run_daily_signal_export_output_artifact_schema_is_runtime_safe(tmp_path: Path) -> None:
    data_path = tmp_path / "etf_daily.csv"
    output_dir = tmp_path / "model_signals"
    _synthetic_etf_ohlcv().to_csv(data_path, index=False)

    run_daily_signal_export_from_args(_daily_export_args(data_path, output_dir))

    signals_path = (
        output_dir / "model_version=linear_smooth10_margin002_v1" / "signals.csv"
    )
    signals = pd.read_csv(signals_path, parse_dates=["date"])

    validate_model_signal_artifact(signals)
    assert {
        "date",
        "symbol",
        "model_version",
        "score",
        "predicted_return",
        "rank",
        "signal_bucket",
        "feature_set_version",
        "label_horizon_days",
        "created_at",
    }.issubset(signals.columns)
    forbidden_tokens = ("future", "forward", "realized", "target", "excess_return_vs_")
    assert not [
        column
        for column in signals.columns
        if any(token in column.lower() for token in forbidden_tokens)
    ]
    assert (signals["signal_bucket"] == "top").any()


def test_run_daily_signal_export_merge_existing_deduplicates_keys(tmp_path: Path) -> None:
    data_path = tmp_path / "etf_daily.csv"
    output_dir = tmp_path / "model_signals"
    _synthetic_etf_ohlcv().to_csv(data_path, index=False)

    args = _daily_export_args(data_path, output_dir)
    run_daily_signal_export_from_args(args)
    signals_path = output_dir / "model_version=linear_smooth10_margin002_v1" / "signals.csv"
    first_signals = pd.read_csv(signals_path)
    duplicated = pd.concat([first_signals, first_signals.iloc[[0]]], ignore_index=True)
    duplicated.to_csv(signals_path, index=False)

    run_daily_signal_export_from_args(args)

    merged_signals = pd.read_csv(signals_path, parse_dates=["date"])
    assert not merged_signals.duplicated(["date", "symbol", "model_version"]).any()
    assert len(merged_signals) == len(first_signals)
