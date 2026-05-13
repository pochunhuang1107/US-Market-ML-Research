from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli.run_model_signal_export import run_model_signal_export_from_args


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


def test_run_model_signal_export_from_args_writes_csv_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_path = tmp_path / "etf_daily.csv"
    output_path = tmp_path / "curated/model_signals/model_version=linear_v1/signals.csv"
    _synthetic_etf_ohlcv().to_csv(data_path, index=False)

    return_code = run_model_signal_export_from_args(
        [
            "--data-path",
            str(data_path),
            "--output-path",
            str(output_path),
            "--candidate",
            "baseline:linear_smooth10_margin002:linear_regression:1:42:10:0.002",
            "--model-version",
            "linear_smooth10_margin002_v1",
            "--feature-set-version",
            "default_features_v1",
            "--created-at",
            "2026-05-13T00:00:00+00:00",
            "--train-start",
            "2020-08-01",
            "--train-end",
            "2020-12-31",
            "--validation-start",
            "2021-01-01",
            "--validation-end",
            "2021-02-28",
            "--export-start",
            "2021-03-01",
            "--export-end",
            "2021-03-31",
            "--format",
            "csv",
        ]
    )

    captured = capsys.readouterr()
    artifact = pd.read_csv(output_path)

    assert return_code == 0
    assert "model signal export" in captured.out
    assert str(output_path) in captured.out
    assert output_path.exists()
    assert artifact.columns.tolist() == [
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
    ]
    assert set(artifact["model_version"]) == {"linear_smooth10_margin002_v1"}
    assert set(artifact["feature_set_version"]) == {"default_features_v1"}
    assert set(artifact["label_horizon_days"]) == {20}
    assert set(artifact["signal_bucket"]) == {"top", "other"}
    assert pd.api.types.is_numeric_dtype(artifact["score"])
    assert artifact.duplicated(["date", "symbol", "model_version"]).sum() == 0


def test_run_model_signal_export_from_args_excludes_forbidden_runtime_columns(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "etf_daily.csv"
    output_path = tmp_path / "signals.csv"
    _synthetic_etf_ohlcv().to_csv(data_path, index=False)

    run_model_signal_export_from_args(
        [
            "--data-path",
            str(data_path),
            "--output-path",
            str(output_path),
            "--candidate",
            "baseline:linear_smooth10_margin002:linear_regression:1:42:10:0.002",
            "--model-version",
            "linear_smooth10_margin002_v1",
            "--train-start",
            "2020-08-01",
            "--train-end",
            "2020-12-31",
            "--validation-start",
            "2021-01-01",
            "--validation-end",
            "2021-02-28",
            "--export-start",
            "2021-03-01",
            "--export-end",
            "2021-03-31",
            "--format",
            "csv",
        ]
    )

    artifact = pd.read_csv(output_path)
    forbidden_columns = {
        column
        for column in artifact.columns
        if column in {"future_return", "forward_return", "realized_return", "target_return"}
        or column.startswith(
            ("future_return_", "forward_return_", "realized_return_", "target_return_")
        )
        or column == "excess_return_vs_spy_20d"
    }

    assert forbidden_columns == set()
