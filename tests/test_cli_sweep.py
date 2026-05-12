from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from us_market_ml.cli.run_sweep import (
    parse_float_list,
    parse_fold,
    parse_hidden_units,
    parse_int_list,
    parse_name_path,
    run_sweep_from_args,
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


def _write_csv(path: Path) -> None:
    _synthetic_etf_ohlcv().to_csv(path, index=False)


def test_run_sweep_from_args_runs_tiny_baseline_sweep(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_path = tmp_path / "etf_daily.csv"
    output_dir = tmp_path / "sweeps"
    _write_csv(data_path)

    return_code = run_sweep_from_args(
        [
            "--run-name",
            "cli",
            "--output-dir",
            str(output_dir),
            "--universe",
            f"core:{data_path}",
            "--fold",
            "fold_1:2020-01-01:2020-12-31:2021-01-01:2021-03-31:2021-04-01:2021-06-30",
            "--model-family",
            "baseline",
            "--baseline-models",
            "ridge",
            "--top-k",
            "2",
            "--label-horizon-days",
            "20",
        ]
    )

    captured = capsys.readouterr()
    results = pd.read_csv(output_dir / "cli_results.csv")
    assert return_code == 0
    assert "structured sweep" in captured.out
    assert "result rows: 3" in captured.out
    assert (output_dir / "cli_results.csv").exists()
    assert (output_dir / "cli_completed_keys.txt").exists()
    assert results["label_horizon_days"].tolist() == [20, 20, 20]
    assert results["scale_target"].tolist() == [False, False, False]


def test_run_sweep_from_args_accepts_new_tree_baseline_model_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_model_types: list[str] = []

    def fake_run_sweep(config: Any) -> pd.DataFrame:
        observed_model_types.extend(model.model_type for model in config.models)
        return pd.DataFrame()

    monkeypatch.setattr("us_market_ml.cli.run_sweep.run_sweep", fake_run_sweep)

    return_code = run_sweep_from_args(
        [
            "--run-name",
            "cli_tree_names",
            "--output-dir",
            str(tmp_path / "sweeps"),
            "--universe",
            "core:data/curated/etf_daily_ohlcv.parquet",
            "--fold",
            "fold_1:2020-01-01:2020-12-31:2021-01-01:2021-03-31:2021-04-01:2021-06-30",
            "--model-family",
            "baseline",
            "--baseline-models",
            "ridge,linear_regression,gradient_boosting,extra_trees,random_forest",
            "--top-k",
            "2",
        ]
    )

    assert return_code == 0
    assert observed_model_types == [
        "ridge",
        "linear_regression",
        "gradient_boosting",
        "extra_trees",
        "random_forest",
    ]


def test_run_sweep_from_args_accepts_scale_target_for_mlp(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("torch")
    data_path = tmp_path / "etf_daily.csv"
    output_dir = tmp_path / "sweeps"
    _write_csv(data_path)

    return_code = run_sweep_from_args(
        [
            "--run-name",
            "cli_mlp_scaled",
            "--output-dir",
            str(output_dir),
            "--universe",
            f"core:{data_path}",
            "--fold",
            "fold_1:2020-01-01:2020-12-31:2021-01-01:2021-03-31:2021-04-01:2021-06-30",
            "--model-family",
            "mlp",
            "--hidden-units",
            "8",
            "--learning-rates",
            "0.001",
            "--dropouts",
            "0.0",
            "--seeds",
            "42",
            "--max-epochs",
            "1",
            "--patience",
            "1",
            "--batch-size",
            "256",
            "--top-k",
            "2",
            "--scale-target",
        ]
    )

    captured = capsys.readouterr()
    results = pd.read_csv(output_dir / "cli_mlp_scaled_results.csv")
    assert return_code == 0
    assert "result rows: 3" in captured.out
    assert "scale_target" in results.columns
    assert results["scale_target"].tolist() == [True, True, True]


def test_cli_parser_helpers() -> None:
    universe = parse_name_path("core:data/cache.csv")
    fold = parse_fold(
        "fold_1:2020-01-01:2020-12-31:2021-01-01:2021-03-31:2021-04-01:2021-06-30"
    )

    assert universe.name == "core"
    assert universe.data_path == Path("data/cache.csv")
    assert fold.name == "fold_1"
    assert parse_int_list("1, 2,3") == [1, 2, 3]
    assert parse_float_list("0.001, 0.01") == [0.001, 0.01]
    assert parse_hidden_units("128, 64") == (128, 64)


@pytest.mark.parametrize(
    ("parser", "value"),
    [
        (parse_name_path, "missing_separator"),
        (parse_fold, "too:few:parts"),
        (parse_int_list, ""),
        (parse_float_list, "not_float"),
        (parse_hidden_units, "0"),
    ],
)
def test_cli_parser_helpers_reject_invalid_values(parser: object, value: str) -> None:
    with pytest.raises(Exception):  # noqa: B017 - parser helpers expose argparse errors.
        parser(value)  # type: ignore[operator]
