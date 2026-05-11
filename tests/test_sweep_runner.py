from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.experiments import (
    SweepFoldConfig,
    SweepModelConfig,
    SweepRunConfig,
    SweepSignalConfig,
    SweepUniverseConfig,
    build_sweep_run_key,
    run_sweep,
    summarize_sweep_results,
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


def _fold() -> SweepFoldConfig:
    return SweepFoldConfig(
        name="fold_1",
        train_start="2020-01-01",
        train_end="2020-12-31",
        validation_start="2021-01-01",
        validation_end="2021-03-31",
        test_start="2021-04-01",
        test_end="2021-06-30",
    )


def _write_csv(path: Path) -> None:
    _synthetic_etf_ohlcv().to_csv(path, index=False)


def _baseline_config(tmp_path: Path, data_path: Path, run_name: str = "sweep") -> SweepRunConfig:
    return SweepRunConfig(
        universes=[SweepUniverseConfig(name="core", data_path=data_path)],
        folds=[_fold()],
        models=[SweepModelConfig(model_family="baseline", model_type="ridge")],
        signals=[SweepSignalConfig(top_k=2)],
        output_dir=tmp_path / "sweeps",
        run_name=run_name,
    )


def test_build_sweep_run_key_is_deterministic_and_changes_with_hyperparameters(
    tmp_path: Path,
) -> None:
    universe = SweepUniverseConfig(name="core", data_path=tmp_path / "data.csv")
    fold = _fold()
    signal = SweepSignalConfig(top_k=2)
    model = SweepModelConfig(
        model_family="mlp",
        hidden_units=(16,),
        learning_rate=0.001,
        dropout=0.1,
    )

    assert build_sweep_run_key(universe, fold, model, signal) == build_sweep_run_key(
        universe, fold, model, signal
    )

    changed_model = SweepModelConfig(
        model_family="mlp",
        hidden_units=(32,),
        learning_rate=0.001,
        dropout=0.1,
    )
    assert build_sweep_run_key(universe, fold, model, signal) != build_sweep_run_key(
        universe, fold, changed_model, signal
    )


def test_run_sweep_runs_baseline_and_writes_results_and_completed_checkpoint(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "etf_daily.csv"
    _write_csv(data_path)
    config = _baseline_config(tmp_path, data_path)

    results = run_sweep(config)

    results_path = tmp_path / "sweeps" / "sweep_results.csv"
    completed_path = tmp_path / "sweeps" / "sweep_completed_keys.txt"
    assert len(results) == 3
    assert results_path.exists()
    assert completed_path.exists()
    assert pd.read_csv(results_path)["split"].tolist() == ["train", "validation", "test"]
    assert len(completed_path.read_text().splitlines()) == 1


def test_run_sweep_resume_skips_completed_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_path = tmp_path / "etf_daily.csv"
    _write_csv(data_path)
    config = _baseline_config(tmp_path, data_path)
    first_results = run_sweep(config)

    def fail_if_called(*args: object, **kwargs: object) -> object:
        raise AssertionError("completed run should have been skipped")

    monkeypatch.setattr(
        "us_market_ml.experiments.sweep_runner.run_baseline_e2e_experiment",
        fail_if_called,
    )

    resumed_results = run_sweep(config)

    assert len(resumed_results) == len(first_results) == 3
    pd.testing.assert_frame_equal(
        resumed_results.reset_index(drop=True),
        pd.read_csv(tmp_path / "sweeps" / "sweep_results.csv").reset_index(drop=True),
    )


def test_run_sweep_records_failed_run_and_continues(tmp_path: Path) -> None:
    data_path = tmp_path / "etf_daily.csv"
    _write_csv(data_path)
    config = SweepRunConfig(
        universes=[SweepUniverseConfig(name="core", data_path=data_path)],
        folds=[_fold()],
        models=[
            SweepModelConfig(model_family="baseline", model_type="not_a_model"),
            SweepModelConfig(model_family="baseline", model_type="ridge"),
        ],
        signals=[SweepSignalConfig(top_k=2)],
        output_dir=tmp_path / "sweeps",
        run_name="failure_case",
    )

    results = run_sweep(config)

    failed_path = tmp_path / "sweeps" / "failure_case_failed_keys.txt"
    assert len(results) == 3
    assert failed_path.exists()
    assert "not_a_model" in failed_path.read_text()


def test_summarize_sweep_results_groups_test_rows() -> None:
    results = pd.DataFrame(
        [
            {
                "run_key": "run_1",
                "universe": "core",
                "split": "train",
                "model_family": "baseline",
                "model_type": "ridge",
                "top_k": 2,
                "hidden_units": "",
                "learning_rate": float("nan"),
                "dropout": float("nan"),
                "total_return": -1.0,
                "sharpe_like": 0.0,
                "max_drawdown": -0.2,
            },
            {
                "run_key": "run_1",
                "universe": "core",
                "split": "test",
                "model_family": "baseline",
                "model_type": "ridge",
                "top_k": 2,
                "hidden_units": "",
                "learning_rate": float("nan"),
                "dropout": float("nan"),
                "total_return": 0.1,
                "sharpe_like": 1.0,
                "max_drawdown": -0.05,
            },
            {
                "run_key": "run_2",
                "universe": "core",
                "split": "test",
                "model_family": "baseline",
                "model_type": "ridge",
                "top_k": 2,
                "hidden_units": "",
                "learning_rate": float("nan"),
                "dropout": float("nan"),
                "total_return": -0.05,
                "sharpe_like": -0.5,
                "max_drawdown": -0.1,
            },
        ]
    )

    summary = summarize_sweep_results(results)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["avg_test_return"] == pytest.approx(0.025)
    assert row["avg_sharpe_like"] == pytest.approx(0.25)
    assert row["avg_max_drawdown"] == pytest.approx(-0.075)
    assert row["positive_rate"] == pytest.approx(0.5)
    assert row["runs"] == 2


def test_run_sweep_runs_tiny_mlp_when_torch_is_available(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    data_path = tmp_path / "etf_daily.csv"
    _write_csv(data_path)
    config = SweepRunConfig(
        universes=[SweepUniverseConfig(name="core", data_path=data_path)],
        folds=[_fold()],
        models=[
            SweepModelConfig(
                model_family="mlp",
                hidden_units=(16,),
                dropout=0.0,
                max_epochs=3,
                patience=2,
                batch_size=16,
            )
        ],
        signals=[SweepSignalConfig(top_k=2)],
        output_dir=tmp_path / "sweeps",
        run_name="mlp",
    )

    results = run_sweep(config)

    assert len(results) == 3
    assert results["model_family"].tolist() == ["mlp", "mlp", "mlp"]
    assert results["epochs_trained"].notna().all()
