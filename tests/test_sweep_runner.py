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
from us_market_ml.experiments.sweep_runner import (
    RESULT_COLUMNS,
    _run_one_configuration,
    _to_walk_forward_fold,
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


def _header_count(path: Path) -> int:
    header = ",".join(RESULT_COLUMNS)
    return sum(line == header for line in path.read_text().splitlines())


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
    assert build_sweep_run_key(
        universe,
        fold,
        model,
        signal,
        label_horizon_days=0,
    ) != build_sweep_run_key(
        universe,
        fold,
        model,
        signal,
        label_horizon_days=20,
    )

    scaled_model = SweepModelConfig(
        model_family="mlp",
        hidden_units=(16,),
        learning_rate=0.001,
        dropout=0.1,
        scale_target=True,
    )
    scaled_key = build_sweep_run_key(universe, fold, scaled_model, signal)
    unscaled_key = build_sweep_run_key(universe, fold, model, signal)
    assert scaled_key != unscaled_key
    assert "scaletarget-true" in scaled_key
    assert "scaletarget-false" in unscaled_key

    smoothed_signal = SweepSignalConfig(top_k=2, min_holding_days=3, switch_margin=0.02)
    smoothed_key = build_sweep_run_key(universe, fold, model, smoothed_signal)
    assert smoothed_key != unscaled_key
    assert "hold-3d" in smoothed_key
    assert "margin-0.02" in smoothed_key

    cost_key = build_sweep_run_key(universe, fold, model, signal, transaction_cost_bps=10.0)
    assert cost_key != unscaled_key
    assert "cost-10bps" in cost_key


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
    assert results["label_horizon_days"].tolist() == [0, 0, 0]
    assert results["transaction_cost_bps"].tolist() == [0.0, 0.0, 0.0]
    assert results["scale_target"].tolist() == [False, False, False]
    assert results_path.exists()
    assert completed_path.exists()
    assert pd.read_csv(results_path)["split"].tolist() == ["train", "validation", "test"]
    assert _header_count(results_path) == 1
    assert len(completed_path.read_text().splitlines()) == 1


def test_run_sweep_result_rows_include_label_horizon_days(tmp_path: Path) -> None:
    data_path = tmp_path / "etf_daily.csv"
    _write_csv(data_path)
    config = SweepRunConfig(
        universes=[SweepUniverseConfig(name="core", data_path=data_path)],
        folds=[_fold()],
        models=[SweepModelConfig(model_family="baseline", model_type="ridge")],
        signals=[SweepSignalConfig(top_k=2)],
        label_horizon_days=20,
        output_dir=tmp_path / "sweeps",
        run_name="purged",
    )

    results = run_sweep(config)

    assert "label_horizon_days" in results.columns
    assert results["label_horizon_days"].tolist() == [20, 20, 20]


def test_run_sweep_result_rows_include_smoothing_parameters(tmp_path: Path) -> None:
    data_path = tmp_path / "etf_daily.csv"
    _write_csv(data_path)
    config = SweepRunConfig(
        universes=[SweepUniverseConfig(name="core", data_path=data_path)],
        folds=[_fold()],
        models=[SweepModelConfig(model_family="baseline", model_type="ridge")],
        signals=[SweepSignalConfig(top_k=1, min_holding_days=3, switch_margin=0.05)],
        output_dir=tmp_path / "sweeps",
        run_name="smoothed",
    )

    results = run_sweep(config)

    assert "min_holding_days" in results.columns
    assert "switch_margin" in results.columns
    assert results["min_holding_days"].tolist() == [3, 3, 3]
    assert results["switch_margin"].tolist() == [0.05, 0.05, 0.05]


def test_run_sweep_result_rows_include_transaction_cost_bps(tmp_path: Path) -> None:
    data_path = tmp_path / "etf_daily.csv"
    _write_csv(data_path)
    config = SweepRunConfig(
        universes=[SweepUniverseConfig(name="core", data_path=data_path)],
        folds=[_fold()],
        models=[SweepModelConfig(model_family="baseline", model_type="ridge")],
        signals=[SweepSignalConfig(top_k=1)],
        transaction_cost_bps_values=[10.0],
        output_dir=tmp_path / "sweeps",
        run_name="costs",
    )

    results = run_sweep(config)

    assert "transaction_cost_bps" in results.columns
    assert results["transaction_cost_bps"].tolist() == [10.0, 10.0, 10.0]


def test_run_one_configuration_result_rows_include_mlp_scale_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe = SweepUniverseConfig(name="core", data_path=tmp_path / "etf_daily.csv")
    fold_config = _fold()
    signal = SweepSignalConfig(top_k=2)
    model = SweepModelConfig(model_family="mlp", hidden_units=(16,), scale_target=True)
    config = SweepRunConfig(
        universes=[universe],
        folds=[fold_config],
        models=[model],
        signals=[signal],
    )

    def fake_run_mlp(
        raw_df: pd.DataFrame,
        fold: object,
        model_config: SweepModelConfig,
        signal_config: SweepSignalConfig,
        run_config: SweepRunConfig,
        transaction_cost_bps: float,
    ) -> pd.DataFrame:
        assert model_config.scale_target is True
        assert transaction_cost_bps == 0.0
        return pd.DataFrame(
            [
                {
                    "split": "test",
                    "model_type": "mlp",
                    "target_col": run_config.target_col,
                    "total_return": 0.1,
                    "mean_daily_return": 0.001,
                    "daily_volatility": 0.01,
                    "sharpe_like": 1.0,
                    "max_drawdown": -0.05,
                    "final_equity": 1.1,
                    "epochs_trained": 3,
                }
            ]
        )

    monkeypatch.setattr("us_market_ml.experiments.sweep_runner._run_mlp", fake_run_mlp)

    rows = _run_one_configuration(
        raw_df=pd.DataFrame(),
        universe=universe,
        fold=_to_walk_forward_fold(fold_config),
        model=model,
        signal=signal,
        config=config,
        run_key="run_1",
    )

    assert rows[0]["scale_target"] is True


def test_run_sweep_appends_multiple_runs_without_duplicate_result_headers(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "etf_daily.csv"
    _write_csv(data_path)
    config = SweepRunConfig(
        universes=[SweepUniverseConfig(name="core", data_path=data_path)],
        folds=[_fold()],
        models=[SweepModelConfig(model_family="baseline", model_type="ridge")],
        signals=[SweepSignalConfig(top_k=1), SweepSignalConfig(top_k=2)],
        output_dir=tmp_path / "sweeps",
        run_name="append_case",
    )

    results = run_sweep(config)

    results_path = tmp_path / "sweeps" / "append_case_results.csv"
    completed_path = tmp_path / "sweeps" / "append_case_completed_keys.txt"
    assert len(results) == 6
    assert len(pd.read_csv(results_path)) == 6
    assert _header_count(results_path) == 1
    assert len(completed_path.read_text().splitlines()) == 2


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
    assert _header_count(tmp_path / "sweeps" / "sweep_results.csv") == 1
    pd.testing.assert_frame_equal(
        resumed_results.reset_index(drop=True),
        pd.read_csv(tmp_path / "sweeps" / "sweep_results.csv").reset_index(drop=True),
    )


def test_run_sweep_resume_handles_empty_existing_results_file(tmp_path: Path) -> None:
    data_path = tmp_path / "etf_daily.csv"
    _write_csv(data_path)
    output_dir = tmp_path / "sweeps"
    output_dir.mkdir()
    (output_dir / "empty_results.csv").touch()
    config = _baseline_config(tmp_path, data_path, run_name="empty")

    results = run_sweep(config)

    results_path = output_dir / "empty_results.csv"
    assert len(results) == 3
    assert len(pd.read_csv(results_path)) == 3
    assert _header_count(results_path) == 1


def test_run_sweep_resume_false_resets_old_files_and_reruns_from_scratch(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "etf_daily.csv"
    _write_csv(data_path)
    config = _baseline_config(tmp_path, data_path, run_name="clean_start")
    first_results = run_sweep(config)
    output_dir = tmp_path / "sweeps"
    failed_path = output_dir / "clean_start_failed_keys.txt"
    failed_path.write_text("old_run\told error\n")

    clean_config = SweepRunConfig(
        universes=config.universes,
        folds=config.folds,
        models=config.models,
        signals=config.signals,
        target_col=config.target_col,
        initial_equity=config.initial_equity,
        prediction_col=config.prediction_col,
        output_dir=config.output_dir,
        run_name=config.run_name,
        resume=False,
    )

    rerun_results = run_sweep(clean_config)

    results_path = output_dir / "clean_start_results.csv"
    completed_path = output_dir / "clean_start_completed_keys.txt"
    assert len(rerun_results) == len(first_results) == 3
    assert len(pd.read_csv(results_path)) == 3
    assert _header_count(results_path) == 1
    assert len(completed_path.read_text().splitlines()) == 1
    assert failed_path.read_text() == ""


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
    failed_lines = failed_path.read_text().splitlines()
    assert len(failed_lines) == 1
    run_key, error = failed_lines[0].split("\t", maxsplit=1)
    assert "not_a_model" in run_key
    assert "ValueError:" in error


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
    assert not bool(row["scale_target"])


def test_summarize_sweep_results_groups_by_scale_target_for_mlp() -> None:
    results = pd.DataFrame(
        [
            {
                "run_key": "run_1",
                "universe": "core",
                "split": "test",
                "model_family": "mlp",
                "model_type": "mlp",
                "top_k": 2,
                "hidden_units": "16",
                "learning_rate": 0.001,
                "dropout": 0.0,
                "scale_target": False,
                "total_return": 0.1,
                "sharpe_like": 1.0,
                "max_drawdown": -0.05,
            },
            {
                "run_key": "run_2",
                "universe": "core",
                "split": "test",
                "model_family": "mlp",
                "model_type": "mlp",
                "top_k": 2,
                "hidden_units": "16",
                "learning_rate": 0.001,
                "dropout": 0.0,
                "scale_target": True,
                "total_return": 0.2,
                "sharpe_like": 2.0,
                "max_drawdown": -0.04,
            },
        ]
    )

    summary = summarize_sweep_results(results)

    assert "scale_target" in summary.columns
    assert len(summary) == 2
    assert set(summary["scale_target"].tolist()) == {False, True}


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
    assert results["scale_target"].tolist() == [False, False, False]
    assert results["epochs_trained"].notna().all()
