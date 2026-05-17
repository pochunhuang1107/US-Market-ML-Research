from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli import run_tiingo_modern_baseline_sweep as tiingo_cli
from us_market_ml.cli.run_tiingo_modern_baseline_sweep import (
    BASELINE_MODEL_TYPES,
    DATASET_SOURCE_LABEL,
    MIN_HOLDING_DAYS_VALUES,
    SWITCH_MARGIN_VALUES,
    TOP_K_VALUES,
    TRANSACTION_COST_BPS_VALUES,
    build_tiingo_modern_baseline_sweep_plan,
    run_tiingo_modern_baseline_sweep_from_args,
)
from us_market_ml.experiments import SweepFoldConfig


def test_tiingo_modern_baseline_plan_uses_tiingo_only_classical_defaults(
    tmp_path: Path,
) -> None:
    plan = build_tiingo_modern_baseline_sweep_plan(
        market_dataset_glob="/tmp/Market Data/tiingo/**/*.parquet",
        fold_plans=[_fold_plan()],
        output_dir=tmp_path,
        run_name="tiingo",
        resume=True,
        max_workers=4,
    )
    config = plan.config

    assert config.universes[0].name == "tiingo_modern_dynamic"
    assert str(config.universes[0].data_path) == "/tmp/Market Data/tiingo/**/*.parquet"
    assert config.universe_mode == "dynamic"
    assert config.feature_selection_scope == "fold_train"
    assert config.min_history_rows == 252
    assert config.target_col == "excess_return_vs_spy_20d"
    assert config.label_horizon_days == 20
    assert config.feature_set_versions == ["default_features_v1"]
    assert [model.model_type for model in config.models] == list(BASELINE_MODEL_TYPES)
    assert all(model.model_family == "baseline" for model in config.models)
    assert not any(model.model_family == "mlp" for model in config.models)
    assert {signal.top_k for signal in config.signals} == set(TOP_K_VALUES)
    assert {signal.min_holding_days for signal in config.signals} == set(
        MIN_HOLDING_DAYS_VALUES
    )
    assert {signal.switch_margin for signal in config.signals} == set(SWITCH_MARGIN_VALUES)
    assert {signal.allocation_mode for signal in config.signals} == {"equal_weight"}
    assert config.transaction_cost_bps_values == list(TRANSACTION_COST_BPS_VALUES)
    assert plan.estimated_total_configs == 405


def test_tiingo_modern_baseline_cli_dry_run_writes_preflight_and_counts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_path = tmp_path / "market.csv"
    _synthetic_market_data().to_csv(data_path, index=False)

    exit_code = run_tiingo_modern_baseline_sweep_from_args(
        [
            "--dry-run",
            "--market-dataset-glob",
            str(data_path),
            "--output-dir",
            str(tmp_path / "study"),
            "--run-name",
            "tiingo_modern_baseline_sweep",
            "--start-date",
            "2020-01-01",
            "--end-date",
            "2021-04-30",
            "--train-window-years",
            "1",
            "--validation-window-months",
            "1",
            "--test-window-months",
            "3",
            "--step-months",
            "6",
            "--min-history-rows",
            "5",
            "--min-train-eligible-rows",
            "1",
            "--max-workers",
            "2",
        ]
    )

    captured = capsys.readouterr()
    output_dir = tmp_path / "study"
    preflight_csv = output_dir / "tiingo_modern_baseline_sweep_preflight.csv"
    preflight_md = output_dir / "tiingo_modern_baseline_sweep_preflight.md"
    skipped = output_dir / "tiingo_modern_baseline_sweep_skipped_keys.txt"
    preflight = pd.read_csv(preflight_csv)

    assert exit_code == 0
    assert preflight_csv.exists()
    assert preflight_md.exists()
    assert skipped.exists()
    assert len(preflight) == 405
    assert set(preflight["dataset_source"]) == {DATASET_SOURCE_LABEL}
    assert set(preflight["model_family"]) == {"baseline"}
    assert set(preflight["model_type"]) == set(BASELINE_MODEL_TYPES)
    assert set(preflight["top_k"]) == set(TOP_K_VALUES)
    assert set(preflight["transaction_cost_bps"]) == set(TRANSACTION_COST_BPS_VALUES)
    assert set(preflight["min_holding_days"]) == set(MIN_HOLDING_DAYS_VALUES)
    assert set(preflight["switch_margin"].round(3)) == {0.0, 0.002, 0.005}
    assert preflight["selected_feature_count"].min() > 0
    assert preflight["valid_config"].all()

    assert "dataset source label: tiingo_only" in captured.out
    assert "total configs/runs: 405" in captured.out
    assert "valid configs/runs: 405" in captured.out
    assert "skipped configs/runs: 0" in captured.out
    assert "already completed configs if RESUME=1: 0" in captured.out
    assert "failed checkpoint count if RESUME=1: 0" in captured.out
    assert "pending configs for this batch: 405" in captured.out
    assert "model count: 5" in captured.out
    assert "model types: linear_regression, ridge, random_forest, extra_trees, gradient_boosting" in captured.out

    markdown = preflight_md.read_text(encoding="utf-8")
    assert "Dataset source: `tiingo_only`" in markdown
    assert "Model types:" in markdown
    assert "Momentum baseline: deferred" in markdown
    assert "MLP is intentionally deferred" in markdown
    assert "Ctrl+C / Resume Notes" in markdown


def test_tiingo_modern_baseline_missing_default_path_fails_helpfully(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_default = str(tmp_path / "missing" / "**" / "*.parquet")
    monkeypatch.delenv("MARKET_DATASET_GLOB", raising=False)
    monkeypatch.setattr(tiingo_cli, "DEFAULT_DATASET_GLOB", missing_default)

    with pytest.raises(FileNotFoundError, match="default Tiingo-only dataset"):
        run_tiingo_modern_baseline_sweep_from_args(
            [
                "--dry-run",
                "--output-dir",
                str(tmp_path / "study"),
                "--start-date",
                "2020-01-01",
                "--end-date",
                "2021-04-30",
                "--train-window-years",
                "1",
                "--validation-window-months",
                "1",
                "--test-window-months",
                "1",
            ]
        )


def test_tiingo_modern_baseline_market_dataset_env_override_works(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_path = tmp_path / "override_market.csv"
    _synthetic_market_data().to_csv(data_path, index=False)
    monkeypatch.setenv("MARKET_DATASET_GLOB", str(data_path))

    exit_code = run_tiingo_modern_baseline_sweep_from_args(
        [
            "--dry-run",
            "--output-dir",
            str(tmp_path / "study"),
            "--start-date",
            "2020-01-01",
            "--end-date",
            "2021-04-30",
            "--train-window-years",
            "1",
            "--validation-window-months",
            "1",
            "--test-window-months",
            "3",
            "--step-months",
            "6",
            "--min-history-rows",
            "5",
            "--min-train-eligible-rows",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"dataset glob: {data_path}" in captured.out


def test_tiingo_modern_baseline_min_train_guard_marks_configs_invalid(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "market.csv"
    _synthetic_market_data().to_csv(data_path, index=False)

    exit_code = run_tiingo_modern_baseline_sweep_from_args(
        [
            "--dry-run",
            "--market-dataset-glob",
            str(data_path),
            "--output-dir",
            str(tmp_path / "study"),
            "--start-date",
            "2020-01-01",
            "--end-date",
            "2021-04-30",
            "--train-window-years",
            "1",
            "--validation-window-months",
            "1",
            "--test-window-months",
            "3",
            "--step-months",
            "6",
            "--min-history-rows",
            "5",
            "--min-train-eligible-rows",
            "999999",
        ]
    )

    output_dir = tmp_path / "study"
    preflight = pd.read_csv(output_dir / "tiingo_modern_baseline_sweep_preflight.csv")
    skipped_keys = [
        line
        for line in (output_dir / "tiingo_modern_baseline_sweep_skipped_keys.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    assert exit_code == 0
    assert not preflight["valid_config"].any()
    assert preflight["invalid_reason"].str.contains(
        "train_eligible_row_count below min_train_eligible_rows",
        regex=False,
    ).all()
    assert len(skipped_keys) == len(preflight)


def test_tiingo_modern_baseline_shell_runner_has_ctrl_c_safe_helpers() -> None:
    script_path = Path("scripts/run_tiingo_modern_baseline_sweep.sh")
    script = script_path.read_text(encoding="utf-8")

    assert os.access(script_path, os.X_OK)
    assert "set -euo pipefail" in script
    assert "MARKET_DATASET_GLOB=" in script
    assert "OUTPUT_DIR=" in script
    assert "RUN_NAME=" in script
    assert "START_DATE=" in script
    assert "END_DATE=" in script
    assert "MAX_WORKERS=" in script
    assert "RESUME=" in script
    assert "DRY_RUN=" in script
    assert "STEP_MONTHS=" in script
    assert "TRAIN_WINDOW_YEARS=" in script
    assert "VALIDATION_WINDOW_MONTHS=" in script
    assert "TEST_WINDOW_MONTHS=" in script
    assert "terminate_process_tree" in script
    assert "wait_with_timeout" in script
    assert "cleanup_on_interrupt" in script
    assert "forwarding SIGINT to study process" in script
    assert "sending SIGTERM" in script
    assert "sending SIGKILL" in script
    assert "sleep 60" in script
    assert "completed %s / pending %s / total %s" in script
    assert "local -n" not in script

    subprocess.run(["bash", "-n", str(script_path)], check=True)
    subprocess.run(
        [
            "bash",
            "-c",
            (
                "TIINGO_BASELINE_SWEEP_TEST_HELPERS=1; "
                "source scripts/run_tiingo_modern_baseline_sweep.sh; "
                "declare -f terminate_process_tree >/dev/null; "
                "declare -f wait_with_timeout >/dev/null"
            ),
        ],
        check=True,
    )


def _fold_plan() -> tiingo_cli.TiingoModernBaselineFoldPlan:
    return tiingo_cli.TiingoModernBaselineFoldPlan(
        fold=SweepFoldConfig(
            name="tw1y_v1m_t1m_q01_20210201_20210228",
            train_start="2020-01-01",
            train_end="2020-12-31",
            validation_start="2021-01-01",
            validation_end="2021-01-31",
            test_start="2021-02-01",
            test_end="2021-02-28",
        ),
        train_window_years=1,
        validation_window_months=1,
        test_window_months=1,
    )


def _synthetic_market_data() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2020-01-01", "2021-04-30", freq="D")
    symbols = ("SPY", "QQQ", "TLT", "IWM", "GLD", "VNQ")
    for symbol_index, symbol in enumerate(symbols):
        base_price = 100.0 + symbol_index * 10.0
        for day_index, date in enumerate(dates):
            close = base_price + day_index * (0.1 + symbol_index * 0.02)
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close - 0.1,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1_000_000 + symbol_index * 10_000 + day_index,
                    "vwap": close,
                    "year": date.year,
                    "month": date.month,
                    "transactions": 10_000 + day_index,
                }
            )
    return pd.DataFrame(rows)
