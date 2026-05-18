from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli.run_tiingo_focused_linear_ridge_confirmation import (
    DATASET_SOURCE_LABEL,
    DEFAULT_TRANSACTION_COST_BPS_VALUES,
    EXPLICIT_CANDIDATE_SPECS,
    build_focused_confirmation_folds,
    build_focused_confirmation_plan,
    run_tiingo_focused_linear_ridge_confirmation_from_args,
)


def test_focused_confirmation_default_scale_is_smaller_than_core_linear_full_run(
    tmp_path: Path,
) -> None:
    folds = build_focused_confirmation_folds()
    plan = build_focused_confirmation_plan(
        market_dataset_glob="/tmp/Market Data/tiingo/**/*.parquet",
        fold_plans=folds,
        output_dir=tmp_path,
        run_name="focused",
        resume=True,
        max_workers=4,
    )

    assert len(folds) == 54
    assert len(EXPLICIT_CANDIDATE_SPECS) == 11
    assert plan.estimated_total_configs == 1188
    assert plan.estimated_total_configs < 36_936


def test_focused_confirmation_cli_dry_run_writes_preflight_and_counts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_path = tmp_path / "market.csv"
    _synthetic_market_data().to_csv(data_path, index=False)

    exit_code = run_tiingo_focused_linear_ridge_confirmation_from_args(
        [
            "--dry-run",
            "--market-dataset-glob",
            str(data_path),
            "--output-dir",
            str(tmp_path / "study"),
            "--run-name",
            "tiingo_focused_linear_ridge_confirmation",
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
    preflight_csv = output_dir / "tiingo_focused_linear_ridge_confirmation_preflight.csv"
    preflight_md = output_dir / "tiingo_focused_linear_ridge_confirmation_preflight.md"
    skipped = output_dir / "tiingo_focused_linear_ridge_confirmation_skipped_keys.txt"
    preflight = pd.read_csv(preflight_csv)

    assert exit_code == 0
    assert preflight_csv.exists()
    assert preflight_md.exists()
    assert skipped.exists()
    assert len(preflight) == 22
    assert set(preflight["dataset_source"]) == {DATASET_SOURCE_LABEL}
    assert set(preflight["candidate_role"]) == {
        "balanced_reference",
        "drawdown_control_reference",
        "return_reference",
    }
    assert set(preflight["model_family"]) == {"baseline"}
    assert set(preflight["model_type"]) == {"linear_regression", "ridge"}
    assert set(preflight["top_k"]) == {1, 3, 5}
    assert set(preflight["transaction_cost_bps"]) == set(
        DEFAULT_TRANSACTION_COST_BPS_VALUES
    )
    assert 5.0 not in set(preflight["transaction_cost_bps"])
    assert preflight["selected_feature_count"].min() > 0
    assert preflight["valid_config"].all()

    assert "total configs/runs: 22" in captured.out
    assert "valid configs/runs: 22" in captured.out
    assert "skipped configs/runs: 0" in captured.out
    assert "pending configs for this batch: 22" in captured.out
    assert "candidate spec count: 11" in captured.out
    assert "model types: linear_regression, ridge" in captured.out
    assert "top_k values: [1, 3, 5]" in captured.out
    assert "transaction cost values: [10.0, 20.0]" in captured.out
    assert "tree_screen: not run" in captured.out
    assert "MLP: not run" in captured.out

    markdown = preflight_md.read_text(encoding="utf-8")
    assert "Tiingo Focused Linear/Ridge Confirmation Preflight" in markdown
    assert "Candidate spec count: `11`" in markdown
    assert "return_reference" in markdown
    assert "balanced_reference" in markdown
    assert "drawdown_control_reference" in markdown
    assert "Transaction costs bps: `[10.0, 20.0]`" in markdown
    assert "Ctrl+C / Resume Notes" in markdown
    assert "`tree_screen` and MLP are intentionally not part of this run" in markdown


def test_focused_confirmation_env_transaction_cost_override_works(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_path = tmp_path / "market.csv"
    _synthetic_market_data().to_csv(data_path, index=False)
    monkeypatch.setenv("TRANSACTION_COST_BPS_VALUES", "20")

    exit_code = run_tiingo_focused_linear_ridge_confirmation_from_args(
        [
            "--dry-run",
            "--market-dataset-glob",
            str(data_path),
            "--output-dir",
            str(tmp_path / "study"),
            "--run-name",
            "override_cost",
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

    preflight = pd.read_csv(tmp_path / "study" / "override_cost_preflight.csv")
    assert exit_code == 0
    assert len(preflight) == 11
    assert set(preflight["transaction_cost_bps"]) == {20.0}


def test_focused_confirmation_shell_runner_has_ctrl_c_safe_helpers() -> None:
    script_path = Path("scripts/run_tiingo_focused_linear_ridge_confirmation.sh")
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
    assert "MIN_HISTORY_ROWS=" in script
    assert "MIN_TRAIN_ELIGIBLE_ROWS=" in script
    assert "STEP_MONTHS=" in script
    assert "TRAIN_WINDOW_YEARS=" in script
    assert "VALIDATION_WINDOW_MONTHS=" in script
    assert "TEST_WINDOW_MONTHS=" in script
    assert "TRANSACTION_COST_BPS_VALUES=" in script
    assert "run_tiingo_focused_linear_ridge_confirmation" in script
    assert "terminate_process_tree" in script
    assert "wait_with_timeout" in script
    assert "cleanup_on_interrupt" in script
    assert "forwarding SIGINT to study process" in script
    assert "sending SIGTERM" in script
    assert "sending SIGKILL" in script
    assert "sleep 60" in script
    assert "completed %s / pending %s / total %s" in script
    assert "Run-count summary before starting" in script
    assert "local -n" not in script

    subprocess.run(["bash", "-n", str(script_path)], check=True)
    subprocess.run(
        [
            "bash",
            "-c",
            (
                "TIINGO_FOCUSED_LINEAR_RIDGE_CONFIRMATION_TEST_HELPERS=1; "
                "source scripts/run_tiingo_focused_linear_ridge_confirmation.sh; "
                "declare -f terminate_process_tree >/dev/null; "
                "declare -f wait_with_timeout >/dev/null"
            ),
        ],
        check=True,
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
