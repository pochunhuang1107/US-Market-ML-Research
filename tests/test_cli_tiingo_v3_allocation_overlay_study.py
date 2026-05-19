from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd

from us_market_ml.allocation_overlays import default_allocation_overlay_specs
from us_market_ml.cli.run_tiingo_v3_allocation_overlay_study import (
    build_tiingo_v3_allocation_overlay_study_folds,
    build_tiingo_v3_allocation_overlay_study_plan,
    run_tiingo_v3_allocation_overlay_study_from_args,
)


def test_allocation_overlay_study_default_scale_matches_expected_grid(tmp_path: Path) -> None:
    folds = build_tiingo_v3_allocation_overlay_study_folds()
    overlays = default_allocation_overlay_specs()
    plan = build_tiingo_v3_allocation_overlay_study_plan(
        market_dataset_glob="/tmp/Market Data/tiingo/**/*.parquet",
        fold_plans=folds,
        overlay_specs=overlays,
        output_dir=tmp_path,
        run_name="allocation_overlay",
        resume=True,
        max_workers=4,
    )

    assert len(folds) == 54
    assert len(overlays) == 9
    assert plan.estimated_total_overlay_evaluations == 486


def test_allocation_overlay_cli_dry_run_writes_preflight_and_counts(
    tmp_path: Path,
    capsys,
) -> None:
    data_path = tmp_path / "market.csv"
    _synthetic_market_data().to_csv(data_path, index=False)
    output_dir = tmp_path / "study"

    exit_code = run_tiingo_v3_allocation_overlay_study_from_args(
        [
            "--dry-run",
            "--market-dataset-glob",
            str(data_path),
            "--output-dir",
            str(output_dir),
            "--run-name",
            "tiingo_v3_allocation_overlay_study",
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
    preflight_csv = output_dir / "tiingo_v3_allocation_overlay_preflight.csv"
    preflight_md = output_dir / "tiingo_v3_allocation_overlay_preflight.md"
    completed = output_dir / "tiingo_v3_allocation_overlay_completed_keys.txt"
    failed = output_dir / "tiingo_v3_allocation_overlay_failed_keys.txt"
    terminal_log = output_dir / "tiingo_v3_allocation_overlay_terminal.log"
    preflight = pd.read_csv(preflight_csv)

    assert exit_code == 0
    assert preflight_csv.exists()
    assert preflight_md.exists()
    assert completed.exists()
    assert failed.exists()
    assert terminal_log.exists()
    assert len(preflight) == 9
    assert preflight["overlay_name"].nunique() == 9
    assert preflight["fold_name"].nunique() == 1
    assert set(preflight["feature_set_version"]) == {"benchmark_gap_features_v3"}
    assert set(preflight["model_type"]) == {"ridge"}
    assert set(preflight["top_k"]) == {5}
    assert preflight["valid_config"].all()

    assert "generated fold count: 1" in captured.out
    assert "overlay count: 9" in captured.out
    assert "total overlay evaluations: 9" in captured.out
    assert "equal_weight_top5_baseline" in captured.out
    assert "tree_screen: not run" in captured.out
    assert "MLP: not run" in captured.out
    assert "feature v4: not added" in captured.out
    assert "production signal export: not run" in captured.out

    markdown = preflight_md.read_text(encoding="utf-8")
    assert "Tiingo V3 Allocation Overlay Study Preflight" in markdown
    assert "Overlay count: `9`" in markdown
    assert "Total overlay evaluations: `9`" in markdown
    assert "Model fitting is shared once per pending fold" in markdown
    assert "feature v4, S3 readers, and production signal export" in markdown


def test_allocation_overlay_cli_writes_results_and_checkpoints(tmp_path: Path) -> None:
    data_path = tmp_path / "market.csv"
    _synthetic_market_data().to_csv(data_path, index=False)
    output_dir = tmp_path / "study"

    exit_code = run_tiingo_v3_allocation_overlay_study_from_args(
        [
            "--market-dataset-glob",
            str(data_path),
            "--output-dir",
            str(output_dir),
            "--run-name",
            "tiingo_v3_allocation_overlay_study",
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
            "--no-resume",
        ]
    )

    results = pd.read_csv(output_dir / "tiingo_v3_allocation_overlay_results.csv")
    completed = (
        output_dir / "tiingo_v3_allocation_overlay_completed_keys.txt"
    ).read_text(encoding="utf-8")

    assert exit_code == 0
    assert len(results) == 9
    assert results["overlay_name"].nunique() == 9
    assert set(results["feature_set_version"]) == {"benchmark_gap_features_v3"}
    assert set(results["model_type"]) == {"ridge"}
    assert results["overlay_weight_valid"].all()
    assert len([line for line in completed.splitlines() if line.strip()]) == 9


def test_allocation_overlay_shell_runner_is_local_and_executable() -> None:
    script_path = Path("scripts/run_tiingo_v3_allocation_overlay_study.sh")
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
    assert "FEATURE_SET_VERSION=" in script
    assert "benchmark_gap_features_v3" in script
    assert "TRANSACTION_COST_BPS=" in script
    assert "run_tiingo_v3_allocation_overlay_study" in script
    assert "create_tiingo_v3_allocation_overlay_report" in script
    assert "total overlay evaluations" in script
    assert "overlay names" in script
    assert "sleep 60" in script
    assert "No S3/cloud/GPU" in script
    assert "no MLP" in script
    assert "no tree models" in script
    assert "no feature v4" in script
    assert "no production signal export" in script
    assert "cleanup_on_interrupt" in script
    assert "terminate_process_tree" in script
    assert "wait_with_timeout" in script

    subprocess.run(["bash", "-n", str(script_path)], check=True)
    subprocess.run(
        [
            "bash",
            "-c",
            (
                "TIINGO_V3_ALLOCATION_OVERLAY_STUDY_TEST_HELPERS=1; "
                "source scripts/run_tiingo_v3_allocation_overlay_study.sh; "
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
