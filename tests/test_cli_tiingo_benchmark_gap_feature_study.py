from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli.run_tiingo_benchmark_gap_feature_study import (
    BENCHMARK_GAP_FEATURE_SET_VERSION,
    DEFAULT_FEATURE_SET_VERSION,
    DEFAULT_TRANSACTION_COST_BPS_VALUES,
    build_tiingo_benchmark_gap_feature_study_folds,
    build_tiingo_benchmark_gap_feature_study_plan,
    run_tiingo_benchmark_gap_feature_study_from_args,
)


def test_benchmark_gap_feature_study_default_scale_is_compact(tmp_path: Path) -> None:
    folds = build_tiingo_benchmark_gap_feature_study_folds()
    plan = build_tiingo_benchmark_gap_feature_study_plan(
        market_dataset_glob="/tmp/Market Data/tiingo/**/*.parquet",
        fold_plans=folds,
        output_dir=tmp_path,
        run_name="benchmark_gap",
        resume=True,
        max_workers=4,
    )

    assert len(folds) == 54
    assert len(plan.candidate_specs) == 3
    assert plan.feature_set_versions == (
        DEFAULT_FEATURE_SET_VERSION,
        BENCHMARK_GAP_FEATURE_SET_VERSION,
    )
    assert plan.estimated_total_configs == 324
    assert plan.estimated_total_configs < 1188


def test_benchmark_gap_feature_study_cli_dry_run_writes_preflight_and_counts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_path = tmp_path / "market.csv"
    _synthetic_market_data().to_csv(data_path, index=False)

    exit_code = run_tiingo_benchmark_gap_feature_study_from_args(
        [
            "--dry-run",
            "--market-dataset-glob",
            str(data_path),
            "--output-dir",
            str(tmp_path / "study"),
            "--run-name",
            "tiingo_benchmark_gap_feature_study",
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
    preflight_csv = output_dir / "tiingo_benchmark_gap_feature_study_preflight.csv"
    preflight_md = output_dir / "tiingo_benchmark_gap_feature_study_preflight.md"
    skipped = output_dir / "tiingo_benchmark_gap_feature_study_skipped_keys.txt"
    preflight = pd.read_csv(preflight_csv)

    assert exit_code == 0
    assert preflight_csv.exists()
    assert preflight_md.exists()
    assert skipped.exists()
    assert len(preflight) == 6
    assert set(preflight["feature_set_version"]) == {
        DEFAULT_FEATURE_SET_VERSION,
        BENCHMARK_GAP_FEATURE_SET_VERSION,
    }
    assert set(preflight["candidate_role"]) == {
        "historical_reference",
        "return_reference",
        "selected_candidate",
    }
    assert set(preflight["model_family"]) == {"baseline"}
    assert set(preflight["model_type"]) == {"linear_regression", "ridge"}
    assert set(preflight["transaction_cost_bps"]) == set(DEFAULT_TRANSACTION_COST_BPS_VALUES)
    assert preflight["selected_feature_count"].min() > 0
    assert preflight["valid_config"].all()

    feature_counts = preflight.groupby("feature_set_version")[
        "selected_feature_count"
    ].max()
    assert (
        feature_counts[BENCHMARK_GAP_FEATURE_SET_VERSION]
        > feature_counts[DEFAULT_FEATURE_SET_VERSION]
    )

    assert "total configs/runs: 6" in captured.out
    assert "valid configs/runs: 6" in captured.out
    assert "pending configs for this batch: 6" in captured.out
    assert "feature_set_versions: ['default_features_v1', 'benchmark_gap_features_v1']" in captured.out
    assert "candidate spec count: 3" in captured.out
    assert "tree_screen: not run" in captured.out
    assert "MLP: not run" in captured.out
    assert "production signal export: not run" in captured.out

    markdown = preflight_md.read_text(encoding="utf-8")
    assert "Tiingo Benchmark-Gap Feature Study Preflight" in markdown
    assert "Feature set versions: `['default_features_v1', 'benchmark_gap_features_v1']`" in markdown
    assert "Candidate spec count: `3`" in markdown
    assert "`tree_screen`, MLP, and production signal export are intentionally not part" in markdown


def test_benchmark_gap_feature_study_can_disable_reference_candidates(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "market.csv"
    _synthetic_market_data().to_csv(data_path, index=False)

    exit_code = run_tiingo_benchmark_gap_feature_study_from_args(
        [
            "--dry-run",
            "--no-include-reference-candidates",
            "--market-dataset-glob",
            str(data_path),
            "--output-dir",
            str(tmp_path / "study"),
            "--run-name",
            "selected_only",
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

    preflight = pd.read_csv(tmp_path / "study" / "selected_only_preflight.csv")

    assert exit_code == 0
    assert len(preflight) == 2
    assert set(preflight["candidate_role"]) == {"selected_candidate"}


def test_benchmark_gap_feature_study_shell_runner_has_ctrl_c_safe_helpers() -> None:
    script_path = Path("scripts/run_tiingo_benchmark_gap_feature_study.sh")
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
    assert "CREATE_REPORT=" in script
    assert "INCLUDE_REFERENCE_CANDIDATES=" in script
    assert "FEATURE_SET_VERSIONS=" in script
    assert "run_tiingo_benchmark_gap_feature_study" in script
    assert "create_tiingo_benchmark_gap_feature_report" in script
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
                "TIINGO_BENCHMARK_GAP_FEATURE_STUDY_TEST_HELPERS=1; "
                "source scripts/run_tiingo_benchmark_gap_feature_study.sh; "
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

