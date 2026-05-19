from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd

from us_market_ml.cli.run_tiingo_selected_candidate_rich_replay import (
    DAILY_HOLDINGS_COLUMNS,
    DAILY_STRATEGY_EQUITY_COLUMNS,
    FOLD_PERFORMANCE_COLUMNS,
    REBALANCE_LOG_COLUMNS,
    SELECTED_SYMBOL_SUMMARY_COLUMNS,
    UNDERPERFORMANCE_WINDOWS_COLUMNS,
    build_tiingo_selected_candidate_rich_replay,
    build_tiingo_selected_candidate_rich_replay_folds,
    run_tiingo_selected_candidate_rich_replay_from_args,
    write_tiingo_selected_candidate_rich_replay_artifacts,
)


def test_rich_replay_default_fold_count_matches_selected_grid() -> None:
    folds = build_tiingo_selected_candidate_rich_replay_folds()

    assert len(folds) == 54


def test_rich_replay_builds_daily_level_artifacts(tmp_path: Path) -> None:
    data_path = tmp_path / "market.csv"
    _synthetic_market_data().to_csv(data_path, index=False)

    artifacts = build_tiingo_selected_candidate_rich_replay(
        market_dataset_glob=data_path,
        start_date="2020-01-01",
        end_date="2021-04-30",
        train_window_years=(1,),
        validation_window_months=(1,),
        test_window_months=(3,),
        step_months=6,
        transaction_cost_bps=10.0,
        include_references=True,
        min_history_rows=5,
        min_train_eligible_rows=1,
    )

    assert artifacts.daily_strategy_equity.columns.tolist() == DAILY_STRATEGY_EQUITY_COLUMNS
    assert artifacts.daily_holdings.columns.tolist() == DAILY_HOLDINGS_COLUMNS
    assert artifacts.rebalance_log.columns.tolist() == REBALANCE_LOG_COLUMNS
    assert artifacts.fold_performance.columns.tolist() == FOLD_PERFORMANCE_COLUMNS
    assert artifacts.underperformance_windows.columns.tolist() == UNDERPERFORMANCE_WINDOWS_COLUMNS
    assert artifacts.selected_symbol_summary.columns.tolist() == SELECTED_SYMBOL_SUMMARY_COLUMNS
    assert not artifacts.daily_strategy_equity.empty
    assert not artifacts.daily_holdings.empty
    assert not artifacts.rebalance_log.empty
    assert not artifacts.fold_performance.empty
    assert not artifacts.underperformance_windows.empty
    assert not artifacts.selected_symbol_summary.empty
    assert {"strategy_equity", "spy_equity", "qqq_equity"}.issubset(
        artifacts.daily_strategy_equity.columns
    )
    assert {"symbol", "target_weight", "selected"}.issubset(artifacts.daily_holdings.columns)
    assert set(artifacts.underperformance_windows["benchmark_symbol"]) == {"SPY", "QQQ"}
    assert set(artifacts.reference_performance["reference_role"]) == {
        "historical_champion",
        "return_reference",
        "drawdown_guardrail",
    }
    assert not artifacts.feature_set_comparison.empty


def test_rich_replay_cli_writes_required_csvs_and_markdown(
    tmp_path: Path,
    capsys,
) -> None:
    data_path = tmp_path / "market.csv"
    _synthetic_market_data().to_csv(data_path, index=False)
    output_dir = tmp_path / "replay"

    return_code = run_tiingo_selected_candidate_rich_replay_from_args(
        [
            "--market-dataset-glob",
            str(data_path),
            "--output-dir",
            str(output_dir),
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
            "--transaction-cost-bps",
            "10",
            "--feature-set-version",
            "default_features_v1",
            "--min-history-rows",
            "5",
            "--min-train-eligible-rows",
            "1",
        ]
    )

    captured = capsys.readouterr()

    assert return_code == 0
    assert "generated fold count: 1" in captured.out
    assert "Equivalent Python command:" in captured.out

    required_names = [
        "daily_strategy_equity.csv",
        "daily_holdings.csv",
        "rebalance_log.csv",
        "trade_turnover_diagnostics.csv",
        "fold_performance.csv",
        "underperformance_windows.csv",
        "selected_symbol_summary.csv",
        "rich_replay_summary.csv",
        "reference_performance.csv",
        "feature_set_comparison.csv",
        "v2_vs_v1_underperformance_delta.csv",
        "v2_selected_symbol_summary.csv",
        "rich_replay_report.md",
    ]
    for name in required_names:
        assert (output_dir / name).exists()

    daily_equity = pd.read_csv(output_dir / "daily_strategy_equity.csv")
    daily_holdings = pd.read_csv(output_dir / "daily_holdings.csv")
    fold_performance = pd.read_csv(output_dir / "fold_performance.csv")
    underperformance = pd.read_csv(output_dir / "underperformance_windows.csv")
    selected_summary = pd.read_csv(output_dir / "selected_symbol_summary.csv")
    markdown = (output_dir / "rich_replay_report.md").read_text(encoding="utf-8")

    assert {"strategy_equity", "spy_equity", "qqq_equity"}.issubset(daily_equity.columns)
    assert {"symbol", "target_weight", "selected"}.issubset(daily_holdings.columns)
    assert not fold_performance.empty
    assert not underperformance.empty
    assert not selected_summary.empty
    assert "SPY" in markdown
    assert "QQQ" in markdown
    assert "Decision:" in markdown
    assert "Artifact limitations" in markdown
    assert "not production-ready" in markdown
    assert "historical ML champion" in markdown


def test_rich_replay_cli_accepts_feature_set_version_in_dry_run(
    tmp_path: Path,
    capsys,
) -> None:
    data_path = tmp_path / "market.csv"
    _synthetic_market_data().to_csv(data_path, index=False)
    output_dir = tmp_path / "replay"

    return_code = run_tiingo_selected_candidate_rich_replay_from_args(
        [
            "--market-dataset-glob",
            str(data_path),
            "--output-dir",
            str(output_dir),
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
            "--feature-set-version",
            "benchmark_gap_features_v2",
            "--reference-feature-set-versions",
            "default_features_v1",
            "benchmark_gap_features_v1",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()

    assert return_code == 0
    assert "feature_set_version: benchmark_gap_features_v2" in captured.out
    assert "reference_feature_set_versions" in captured.out
    assert "--feature-set-version" in captured.out
    assert "benchmark_gap_features_v2" in captured.out


def test_rich_replay_writes_expected_files_with_benchmark_gap_features_v2(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "market.csv"
    _synthetic_market_data(end="2023-12-31").to_csv(data_path, index=False)
    output_dir = tmp_path / "replay_v2"

    artifacts = build_tiingo_selected_candidate_rich_replay(
        market_dataset_glob=data_path,
        start_date="2020-01-01",
        end_date="2023-12-31",
        train_window_years=(1,),
        validation_window_months=(1,),
        test_window_months=(3,),
        step_months=12,
        transaction_cost_bps=10.0,
        include_references=False,
        feature_set_version="benchmark_gap_features_v2",
        reference_feature_set_versions=(
            "default_features_v1",
            "benchmark_gap_features_v1",
        ),
        min_history_rows=5,
        min_train_eligible_rows=1,
    )
    outputs = write_tiingo_selected_candidate_rich_replay_artifacts(
        artifacts=artifacts,
        output_dir=output_dir,
    )

    assert set(outputs) >= {
        "daily_strategy_equity.csv",
        "daily_holdings.csv",
        "rebalance_log.csv",
        "trade_turnover_diagnostics.csv",
        "fold_performance.csv",
        "underperformance_windows.csv",
        "selected_symbol_summary.csv",
        "rich_replay_summary.csv",
        "reference_performance.csv",
        "feature_set_comparison.csv",
        "v2_vs_v1_underperformance_delta.csv",
        "v2_selected_symbol_summary.csv",
        "rich_replay_report.md",
    }
    summary = pd.read_csv(output_dir / "rich_replay_summary.csv")
    references = pd.read_csv(output_dir / "reference_performance.csv")
    comparison = pd.read_csv(output_dir / "feature_set_comparison.csv")
    markdown = (output_dir / "rich_replay_report.md").read_text(encoding="utf-8")

    assert summary.loc[0, "feature_set_version"] == "benchmark_gap_features_v2"
    assert set(references["feature_set_version"]) == {
        "default_features_v1",
        "benchmark_gap_features_v1",
    }
    assert "SPY buy-and-hold" in set(comparison["reference_label"])
    assert "QQQ buy-and-hold" in set(comparison["reference_label"])
    assert "Feature set used: `benchmark_gap_features_v2`" in markdown
    assert "Comparison vs default_features_v1 selected candidate" in markdown
    assert "Comparison vs benchmark_gap_features_v1 selected candidate" in markdown


def test_rich_replay_shell_runner_is_local_and_executable() -> None:
    script_path = Path("scripts/run_tiingo_selected_candidate_rich_replay.sh")
    script = script_path.read_text(encoding="utf-8")

    assert os.access(script_path, os.X_OK)
    assert "set -euo pipefail" in script
    assert "MARKET_DATASET_GLOB=" in script
    assert "OUTPUT_DIR=" in script
    assert "START_DATE=" in script
    assert "END_DATE=" in script
    assert "MAX_WORKERS=" in script
    assert "TRAIN_WINDOW_YEARS=" in script
    assert "VALIDATION_WINDOW_MONTHS=" in script
    assert "TEST_WINDOW_MONTHS=" in script
    assert "TRANSACTION_COST_BPS=" in script
    assert "FEATURE_SET_VERSION=" in script
    assert "run_tiingo_selected_candidate_rich_replay" in script
    assert "--feature-set-version" in script
    assert "--dry-run" in script
    assert "Equivalent Python command" in script
    assert "No S3/cloud/GPU" in script
    assert "no MLP" in script
    assert "no tree models" in script
    assert "cleanup_on_interrupt" in script

    subprocess.run(["bash", "-n", str(script_path)], check=True)
    subprocess.run(
        [
            "bash",
            "-c",
            (
                "TIINGO_SELECTED_CANDIDATE_RICH_REPLAY_TEST_HELPERS=1; "
                "source scripts/run_tiingo_selected_candidate_rich_replay.sh; "
                "declare -f append_values_arg >/dev/null"
            ),
        ],
        check=True,
    )


def test_rich_replay_v2_shell_runner_is_local_and_executable() -> None:
    script_path = Path("scripts/run_tiingo_selected_candidate_rich_replay_v2.sh")
    script = script_path.read_text(encoding="utf-8")

    assert os.access(script_path, os.X_OK)
    assert "set -euo pipefail" in script
    assert "OUTPUT_DIR=" in script
    assert "tiingo_selected_candidate_rich_replay_v2" in script
    assert "FEATURE_SET_VERSION=" in script
    assert "benchmark_gap_features_v2" in script
    assert "REFERENCE_FEATURE_SET_VERSIONS=" in script
    assert "default_features_v1 benchmark_gap_features_v1" in script
    assert "--feature-set-version" in script
    assert "--reference-feature-set-versions" in script
    assert "--dry-run" in script
    assert "No S3/cloud/GPU" in script
    assert "no MLP" in script
    assert "no tree models" in script
    assert "no production signal export" in script

    subprocess.run(["bash", "-n", str(script_path)], check=True)
    subprocess.run(
        [
            "bash",
            "-c",
            (
                "TIINGO_SELECTED_CANDIDATE_RICH_REPLAY_TEST_HELPERS=1; "
                "source scripts/run_tiingo_selected_candidate_rich_replay_v2.sh; "
                "declare -f append_values_arg >/dev/null"
            ),
        ],
        check=True,
    )


def _synthetic_market_data(
    start: str = "2020-01-01",
    end: str = "2021-04-30",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range(start, end, freq="D")
    symbols = ("SPY", "QQQ", "TLT", "IWM", "GLD", "VNQ")
    for symbol_index, symbol in enumerate(symbols):
        base_price = 100.0 + symbol_index * 12.0
        drift = 0.08 + symbol_index * 0.015
        cycle = symbol_index + 2
        for day_index, date in enumerate(dates):
            close = base_price + day_index * drift + (day_index % 19) * cycle * 0.025
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close - 0.2,
                    "high": close + 0.7,
                    "low": close - 0.8,
                    "close": close,
                    "volume": 1_000_000 + symbol_index * 50_000 + day_index * 1_000,
                    "vwap": close - 0.03,
                    "year": date.year,
                    "month": date.month,
                    "transactions": 10_000 + day_index,
                }
            )
    return pd.DataFrame(rows)
