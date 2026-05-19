from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.cli.run_tiingo_qqq_defensive_timing_validation import (
    COST_SENSITIVITY_NAME,
    COVERAGE_NAME,
    DAILY_EQUITY_NAME,
    FULL_PERIOD_NAME,
    PREFLIGHT_MD_NAME,
    SUBPERIODS_NAME,
    TRADES_NAME,
    WALK_FORWARD_SUMMARY_NAME,
    WALK_FORWARD_WINDOWS_NAME,
    run_tiingo_qqq_defensive_timing_validation_from_args,
)


def test_validation_cli_dry_run_prints_coverage_windows_and_guardrails(
    tmp_path: Path,
    capsys,
) -> None:
    data_path = tmp_path / "market.csv"
    _synthetic_market_data().to_csv(data_path, index=False)
    output_dir = tmp_path / "validation"

    exit_code = run_tiingo_qqq_defensive_timing_validation_from_args(
        [
            "--dry-run",
            "--market-dataset-glob",
            str(data_path),
            "--output-dir",
            str(output_dir),
            "--start-date",
            "auto",
            "--end-date",
            "2025-06-30",
            "--transaction-cost-bps",
            "10",
        ]
    )

    captured = capsys.readouterr()
    preflight = output_dir / PREFLIGHT_MD_NAME

    assert exit_code == 0
    assert preflight.exists()
    assert not (output_dir / FULL_PERIOD_NAME).exists()
    assert "qqq_first_date: 2019-01-01" in captured.out
    assert "spy_first_date: 2019-01-01" in captured.out
    assert "common_start_date: 2019-01-01" in captured.out
    assert "selected_start_date: 2019-01-01" in captured.out
    assert "rolling window count:" in captured.out
    assert "rolling window count 1y:" in captured.out
    assert "rolling window count 5y:" in captured.out
    assert "strategy_name: qqq_sma190_buffer_p0p5_hold0_to_spy" in captured.out
    assert "risk_on_asset: QQQ" in captured.out
    assert "risk_off_asset: SPY" in captured.out
    assert "signal_name: qqq_sma_price_filter" in captured.out
    assert "No S3/cloud/GPU" in captured.out
    assert "no tree_screen" in captured.out
    assert "no tree models" in captured.out
    assert "no MLP" in captured.out
    assert "no feature v4" in captured.out
    assert "No production signal export" in captured.out
    assert "Production-ready status: false" in captured.out

    markdown = preflight.read_text(encoding="utf-8")
    assert "Tiingo QQQ Defensive Timing Validation Preflight" in markdown
    assert "Longer-history available locally: `false`" in markdown
    assert "Coverage note: Longer-history validation is limited" in markdown
    assert "Total rolling windows" in markdown
    assert "Strategy name: `qqq_sma190_buffer_p0p5_hold0_to_spy`" in markdown
    assert "No benchmark_gap_features_v4" in markdown
    assert "No production signal export" in markdown
    assert "No production-ready label" in markdown
    assert "close data available on date T" in markdown


def test_validation_cli_writes_required_artifacts_and_cost_sensitivity(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "market.csv"
    _synthetic_market_data().to_csv(data_path, index=False)
    output_dir = tmp_path / "validation"

    exit_code = run_tiingo_qqq_defensive_timing_validation_from_args(
        [
            "--market-dataset-glob",
            str(data_path),
            "--output-dir",
            str(output_dir),
            "--start-date",
            "auto",
            "--end-date",
            "2025-06-30",
            "--transaction-cost-bps",
            "10",
        ]
    )

    coverage = pd.read_csv(output_dir / COVERAGE_NAME)
    full = pd.read_csv(output_dir / FULL_PERIOD_NAME)
    walk_windows = pd.read_csv(output_dir / WALK_FORWARD_WINDOWS_NAME)
    walk_summary = pd.read_csv(output_dir / WALK_FORWARD_SUMMARY_NAME)
    subperiods = pd.read_csv(output_dir / SUBPERIODS_NAME)
    cost = pd.read_csv(output_dir / COST_SENSITIVITY_NAME)
    daily_equity = pd.read_csv(output_dir / DAILY_EQUITY_NAME)
    trades = pd.read_csv(output_dir / TRADES_NAME)

    assert exit_code == 0
    assert coverage.loc[0, "qqq_first_date"] == "2019-01-01"
    assert coverage.loc[0, "selected_start_date"] == "2019-01-01"
    assert not bool(coverage.loc[0, "longer_history_available"])
    assert full.loc[0, "strategy_name"] == "qqq_sma190_buffer_p0p5_hold0_to_spy"
    assert full.loc[0, "risk_on_asset"] == "QQQ"
    assert full.loc[0, "risk_off_asset"] == "SPY"
    assert full.loc[0, "slow_sma_days"] == 190
    assert full.loc[0, "buffer_pct"] == 0.005
    assert full.loc[0, "min_hold_days"] == 0
    assert full.loc[0, "research_status"] == "research_only_defensive_candidate"
    assert not bool(full.loc[0, "production_ready"])
    assert {"1y", "2y", "3y", "5y"}.issubset(set(walk_summary["window_group"]))
    assert {"window_label", "strategy_vs_qqq", "drawdown_improvement_vs_qqq"}.issubset(
        walk_windows.columns
    )
    assert {"covid_crash_recovery", "bear_2022", "ai_bull_2023_2024"}.issubset(
        set(subperiods["period_name"])
    )
    assert set(cost["transaction_cost_bps"]) == {0.0, 5.0, 10.0, 20.0}
    assert cost.loc[cost["transaction_cost_bps"].eq(10.0), "is_anchor_cost"].all()
    assert {
        "decision_at_cost",
        "decision_changed_vs_anchor",
        "qqq_return_capture",
        "estimated_transaction_cost",
    }.issubset(cost.columns)
    assert {
        "qqq_sma190_buffer_p0p5_hold0_to_spy",
        "qqq_buy_hold",
        "spy_buy_hold",
        "cash",
    }.issubset(set(daily_equity["strategy_name"]))
    assert {"date", "active_asset", "daily_return", "equity", "drawdown"}.issubset(
        daily_equity.columns
    )
    assert {"previous_asset", "new_asset", "estimated_transaction_cost", "reason"}.issubset(
        trades.columns
    )


def test_validation_shell_runner_is_local_and_executable() -> None:
    script_path = Path("scripts/run_tiingo_qqq_defensive_timing_validation.sh")
    script = script_path.read_text(encoding="utf-8")

    assert os.access(script_path, os.X_OK)
    assert "set -euo pipefail" in script
    assert "MARKET_DATASET_GLOB=" in script
    assert "artifacts/validation/tiingo_qqq_defensive_timing_validation" in script
    assert "START_DATE=" in script
    assert "auto" in script
    assert "END_DATE=" in script
    assert "2026-05-15" in script
    assert "TRANSACTION_COST_BPS=" in script
    assert "STRATEGY_NAME=" in script
    assert "qqq_sma190_buffer_p0p5_hold0_to_spy" in script
    assert "SLOW_SMA_DAYS=" in script
    assert "BUFFER_PCT=" in script
    assert "MIN_HOLD_DAYS=" in script
    assert "RISK_OFF_ASSET=" in script
    assert "DRY_RUN=" in script
    assert "Required symbols: QQQ, SPY" in script
    assert "no S3/cloud/GPU" in script
    assert "no tree_screen" in script
    assert "no tree models" in script
    assert "no MLP" in script
    assert "no feature v4" in script
    assert "no production signal export" in script
    assert "create_tiingo_qqq_defensive_timing_validation_report" in script

    subprocess.run(["bash", "-n", str(script_path)], check=True)


def test_validation_paths_do_not_use_forbidden_runtime_imports() -> None:
    paths = [
        Path("src/us_market_ml/qqq_timing.py"),
        Path("src/us_market_ml/cli/run_tiingo_qqq_defensive_timing_validation.py"),
        Path("src/us_market_ml/cli/create_tiingo_qqq_defensive_timing_validation_report.py"),
        Path("scripts/run_tiingo_qqq_defensive_timing_validation.sh"),
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert "read_s3" not in combined
    assert "s3_reader" not in combined
    assert "run_mlp" not in combined
    assert "tree_screen(" not in combined
    assert "run_model_signal_export" not in combined
    assert "run_daily_signal_export" not in combined
    assert "add_benchmark_gap_v4" not in combined


def _synthetic_market_data() -> pd.DataFrame:
    dates = pd.bdate_range("2019-01-01", periods=1700)
    qqq = np.concatenate(
        [
            np.linspace(100.0, 160.0, 300),
            np.linspace(158.0, 82.0, 300),
            np.linspace(84.0, 185.0, 300),
            np.linspace(183.0, 92.0, 300),
            np.linspace(94.0, 210.0, 500),
        ]
    )
    price_map = {
        "QQQ": qqq,
        "SPY": np.linspace(100.0, 165.0, len(dates)),
    }
    rows: list[dict[str, object]] = []
    for symbol, values in price_map.items():
        for date, close in zip(dates, values, strict=True):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close - 0.1,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1_000_000,
                    "vwap": close,
                }
            )
    return pd.DataFrame(rows)
