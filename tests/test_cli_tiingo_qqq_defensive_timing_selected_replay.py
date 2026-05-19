from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.cli.run_tiingo_qqq_defensive_timing_selected_replay import (
    DAILY_EQUITY_NAME,
    DECISION_MATRIX_NAME,
    SUBPERIOD_SUMMARY_NAME,
    SUMMARY_NAME,
    TRADES_NAME,
    run_tiingo_qqq_defensive_timing_selected_replay_from_args,
)


def test_selected_replay_cli_dry_run_prints_config_and_guardrails(
    tmp_path: Path,
    capsys,
) -> None:
    data_path = tmp_path / "market.csv"
    _synthetic_market_data().to_csv(data_path, index=False)
    output_dir = tmp_path / "replay"

    exit_code = run_tiingo_qqq_defensive_timing_selected_replay_from_args(
        [
            "--dry-run",
            "--market-dataset-glob",
            str(data_path),
            "--output-dir",
            str(output_dir),
            "--start-date",
            "2019-01-01",
            "--end-date",
            "2021-06-30",
            "--transaction-cost-bps",
            "10",
        ]
    )

    captured = capsys.readouterr()
    preflight = output_dir / "tiingo_qqq_sma190_defensive_replay_preflight.md"

    assert exit_code == 0
    assert preflight.exists()
    assert not (output_dir / SUMMARY_NAME).exists()
    assert "strategy_name: qqq_sma190_buffer_p0p5_hold0_to_spy" in captured.out
    assert "risk_on_asset: QQQ" in captured.out
    assert "risk_off_asset: SPY" in captured.out
    assert "signal_name: qqq_sma_price_filter" in captured.out
    assert "slow_sma_days: 190" in captured.out
    assert "buffer_pct: 0.005" in captured.out
    assert "min_hold_days: 0" in captured.out
    assert "required symbols: QQQ, SPY" in captured.out
    assert "No S3/cloud/GPU" in captured.out
    assert "no tree_screen" in captured.out
    assert "no tree models" in captured.out
    assert "no MLP" in captured.out
    assert "no feature v4" in captured.out
    assert "No production signal export" in captured.out
    assert "Production-ready status: false" in captured.out

    markdown = preflight.read_text(encoding="utf-8")
    assert "Selected Replay Preflight" in markdown
    assert "Strategy name: `qqq_sma190_buffer_p0p5_hold0_to_spy`" in markdown
    assert "Risk-off asset: `SPY`" in markdown
    assert "No benchmark_gap_features_v4" in markdown
    assert "No production signal export" in markdown
    assert "No production-ready label" in markdown
    assert "close data available on date T" in markdown


def test_selected_replay_cli_writes_required_artifacts(tmp_path: Path) -> None:
    data_path = tmp_path / "market.csv"
    _synthetic_market_data().to_csv(data_path, index=False)
    output_dir = tmp_path / "replay"

    exit_code = run_tiingo_qqq_defensive_timing_selected_replay_from_args(
        [
            "--market-dataset-glob",
            str(data_path),
            "--output-dir",
            str(output_dir),
            "--start-date",
            "2019-01-01",
            "--end-date",
            "2021-06-30",
            "--transaction-cost-bps",
            "10",
        ]
    )

    summary = pd.read_csv(output_dir / SUMMARY_NAME)
    daily_equity = pd.read_csv(output_dir / DAILY_EQUITY_NAME)
    trades = pd.read_csv(output_dir / TRADES_NAME)
    subperiods = pd.read_csv(output_dir / SUBPERIOD_SUMMARY_NAME)
    decision = pd.read_csv(output_dir / DECISION_MATRIX_NAME)

    assert exit_code == 0
    assert summary.loc[0, "strategy_name"] == "qqq_sma190_buffer_p0p5_hold0_to_spy"
    assert summary.loc[0, "risk_on_asset"] == "QQQ"
    assert summary.loc[0, "risk_off_asset"] == "SPY"
    assert summary.loc[0, "signal_name"] == "qqq_sma_price_filter"
    assert summary.loc[0, "slow_sma_days"] == 190
    assert summary.loc[0, "buffer_pct"] == 0.005
    assert summary.loc[0, "min_hold_days"] == 0
    assert {
        "qqq_sma190_buffer_p0p5_hold0_to_spy",
        "qqq_buy_hold",
        "spy_buy_hold",
        "cash",
    }.issubset(set(daily_equity["strategy_name"]))
    assert {"date", "strategy_name", "active_asset", "daily_return", "equity", "drawdown"}.issubset(
        daily_equity.columns
    )
    assert not trades.empty
    assert {
        "date",
        "signal_date",
        "previous_asset",
        "new_asset",
        "turnover",
        "estimated_transaction_cost",
        "reason",
    }.issubset(trades.columns)
    assert not subperiods.empty
    assert {
        "period_name",
        "strategy_total_return",
        "qqq_buy_hold_total_return",
        "spy_buy_hold_total_return",
        "strategy_vs_qqq",
        "strategy_vs_spy",
        "strategy_max_drawdown",
        "qqq_max_drawdown",
        "spy_max_drawdown",
        "exposure_rate_to_qqq",
        "exposure_rate_to_spy",
        "transaction_count",
    }.issubset(subperiods.columns)
    assert "full" in set(subperiods["period_name"])
    assert decision.loc[0, "research_status"] == "research_only_defensive_candidate"
    assert not bool(decision.loc[0, "production_ready"])


def test_selected_replay_shell_runner_is_local_and_executable() -> None:
    script_path = Path("scripts/run_tiingo_qqq_defensive_timing_selected_replay.sh")
    script = script_path.read_text(encoding="utf-8")

    assert os.access(script_path, os.X_OK)
    assert "set -euo pipefail" in script
    assert "MARKET_DATASET_GLOB=" in script
    assert "artifacts/replays/tiingo_qqq_sma190_defensive_replay" in script
    assert "START_DATE=" in script
    assert "2019-01-01" in script
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
    assert "create_tiingo_qqq_defensive_timing_selected_replay_report" in script

    subprocess.run(["bash", "-n", str(script_path)], check=True)


def test_selected_replay_paths_do_not_use_forbidden_runtime_imports() -> None:
    paths = [
        Path("src/us_market_ml/qqq_timing.py"),
        Path("src/us_market_ml/cli/run_tiingo_qqq_defensive_timing_selected_replay.py"),
        Path("src/us_market_ml/cli/create_tiingo_qqq_defensive_timing_selected_replay_report.py"),
        Path("scripts/run_tiingo_qqq_defensive_timing_selected_replay.sh"),
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
    dates = pd.bdate_range("2019-01-01", periods=650)
    qqq = np.concatenate(
        [
            np.linspace(100.0, 150.0, 260),
            np.linspace(149.0, 92.0, 170),
            np.linspace(94.0, 142.0, 220),
        ]
    )
    price_map = {
        "QQQ": qqq,
        "SPY": np.linspace(100.0, 132.0, len(dates)),
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
