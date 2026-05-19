from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.cli.run_tiingo_qqq_defensive_timing_narrow_sweep import (
    run_tiingo_qqq_defensive_timing_narrow_sweep_from_args,
)


def test_defensive_timing_narrow_sweep_cli_dry_run_prints_counts_and_guardrails(
    tmp_path: Path,
    capsys,
) -> None:
    data_path = tmp_path / "market.csv"
    _synthetic_market_data(symbols=("QQQ", "SPY", "TLT")).to_csv(data_path, index=False)
    output_dir = tmp_path / "narrow"

    exit_code = run_tiingo_qqq_defensive_timing_narrow_sweep_from_args(
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
    preflight = output_dir / "tiingo_qqq_defensive_timing_narrow_preflight.md"

    assert exit_code == 0
    assert preflight.exists()
    assert not (output_dir / "tiingo_qqq_defensive_timing_narrow_results.csv").exists()
    assert "timing config count: 426" in captured.out
    assert "total result row count: 429" in captured.out
    assert "family count total_return_spy_defensive: 144" in captured.out
    assert "family count drawdown_adjusted_cash_defensive: 90" in captured.out
    assert "family count low_turnover_crossover_cash: 192" in captured.out
    assert "required symbols: QQQ, SPY" in captured.out
    assert "benchmark references: qqq_buy_hold, spy_buy_hold, cash" in captured.out
    assert "missing optional symbols: IEF, SHY" in captured.out
    assert "warning: Optional risk-off asset IEF is missing" in captured.out
    assert "No S3/cloud/GPU" in captured.out
    assert "no tree_screen" in captured.out
    assert "no tree models" in captured.out
    assert "no MLP" in captured.out
    assert "no feature v4" in captured.out
    assert "No production signal export" in captured.out
    assert "Market Data repo mutation: false" in captured.out

    markdown = preflight.read_text(encoding="utf-8")
    assert "Tiingo QQQ Defensive Timing Narrow Sweep Preflight" in markdown
    assert "Timing config count: `426`" in markdown
    assert "Total result row count: `429`" in markdown
    assert "Optional missing: `IEF, SHY`" in markdown
    assert "Total-return SPY defensive" in markdown
    assert "Drawdown-adjusted cash defensive" in markdown
    assert "Low-turnover crossover cash" in markdown
    assert "No benchmark_gap_features_v4" in markdown
    assert "No production signal export" in markdown
    assert "No production-ready label" in markdown
    assert "close data available on date T" in markdown


def test_defensive_timing_narrow_sweep_cli_writes_required_outputs(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "market.csv"
    _synthetic_market_data(symbols=("QQQ", "SPY", "IEF", "TLT", "SHY")).to_csv(
        data_path,
        index=False,
    )
    output_dir = tmp_path / "narrow"

    exit_code = run_tiingo_qqq_defensive_timing_narrow_sweep_from_args(
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

    results = pd.read_csv(output_dir / "tiingo_qqq_defensive_timing_narrow_results.csv")
    daily_equity = pd.read_csv(
        output_dir / "tiingo_qqq_defensive_timing_narrow_daily_equity.csv"
    )
    trades = pd.read_csv(output_dir / "tiingo_qqq_defensive_timing_narrow_trades.csv")
    subperiod_results = pd.read_csv(
        output_dir / "tiingo_qqq_defensive_timing_narrow_subperiod_results.csv"
    )

    assert exit_code == 0
    assert len(results) == 429
    assert {"qqq_buy_hold", "spy_buy_hold", "cash"}.issubset(results["strategy_name"])
    assert {
        "family",
        "slow_sma_days",
        "fast_sma_days",
        "buffer_pct",
        "min_hold_days",
        "beats_qqq_return",
        "materially_improves_qqq_drawdown",
    }.issubset(results.columns)
    assert {
        "total_return_spy_defensive",
        "drawdown_adjusted_cash_defensive",
        "low_turnover_crossover_cash",
    }.issubset(set(results["family"]))
    assert not daily_equity.empty
    assert {"date", "strategy_name", "family", "active_asset", "daily_return"}.issubset(
        daily_equity.columns
    )
    assert {"date", "strategy_name", "family", "previous_asset", "new_asset"}.issubset(
        trades.columns
    )
    assert not subperiod_results.empty
    assert {
        "strategy_name",
        "period_name",
        "period_start",
        "period_end",
        "total_return",
        "qqq_buy_hold_total_return",
        "spy_buy_hold_total_return",
        "strategy_vs_qqq",
        "max_drawdown",
        "qqq_max_drawdown",
        "transaction_count",
        "exposure_rate_to_qqq",
    }.issubset(subperiod_results.columns)
    assert "full" in set(subperiod_results["period_name"])
    assert not any("production" in column.lower() for column in results.columns)


def test_defensive_timing_narrow_sweep_shell_runner_is_local_and_executable() -> None:
    script_path = Path("scripts/run_tiingo_qqq_defensive_timing_narrow_sweep.sh")
    script = script_path.read_text(encoding="utf-8")

    assert os.access(script_path, os.X_OK)
    assert "set -euo pipefail" in script
    assert "MARKET_DATASET_GLOB=" in script
    assert "OUTPUT_DIR=" in script
    assert "REPORT_OUTPUT_DIR=" in script
    assert "START_DATE=" in script
    assert "END_DATE=" in script
    assert "TRANSACTION_COST_BPS=" in script
    assert "DRY_RUN=" in script
    assert "run_tiingo_qqq_defensive_timing_narrow_sweep" in script
    assert "create_tiingo_qqq_defensive_timing_narrow_report" in script
    assert "Required symbols: QQQ, SPY" in script
    assert "qqq_buy_hold, spy_buy_hold, cash" in script
    assert "no S3/cloud/GPU" in script
    assert "no tree_screen" in script
    assert "no tree models" in script
    assert "no MLP" in script
    assert "no feature v4" in script
    assert "no production signal export" in script

    subprocess.run(["bash", "-n", str(script_path)], check=True)


def test_defensive_timing_narrow_paths_do_not_use_forbidden_runtime_imports() -> None:
    paths = [
        Path("src/us_market_ml/qqq_timing.py"),
        Path("src/us_market_ml/cli/run_tiingo_qqq_defensive_timing_narrow_sweep.py"),
        Path("src/us_market_ml/cli/create_tiingo_qqq_defensive_timing_narrow_report.py"),
        Path("scripts/run_tiingo_qqq_defensive_timing_narrow_sweep.sh"),
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert "read_s3" not in combined
    assert "s3_reader" not in combined
    assert "run_mlp" not in combined
    assert "tree_screen(" not in combined
    assert "run_model_signal_export" not in combined
    assert "run_daily_signal_export" not in combined
    assert "add_benchmark_gap_v4" not in combined


def _synthetic_market_data(symbols: tuple[str, ...]) -> pd.DataFrame:
    dates = pd.bdate_range("2019-01-01", periods=650)
    qqq = np.concatenate(
        [
            np.linspace(100.0, 150.0, 260),
            np.linspace(149.0, 95.0, 170),
            np.linspace(96.0, 135.0, 220),
        ]
    )
    price_map = {
        "QQQ": qqq,
        "SPY": np.linspace(100.0, 132.0, len(dates)),
        "IEF": np.linspace(80.0, 86.0, len(dates)),
        "TLT": np.linspace(90.0, 99.0, len(dates)),
        "SHY": np.linspace(70.0, 72.0, len(dates)),
    }
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        for date, close in zip(dates, price_map[symbol], strict=True):
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
