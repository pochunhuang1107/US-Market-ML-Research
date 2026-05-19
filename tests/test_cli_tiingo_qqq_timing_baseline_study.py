from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.cli.run_tiingo_qqq_timing_baseline_study import (
    run_tiingo_qqq_timing_baseline_study_from_args,
)


def test_qqq_timing_cli_dry_run_writes_preflight_and_guardrails(
    tmp_path: Path,
    capsys,
) -> None:
    data_path = tmp_path / "market.csv"
    _synthetic_market_data(symbols=("QQQ", "SPY", "TLT")).to_csv(data_path, index=False)
    output_dir = tmp_path / "study"

    exit_code = run_tiingo_qqq_timing_baseline_study_from_args(
        [
            "--dry-run",
            "--market-dataset-glob",
            str(data_path),
            "--output-dir",
            str(output_dir),
            "--start-date",
            "2019-01-01",
            "--end-date",
            "2020-06-30",
            "--transaction-cost-bps",
            "10",
        ]
    )

    captured = capsys.readouterr()
    preflight = output_dir / "tiingo_qqq_timing_baseline_preflight.md"

    assert exit_code == 0
    assert preflight.exists()
    assert not (output_dir / "tiingo_qqq_timing_baseline_results.csv").exists()
    assert "strategy count:" in captured.out
    assert "required symbols: QQQ, SPY" in captured.out
    assert "missing optional symbols: IEF, SHY" in captured.out
    assert "warning: Skipping qqq_momentum_drawdown_guard_to_ief" in captured.out
    assert "No S3/cloud/GPU" in captured.out
    assert "no tree_screen" in captured.out
    assert "no MLP" in captured.out
    assert "no feature v4" in captured.out
    assert "no production export" in captured.out
    assert "Fixed QQQ core portfolio: false" in captured.out

    markdown = preflight.read_text(encoding="utf-8")
    assert "Tiingo QQQ Timing Baseline Study Preflight" in markdown
    assert "Strategy count: `15`" in markdown
    assert "Optional missing: `IEF, SHY`" in markdown
    assert "No benchmark_gap_features_v4" in markdown
    assert "No production signal export" in markdown
    assert "No production-ready label" in markdown


def test_qqq_timing_cli_writes_results_daily_equity_and_trades(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "market.csv"
    _synthetic_market_data(symbols=("QQQ", "SPY", "IEF", "TLT")).to_csv(
        data_path,
        index=False,
    )
    output_dir = tmp_path / "study"

    exit_code = run_tiingo_qqq_timing_baseline_study_from_args(
        [
            "--market-dataset-glob",
            str(data_path),
            "--output-dir",
            str(output_dir),
            "--start-date",
            "2019-01-01",
            "--end-date",
            "2020-06-30",
            "--transaction-cost-bps",
            "10",
        ]
    )

    results = pd.read_csv(output_dir / "tiingo_qqq_timing_baseline_results.csv")
    daily_equity = pd.read_csv(output_dir / "tiingo_qqq_timing_baseline_daily_equity.csv")
    trades = pd.read_csv(output_dir / "tiingo_qqq_timing_baseline_trades.csv")

    assert exit_code == 0
    assert len(results) == 16
    assert {"qqq_buy_hold", "spy_buy_hold", "cash"}.issubset(results["strategy_name"])
    assert "qqq_momentum_drawdown_guard_to_ief" in set(results["strategy_name"])
    assert "qqq_momentum_drawdown_guard_to_tlt" in set(results["strategy_name"])
    assert not daily_equity.empty
    assert not trades.empty
    assert {"date", "strategy_name", "active_asset", "daily_return", "equity"}.issubset(
        daily_equity.columns
    )
    assert {"date", "strategy_name", "previous_asset", "new_asset"}.issubset(
        trades.columns
    )
    assert not any(
        "core" in name.replace("-", "_").split("_")
        for name in results["strategy_name"].astype(str)
    )


def test_qqq_timing_shell_runner_is_local_and_executable() -> None:
    script_path = Path("scripts/run_tiingo_qqq_timing_baseline_study.sh")
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
    assert "run_tiingo_qqq_timing_baseline_study" in script
    assert "create_tiingo_qqq_timing_baseline_report" in script
    assert "Required symbols: QQQ, SPY" in script
    assert "Optional defensive symbols: IEF, TLT, SHY" in script
    assert "no S3/cloud/GPU" in script
    assert "no tree_screen" in script
    assert "no tree models" in script
    assert "no MLP" in script
    assert "no feature v4" in script
    assert "no production signal export" in script
    assert "no fixed QQQ core portfolio" in script
    assert "qqq_core_30" not in script

    subprocess.run(["bash", "-n", str(script_path)], check=True)


def test_qqq_timing_new_paths_do_not_use_forbidden_runtime_imports() -> None:
    paths = [
        Path("src/us_market_ml/qqq_timing.py"),
        Path("src/us_market_ml/cli/run_tiingo_qqq_timing_baseline_study.py"),
        Path("src/us_market_ml/cli/create_tiingo_qqq_timing_baseline_report.py"),
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert "read_s3" not in combined
    assert "s3_reader" not in combined
    assert "run_mlp" not in combined
    assert "tree_screen(" not in combined
    assert "model_signal_export" not in combined
    assert "daily_signal_export" not in combined
    assert "add_benchmark_gap_v4" not in combined


def _synthetic_market_data(symbols: tuple[str, ...]) -> pd.DataFrame:
    dates = pd.bdate_range("2019-01-01", periods=390)
    qqq = np.concatenate(
        [
            np.linspace(100.0, 135.0, 180),
            np.linspace(134.0, 90.0, 210),
        ]
    )
    price_map = {
        "QQQ": qqq,
        "SPY": np.linspace(100.0, 130.0, len(dates)),
        "IEF": np.linspace(80.0, 86.0, len(dates)),
        "TLT": np.linspace(90.0, 99.0, len(dates)),
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
