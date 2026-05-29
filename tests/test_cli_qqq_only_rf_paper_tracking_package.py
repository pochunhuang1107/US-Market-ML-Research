from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest
from test_qqq_only_cross_market_timing_model_zoo import (
    _synthetic_market_data,
    _synthetic_reference_long,
)
from test_qqq_only_rf_out_of_time_confirmation import _out_of_time_inputs

from us_market_ml.cli.create_qqq_only_rf_paper_tracking_package import (
    create_qqq_only_rf_paper_tracking_package_from_args,
)
from us_market_ml.qqq_only_rf_out_of_time_confirmation import (
    build_qqq_only_rf_out_of_time_confirmation_artifacts,
)


def test_paper_tracking_cli_dry_run_writes_preflight_and_guardrails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_inputs(tmp_path)
    output_dir = tmp_path / "paper_tracking"

    exit_code = create_qqq_only_rf_paper_tracking_package_from_args(
        [
            "--dry-run",
            "--market-dataset-glob",
            str(paths["market"]),
            "--reference-series-glob",
            str(paths["reference"]),
            "--out-of-time-summary",
            str(paths["summary"]),
            "--out-of-time-decision-matrix",
            str(paths["decision"]),
            "--out-of-time-results",
            str(paths["results"]),
            "--out-of-time-exposure-log",
            str(paths["exposure"]),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "local-only: true" in captured.out
    assert "reads prior out-of-time artifacts: true" in captured.out
    assert "new broad model zoo: false" in captured.out
    assert "production signal export: false" in captured.out
    assert "model card: false" in captured.out
    assert (output_dir / "qqq_only_rf_paper_tracking_preflight.md").exists()


def test_paper_tracking_cli_run_writes_required_outputs(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    output_dir = tmp_path / "paper_tracking"

    assert (
        create_qqq_only_rf_paper_tracking_package_from_args(
            [
                "--market-dataset-glob",
                str(paths["market"]),
                "--reference-series-glob",
                str(paths["reference"]),
                "--out-of-time-summary",
                str(paths["summary"]),
                "--out-of-time-decision-matrix",
                str(paths["decision"]),
                "--out-of-time-results",
                str(paths["results"]),
                "--out-of-time-exposure-log",
                str(paths["exposure"]),
                "--output-dir",
                str(output_dir),
                "--start-date",
                "2019-01-01",
                "--end-date",
                "2020-06-30",
                "--min-train-rows",
                "40",
            ]
        )
        == 0
    )

    registry = pd.read_csv(output_dir / "qqq_only_rf_paper_tracking_candidate_registry.csv")
    signal = pd.read_csv(output_dir / "qqq_only_rf_paper_tracking_initial_signal.csv")
    matrix = pd.read_csv(output_dir / "qqq_only_rf_paper_tracking_decision_matrix.csv")
    report = output_dir / "qqq_only_rf_paper_tracking_report.md"

    assert len(registry) == 3
    assert registry["production_ready"].eq(False).all()
    assert signal["production_ready"].eq(False).all()
    assert matrix["production_ready"].eq(False).all()
    assert set(signal["held_assets"]).issubset({"QQQ", "cash", "QQQ,cash"})
    assert "## 11. Recommendation:" in report.read_text(encoding="utf-8")
    assert (output_dir / "qqq_only_rf_paper_tracking_equity_curve.png").exists()


def test_paper_tracking_cli_rejects_cloud_and_model_signal_paths(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)

    with pytest.raises(ValueError, match="local path"):
        create_qqq_only_rf_paper_tracking_package_from_args(
            [
                "--market-dataset-glob",
                "s3://bucket/market.parquet",
                "--reference-series-glob",
                str(paths["reference"]),
                "--out-of-time-summary",
                str(paths["summary"]),
                "--out-of-time-decision-matrix",
                str(paths["decision"]),
                "--out-of-time-results",
                str(paths["results"]),
                "--out-of-time-exposure-log",
                str(paths["exposure"]),
                "--output-dir",
                str(tmp_path / "paper_tracking"),
            ]
        )

    with pytest.raises(ValueError, match="artifacts/model_signals"):
        create_qqq_only_rf_paper_tracking_package_from_args(
            [
                "--dry-run",
                "--market-dataset-glob",
                str(paths["market"]),
                "--reference-series-glob",
                str(paths["reference"]),
                "--out-of-time-summary",
                str(paths["summary"]),
                "--out-of-time-decision-matrix",
                str(paths["decision"]),
                "--out-of-time-results",
                str(paths["results"]),
                "--out-of-time-exposure-log",
                str(paths["exposure"]),
                "--output-dir",
                "artifacts/model_signals/qqq_only_rf_paper_tracking",
            ]
        )


def test_paper_tracking_shell_runner_defaults_are_research_only() -> None:
    script_path = Path("scripts/create_qqq_only_rf_paper_tracking_package.sh")
    script = script_path.read_text(encoding="utf-8")

    assert os.access(script_path, os.X_OK)
    assert "set -euo pipefail" in script
    assert (
        'MARKET_DATASET_GLOB="${MARKET_DATASET_GLOB:-/Users/casper/Projects/Research/'
        'Market Data/data/research/etf_daily_prices_tiingo_only/**/*.parquet}"'
    ) in script
    assert (
        'REFERENCE_SERIES_GLOB="${REFERENCE_SERIES_GLOB:-/Users/casper/Projects/Research/'
        'Market Data/data/research/reference_series/**/*.parquet}"'
    ) in script
    assert 'OUTPUT_DIR="${OUTPUT_DIR:-artifacts/reports/qqq_only_rf_paper_tracking}"' in script
    assert 'START_DATE="${START_DATE:-2019-01-01}"' in script
    assert 'END_DATE="${END_DATE:-2026-05-15}"' in script
    assert 'DRY_RUN="${DRY_RUN:-0}"' in script
    assert "local-only: true" in script
    assert "reads local data and prior artifacts only: true" in script
    assert "broad sweep: false" in script
    assert "production signal export: false" in script
    assert "model card: false" in script
    assert "run_model_signal_export" not in script
    assert "run_model_card" not in script
    assert "run_qqq_only_rf_out_of_time_confirmation" not in script
    assert "s3://" not in script.lower()

    subprocess.run(["bash", "-n", str(script_path)], check=True)


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    market_path = tmp_path / "market.csv"
    reference_path = tmp_path / "reference.csv"
    _synthetic_market_data().to_csv(market_path, index=False)
    _synthetic_reference_long().to_csv(reference_path, index=False)

    results, daily_equity, daily_drawdown, exposure_log = _out_of_time_inputs()
    out_of_time = build_qqq_only_rf_out_of_time_confirmation_artifacts(
        results=results,
        daily_equity=daily_equity,
        daily_drawdown=daily_drawdown,
        exposure_log=exposure_log,
    )
    paths = {
        "market": market_path,
        "reference": reference_path,
        "summary": tmp_path / "summary.csv",
        "decision": tmp_path / "decision.csv",
        "results": tmp_path / "results.csv",
        "exposure": tmp_path / "exposure.csv",
    }
    out_of_time.summary.to_csv(paths["summary"], index=False)
    out_of_time.decision_matrix.to_csv(paths["decision"], index=False)
    results.to_csv(paths["results"], index=False)
    exposure_log.to_csv(paths["exposure"], index=False)
    return paths
