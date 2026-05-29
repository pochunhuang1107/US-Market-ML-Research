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

from us_market_ml.cli.update_qqq_only_rf_paper_tracking import (
    update_qqq_only_rf_paper_tracking_from_args,
)
from us_market_ml.qqq_only_rf_paper_tracking import (
    CANDIDATE_IDS,
    CANDIDATE_REGISTRY_JSON_NAME,
    CANDIDATE_REGISTRY_NAME,
    INITIAL_SIGNAL_COLUMNS,
    INITIAL_SIGNAL_NAME,
    build_qqq_only_rf_paper_tracking_candidate_registry,
)


def test_paper_tracking_update_cli_dry_run_writes_preflight_and_guardrails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_inputs(tmp_path)
    output_dir = tmp_path / "updates"

    exit_code = update_qqq_only_rf_paper_tracking_from_args(
        [
            "--dry-run",
            "--market-dataset-glob",
            str(paths["market"]),
            "--reference-series-glob",
            str(paths["reference"]),
            "--package-dir",
            str(paths["package_dir"]),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "local-only: true" in captured.out
    assert "reads locked paper-tracking package registry: true" in captured.out
    assert "new broad model zoo: false" in captured.out
    assert "production signal export: false" in captured.out
    assert "model card: false" in captured.out
    assert (output_dir / "qqq_only_rf_paper_tracking_update_preflight.md").exists()


def test_paper_tracking_update_cli_run_writes_required_outputs(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    output_dir = tmp_path / "updates"

    assert (
        update_qqq_only_rf_paper_tracking_from_args(
            [
                "--market-dataset-glob",
                str(paths["market"]),
                "--reference-series-glob",
                str(paths["reference"]),
                "--package-dir",
                str(paths["package_dir"]),
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

    signal = pd.read_csv(output_dir / "qqq_only_rf_paper_tracking_update_signal.csv")
    summary = pd.read_csv(output_dir / "qqq_only_rf_paper_tracking_update_summary.csv")
    history = pd.read_csv(output_dir / "qqq_only_rf_paper_tracking_update_history.csv")
    matrix = pd.read_csv(output_dir / "qqq_only_rf_paper_tracking_update_decision_matrix.csv")
    report = output_dir / "qqq_only_rf_paper_tracking_update_report.md"

    assert len(signal) == 3
    assert len(summary) == 1
    assert len(history) == 3
    assert len(matrix) == 3
    assert signal["production_ready"].eq(False).all()
    assert summary["production_ready"].eq(False).all()
    assert matrix["production_ready"].eq(False).all()
    assert set(signal["held_assets"]).issubset({"QQQ", "cash", "QQQ,cash"})
    assert "## 8. Recommendation." in report.read_text(encoding="utf-8")
    assert (output_dir / "qqq_only_rf_paper_tracking_update_exposure_snapshot.png").exists()


def test_paper_tracking_update_cli_rejects_cloud_and_model_signal_paths(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)

    with pytest.raises(ValueError, match="local path"):
        update_qqq_only_rf_paper_tracking_from_args(
            [
                "--dry-run",
                "--market-dataset-glob",
                "s3://bucket/market.parquet",
                "--reference-series-glob",
                str(paths["reference"]),
                "--package-dir",
                str(paths["package_dir"]),
                "--output-dir",
                str(tmp_path / "updates"),
            ]
        )

    with pytest.raises(ValueError, match="artifacts/model_signals"):
        update_qqq_only_rf_paper_tracking_from_args(
            [
                "--dry-run",
                "--market-dataset-glob",
                str(paths["market"]),
                "--reference-series-glob",
                str(paths["reference"]),
                "--package-dir",
                str(paths["package_dir"]),
                "--output-dir",
                "artifacts/model_signals/qqq_only_rf_paper_tracking_updates",
            ]
        )


def test_paper_tracking_update_shell_runner_defaults_are_research_only() -> None:
    script_path = Path("scripts/update_qqq_only_rf_paper_tracking.sh")
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
    assert 'PACKAGE_DIR="${PACKAGE_DIR:-artifacts/reports/qqq_only_rf_paper_tracking}"' in script
    assert (
        'OUTPUT_DIR="${OUTPUT_DIR:-artifacts/reports/qqq_only_rf_paper_tracking_updates}"' in script
    )
    assert 'START_DATE="${START_DATE:-2019-01-01}"' in script
    assert 'END_DATE="${END_DATE:-2026-05-15}"' in script
    assert 'DRY_RUN="${DRY_RUN:-0}"' in script
    assert "local-only: true" in script
    assert "reads locked package registry and local data only: true" in script
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
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    _synthetic_market_data().to_csv(market_path, index=False)
    _synthetic_reference_long().to_csv(reference_path, index=False)

    registry = build_qqq_only_rf_paper_tracking_candidate_registry()
    registry.to_csv(package_dir / CANDIDATE_REGISTRY_NAME, index=False)
    registry.to_json(package_dir / CANDIDATE_REGISTRY_JSON_NAME, orient="records", indent=2)
    _initial_signal().to_csv(package_dir / INITIAL_SIGNAL_NAME, index=False)
    return {
        "market": market_path,
        "reference": reference_path,
        "package_dir": package_dir,
    }


def _initial_signal() -> pd.DataFrame:
    rows = []
    for candidate_id in CANDIDATE_IDS:
        rows.append(
            {
                "candidate_id": candidate_id,
                "as_of_date": "2020-01-03",
                "latest_available_price_date": "2020-01-03",
                "target_exposure": 0.0,
                "effective_exposure": 0.0,
                "qqq_weight": 0.0,
                "cash_weight": 1.0,
                "held_assets": "cash",
                "raw_score": 0.40,
                "threshold": 0.50,
                "rebalance_weekday": "FRI",
                "is_rebalance_day": True,
                "next_action_interpretation": "initial research-only snapshot",
                "production_ready": False,
            }
        )
    return pd.DataFrame(rows, columns=INITIAL_SIGNAL_COLUMNS)
