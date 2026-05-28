from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd

from test_qqq_only_cross_market_timing_model_zoo import (
    _synthetic_market_data,
    _synthetic_reference_long,
)

from us_market_ml.cli import run_qqq_only_rf_confirmation as cli
from us_market_ml.qqq_only_rf_confirmation import build_qqq_only_rf_confirmation_specs


def test_cli_dry_run_writes_preflight_and_reports_guardrails(
    tmp_path: Path,
    capsys,
) -> None:
    market_path = tmp_path / "market.csv"
    reference_path = tmp_path / "reference.csv"
    _synthetic_market_data().to_csv(market_path, index=False)
    _synthetic_reference_long().to_csv(reference_path, index=False)
    output_dir = tmp_path / "sweep"

    exit_code = cli.run_qqq_only_rf_confirmation_from_args(
        [
            "--dry-run",
            "--market-dataset-glob",
            str(market_path),
            "--reference-series-glob",
            str(reference_path),
            "--output-dir",
            str(output_dir),
            "--report-output-dir",
            str(tmp_path / "report"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "focused confirmation: true" in captured.out
    assert "broad model zoo: false" in captured.out
    assert "holdings rule: QQQ/cash only" in captured.out
    assert "production signal export: false" in captured.out
    assert (output_dir / "qqq_only_rf_confirmation_preflight.md").exists()


def test_cli_run_writes_outputs_and_resume_does_not_duplicate_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    market_path = tmp_path / "market.csv"
    reference_path = tmp_path / "reference.csv"
    _synthetic_market_data().to_csv(market_path, index=False)
    _synthetic_reference_long().to_csv(reference_path, index=False)
    output_dir = tmp_path / "sweep"
    report_dir = tmp_path / "report"
    specs = build_qqq_only_rf_confirmation_specs()
    monkeypatch.setattr(cli, "build_qqq_only_rf_confirmation_specs", lambda: specs[:3])

    args = [
        "--market-dataset-glob",
        str(market_path),
        "--reference-series-glob",
        str(reference_path),
        "--output-dir",
        str(output_dir),
        "--report-output-dir",
        str(report_dir),
        "--start-date",
        "2019-01-01",
        "--end-date",
        "2020-06-30",
        "--min-train-rows",
        "40",
        "--resume",
    ]

    assert cli.run_qqq_only_rf_confirmation_from_args(args) == 0
    assert cli.run_qqq_only_rf_confirmation_from_args(args) == 0

    results = pd.read_csv(output_dir / "qqq_only_rf_confirmation_results.csv")
    completed = (
        (output_dir / "qqq_only_rf_confirmation_completed_keys.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    exposure = pd.read_csv(output_dir / "qqq_only_rf_confirmation_exposure_log.csv")

    assert len(results) == len(completed) == 3
    assert results["production_ready"].eq(False).all()
    assert set(exposure["held_assets"]).issubset({"QQQ", "cash", "QQQ,cash"})
    assert (report_dir / "qqq_only_rf_confirmation_report.md").exists()
    assert (report_dir / "qqq_only_rf_confirmation_equity_curve_top_candidates.png").exists()


def test_shell_runner_defaults_and_ctrl_c_helpers_are_present() -> None:
    script_path = Path("scripts/run_qqq_only_rf_confirmation.sh")
    script = script_path.read_text(encoding="utf-8")

    assert os.access(script_path, os.X_OK)
    assert "set -euo pipefail" in script
    assert 'MAX_WORKERS="${MAX_WORKERS:-2}"' in script
    assert 'RESUME="${RESUME:-1}"' in script
    assert 'DRY_RUN="${DRY_RUN:-0}"' in script
    assert "terminate_process_tree" in script
    assert "wait_with_timeout" in script
    assert "cleanup_on_interrupt" in script
    assert "sleep 60" in script
    assert "run_model_signal_export" not in script
    assert "run_model_card" not in script
    assert "s3://" not in script.lower()

    subprocess.run(["bash", "-n", str(script_path)], check=True)
    subprocess.run(
        [
            "bash",
            "-c",
            (
                "QQQ_ONLY_RF_CONFIRMATION_TEST_HELPERS=1; "
                "source scripts/run_qqq_only_rf_confirmation.sh; "
                "declare -f terminate_process_tree >/dev/null; "
                "declare -f wait_with_timeout >/dev/null"
            ),
        ],
        check=True,
    )
