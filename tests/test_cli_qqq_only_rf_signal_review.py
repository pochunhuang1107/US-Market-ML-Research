from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from test_qqq_only_rf_signal_review import _signal_review_inputs

from us_market_ml.cli.create_qqq_only_rf_signal_review import (
    create_qqq_only_rf_signal_review_from_args,
)


def test_signal_review_cli_dry_run_writes_preflight_and_guardrails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_inputs(tmp_path)
    output_dir = tmp_path / "review"

    exit_code = create_qqq_only_rf_signal_review_from_args(
        [
            "--dry-run",
            "--results",
            str(paths["results"]),
            "--daily-equity",
            str(paths["daily_equity"]),
            "--daily-drawdown",
            str(paths["daily_drawdown"]),
            "--exposure-log",
            str(paths["exposure_log"]),
            "--summary",
            str(paths["summary"]),
            "--decision-matrix",
            str(paths["decision"]),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "local-only: true" in captured.out
    assert "new model training: false" in captured.out
    assert "broad sweep: false" in captured.out
    assert "production signal export: false" in captured.out
    assert "model card: false" in captured.out
    assert (output_dir / "qqq_only_rf_signal_review_preflight.md").exists()


def test_signal_review_cli_run_writes_required_outputs(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    output_dir = tmp_path / "review"

    assert (
        create_qqq_only_rf_signal_review_from_args(
            [
                "--results",
                str(paths["results"]),
                "--daily-equity",
                str(paths["daily_equity"]),
                "--daily-drawdown",
                str(paths["daily_drawdown"]),
                "--exposure-log",
                str(paths["exposure_log"]),
                "--summary",
                str(paths["summary"]),
                "--decision-matrix",
                str(paths["decision"]),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    summary = pd.read_csv(output_dir / "qqq_only_rf_signal_review_summary.csv")
    matrix = pd.read_csv(output_dir / "qqq_only_rf_signal_review_decision_matrix.csv")
    report = output_dir / "qqq_only_rf_signal_review_report.md"

    assert summary["production_ready"].eq(False).all()
    assert matrix["production_ready"].eq(False).all()
    assert "## 11. Final recommendation:" in report.read_text(encoding="utf-8")
    assert (output_dir / "qqq_only_rf_signal_review_threshold_comparison.png").exists()


def test_signal_review_cli_rejects_cloud_paths(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)

    with pytest.raises(ValueError, match="local path"):
        create_qqq_only_rf_signal_review_from_args(
            [
                "--results",
                "s3://bucket/results.csv",
                "--daily-equity",
                str(paths["daily_equity"]),
                "--daily-drawdown",
                str(paths["daily_drawdown"]),
                "--exposure-log",
                str(paths["exposure_log"]),
                "--summary",
                str(paths["summary"]),
                "--decision-matrix",
                str(paths["decision"]),
                "--output-dir",
                str(tmp_path / "review"),
            ]
        )


def test_signal_review_shell_runner_defaults_are_report_only() -> None:
    script_path = Path("scripts/create_qqq_only_rf_signal_review.sh")
    script = script_path.read_text(encoding="utf-8")

    assert os.access(script_path, os.X_OK)
    assert "set -euo pipefail" in script
    assert "RF_CONFIRMATION_RESULTS" in script
    assert "RF_CONFIRMATION_DAILY_EQUITY" in script
    assert "RF_CONFIRMATION_DECISION_MATRIX" in script
    assert "new model training: false" in script
    assert "broad sweep: false" in script
    assert "production signal export: false" in script
    assert "model card: false" in script
    assert "run_model_signal_export" not in script
    assert "run_model_card" not in script
    assert "run_qqq_only_rf_confirmation" not in script
    assert "s3://" not in script.lower()

    subprocess.run(["bash", "-n", str(script_path)], check=True)


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    results, daily_equity, daily_drawdown, exposure_log, summary, decision = _signal_review_inputs()
    paths = {
        "results": tmp_path / "results.csv",
        "daily_equity": tmp_path / "daily_equity.csv",
        "daily_drawdown": tmp_path / "daily_drawdown.csv",
        "exposure_log": tmp_path / "exposure_log.csv",
        "summary": tmp_path / "summary.csv",
        "decision": tmp_path / "decision.csv",
    }
    results.to_csv(paths["results"], index=False)
    daily_equity.to_csv(paths["daily_equity"], index=False)
    daily_drawdown.to_csv(paths["daily_drawdown"], index=False)
    exposure_log.to_csv(paths["exposure_log"], index=False)
    summary.to_csv(paths["summary"], index=False)
    decision.to_csv(paths["decision"], index=False)
    return paths
