from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from test_qqq_only_rf_out_of_time_confirmation import _out_of_time_inputs

from us_market_ml.cli.create_qqq_only_rf_out_of_time_confirmation_report import (
    create_qqq_only_rf_out_of_time_confirmation_report_from_args,
)


def test_report_cli_dry_run_writes_preflight_and_guardrails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_inputs(tmp_path)
    output_dir = tmp_path / "report"

    exit_code = create_qqq_only_rf_out_of_time_confirmation_report_from_args(
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
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "local-only: true" in captured.out
    assert "new broad model zoo: false" in captured.out
    assert "production signal export: false" in captured.out
    assert (output_dir / "qqq_only_rf_out_of_time_confirmation_report_preflight.md").exists()


def test_report_cli_run_writes_required_outputs(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    output_dir = tmp_path / "report"

    assert (
        create_qqq_only_rf_out_of_time_confirmation_report_from_args(
            [
                "--results",
                str(paths["results"]),
                "--daily-equity",
                str(paths["daily_equity"]),
                "--daily-drawdown",
                str(paths["daily_drawdown"]),
                "--exposure-log",
                str(paths["exposure_log"]),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    summary = pd.read_csv(output_dir / "qqq_only_rf_out_of_time_confirmation_summary.csv")
    matrix = pd.read_csv(output_dir / "qqq_only_rf_out_of_time_confirmation_decision_matrix.csv")
    report = output_dir / "qqq_only_rf_out_of_time_confirmation_report.md"

    assert summary["production_ready"].eq(False).all()
    assert matrix["production_ready"].eq(False).all()
    assert "## 11. Recommendation:" in report.read_text(encoding="utf-8")
    assert (output_dir / "qqq_only_rf_out_of_time_cost_stability.png").exists()


def test_report_cli_rejects_cloud_paths(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)

    with pytest.raises(ValueError, match="local path"):
        create_qqq_only_rf_out_of_time_confirmation_report_from_args(
            [
                "--results",
                "s3://bucket/results.csv",
                "--daily-equity",
                str(paths["daily_equity"]),
                "--daily-drawdown",
                str(paths["daily_drawdown"]),
                "--exposure-log",
                str(paths["exposure_log"]),
                "--output-dir",
                str(tmp_path / "report"),
            ]
        )


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    results, daily_equity, daily_drawdown, exposure_log = _out_of_time_inputs()
    paths = {
        "results": tmp_path / "results.csv",
        "daily_equity": tmp_path / "daily_equity.csv",
        "daily_drawdown": tmp_path / "daily_drawdown.csv",
        "exposure_log": tmp_path / "exposure_log.csv",
    }
    results.to_csv(paths["results"], index=False)
    daily_equity.to_csv(paths["daily_equity"], index=False)
    daily_drawdown.to_csv(paths["daily_drawdown"], index=False)
    exposure_log.to_csv(paths["exposure_log"], index=False)
    return paths
