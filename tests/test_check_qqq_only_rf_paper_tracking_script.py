from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_check_qqq_only_rf_paper_tracking_script_exists_and_is_executable() -> None:
    script_path = Path("scripts/check_qqq_only_rf_paper_tracking.sh")

    assert script_path.exists()
    assert os.access(script_path, os.X_OK)


def test_check_qqq_only_rf_paper_tracking_script_is_research_only_wrapper() -> None:
    script_path = Path("scripts/check_qqq_only_rf_paper_tracking.sh")
    script = script_path.read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "research-only: true" in script
    assert "local-only: true" in script
    assert "S3/cloud: false" in script
    assert "broad sweep: false" in script
    assert "production signal export: false" in script
    assert "model card: false" in script
    assert "production_ready always false" in script
    assert "tradable holdings: QQQ/cash only" in script
    assert "output under artifacts/model_signals: false" in script
    assert "artifacts/model_signals/" not in script
    assert "run_model_signal_export" not in script
    assert "run_model_card" not in script
    assert "run_qqq_only_rf_out_of_time_confirmation" not in script
    assert "s3://" not in script.lower()


def test_check_qqq_only_rf_paper_tracking_script_calls_update_and_reads_signal() -> None:
    script = Path("scripts/check_qqq_only_rf_paper_tracking.sh").read_text(encoding="utf-8")

    assert "scripts/update_qqq_only_rf_paper_tracking.sh" in script
    assert 'END_DATE="$latest_price_date"' in script
    assert "qqq_only_rf_paper_tracking_update_signal.csv" in script
    assert "qqq_only_rf_paper_tracking_update_report.md" in script


def test_check_qqq_only_rf_paper_tracking_script_prints_primary_action() -> None:
    script = Path("scripts/check_qqq_only_rf_paper_tracking.sh").read_text(encoding="utf-8")

    assert "PRIMARY CANDIDATE:" in script
    assert "primary_rf_threshold_050" in script
    assert "PRIMARY ACTION:" in script
    assert "MAINTAIN 100% QQQ" in script
    assert "MOVE TO CASH / RISK-OFF" in script
    assert "REVIEW REQUIRED" in script


def test_check_qqq_only_rf_paper_tracking_script_syntax() -> None:
    subprocess.run(
        ["bash", "-n", "scripts/check_qqq_only_rf_paper_tracking.sh"],
        check=True,
    )
