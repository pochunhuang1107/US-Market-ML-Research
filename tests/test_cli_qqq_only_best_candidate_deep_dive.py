from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli.create_qqq_only_best_candidate_deep_dive import (
    create_qqq_only_best_candidate_deep_dive_from_args,
)
from test_qqq_only_best_candidate_deep_dive import _synthetic_inputs


def test_cli_dry_run_writes_preflight_and_reports_guardrails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results, daily_equity, daily_drawdown, exposure_log, summary, decision = _synthetic_inputs()
    paths = _write_inputs(tmp_path, results, daily_equity, daily_drawdown, exposure_log, summary, decision)
    output_dir = tmp_path / "deep_dive"

    exit_code = create_qqq_only_best_candidate_deep_dive_from_args(
        [
            "--dry-run",
            "--model-zoo-results",
            str(paths["results"]),
            "--model-zoo-daily-equity",
            str(paths["daily_equity"]),
            "--model-zoo-daily-drawdown",
            str(paths["daily_drawdown"]),
            "--model-zoo-exposure-log",
            str(paths["exposure_log"]),
            "--model-zoo-summary",
            str(paths["summary"]),
            "--model-zoo-decision-matrix",
            str(paths["decision"]),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    preflight = output_dir / "qqq_only_best_candidate_deep_dive_preflight.md"
    assert exit_code == 0
    assert preflight.exists()
    assert "local-only: true" in captured.out
    assert "new model training: false" in captured.out
    assert "broad sweep: false" in captured.out
    assert "production signal export: false" in captured.out
    assert "model card: false" in captured.out
    text = preflight.read_text(encoding="utf-8")
    assert "Broad sweep: `False`" in text
    assert "Production signal export: `False`" in text
    assert "Model card: `False`" in text


def test_cli_run_writes_outputs(tmp_path: Path) -> None:
    results, daily_equity, daily_drawdown, exposure_log, summary, decision = _synthetic_inputs()
    paths = _write_inputs(tmp_path, results, daily_equity, daily_drawdown, exposure_log, summary, decision)
    output_dir = tmp_path / "deep_dive"

    assert (
        create_qqq_only_best_candidate_deep_dive_from_args(
            [
                "--model-zoo-results",
                str(paths["results"]),
                "--model-zoo-daily-equity",
                str(paths["daily_equity"]),
                "--model-zoo-daily-drawdown",
                str(paths["daily_drawdown"]),
                "--model-zoo-exposure-log",
                str(paths["exposure_log"]),
                "--model-zoo-summary",
                str(paths["summary"]),
                "--model-zoo-decision-matrix",
                str(paths["decision"]),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    matrix = pd.read_csv(output_dir / "qqq_only_best_candidate_deep_dive_decision_matrix.csv")
    report = output_dir / "qqq_only_best_candidate_deep_dive_report.md"
    assert matrix["production_ready"].eq(False).all()
    assert "keep_for_confirmation" in set(matrix["decision"])
    assert report.exists()
    assert "## 11. Decision and recommendation." in report.read_text(encoding="utf-8")


def test_cli_rejects_cloud_paths(tmp_path: Path) -> None:
    results, daily_equity, daily_drawdown, exposure_log, summary, decision = _synthetic_inputs()
    paths = _write_inputs(tmp_path, results, daily_equity, daily_drawdown, exposure_log, summary, decision)

    with pytest.raises(ValueError, match="local path"):
        create_qqq_only_best_candidate_deep_dive_from_args(
            [
                "--model-zoo-results",
                "s3://bucket/results.csv",
                "--model-zoo-daily-equity",
                str(paths["daily_equity"]),
                "--model-zoo-daily-drawdown",
                str(paths["daily_drawdown"]),
                "--model-zoo-exposure-log",
                str(paths["exposure_log"]),
                "--model-zoo-summary",
                str(paths["summary"]),
                "--model-zoo-decision-matrix",
                str(paths["decision"]),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )


def test_shell_runner_guardrails() -> None:
    script_path = Path("scripts/create_qqq_only_best_candidate_deep_dive.sh")
    script = script_path.read_text(encoding="utf-8")

    assert os.access(script_path, os.X_OK)
    assert "set -euo pipefail" in script
    assert "MODEL_ZOO_RESULTS=" in script
    assert "MODEL_ZOO_DAILY_EQUITY=" in script
    assert "MODEL_ZOO_EXPOSURE_LOG=" in script
    assert "reads existing artifacts only: true" in script
    assert "new model training: false" in script
    assert "broad sweep: false" in script
    assert "production signal export: false" in script
    assert "model card: false" in script
    assert "generated artifacts remain uncommitted" in script
    assert "run_daily_signal_export" not in script
    assert "run_model_card" not in script
    assert "model_signal_export" not in script
    assert "s3://" not in script.lower()

    subprocess.run(["bash", "-n", str(script_path)], check=True)


def _write_inputs(
    tmp_path: Path,
    results: pd.DataFrame,
    daily_equity: pd.DataFrame,
    daily_drawdown: pd.DataFrame,
    exposure_log: pd.DataFrame,
    summary: pd.DataFrame,
    decision: pd.DataFrame,
) -> dict[str, Path]:
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
