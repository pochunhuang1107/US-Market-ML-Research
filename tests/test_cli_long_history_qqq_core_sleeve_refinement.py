from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli.run_long_history_qqq_core_sleeve_refinement import (
    run_long_history_qqq_core_sleeve_refinement_from_args,
)


def test_core_sleeve_refinement_cli_dry_run_writes_preflight_and_prints_modes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_inputs(tmp_path)
    output_dir = tmp_path / "sweep"
    report_output_dir = tmp_path / "report"

    exit_code = run_long_history_qqq_core_sleeve_refinement_from_args(
        [
            "--dry-run",
            "--spy-core-sleeve-results",
            str(paths["previous_results"]),
            "--spy-core-sleeve-daily-equity",
            str(paths["previous_daily_equity"]),
            "--candidate-deep-dive-daily-equity",
            str(paths["daily_equity"]),
            "--candidate-deep-dive-summary",
            str(paths["summary"]),
            "--candidate-deep-dive-decision-matrix",
            str(paths["decision"]),
            "--output-dir",
            str(output_dir),
            "--report-output-dir",
            str(report_output_dir),
        ]
    )

    captured = capsys.readouterr()
    preflight_path = output_dir / "long_history_qqq_core_sleeve_refinement_preflight.md"

    assert exit_code == 0
    assert preflight_path.exists()
    assert "local-only: true" in captured.out
    assert "uses S3/cloud: false" in captured.out
    assert "broad model zoo: false" in captured.out
    assert "MLP batch: false" in captured.out
    assert "production signal export: false" in captured.out
    assert "model card: false" in captured.out
    assert "Accounting modes:" in captured.out
    assert "Fine core weight grid:" in captured.out
    preflight = preflight_path.read_text(encoding="utf-8")
    assert "Broad model zoo: `false`" in preflight
    assert "MLP batch: `false`" in preflight
    assert "Production signal export: `false`" in preflight
    assert "Model card: `false`" in preflight
    assert "daily_rebalanced_fixed_weights" in preflight
    assert "monthly_rebalanced_weights" in preflight
    assert "quarterly_rebalanced_weights" in preflight
    assert "candidate_rebalance_aligned_weights" in preflight


def test_core_sleeve_refinement_cli_rejects_cloud_paths(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)

    with pytest.raises(ValueError, match="local path/glob"):
        run_long_history_qqq_core_sleeve_refinement_from_args(
            [
                "--dry-run",
                "--spy-core-sleeve-results",
                "https://example.com/results.csv",
                "--spy-core-sleeve-daily-equity",
                str(paths["previous_daily_equity"]),
                "--candidate-deep-dive-daily-equity",
                str(paths["daily_equity"]),
                "--candidate-deep-dive-summary",
                str(paths["summary"]),
                "--candidate-deep-dive-decision-matrix",
                str(paths["decision"]),
                "--output-dir",
                str(tmp_path / "sweep"),
            ]
        )


def test_core_sleeve_refinement_shell_runner_guardrails() -> None:
    script_path = Path("scripts/run_long_history_qqq_core_sleeve_refinement.sh")
    script = script_path.read_text(encoding="utf-8")

    assert os.access(script_path, os.X_OK)
    assert "set -euo pipefail" in script
    assert "SPY_CORE_SLEEVE_RESULTS=" in script
    assert "SPY_CORE_SLEEVE_DAILY_EQUITY=" in script
    assert "CANDIDATE_DEEP_DIVE_DAILY_EQUITY=" in script
    assert "CANDIDATE_DEEP_DIVE_SUMMARY=" in script
    assert "CANDIDATE_DEEP_DIVE_DECISION_MATRIX=" in script
    assert "OUTPUT_DIR=" in script
    assert "REPORT_OUTPUT_DIR=" in script
    assert "CANDIDATE_IDS=" in script
    assert "PORTFOLIO_ACCOUNTING_MODES=" in script
    assert "DRY_RUN=\"${DRY_RUN:-0}\"" in script
    assert "Generated artifacts remain uncommitted" in script
    assert "run_daily_signal_export" not in script
    assert "run_model_card" not in script
    assert "model_signal_export" not in script
    assert "s3://" not in script.lower()

    subprocess.run(["bash", "-n", str(script_path)], check=True)


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    daily_equity = _daily_equity_frame()
    previous_results = pd.DataFrame(
        [
            {
                "candidate_id": "candidate_0001",
                "spy_core_weight": 0.4,
                "candidate_sleeve_weight": 0.6,
                "rebalance_policy": "daily_rebalanced_fixed_weights",
                "total_return": 4.58,
                "strategy_vs_qqq": 0.78,
                "max_drawdown": -0.348,
                "return_to_drawdown": 13.1,
                "decision": "best_core_sleeve_candidate",
            }
        ]
    )
    previous_daily_equity = pd.DataFrame(
        {
            "date": ["2020-01-01"],
            "candidate_id": ["candidate_0001"],
            "equity": [1.0],
        }
    )
    summary = pd.DataFrame(
        [
            _source_row("candidate_0001", "elasticnet"),
            _source_row("candidate_0003", "ridge"),
        ]
    )
    decision = pd.DataFrame(
        [
            _decision_row("candidate_0001"),
            _decision_row("candidate_0003"),
        ]
    )
    paths = {
        "previous_results": tmp_path / "previous_results.csv",
        "previous_daily_equity": tmp_path / "previous_daily_equity.csv",
        "daily_equity": tmp_path / "daily_equity.csv",
        "summary": tmp_path / "summary.csv",
        "decision": tmp_path / "decision.csv",
    }
    previous_results.to_csv(paths["previous_results"], index=False)
    previous_daily_equity.to_csv(paths["previous_daily_equity"], index=False)
    daily_equity.to_csv(paths["daily_equity"], index=False)
    summary.to_csv(paths["summary"], index=False)
    decision.to_csv(paths["decision"], index=False)
    return paths


def _daily_equity_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    for candidate_id in ("candidate_0001", "candidate_0003"):
        specs = [
            ("candidate_strategy", [0.0, 0.02, -0.01, 0.01, 0.02]),
            ("QQQ buy-and-hold", [0.0, 0.01, -0.005, 0.01, 0.01]),
            ("SPY buy-and-hold", [0.0, 0.005, -0.003, 0.006, 0.006]),
        ]
        for strategy_name, returns in specs:
            equity = pd.Series(returns).add(1.0).cumprod()
            drawdown = equity / equity.cummax() - 1.0
            for idx, date in enumerate(dates):
                rows.append(
                    {
                        "date": date,
                        "candidate_id": candidate_id,
                        "candidate_label": f"{candidate_id} label",
                        "strategy_name": strategy_name,
                        "active_symbol": "AAA",
                        "active_symbols": "AAA",
                        "daily_return": returns[idx],
                        "equity": equity.iloc[idx],
                        "drawdown": drawdown.iloc[idx],
                        "gross_return": returns[idx],
                        "transaction_cost": 0.0,
                        "turnover": 1.0
                        if strategy_name == "candidate_strategy" and idx == 2
                        else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def _source_row(candidate_id: str, model_type: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "candidate_label": f"{candidate_id} label",
        "model_type": model_type,
        "training_mode": "recency_weighted_expanding"
        if model_type == "elasticnet"
        else "frozen_base_1999_2018",
        "target_variant": "excess_return_vs_qqq_126d"
        if model_type == "elasticnet"
        else "excess_return_vs_qqq_63d",
        "feature_set_version": "long_history_regime_features_v1",
        "allocation_rule": "top1",
        "rebalance_frequency": "quarterly",
        "min_holding_days": 20 if model_type == "elasticnet" else 0,
        "switch_margin": 0.0,
        "transaction_cost_bps": 10.0,
        "recency_decay_profile": "medium_decay" if model_type == "elasticnet" else pd.NA,
        "base_weight": pd.NA,
        "recent_window_years": pd.NA,
        "finetune_window_years": pd.NA,
        "finetune_lr_scale": pd.NA,
        "total_return": 1.0,
        "max_drawdown": -0.3,
        "return_to_drawdown": 3.3,
        "transaction_count": 5,
        "average_turnover": 0.01,
        "estimated_transaction_cost": 0.02,
        "qqq_buy_hold_total_return": 0.8,
        "spy_buy_hold_total_return": 0.6,
        "strategy_vs_qqq": 0.2,
        "strategy_vs_spy": 0.4,
        "positive_vs_qqq_window_rate": 0.6,
        "qqq_underperformance_rate": 0.4,
        "worst_strategy_vs_qqq": -0.1,
        "top_symbol": "XLE" if candidate_id == "candidate_0001" else "ARKK",
        "top_symbol_exposure_pct": 0.44,
        "top3_symbol_exposure_pct": 0.86,
        "shortlist_decision": "keep_for_deep_dive",
    }


def _decision_row(candidate_id: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "decision": "keep_for_targeted_sweep",
        "rationale": "research only",
        "cost_20bps_strategy_vs_qqq": 0.4,
        "cost_30bps_strategy_vs_qqq": 0.3,
    }
