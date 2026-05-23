from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli.run_long_history_qqq_shortlist_validation import (
    run_long_history_qqq_shortlist_validation_from_args,
)


def test_shortlist_validation_cli_dry_run_writes_candidates_and_preflight(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_csv = tmp_path / "results.csv"
    decision_csv = tmp_path / "decision.csv"
    market_csv = tmp_path / "market.csv"
    output_dir = tmp_path / "validation"
    pd.DataFrame([_result_row("run_1")]).to_csv(results_csv, index=False)
    pd.DataFrame(
        {
            "candidate_id": ["candidate_0001"],
            "candidate_label": ["candidate 1"],
            "decision": ["best_candidate"],
        }
    ).to_csv(decision_csv, index=False)
    _market_frame().to_csv(market_csv, index=False)

    exit_code = run_long_history_qqq_shortlist_validation_from_args(
        [
            "--dry-run",
            "--model-zoo-results",
            str(results_csv),
            "--model-zoo-decision-matrix",
            str(decision_csv),
            "--market-dataset-glob",
            str(market_csv),
            "--output-dir",
            str(output_dir),
            "--top-n",
            "1",
        ]
    )

    captured = capsys.readouterr()
    candidates_path = output_dir / "long_history_qqq_shortlist_validation_candidates.csv"
    preflight_path = output_dir / "long_history_qqq_shortlist_validation_preflight.md"

    assert exit_code == 0
    assert candidates_path.exists()
    assert preflight_path.exists()
    assert "local-only: true" in captured.out
    assert "uses S3/cloud: false" in captured.out
    assert "production signal export: false" in captured.out
    assert "model card: false" in captured.out
    assert "Selected candidates:" in captured.out
    preflight = preflight_path.read_text(encoding="utf-8")
    assert "Broad sweep: `false`" in preflight
    assert "MLP batch: `false`" in preflight
    assert "Production signal export: `false`" in preflight


def test_shortlist_validation_cli_rejects_cloud_paths(tmp_path: Path) -> None:
    results_csv = tmp_path / "results.csv"
    decision_csv = tmp_path / "decision.csv"
    pd.DataFrame([_result_row("run_1")]).to_csv(results_csv, index=False)
    pd.DataFrame(
        {
            "candidate_id": ["candidate_0001"],
            "candidate_label": ["candidate 1"],
            "decision": ["best_candidate"],
        }
    ).to_csv(decision_csv, index=False)

    with pytest.raises(ValueError, match="local path/glob"):
        run_long_history_qqq_shortlist_validation_from_args(
            [
                "--dry-run",
                "--model-zoo-results",
                str(results_csv),
                "--model-zoo-decision-matrix",
                str(decision_csv),
                "--market-dataset-glob",
                "s3://bucket/data/*.parquet",
                "--output-dir",
                str(tmp_path / "validation"),
            ]
        )


def test_shortlist_validation_shell_runner_guardrails() -> None:
    script_path = Path("scripts/run_long_history_qqq_shortlist_validation.sh")
    script = script_path.read_text(encoding="utf-8")

    assert os.access(script_path, os.X_OK)
    assert "set -euo pipefail" in script
    assert "MODEL_ZOO_RESULTS=" in script
    assert "MODEL_ZOO_DECISION_MATRIX=" in script
    assert "MARKET_DATASET_GLOB=" in script
    assert "DRY_RUN=\"${DRY_RUN:-0}\"" in script
    assert "run_daily_signal_export" not in script
    assert "run_model_card" not in script
    assert "model_signal_export" not in script
    assert "s3://" not in script.lower()

    subprocess.run(["bash", "-n", str(script_path)], check=True)


def _result_row(run_key: str) -> dict[str, object]:
    return {
        "run_key": run_key,
        "model_type": "ridge",
        "training_mode": "frozen_base_1999_2018",
        "target_variant": "excess_return_vs_qqq_63d",
        "feature_set_version": "long_history_regime_features_v1",
        "allocation_rule": "top1",
        "rebalance_frequency": "quarterly",
        "min_holding_days": 0,
        "switch_margin": 0.0,
        "transaction_cost_bps": 10.0,
        "recency_decay_profile": pd.NA,
        "base_weight": pd.NA,
        "recent_window_years": pd.NA,
        "finetune_window_years": pd.NA,
        "finetune_lr_scale": pd.NA,
        "total_return": 2.0,
        "annualized_return": 0.2,
        "max_drawdown": -0.3,
        "sharpe_like": 0.1,
        "return_to_drawdown": 6.6,
        "transaction_count": 10,
        "average_turnover": 0.01,
        "estimated_transaction_cost": 0.02,
        "qqq_buy_hold_total_return": 1.0,
        "spy_buy_hold_total_return": 0.8,
        "strategy_vs_qqq": 1.0,
        "strategy_vs_spy": 1.2,
        "qqq_return_capture": 2.0,
        "beats_qqq_return": True,
        "beats_qqq_return_to_drawdown": True,
        "drawdown_improvement_vs_qqq": 0.02,
        "qqq_buy_hold_max_drawdown": -0.35,
        "spy_buy_hold_max_drawdown": -0.3,
        "qqq_buy_hold_return_to_drawdown": 2.8,
        "spy_buy_hold_return_to_drawdown": 2.6,
        "decision": "unclassified",
    }


def _market_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2020-01-01", periods=2, freq="D")
    for symbol in ["QQQ", "SPY"]:
        for day, date in enumerate(dates):
            close = 100.0 + day
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 1_000_000,
                    "vwap": close,
                }
            )
    return pd.DataFrame(rows)
