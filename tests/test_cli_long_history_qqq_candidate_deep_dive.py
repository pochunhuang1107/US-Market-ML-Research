from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli.run_long_history_qqq_candidate_deep_dive import (
    run_long_history_qqq_candidate_deep_dive_from_args,
)


def test_candidate_deep_dive_cli_dry_run_writes_preflight_and_selects_defaults(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_csv = tmp_path / "results.csv"
    candidates_csv = tmp_path / "candidates.csv"
    decision_csv = tmp_path / "decision.csv"
    market_csv = tmp_path / "market.csv"
    output_dir = tmp_path / "deep_dive"
    pd.DataFrame([{"run_key": "run_1"}, {"run_key": "run_2"}]).to_csv(
        results_csv,
        index=False,
    )
    pd.DataFrame(
        [
            _candidate_row("candidate_0001"),
            _candidate_row("candidate_0003", model_type="ridge"),
        ]
    ).to_csv(candidates_csv, index=False)
    pd.DataFrame(
        {
            "candidate_id": ["candidate_0001", "candidate_0003"],
            "decision": ["keep_for_deep_dive", "robust_candidate"],
            "rationale": ["one", "three"],
        }
    ).to_csv(decision_csv, index=False)
    _market_frame().to_csv(market_csv, index=False)

    exit_code = run_long_history_qqq_candidate_deep_dive_from_args(
        [
            "--dry-run",
            "--model-zoo-results",
            str(results_csv),
            "--shortlist-candidates",
            str(candidates_csv),
            "--shortlist-decision-matrix",
            str(decision_csv),
            "--market-dataset-glob",
            str(market_csv),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    preflight_path = output_dir / "long_history_qqq_candidate_deep_dive_preflight.md"

    assert exit_code == 0
    assert preflight_path.exists()
    assert "local-only: true" in captured.out
    assert "uses S3/cloud: false" in captured.out
    assert "production signal export: false" in captured.out
    assert "model card: false" in captured.out
    assert "Selected candidates:" in captured.out
    assert "candidate_0001" in captured.out
    assert "candidate_0003" in captured.out
    preflight = preflight_path.read_text(encoding="utf-8")
    assert "Broad sweep: `false`" in preflight
    assert "MLP batch: `false`" in preflight
    assert "Production signal export: `false`" in preflight
    assert "Model card: `false`" in preflight


def test_candidate_deep_dive_cli_rejects_cloud_paths(tmp_path: Path) -> None:
    results_csv = tmp_path / "results.csv"
    candidates_csv = tmp_path / "candidates.csv"
    decision_csv = tmp_path / "decision.csv"
    pd.DataFrame([{"run_key": "run_1"}]).to_csv(results_csv, index=False)
    pd.DataFrame([_candidate_row("candidate_0001")]).to_csv(candidates_csv, index=False)
    pd.DataFrame(
        {
            "candidate_id": ["candidate_0001"],
            "decision": ["keep_for_deep_dive"],
        }
    ).to_csv(decision_csv, index=False)

    with pytest.raises(ValueError, match="local path/glob"):
        run_long_history_qqq_candidate_deep_dive_from_args(
            [
                "--dry-run",
                "--model-zoo-results",
                str(results_csv),
                "--shortlist-candidates",
                str(candidates_csv),
                "--shortlist-decision-matrix",
                str(decision_csv),
                "--market-dataset-glob",
                "s3://bucket/data/*.parquet",
                "--output-dir",
                str(tmp_path / "deep_dive"),
            ]
        )


def test_candidate_deep_dive_shell_runner_guardrails() -> None:
    script_path = Path("scripts/run_long_history_qqq_candidate_deep_dive.sh")
    script = script_path.read_text(encoding="utf-8")

    assert os.access(script_path, os.X_OK)
    assert "set -euo pipefail" in script
    assert "MODEL_ZOO_RESULTS=" in script
    assert "SHORTLIST_CANDIDATES=" in script
    assert "SHORTLIST_DECISION_MATRIX=" in script
    assert "MARKET_DATASET_GLOB=" in script
    assert "CANDIDATE_IDS=" in script
    assert "DRY_RUN=\"${DRY_RUN:-0}\"" in script
    assert "Generated artifacts remain uncommitted" in script
    assert "run_daily_signal_export" not in script
    assert "run_model_card" not in script
    assert "model_signal_export" not in script
    assert "s3://" not in script.lower()

    subprocess.run(["bash", "-n", str(script_path)], check=True)


def _candidate_row(
    candidate_id: str,
    *,
    model_type: str = "elasticnet",
) -> dict[str, object]:
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
        "decision": "best_candidate",
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
