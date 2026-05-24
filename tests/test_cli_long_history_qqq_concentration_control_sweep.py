from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli.run_long_history_qqq_concentration_control_sweep import (
    run_long_history_qqq_concentration_control_sweep_from_args,
)


def test_concentration_control_sweep_cli_dry_run_writes_candidates_and_preflight(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_zoo_csv = tmp_path / "model_zoo.csv"
    summary_csv = tmp_path / "summary.csv"
    decision_csv = tmp_path / "decision.csv"
    selected_symbols_csv = tmp_path / "selected_symbols.csv"
    market_csv = tmp_path / "market.csv"
    output_dir = tmp_path / "sweep"
    report_output_dir = tmp_path / "report"
    pd.DataFrame([{"run_key": f"run_{index}"} for index in range(200)]).to_csv(
        model_zoo_csv,
        index=False,
    )
    pd.DataFrame(
        [
            _source_row("candidate_0001", "elasticnet", "XLE", 0.44, 0.86, 4),
            _source_row("candidate_0003", "ridge", "ARKK", 0.34, 0.73, 8),
        ]
    ).to_csv(summary_csv, index=False)
    pd.DataFrame(
        {
            "candidate_id": ["candidate_0001", "candidate_0003"],
            "decision": ["keep_for_targeted_sweep", "keep_for_signal_review"],
            "rationale": ["one", "three"],
        }
    ).to_csv(decision_csv, index=False)
    pd.DataFrame(
        {
            "date": ["2020-01-01"],
            "candidate_id": ["candidate_0001"],
            "symbol": ["XLE"],
        }
    ).to_csv(selected_symbols_csv, index=False)
    _market_frame().to_csv(market_csv, index=False)

    exit_code = run_long_history_qqq_concentration_control_sweep_from_args(
        [
            "--dry-run",
            "--model-zoo-results",
            str(model_zoo_csv),
            "--candidate-deep-dive-summary",
            str(summary_csv),
            "--candidate-deep-dive-decision-matrix",
            str(decision_csv),
            "--candidate-deep-dive-selected-symbols",
            str(selected_symbols_csv),
            "--market-dataset-glob",
            str(market_csv),
            "--output-dir",
            str(output_dir),
            "--report-output-dir",
            str(report_output_dir),
        ]
    )

    captured = capsys.readouterr()
    candidates_path = output_dir / "long_history_qqq_concentration_control_candidates.csv"
    preflight_path = output_dir / "long_history_qqq_concentration_control_preflight.md"

    assert exit_code == 0
    assert candidates_path.exists()
    assert preflight_path.exists()
    assert len(pd.read_csv(candidates_path)) == 34
    assert "local-only: true" in captured.out
    assert "uses S3/cloud: false" in captured.out
    assert "broad sweep: false" in captured.out
    assert "MLP batch: false" in captured.out
    assert "production signal export: false" in captured.out
    assert "model card: false" in captured.out
    assert "generated concentration-control config count: 34" in captured.out
    assert "smaller than model zoo: True" in captured.out
    preflight = preflight_path.read_text(encoding="utf-8")
    assert "Broad sweep: `false`" in preflight
    assert "MLP batch: `false`" in preflight
    assert "Production signal export: `false`" in preflight
    assert "Model card: `false`" in preflight


def test_concentration_control_sweep_cli_rejects_cloud_paths(tmp_path: Path) -> None:
    model_zoo_csv = tmp_path / "model_zoo.csv"
    summary_csv = tmp_path / "summary.csv"
    decision_csv = tmp_path / "decision.csv"
    selected_symbols_csv = tmp_path / "selected_symbols.csv"
    pd.DataFrame([{"run_key": "run_1"}]).to_csv(model_zoo_csv, index=False)
    pd.DataFrame([_source_row("candidate_0001", "elasticnet", "XLE", 0.44, 0.86, 4)]).to_csv(
        summary_csv,
        index=False,
    )
    pd.DataFrame(
        {
            "candidate_id": ["candidate_0001"],
            "decision": ["keep_for_targeted_sweep"],
            "rationale": ["one"],
        }
    ).to_csv(decision_csv, index=False)
    pd.DataFrame({"candidate_id": ["candidate_0001"], "symbol": ["XLE"]}).to_csv(
        selected_symbols_csv,
        index=False,
    )

    with pytest.raises(ValueError, match="local path/glob"):
        run_long_history_qqq_concentration_control_sweep_from_args(
            [
                "--dry-run",
                "--model-zoo-results",
                str(model_zoo_csv),
                "--candidate-deep-dive-summary",
                str(summary_csv),
                "--candidate-deep-dive-decision-matrix",
                str(decision_csv),
                "--candidate-deep-dive-selected-symbols",
                str(selected_symbols_csv),
                "--market-dataset-glob",
                "s3://bucket/data/*.parquet",
                "--output-dir",
                str(tmp_path / "sweep"),
            ]
        )


def test_concentration_control_shell_runner_guardrails() -> None:
    script_path = Path("scripts/run_long_history_qqq_concentration_control_sweep.sh")
    script = script_path.read_text(encoding="utf-8")

    assert os.access(script_path, os.X_OK)
    assert "set -euo pipefail" in script
    assert "MODEL_ZOO_RESULTS=" in script
    assert "CANDIDATE_DEEP_DIVE_SUMMARY=" in script
    assert "CANDIDATE_DEEP_DIVE_DECISION_MATRIX=" in script
    assert "CANDIDATE_DEEP_DIVE_SELECTED_SYMBOLS=" in script
    assert "MARKET_DATASET_GLOB=" in script
    assert "OUTPUT_DIR=" in script
    assert "REPORT_OUTPUT_DIR=" in script
    assert "CANDIDATE_IDS=" in script
    assert "DRY_RUN=\"${DRY_RUN:-0}\"" in script
    assert "Generated artifacts remain uncommitted" in script
    assert "run_daily_signal_export" not in script
    assert "run_model_card" not in script
    assert "model_signal_export" not in script
    assert "s3://" not in script.lower()

    subprocess.run(["bash", "-n", str(script_path)], check=True)


def _source_row(
    candidate_id: str,
    model_type: str,
    top_symbol: str,
    top_exposure: float,
    top3_exposure: float,
    unique_count: int,
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
        "total_return": 6.0,
        "strategy_vs_qqq": 2.0,
        "return_to_drawdown": 12.0,
        "positive_vs_qqq_window_rate": 0.56,
        "worst_strategy_vs_qqq": -0.4,
        "top_symbol": top_symbol,
        "top_symbol_exposure_pct": top_exposure,
        "top3_symbol_exposure_pct": top3_exposure,
        "unique_symbol_count": unique_count,
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
