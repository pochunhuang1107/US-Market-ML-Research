from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli.create_tiingo_modern_baseline_sweep_report import (
    DECISION_MATRIX_COLUMNS,
    SUMMARY_COLUMNS,
    build_tiingo_modern_baseline_sweep_report,
    run_tiingo_modern_baseline_sweep_report_from_args,
)


def test_tiingo_modern_baseline_report_builds_summary_and_decision_matrix(
    tmp_path: Path,
) -> None:
    results_csv = tmp_path / "results.csv"
    _synthetic_results().to_csv(results_csv, index=False)

    report_data = build_tiingo_modern_baseline_sweep_report(results_csv=results_csv)

    assert list(report_data.candidate_summary.columns) == SUMMARY_COLUMNS
    assert list(report_data.decision_matrix.columns) == DECISION_MATRIX_COLUMNS
    assert report_data.dataset_source == "tiingo_only"
    assert set(report_data.candidate_summary["dataset_source"]) == {"tiingo_only"}
    assert set(report_data.decision_matrix["dataset_source"]) == {"tiingo_only"}
    assert set(report_data.candidate_summary["sweep_profile"]) == {"core_linear"}
    assert set(report_data.decision_matrix["sweep_profile"]) == {"core_linear"}
    assert len(report_data.candidate_summary) == 160
    assert len(report_data.decision_matrix) == 160
    assert "mlp" not in set(report_data.candidate_summary["model_family"])

    leader = report_data.candidate_summary.iloc[0]
    assert leader["model_type"] == "gradient_boosting"
    assert leader["average_test_return"] == pytest.approx(0.067)


def test_tiingo_modern_baseline_report_cli_writes_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_csv = tmp_path / "results.csv"
    output_dir = tmp_path / "report"
    _synthetic_results().to_csv(results_csv, index=False)

    return_code = run_tiingo_modern_baseline_sweep_report_from_args(
        [
            "--results-csv",
            str(results_csv),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    summary_path = output_dir / "tiingo_modern_baseline_summary.csv"
    decision_path = output_dir / "tiingo_modern_baseline_decision_matrix.csv"
    markdown_path = output_dir / "tiingo_modern_baseline_report.md"

    assert return_code == 0
    assert "Tiingo-only modern baseline sweep report" in captured.out
    assert "dataset source: tiingo_only" in captured.out
    assert summary_path.exists()
    assert decision_path.exists()
    assert markdown_path.exists()

    summary = pd.read_csv(summary_path)
    matrix = pd.read_csv(decision_path)
    markdown = markdown_path.read_text(encoding="utf-8")

    assert set(summary["dataset_source"]) == {"tiingo_only"}
    assert set(matrix["dataset_source"]) == {"tiingo_only"}
    assert set(summary["sweep_profile"]) == {"core_linear"}
    assert set(matrix["sweep_profile"]) == {"core_linear"}
    assert "Executive Summary" in markdown
    assert "Best Candidates By Return" in markdown
    assert "Best Candidates By Balanced Risk/Return" in markdown
    assert "Drawdown-Sensitive Candidates" in markdown
    assert "Cost Sensitivity" in markdown
    assert "Smoothing Sensitivity" in markdown
    assert "Window Stability" in markdown
    assert "Recommended Baseline Shortlist For Deep Dive" in markdown
    assert "MLP is intentionally deferred" in markdown
    assert "Tiingo-only research source" in markdown
    assert "Massive app data may not match research results exactly" in markdown
    assert "Still not production-ready" in markdown
    assert "ETF survivorship bias is reduced" in markdown


def _synthetic_results() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    model_returns = {
        "linear_regression": 0.020,
        "ridge": 0.025,
        "random_forest": 0.030,
        "extra_trees": 0.035,
        "gradient_boosting": 0.045,
    }
    folds = [
        ("tw2y_v6m_t3m_q01_20220201_20220430", -0.005),
        ("tw2y_v6m_t6m_q02_20220501_20221031", 0.005),
        ("tw3y_v12m_t3m_q01_20230101_20230331", 0.010),
        ("tw5y_v12m_t12m_q01_20240101_20241231", 0.015),
    ]
    for model_type, base_return in model_returns.items():
        for top_k in (1, 3):
            for min_holding_days, switch_margin in [(5, 0.0), (10, 0.002)]:
                for cost in (5.0, 20.0):
                    for fold_name, fold_adjustment in folds:
                        top_k_bonus = 0.004 if top_k == 3 else 0.0
                        smoothing_bonus = 0.003 if min_holding_days == 10 else 0.0
                        cost_penalty = 0.0 if cost == 5.0 else 0.010
                        total_return = (
                            base_return
                            + top_k_bonus
                            + smoothing_bonus
                            + fold_adjustment
                            - cost_penalty
                        )
                        rows.append(
                            {
                                "run_key": (
                                    f"{model_type}-{top_k}-{min_holding_days}-"
                                    f"{switch_margin:g}-{cost:g}-{fold_name}"
                                ),
                                "universe": "tiingo_modern_dynamic",
                                "data_path": "/tmp/tiingo.csv",
                                "sweep_profile": "core_linear",
                                "fold": fold_name,
                                "split": "test",
                                "model_family": "baseline",
                                "model_type": model_type,
                                "feature_set_version": "default_features_v1",
                                "feature_count": 12,
                                "hidden_units": "none",
                                "dropout": "",
                                "scale_target": False,
                                "learning_rate": "",
                                "weight_decay": "",
                                "batch_size": "",
                                "max_epochs": "",
                                "patience": "",
                                "random_state": 42,
                                "top_k": top_k,
                                "min_holding_days": min_holding_days,
                                "switch_margin": switch_margin,
                                "allocation_mode": "equal_weight",
                                "score_temperature": 1.0,
                                "max_weight": 1.0,
                                "transaction_cost_bps": cost,
                                "target_col": "excess_return_vs_spy_20d",
                                "label_horizon_days": 20,
                                "total_return": total_return,
                                "mean_daily_return": total_return / 63,
                                "daily_volatility": 0.01,
                                "sharpe_like": total_return / 0.05,
                                "max_drawdown": -0.10 + fold_adjustment / 4,
                                "final_equity": 1.0 + total_return,
                                "total_transaction_cost": 0.001,
                                "mean_daily_turnover": 0.10,
                                "max_daily_turnover": 0.50,
                                "epochs_trained": "",
                            }
                        )
    rows.append({**rows[0], "split": "train", "run_key": "ignored-train-row"})
    rows.append({**rows[0], "model_family": "mlp", "model_type": "mlp", "run_key": "ignored-mlp"})
    return pd.DataFrame(rows)
