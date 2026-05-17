from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli.create_expanded_dynamic_candidate_confirmation_report import (
    DECISION_MATRIX_COLUMNS,
    SUMMARY_COLUMNS,
    build_expanded_dynamic_candidate_confirmation_report,
    run_expanded_dynamic_candidate_confirmation_report_from_args,
)


def test_expanded_dynamic_report_builds_summary_and_labels(tmp_path: Path) -> None:
    results_csv = tmp_path / "results.csv"
    _synthetic_results().to_csv(results_csv, index=False)

    report_data = build_expanded_dynamic_candidate_confirmation_report(
        results_csv=results_csv,
    )

    assert list(report_data.candidate_summary.columns) == SUMMARY_COLUMNS
    assert list(report_data.decision_matrix.columns) == DECISION_MATRIX_COLUMNS
    assert len(report_data.candidate_summary) == 4
    assert len(report_data.decision_matrix) == 4
    assert report_data.benchmark_columns_available is True

    rows = report_data.candidate_summary.set_index("candidate_id")
    assert rows.loc[
        "linear_regression:top1:equal_weight",
        "role_label",
    ] == "dynamic version of fixed-fold return/reference/champion family"
    assert rows.loc[
        "linear_regression:top3:equal_weight",
        "role_label",
    ] == "champion-family diversified reference"
    assert rows.loc["ridge:top3:equal_weight", "role_label"] == "balanced reference"
    assert rows.loc[
        "ridge:top5:equal_weight",
        "role_label",
    ] == "longer-window balanced candidate"
    assert rows.loc["ridge:top5:equal_weight", "average_test_return"] == pytest.approx(
        0.04325,
    )
    assert rows.loc["ridge:top5:equal_weight", "cost20_minus_cost10"] == pytest.approx(
        -0.006,
    )


def test_expanded_dynamic_report_cli_writes_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_csv = tmp_path / "results.csv"
    output_dir = tmp_path / "report"
    _synthetic_results().to_csv(results_csv, index=False)

    return_code = run_expanded_dynamic_candidate_confirmation_report_from_args(
        [
            "--results-csv",
            str(results_csv),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert return_code == 0
    assert "expanded dynamic candidate confirmation report" in captured.out

    summary_path = output_dir / "expanded_dynamic_candidate_summary.csv"
    decision_path = output_dir / "expanded_dynamic_candidate_decision_matrix.csv"
    markdown_path = output_dir / "expanded_dynamic_candidate_report.md"
    assert summary_path.exists()
    assert decision_path.exists()
    assert markdown_path.exists()

    summary = pd.read_csv(summary_path)
    matrix = pd.read_csv(decision_path)
    markdown = markdown_path.read_text(encoding="utf-8")

    assert len(summary) == 4
    assert len(matrix) == 4
    assert "Executive Summary" in markdown
    assert "Candidate Ranking Table" in markdown
    assert "Decision Matrix" in markdown
    assert "Window Stability" in markdown
    assert "Cost Sensitivity" in markdown
    assert "Drawdown" in markdown
    assert "Recommendation" in markdown
    assert (
        "does not fully solve historical ETF survivorship bias"
        in markdown
    )
    assert "longer-window balanced candidate" in markdown


def _synthetic_results() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    candidates = [
        ("linear_regression", 1, 0.020, -0.16),
        ("linear_regression", 3, 0.016, -0.14),
        ("ridge", 3, 0.026, -0.13),
        ("ridge", 5, 0.040, -0.12),
    ]
    folds = [
        ("tw3y_v6m_t3m_q01_20200101_20200331", -0.010),
        ("tw3y_v6m_t6m_q02_20200401_20200930", 0.005),
        ("tw5y_v12m_t3m_q01_20210101_20210331", 0.010),
        ("tw5y_v12m_t6m_q02_20210401_20210930", 0.020),
    ]
    for model_type, top_k, base_return, base_drawdown in candidates:
        for fold_name, fold_adjustment in folds:
            for cost in (10.0, 20.0):
                cost_penalty = 0.0 if cost == 10.0 else 0.006
                total_return = base_return + fold_adjustment - cost_penalty
                rows.append(
                    {
                        "run_key": f"{model_type}-{top_k}-{fold_name}-{cost:g}",
                        "universe": "expanded_dynamic",
                        "data_path": "/tmp/local.csv",
                        "fold": fold_name,
                        "split": "test",
                        "model_family": "baseline",
                        "model_type": model_type,
                        "feature_set_version": "default_features_v1",
                        "feature_count": 12,
                        "hidden_units": "",
                        "dropout": "",
                        "scale_target": False,
                        "learning_rate": "",
                        "weight_decay": "",
                        "batch_size": "",
                        "max_epochs": "",
                        "patience": "",
                        "random_state": 42,
                        "top_k": top_k,
                        "min_holding_days": 10,
                        "switch_margin": 0.002,
                        "allocation_mode": "equal_weight",
                        "score_temperature": 1.0,
                        "max_weight": 1.0,
                        "transaction_cost_bps": cost,
                        "target_col": "excess_return_vs_spy_20d",
                        "label_horizon_days": 20,
                        "total_return": total_return,
                        "mean_daily_return": total_return / 63,
                        "daily_volatility": 0.01,
                        "sharpe_like": total_return / 0.10,
                        "max_drawdown": base_drawdown + fold_adjustment / 2,
                        "final_equity": 1.0 + total_return,
                        "total_transaction_cost": 0.001,
                        "mean_daily_turnover": 0.10,
                        "max_daily_turnover": 0.50,
                        "epochs_trained": "",
                        "spy_total_return": 0.015 + fold_adjustment / 2,
                        "qqq_total_return": 0.020 + fold_adjustment / 2,
                    }
                )
    rows.append({**rows[0], "split": "train", "run_key": "ignored-train-row"})
    return pd.DataFrame(rows)
