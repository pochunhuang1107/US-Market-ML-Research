from __future__ import annotations

from pathlib import Path

import pandas as pd

from us_market_ml.cli.create_tiingo_focused_linear_ridge_confirmation_report import (
    DECISION_MATRIX_COLUMNS,
    SUMMARY_COLUMNS,
    build_tiingo_focused_linear_ridge_confirmation_report,
    run_tiingo_focused_linear_ridge_confirmation_report_from_args,
)
from us_market_ml.cli.run_tiingo_focused_linear_ridge_confirmation import (
    EXPLICIT_CANDIDATE_SPECS,
    SWEEP_PROFILE,
)


def test_focused_confirmation_report_builds_summary_and_decision_matrix(
    tmp_path: Path,
) -> None:
    results_csv = tmp_path / "focused_results.csv"
    _synthetic_results().to_csv(results_csv, index=False)

    report = build_tiingo_focused_linear_ridge_confirmation_report(
        results_csv=results_csv,
    )

    assert list(report.candidate_summary.columns) == SUMMARY_COLUMNS
    assert list(report.decision_matrix.columns) == DECISION_MATRIX_COLUMNS
    assert report.dataset_source == "tiingo_only"
    assert set(report.candidate_summary["dataset_source"]) == {"tiingo_only"}
    assert set(report.candidate_summary["sweep_profile"]) == {SWEEP_PROFILE}
    assert set(report.decision_matrix["candidate_role"]) == {
        "balanced_reference",
        "drawdown_control_reference",
        "return_reference",
    }
    assert len(report.decision_matrix) == len(EXPLICIT_CANDIDATE_SPECS)
    assert set(report.cost_sensitivity["transaction_cost_bps"]) == {10.0, 20.0}
    assert "mlp" not in set(report.decision_matrix["model_type"])

    recommendations = set(report.decision_matrix["recommendation"])
    assert "keep as return reference" in recommendations
    assert "keep as drawdown-control guardrail" in recommendations
    assert recommendations.intersection(
        {"candidate to deep-dive further", "keep as balanced candidate"}
    )


def test_focused_confirmation_report_cli_writes_outputs(
    tmp_path: Path,
    capsys,
) -> None:
    results_csv = tmp_path / "focused_results.csv"
    output_dir = tmp_path / "report"
    _synthetic_results().to_csv(results_csv, index=False)

    return_code = run_tiingo_focused_linear_ridge_confirmation_report_from_args(
        [
            "--results-csv",
            str(results_csv),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    summary_path = output_dir / "tiingo_focused_linear_ridge_confirmation_summary.csv"
    decision_path = (
        output_dir / "tiingo_focused_linear_ridge_confirmation_decision_matrix.csv"
    )
    markdown_path = output_dir / "tiingo_focused_linear_ridge_confirmation_report.md"

    assert return_code == 0
    assert "Tiingo focused linear/ridge confirmation report" in captured.out
    assert "dataset source: tiingo_only" in captured.out
    assert summary_path.exists()
    assert decision_path.exists()
    assert markdown_path.exists()

    summary = pd.read_csv(summary_path)
    matrix = pd.read_csv(decision_path)
    markdown = markdown_path.read_text(encoding="utf-8")

    assert set(summary["candidate_role"]) == {
        "balanced_reference",
        "drawdown_control_reference",
        "return_reference",
    }
    assert len(matrix) == len(EXPLICIT_CANDIDATE_SPECS)
    assert "Executive Summary" in markdown
    assert "Candidate Families Tested" in markdown
    assert "Return References" in markdown
    assert "Balanced Top3/Top5 Candidates" in markdown
    assert "Drawdown-Control Candidates" in markdown
    assert "Cost Sensitivity 10bps vs 20bps" in markdown
    assert "Test-Window Stability 3m vs 6m vs 12m" in markdown
    assert "Train-Window Stability 3y vs 5y" in markdown
    assert "Decision Matrix" in markdown
    assert "Recommendation" in markdown
    assert "Candidate to deep-dive further" in markdown
    assert "Candidate to keep as return reference" in markdown
    assert "Candidate to keep as drawdown-control guardrail" in markdown
    assert "Whether `tree_screen` should run next" in markdown
    assert "Whether MLP should still wait" in markdown
    assert "Tiingo-only source" in markdown
    assert "Massive app data may not match" in markdown
    assert "ETF survivorship bias is not fully eliminated" in markdown
    assert "Not production-ready" in markdown


def _synthetic_results() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec_index, spec in enumerate(EXPLICIT_CANDIDATE_SPECS, start=1):
        for train_years in (3, 5):
            for test_months in (3, 6, 12):
                for cost in (10.0, 20.0):
                    base_return = _base_return(spec.candidate_role, spec.top_k)
                    train_bonus = 0.010 if train_years == 5 else 0.0
                    test_bonus = {3: 0.000, 6: 0.010, 12: 0.020}[test_months]
                    cost_penalty = 0.000 if cost == 10.0 else 0.018
                    total_return = base_return + train_bonus + test_bonus - cost_penalty
                    fold = _fold_name(train_years, 12, test_months, spec_index)
                    rows.append(
                        {
                            "run_key": (
                                f"{spec.model_type}-{spec.top_k}-{spec.min_holding_days}-"
                                f"{spec.switch_margin:g}-{cost:g}-{fold}"
                            ),
                            "universe": "tiingo_focused_linear_ridge_dynamic",
                            "data_path": "/tmp/tiingo.csv",
                            "sweep_profile": SWEEP_PROFILE,
                            "fold": fold,
                            "split": "test",
                            "model_family": "baseline",
                            "model_type": spec.model_type,
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
                            "top_k": spec.top_k,
                            "min_holding_days": spec.min_holding_days,
                            "switch_margin": spec.switch_margin,
                            "allocation_mode": "equal_weight",
                            "score_temperature": 1.0,
                            "max_weight": 1.0,
                            "transaction_cost_bps": cost,
                            "target_col": "excess_return_vs_spy_20d",
                            "label_horizon_days": 20,
                            "total_return": total_return,
                            "mean_daily_return": total_return / 63,
                            "daily_volatility": 0.01,
                            "sharpe_like": 0.2 + total_return,
                            "max_drawdown": _drawdown(spec.candidate_role, spec.top_k)
                            + test_months / 1000,
                            "final_equity": 1.0 + total_return,
                            "total_transaction_cost": cost / 10_000,
                            "mean_daily_turnover": 0.04 + spec.top_k / 100,
                            "max_daily_turnover": 0.50,
                            "epochs_trained": "",
                        }
                    )

    rows.append(
        {
            **rows[0],
            "split": "train",
            "total_return": 99.0,
            "run_key": "ignored-train-row",
        }
    )
    rows.append(
        {
            **rows[0],
            "model_family": "mlp",
            "model_type": "mlp",
            "run_key": "ignored-mlp-row",
        }
    )
    return pd.DataFrame(rows)


def _base_return(role: str, top_k: int) -> float:
    if role == "return_reference":
        return 0.080
    if role == "balanced_reference":
        return 0.065 + top_k / 1000
    return 0.035


def _drawdown(role: str, top_k: int) -> float:
    if role == "drawdown_control_reference":
        return -0.105
    if role == "balanced_reference":
        return -0.205 + top_k / 100
    return -0.285


def _fold_name(
    train_years: int,
    validation_months: int,
    test_months: int,
    index: int,
) -> str:
    start = pd.Timestamp("2023-01-01") + pd.DateOffset(months=(index - 1) % 3)
    end = start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
    return (
        f"tw{train_years}y_v{validation_months}m_t{test_months}m_q{index:02d}_"
        f"{start:%Y%m%d}_{end:%Y%m%d}"
    )
