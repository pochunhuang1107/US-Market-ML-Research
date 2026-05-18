from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli.create_tiingo_selected_candidate_evaluation_bundle import (
    COST_SENSITIVITY_COLUMNS,
    DECISION_MATRIX_COLUMNS,
    SUMMARY_COLUMNS,
    WINDOW_STABILITY_COLUMNS,
    build_tiingo_selected_candidate_evaluation_bundle,
    run_tiingo_selected_candidate_evaluation_bundle_from_args,
)


def test_selected_candidate_evaluation_builds_expected_tables(tmp_path: Path) -> None:
    paths = _write_synthetic_inputs(tmp_path)

    report = build_tiingo_selected_candidate_evaluation_bundle(
        results_csv=paths["results"],
        summary_csv=paths["summary"],
        decision_matrix_csv=paths["decision"],
    )

    assert list(report.candidate_summary.columns) == SUMMARY_COLUMNS
    assert list(report.decision_matrix.columns) == DECISION_MATRIX_COLUMNS
    assert list(report.window_stability.columns) == WINDOW_STABILITY_COLUMNS
    assert list(report.cost_sensitivity.columns) == COST_SENSITIVITY_COLUMNS
    assert report.dataset_source == "tiingo_only"
    assert set(report.candidate_summary["candidate_role"]) == {
        "primary_balanced_candidate",
        "return_reference",
        "drawdown_guardrail",
    }
    assert set(report.candidate_summary["decision_anchor_bps"]) == {10.0}
    assert set(report.cost_sensitivity["transaction_cost_bps"]) == {10.0, 20.0}
    assert set(report.window_stability["stability_dimension"]) == {
        "train_window_years",
        "test_window_months",
        "validation_window_months",
    }
    assert set(report.comparison["comparison_name"]) == {
        "primary_vs_return_reference",
        "primary_vs_drawdown_guardrail",
    }

    primary = report.candidate_summary.set_index("candidate_role").loc[
        "primary_balanced_candidate"
    ]
    assert primary["candidate_label"] == "ridge / top5 / hold 20 / margin 0.005"
    assert primary["average_test_return_10bps"] == pytest.approx(0.093)
    assert primary["positive_window_rate_20bps"] == pytest.approx(1.0)
    assert primary["cost_delta_20_vs_10"] == pytest.approx(-0.008)
    assert primary["recommendation"] == "primary research baseline candidate"

    decision = report.decision_matrix.set_index("candidate_role").loc[
        "primary_balanced_candidate"
    ]
    assert decision["average_test_return_gate"] == "pass"
    assert decision["positive_window_gate"] == "pass"
    assert decision["cost_sensitivity_gate"] == "pass"
    assert decision["drawdown_vs_top1_gate"] == "pass"
    assert decision["evidence_scope_gate"] == "pass"
    assert bool(decision["promote_primary_candidate"])


def test_selected_candidate_evaluation_cli_writes_outputs_and_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_synthetic_inputs(tmp_path)
    output_dir = tmp_path / "bundle"

    return_code = run_tiingo_selected_candidate_evaluation_bundle_from_args(
        [
            "--results-csv",
            str(paths["results"]),
            "--summary-csv",
            str(paths["summary"]),
            "--decision-matrix-csv",
            str(paths["decision"]),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    required_outputs = [
        "selected_candidate_summary.csv",
        "selected_candidate_decision_matrix.csv",
        "selected_candidate_window_stability.csv",
        "selected_candidate_cost_sensitivity.csv",
        "selected_candidate_comparison.csv",
        "selected_candidate_evaluation_report.md",
    ]

    assert return_code == 0
    assert output_dir.exists()
    assert "Tiingo selected-candidate evaluation bundle" in captured.out
    assert "decision anchor: 10bps" in captured.out
    assert "20bps sensitivity: 20bps" in captured.out
    for filename in required_outputs:
        assert (output_dir / filename).exists()

    assert (output_dir / "selected_candidate_model_card_draft.md").exists()

    markdown = (output_dir / "selected_candidate_evaluation_report.md").read_text(
        encoding="utf-8"
    )
    summary = pd.read_csv(output_dir / "selected_candidate_summary.csv")
    cost = pd.read_csv(output_dir / "selected_candidate_cost_sensitivity.csv")

    assert "ridge / top5 / hold 20 / margin 0.005" in markdown
    assert "ridge / top1 / hold 20 / margin 0.005" in markdown
    assert "ridge / top5 / hold 5 / margin 0.002" in markdown
    assert "The decision anchor is `10bps`" in markdown
    assert "`20bps` is included as cost sensitivity" in markdown
    assert "average_test_return_10bps" in markdown
    assert "average_test_return_20bps" in markdown
    assert "Primary vs Return Reference" in markdown
    assert "Primary vs Drawdown Guardrail" in markdown
    assert "Window Stability" in markdown
    assert "Cost Sensitivity" in markdown
    assert "Test-window stability is still weak" in markdown
    assert "ETF survivorship bias is not fully eliminated" in markdown
    assert "Tiingo-only results may not match Massive app data" in markdown
    assert "This is not production-ready" in markdown
    assert "Do not run `tree_screen` immediately" in markdown
    assert "Do not run MLP immediately" in markdown
    assert "primary research baseline candidate" in markdown
    assert "return reference" in markdown
    assert "drawdown guardrail" in markdown
    assert "Run `tree_screen`" not in markdown
    assert "Run MLP" not in markdown
    assert set(summary["candidate_role"]) == {
        "primary_balanced_candidate",
        "return_reference",
        "drawdown_guardrail",
    }
    assert set(cost["transaction_cost_bps"]) == {10.0, 20.0}


def _write_synthetic_inputs(tmp_path: Path) -> dict[str, Path]:
    results = tmp_path / "focused_results.csv"
    summary = tmp_path / "focused_summary.csv"
    decision = tmp_path / "focused_decision_matrix.csv"

    _synthetic_results().to_csv(results, index=False)
    _synthetic_source_summary().to_csv(summary, index=False)
    _synthetic_source_decision_matrix().to_csv(decision, index=False)
    return {"results": results, "summary": summary, "decision": decision}


def _synthetic_results() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specs = [
        (
            "ridge",
            5,
            20,
            0.005,
            "primary",
            0.09,
            -0.11,
            -0.19,
            0.004,
            0.035,
            0.008,
        ),
        (
            "ridge",
            1,
            20,
            0.005,
            "return",
            0.10,
            -0.13,
            -0.30,
            0.006,
            0.050,
            0.010,
        ),
        (
            "ridge",
            5,
            5,
            0.002,
            "guardrail",
            0.08,
            -0.10,
            -0.22,
            0.018,
            0.130,
            0.022,
        ),
    ]
    for spec in specs:
        for cost in (5.0, 10.0, 20.0):
            _append_candidate_rows(rows, spec=spec, cost=cost)
    rows.append({**rows[0], "split": "train", "run_key": "ignored-train-row"})
    rows.append(
        {
            **rows[0],
            "model_family": "mlp",
            "model_type": "mlp",
            "run_key": "ignored-mlp-row",
        }
    )
    return pd.DataFrame(rows)


def _append_candidate_rows(
    rows: list[dict[str, object]],
    *,
    spec: tuple[str, int, int, float, str, float, float, float, float, float, float],
    cost: float,
) -> None:
    (
        model_type,
        top_k,
        min_holding_days,
        switch_margin,
        role,
        base_return,
        average_drawdown,
        worst_drawdown,
        transaction_cost,
        turnover,
        cost_decay,
    ) = spec
    index = 0
    for train_years in (3, 5):
        for test_months in (3, 6, 12):
            for _sequence in (1, 2):
                index += 1
                fold = _fold_name(train_years, 12, test_months, index)
                cost_adjustment = (10.0 - cost) * cost_decay / 10.0
                total_return = base_return + (index % 4) * 0.002 + cost_adjustment
                max_drawdown = average_drawdown + (index % 3) * 0.005
                if index == 12:
                    max_drawdown = worst_drawdown
                rows.append(
                    {
                        "run_key": (
                            f"{role}-{model_type}-{top_k}-{min_holding_days}-"
                            f"{switch_margin:g}-{cost:g}-{fold}"
                        ),
                        "universe": "tiingo_focused_linear_ridge_dynamic",
                        "data_path": "/tmp/tiingo.parquet",
                        "sweep_profile": "focused_linear_ridge_confirmation",
                        "fold": fold,
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
                        "sharpe_like": 0.05 + total_return / 10,
                        "max_drawdown": max_drawdown,
                        "final_equity": 1.0 + total_return,
                        "total_transaction_cost": transaction_cost * cost / 10.0,
                        "mean_daily_turnover": turnover,
                        "max_daily_turnover": 0.50,
                        "epochs_trained": "",
                    }
                )


def _synthetic_source_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _source_row("ridge", 5, 20, 0.005, "candidate to deep-dive further"),
            _source_row("ridge", 1, 20, 0.005, "keep as return reference"),
            _source_row("ridge", 5, 5, 0.002, "keep as drawdown-control guardrail"),
        ]
    )


def _synthetic_source_decision_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _source_row("ridge", 5, 20, 0.005, "keep as balanced candidate"),
            _source_row("ridge", 1, 20, 0.005, "keep as return reference"),
            _source_row("ridge", 5, 5, 0.002, "keep as drawdown-control guardrail"),
        ]
    ).assign(decision_anchor_bps=10.0, rationale="synthetic source rationale")


def _source_row(
    model_type: str,
    top_k: int,
    min_holding_days: int,
    switch_margin: float,
    recommendation: str,
) -> dict[str, object]:
    return {
        "model_type": model_type,
        "top_k": top_k,
        "min_holding_days": min_holding_days,
        "switch_margin": switch_margin,
        "transaction_cost_bps": 10.0,
        "recommendation": recommendation,
    }


def _fold_name(
    train_years: int,
    validation_months: int,
    test_months: int,
    index: int,
) -> str:
    start = pd.Timestamp("2021-01-01") + pd.DateOffset(months=index)
    end = start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
    return (
        f"tw{train_years}y_v{validation_months}m_t{test_months}m_q{index:02d}_"
        f"{start:%Y%m%d}_{end:%Y%m%d}"
    )
