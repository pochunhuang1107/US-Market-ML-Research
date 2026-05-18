from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from us_market_ml.cli.create_tiingo_core_linear_deep_dive_bundles import (
    DECISION_MATRIX_COLUMNS,
    SUMMARY_COLUMNS,
    build_tiingo_core_linear_deep_dive_bundles,
    run_tiingo_core_linear_deep_dive_bundles_from_args,
)


def test_tiingo_core_linear_deep_dive_builds_shortlist_at_10bps(
    tmp_path: Path,
) -> None:
    results_csv = tmp_path / "core_linear_results.csv"
    _synthetic_results().to_csv(results_csv, index=False)

    report = build_tiingo_core_linear_deep_dive_bundles(
        results_csv=results_csv,
        summary_csv=None,
        decision_matrix_csv=None,
    )

    assert list(report.candidate_summary.columns) == SUMMARY_COLUMNS
    assert list(report.decision_matrix.columns) == DECISION_MATRIX_COLUMNS
    assert len(report.candidate_summary) == 8
    assert len(report.bundles) == 8
    assert set(report.candidate_summary["transaction_cost_bps"]) == {10.0}
    assert set(report.candidate_summary["decision_anchor_bps"]) == {10.0}

    labels = set(report.candidate_summary["candidate_label"])
    assert (
        "ridge / top1 / hold 20 / margin 0.005 / cost 10bps / tw5y-v12m-t12m"
        in labels
    )
    assert (
        "linear_regression / top1 / hold 20 / margin 0.002 / cost 10bps / "
        "tw5y-v12m-t12m"
        in labels
    )
    assert (
        "ridge / top5 / hold 10 / margin 0 / cost 10bps / tw5y-v12m-t12m"
        in labels
    )
    assert (
        "linear_regression / top5 / hold 10 / margin 0.005 / cost 10bps / "
        "tw3y-v12m-t3m"
        in labels
    )
    assert (
        "ridge / top1 / hold 20 / margin 0.005 / cost 10bps / tw5y-v12m-t3m"
        in labels
    )

    recommendation_labels = set(report.candidate_summary["recommendation_label"])
    assert "keep as return reference" in recommendation_labels
    assert "keep as balanced reference" in recommendation_labels
    assert "keep as drawdown-control reference" in recommendation_labels

    first_bundle = report.bundles[0]
    assert set(first_bundle.cost_sensitivity["transaction_cost_bps"]) == {5.0, 10.0, 20.0}
    assert first_bundle.cost_sensitivity["is_anchor_cost"].sum() == 1
    assert first_bundle.fold_results["total_return"].max() < 99.0


def test_tiingo_core_linear_deep_dive_cli_writes_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_csv = tmp_path / "core_linear_results.csv"
    summary_csv = tmp_path / "missing_summary.csv"
    decision_csv = tmp_path / "missing_decision.csv"
    output_dir = tmp_path / "deep_dive"
    _synthetic_results().to_csv(results_csv, index=False)

    return_code = run_tiingo_core_linear_deep_dive_bundles_from_args(
        [
            "--results-csv",
            str(results_csv),
            "--summary-csv",
            str(summary_csv),
            "--decision-matrix-csv",
            str(decision_csv),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    summary_path = output_dir / "tiingo_core_linear_deep_dive_summary.csv"
    decision_path = output_dir / "tiingo_core_linear_deep_dive_decision_matrix.csv"
    markdown_path = output_dir / "tiingo_core_linear_deep_dive_report.md"

    assert return_code == 0
    assert output_dir.exists()
    assert "Tiingo core-linear deep-dive bundles" in captured.out
    assert "decision anchor: 10bps" in captured.out
    assert summary_path.exists()
    assert decision_path.exists()
    assert markdown_path.exists()

    summary = pd.read_csv(summary_path)
    decision = pd.read_csv(decision_path)
    markdown = markdown_path.read_text(encoding="utf-8")

    assert len(summary) == 8
    assert len(decision) == 8
    assert set(summary["transaction_cost_bps"]) == {10.0}
    assert "ridge / top1 / hold 20 / margin 0.005 / cost 10bps" in markdown
    assert "linear_regression / top5 / hold 10 / margin 0.005 / cost 10bps" in markdown
    assert "Top1 vs top3 vs top5 comparison" in markdown
    assert "keep as balanced reference" in markdown
    assert "Run `tree_screen` only after" in markdown
    assert "Do not run MLP yet" in markdown
    assert "Tiingo-only research source" in markdown
    assert "Massive app data may not match research results exactly" in markdown
    assert "ETF survivorship bias" in markdown
    assert "Candidates are not production-ready" in markdown

    candidate_dirs = [path for path in output_dir.iterdir() if path.is_dir()]
    assert len(candidate_dirs) == 8
    first_dir = candidate_dirs[0]
    assert any(path.name.endswith("_candidate_summary.csv") for path in first_dir.iterdir())
    assert any(path.name.endswith("_fold_test_results.csv") for path in first_dir.iterdir())
    assert any(path.name.endswith("_window_metadata.csv") for path in first_dir.iterdir())
    assert any(path.name.endswith("_cost_sensitivity.csv") for path in first_dir.iterdir())
    assert any(path.name.endswith("_smoothing_sensitivity.csv") for path in first_dir.iterdir())
    assert any(path.name.endswith("_window_stability.csv") for path in first_dir.iterdir())
    assert any(path.name.endswith("_report.md") for path in first_dir.iterdir())


def _synthetic_results() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specs = [
        ("ridge", 1, 20, 0.005, 5, 12, 12, [0.20, 0.22, 0.18, 0.25, 0.19], -0.29, 0.09),
        (
            "linear_regression",
            1,
            20,
            0.002,
            5,
            12,
            12,
            [0.18, 0.21, 0.20, 0.17, -0.01],
            -0.30,
            0.08,
        ),
        ("ridge", 3, 10, 0.0, 5, 12, 12, [0.17, 0.20, 0.19, 0.21, 0.16], -0.25, 0.10),
        ("ridge", 5, 10, 0.0, 5, 12, 12, [0.15, 0.18, 0.16, 0.19, 0.14], -0.21, 0.11),
        ("ridge", 5, 20, 0.005, 5, 12, 12, [0.16, 0.19, 0.17, 0.20, 0.15], -0.20, 0.10),
        (
            "linear_regression",
            5,
            10,
            0.005,
            3,
            12,
            3,
            [0.04, 0.05, 0.03, 0.06, -0.01, 0.04],
            -0.11,
            0.12,
        ),
        ("ridge", 5, 5, 0.002, 3, 12, 3, [0.03, 0.05, 0.04, 0.06, 0.02, 0.04], -0.13, 0.13),
        ("ridge", 1, 20, 0.005, 5, 12, 3, [0.08, 0.10, 0.09, 0.11, 0.07], -0.27, 0.30),
    ]
    for spec in specs:
        for cost in (5.0, 10.0, 20.0):
            _append_candidate_rows(rows, spec=spec, cost=cost)

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


def _append_candidate_rows(
    rows: list[dict[str, object]],
    *,
    spec: tuple[str, int, int, float, int, int, int, list[float], float, float],
    cost: float,
) -> None:
    (
        model_type,
        top_k,
        min_holding_days,
        switch_margin,
        train_years,
        validation_months,
        test_months,
        returns,
        worst_drawdown,
        sharpe_base,
    ) = spec
    for index, base_return in enumerate(returns, start=1):
        cost_adjustment = (10.0 - cost) * 0.001
        total_return = base_return + cost_adjustment
        fold = _fold_name(train_years, validation_months, test_months, index)
        rows.append(
            {
                "run_key": (
                    f"{model_type}-{top_k}-{min_holding_days}-"
                    f"{switch_margin:g}-{cost:g}-{fold}"
                ),
                "universe": "tiingo_modern_dynamic",
                "data_path": "/tmp/tiingo.parquet",
                "sweep_profile": "core_linear",
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
                "sharpe_like": sharpe_base + total_return / 10,
                "max_drawdown": worst_drawdown + (index % 3) * 0.01,
                "final_equity": 1.0 + total_return,
                "total_transaction_cost": cost / 10_000,
                "mean_daily_turnover": 0.10 + top_k / 100,
                "max_daily_turnover": 0.50,
                "epochs_trained": "",
            }
        )


def _fold_name(
    train_years: int,
    validation_months: int,
    test_months: int,
    index: int,
) -> str:
    start = pd.Timestamp("2020-01-01") + pd.DateOffset(months=(index - 1) * test_months)
    end = start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
    return (
        f"tw{train_years}y_v{validation_months}m_t{test_months}m_q{index:02d}_"
        f"{start:%Y%m%d}_{end:%Y%m%d}"
    )
