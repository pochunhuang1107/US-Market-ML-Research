from __future__ import annotations

from pathlib import Path

import pandas as pd

from us_market_ml.cli.create_tiingo_v3_allocation_overlay_report import (
    build_tiingo_v3_allocation_overlay_report,
    build_tiingo_v3_allocation_overlay_decision_matrix,
    summarize_tiingo_v3_allocation_overlay_results,
    write_tiingo_v3_allocation_overlay_report,
)
from us_market_ml.cli.run_tiingo_v3_allocation_overlay_study import RESULT_COLUMNS


def test_allocation_overlay_report_summarizes_results_and_identifies_best_overlay(
    tmp_path: Path,
) -> None:
    results_csv = tmp_path / "results.csv"
    _synthetic_results().to_csv(results_csv, index=False)

    report_data = build_tiingo_v3_allocation_overlay_report(
        results_csv=results_csv,
        market_dataset_glob="/tmp/tiingo/**/*.parquet",
    )

    assert set(report_data.summary["overlay_name"]) == {
        "equal_weight_top5_baseline",
        "qqq_core_20_satellite_top5",
        "rank_weighted_top5",
    }
    assert report_data.best_overlay_name == "qqq_core_20_satellite_top5"
    best = report_data.decision_matrix.loc[
        report_data.decision_matrix["overlay_name"].eq("qqq_core_20_satellite_top5")
    ].iloc[0]
    rejected = report_data.decision_matrix.loc[
        report_data.decision_matrix["overlay_name"].eq("rank_weighted_top5")
    ].iloc[0]

    assert best["decision"] == "keep_for_rich_replay"
    assert best["delta_vs_equal_weight_strategy_vs_qqq"] > 0
    assert rejected["decision"] == "reject_no_qqq_gap_improvement"


def test_allocation_overlay_report_writes_required_files(tmp_path: Path) -> None:
    results_csv = tmp_path / "results.csv"
    output_dir = tmp_path / "report"
    _synthetic_results().to_csv(results_csv, index=False)
    report_data = build_tiingo_v3_allocation_overlay_report(
        results_csv=results_csv,
        market_dataset_glob="/tmp/tiingo/**/*.parquet",
    )

    outputs = write_tiingo_v3_allocation_overlay_report(
        report_data=report_data,
        output_dir=output_dir,
    )

    assert set(outputs) == {
        "tiingo_v3_allocation_overlay_summary.csv",
        "tiingo_v3_allocation_overlay_decision_matrix.csv",
        "tiingo_v3_allocation_overlay_report.md",
    }
    summary = pd.read_csv(output_dir / "tiingo_v3_allocation_overlay_summary.csv")
    decision = pd.read_csv(output_dir / "tiingo_v3_allocation_overlay_decision_matrix.csv")
    markdown = (output_dir / "tiingo_v3_allocation_overlay_report.md").read_text(
        encoding="utf-8"
    )

    assert not summary.empty
    assert not decision.empty
    assert "Executive summary" in markdown
    assert "Current v3 baseline" in markdown
    assert "QQQ gap reduction" in markdown
    assert "Rich replay best overlay" in markdown
    assert "qqq_core_20_satellite_top5" in markdown
    assert "Do not mark production-ready" in markdown
    assert "Tiingo-only and research-only" in markdown


def test_allocation_overlay_summary_and_decision_matrix_handle_empty_results() -> None:
    empty = pd.DataFrame(columns=RESULT_COLUMNS)

    summary = summarize_tiingo_v3_allocation_overlay_results(empty)
    decision = build_tiingo_v3_allocation_overlay_decision_matrix(summary)

    assert summary.empty
    assert decision.empty


def _synthetic_results() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    fold_returns = {
        "equal_weight_top5_baseline": (0.05, 0.06),
        "qqq_core_20_satellite_top5": (0.085, 0.095),
        "rank_weighted_top5": (0.04, 0.045),
    }
    for overlay_name, strategy_returns in fold_returns.items():
        for fold_index, strategy_return in enumerate(strategy_returns, start=1):
            spy_return = 0.04 + fold_index * 0.005
            qqq_return = 0.10 + fold_index * 0.01
            drawdown = -0.09 if overlay_name != "rank_weighted_top5" else -0.12
            rows.append(
                {
                    "run_key": f"{overlay_name}_{fold_index}",
                    "dataset_source": "tiingo_only",
                    "study_stage": "allocation_risk_overlay",
                    "run_name": "tiingo_v3_allocation_overlay_study",
                    "overlay_name": overlay_name,
                    "overlay_kind": "synthetic",
                    "fold": f"fold_{fold_index}",
                    "train_window_years": 3,
                    "validation_window_months": 12,
                    "test_window_months": 3,
                    "train_start": "2020-01-01",
                    "train_end": "2022-12-31",
                    "validation_start": "2023-01-01",
                    "validation_end": "2023-12-31",
                    "test_start": "2024-01-01",
                    "test_end": "2024-03-31",
                    "feature_set_version": "benchmark_gap_features_v3",
                    "model_family": "baseline",
                    "model_type": "ridge",
                    "top_k": 5,
                    "allocation_mode": "equal_weight",
                    "min_holding_days": 20,
                    "switch_margin": 0.005,
                    "transaction_cost_bps": 10.0,
                    "target_col": "excess_return_vs_spy_20d",
                    "label_horizon_days": 20,
                    "feature_count": 30,
                    "strategy_total_return": strategy_return,
                    "spy_total_return": spy_return,
                    "qqq_total_return": qqq_return,
                    "strategy_vs_spy": strategy_return - spy_return,
                    "strategy_vs_qqq": strategy_return - qqq_return,
                    "strategy_sharpe_like": 0.10 + strategy_return,
                    "strategy_max_drawdown": drawdown,
                    "total_transaction_cost": 0.01,
                    "mean_daily_turnover": 0.20,
                    "max_daily_turnover": 1.0,
                    "selected_symbol_count_mean": 5.0,
                    "selected_weight_sum_min": 1.0,
                    "selected_weight_sum_max": 1.0,
                    "overlay_weight_valid": True,
                    "spy_sharpe_like": 0.10,
                    "qqq_sharpe_like": 0.12,
                    "spy_max_drawdown": -0.08,
                    "qqq_max_drawdown": -0.13,
                }
            )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)
