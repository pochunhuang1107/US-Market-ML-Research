from __future__ import annotations

from pathlib import Path

import pandas as pd

from us_market_ml.long_history_core_sleeve_decision import (
    DECISION_MATRIX_NAME,
    RECOMMENDATIONS_NAME,
    REPORT_NAME,
    SUMMARY_NAME,
    LongHistoryCoreSleeveDecisionInputs,
    accounting_mode_stability_summary,
    build_long_history_core_sleeve_decision_artifacts,
    build_long_history_core_sleeve_decision_matrix,
    classify_core_sleeve_decision_category,
    read_long_history_core_sleeve_decision_inputs,
    select_long_history_core_sleeve_recommendations,
    write_long_history_core_sleeve_decision_artifacts,
)
from us_market_ml.long_history_core_sleeve_refinement import (
    CANDIDATE_REBALANCE_ALIGNED_WEIGHTS,
    DAILY_REBALANCED_FIXED_WEIGHTS,
    MONTHLY_REBALANCED_WEIGHTS,
    QUARTERLY_REBALANCED_WEIGHTS,
)
from us_market_ml.long_history_core_sleeve_refinement import (
    DECISION_MATRIX_NAME as REFINEMENT_DECISION_MATRIX_NAME,
)
from us_market_ml.long_history_core_sleeve_refinement import (
    RESULTS_NAME as REFINEMENT_RESULTS_NAME,
)
from us_market_ml.long_history_core_sleeve_refinement import (
    ROLLING_WINDOWS_NAME as REFINEMENT_ROLLING_WINDOWS_NAME,
)
from us_market_ml.long_history_core_sleeve_refinement import (
    SUBPERIODS_NAME as REFINEMENT_SUBPERIODS_NAME,
)


def test_loading_refinement_artifacts(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "sweep"
    report_dir = tmp_path / "report"
    sweep_dir.mkdir()
    report_dir.mkdir()
    results = _sample_results()
    subperiods = _sample_subperiods()
    rolling = pd.DataFrame([{"candidate_id": "candidate_0001", "strategy_vs_qqq": 0.1}])

    results.to_csv(sweep_dir / REFINEMENT_RESULTS_NAME, index=False)
    subperiods.to_csv(sweep_dir / REFINEMENT_SUBPERIODS_NAME, index=False)
    rolling.to_csv(sweep_dir / REFINEMENT_ROLLING_WINDOWS_NAME, index=False)
    results.to_csv(sweep_dir / REFINEMENT_DECISION_MATRIX_NAME, index=False)
    results.to_csv(report_dir / "long_history_qqq_core_sleeve_refinement_summary.csv", index=False)
    results.to_csv(
        report_dir / "long_history_qqq_core_sleeve_refinement_decision_matrix.csv",
        index=False,
    )

    loaded = read_long_history_core_sleeve_decision_inputs(
        refinement_sweep_dir=sweep_dir,
        refinement_report_dir=report_dir,
    )

    assert len(loaded.results) == len(results)
    assert len(loaded.subperiods) == len(subperiods)
    assert len(loaded.rolling_windows) == len(rolling)
    assert not loaded.refinement_summary.empty
    assert not loaded.refinement_report_decision_matrix.empty


def test_decision_category_assignment_and_production_ready_false() -> None:
    decision = build_long_history_core_sleeve_decision_matrix(_sample_results())

    assert decision["production_ready"].eq(False).all()
    categories = set(decision["decision_category"])
    assert "raw_return_reference" in categories
    assert "best_core_sleeve_tracking_candidate" in categories
    assert "balanced_core_sleeve_tracking_candidate" in categories
    assert "defensive_reference" in categories
    assert "reject_deprioritize" in categories

    best = decision.loc[
        decision["candidate_id"].eq("candidate_0001")
        & decision["spy_core_weight"].eq(0.35)
        & decision["portfolio_accounting_mode"].eq(MONTHLY_REBALANCED_WEIGHTS)
    ].iloc[0]
    assert best["realistic_accounting_mode_pass_count"] >= 2
    assert classify_core_sleeve_decision_category(best)[0] == (
        "best_core_sleeve_tracking_candidate"
    )


def test_accounting_mode_stability_summary() -> None:
    decision = build_long_history_core_sleeve_decision_matrix(_sample_results())

    summary = accounting_mode_stability_summary(decision)

    assert set(summary["portfolio_accounting_mode"]) >= {
        DAILY_REBALANCED_FIXED_WEIGHTS,
        MONTHLY_REBALANCED_WEIGHTS,
        QUARTERLY_REBALANCED_WEIGHTS,
        CANDIDATE_REBALANCE_ALIGNED_WEIGHTS,
    }
    assert summary.loc[
        summary["portfolio_accounting_mode"].eq(MONTHLY_REBALANCED_WEIGHTS),
        "realistic_accounting_mode",
    ].iloc[0]
    assert not summary.loc[
        summary["portfolio_accounting_mode"].eq(DAILY_REBALANCED_FIXED_WEIGHTS),
        "realistic_accounting_mode",
    ].iloc[0]


def test_recommendation_selection_is_compact_and_research_only() -> None:
    decision = build_long_history_core_sleeve_decision_matrix(_sample_results())

    recommendations = select_long_history_core_sleeve_recommendations(decision)

    tracking = recommendations.loc[
        recommendations["recommendation_action"].isin(["track_next", "rich_replay_next"])
    ]
    assert 2 <= len(tracking) <= 4
    assert set(tracking["candidate_id"]) == {"candidate_0001", "candidate_0003"}
    assert recommendations["production_ready"].eq(False).all()
    assert "broad_sweep" in recommendations["candidate_id"].tolist()


def test_report_writes_required_sections_and_outputs(tmp_path: Path) -> None:
    inputs = LongHistoryCoreSleeveDecisionInputs(
        results=_sample_results(),
        subperiods=_sample_subperiods(),
        rolling_windows=pd.DataFrame(),
        refinement_decision_matrix=_sample_results(),
        refinement_summary=_sample_results(),
        refinement_report_decision_matrix=_sample_results(),
    )
    artifacts = build_long_history_core_sleeve_decision_artifacts(inputs)
    paths = write_long_history_core_sleeve_decision_artifacts(
        artifacts=artifacts,
        output_dir=tmp_path,
    )

    assert paths[SUMMARY_NAME].exists()
    assert paths[DECISION_MATRIX_NAME].exists()
    assert paths[RECOMMENDATIONS_NAME].exists()
    assert paths[REPORT_NAME].exists()
    assert pd.read_csv(paths[DECISION_MATRIX_NAME])["production_ready"].eq(False).all()
    assert pd.read_csv(paths[RECOMMENDATIONS_NAME])["production_ready"].eq(False).all()

    markdown = paths[REPORT_NAME].read_text(encoding="utf-8")
    for section in [
        "## 1. Executive summary",
        "## 2. What changed after realistic accounting",
        "## 3. Standalone candidate comparison",
        "## 4. Candidate_0001 core-sleeve decision",
        "## 5. Candidate_0003 core-sleeve decision",
        "## 6. Accounting-mode stability",
        "## 7. Best portfolios by objective",
        "## 8. Subperiod caveats",
        "## 9. Final recommendation",
        "## 10. Caveats",
    ]:
        assert section in markdown
    assert "Run another broad sweep: `no`" in markdown
    assert "Promote to production: `no`" in markdown
    assert "production_ready` is always false" in markdown
    assert "s3://" not in markdown.lower()
    assert "artifacts/production" not in markdown


def test_no_cloud_model_card_or_production_export_paths_in_script() -> None:
    script = Path("scripts/create_long_history_qqq_core_sleeve_decision_report.sh").read_text(
        encoding="utf-8"
    )

    assert "s3://" not in script.lower()
    assert "gs://" not in script.lower()
    assert "model_card" not in script
    assert "production signal export" in script


def _sample_results() -> pd.DataFrame:
    rows = [
        _row("candidate_0001", 0.0, DAILY_REBALANCED_FIXED_WEIGHTS, 1.60, -0.30, 5.3),
        _row("candidate_0001", 0.0, MONTHLY_REBALANCED_WEIGHTS, 1.60, -0.30, 5.3),
        _row("candidate_0001", 0.35, MONTHLY_REBALANCED_WEIGHTS, 1.28, -0.24, 5.33),
        _row("candidate_0001", 0.35, QUARTERLY_REBALANCED_WEIGHTS, 1.26, -0.245, 5.14),
        _row("candidate_0001", 0.40, MONTHLY_REBALANCED_WEIGHTS, 1.18, -0.23, 5.13),
        _row("candidate_0001", 0.40, CANDIDATE_REBALANCE_ALIGNED_WEIGHTS, 1.20, -0.228, 5.26),
        _row("candidate_0001", 0.60, DAILY_REBALANCED_FIXED_WEIGHTS, 0.95, -0.20, 4.75),
        _row("candidate_0003", 0.0, DAILY_REBALANCED_FIXED_WEIGHTS, 1.35, -0.34, 3.97),
        _row("candidate_0003", 0.25, MONTHLY_REBALANCED_WEIGHTS, 1.18, -0.235, 5.02),
        _row("candidate_0003", 0.25, QUARTERLY_REBALANCED_WEIGHTS, 1.16, -0.236, 4.92),
        _row("candidate_0003", 0.30, MONTHLY_REBALANCED_WEIGHTS, 1.12, -0.225, 4.98),
        _row("candidate_0003", 0.30, CANDIDATE_REBALANCE_ALIGNED_WEIGHTS, 1.14, -0.222, 5.14),
        _row("candidate_0003", 0.40, MONTHLY_REBALANCED_WEIGHTS, 1.03, -0.205, 5.02),
        _row("candidate_0003", 0.40, QUARTERLY_REBALANCED_WEIGHTS, 1.04, -0.204, 5.10),
        _row("candidate_0003", 0.50, DAILY_REBALANCED_FIXED_WEIGHTS, 0.85, -0.205, 4.49),
        _row("candidate_0003", 1.0, DAILY_REBALANCED_FIXED_WEIGHTS, 0.70, -0.22, 3.18),
    ]
    return pd.DataFrame(rows)


def _row(
    candidate_id: str,
    spy_weight: float,
    mode: str,
    total_return: float,
    max_drawdown: float,
    return_to_drawdown: float,
) -> dict[str, object]:
    qqq_return = 1.0
    qqq_drawdown = -0.25
    qqq_rtd = 4.0
    candidate_weight = round(1.0 - spy_weight, 10)
    return {
        "candidate_id": candidate_id,
        "candidate_label": candidate_id,
        "spy_core_weight": spy_weight,
        "candidate_sleeve_weight": candidate_weight,
        "portfolio_accounting_mode": mode,
        "portfolio_name": f"{candidate_id} SPY {spy_weight:.0%} | {mode}",
        "is_core_sleeve": 0.0 < spy_weight < 1.0,
        "total_return": total_return,
        "strategy_vs_qqq": total_return - qqq_return,
        "qqq_return_capture": total_return / qqq_return,
        "max_drawdown": max_drawdown,
        "drawdown_improvement_vs_qqq": max_drawdown - qqq_drawdown,
        "return_to_drawdown": return_to_drawdown,
        "return_to_drawdown_vs_qqq": return_to_drawdown - qqq_rtd,
        "positive_vs_qqq_window_rate": 0.6,
        "worst_strategy_vs_qqq": -0.1,
        "beats_qqq_return": total_return > qqq_return,
        "improves_drawdown_vs_qqq": max_drawdown > qqq_drawdown,
        "improves_return_to_drawdown_vs_qqq": return_to_drawdown > qqq_rtd,
        "captures_90pct_qqq_return": total_return >= 0.9 * qqq_return,
        "production_ready": False,
    }


def _sample_subperiods() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate_id, spy_weight, sleeve_weight, mode in [
        ("candidate_0001", 0.35, 0.65, MONTHLY_REBALANCED_WEIGHTS),
        ("candidate_0003", 0.25, 0.75, MONTHLY_REBALANCED_WEIGHTS),
    ]:
        for period in [
            "covid_crash_recovery",
            "bear_2022",
            "ai_bull_2023_2024",
            "recent_2025_2026",
        ]:
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "spy_core_weight": spy_weight,
                    "candidate_sleeve_weight": sleeve_weight,
                    "portfolio_accounting_mode": mode,
                    "period_name": period,
                    "total_return": 0.2,
                    "qqq_total_return": 0.15,
                    "strategy_vs_qqq": 0.05,
                    "max_drawdown": -0.1,
                    "drawdown_improvement_vs_qqq": 0.02,
                }
            )
    return pd.DataFrame(rows)
