from __future__ import annotations

from pathlib import Path

import pandas as pd

from test_qqq_only_best_candidate_deep_dive import _synthetic_inputs

from us_market_ml.qqq_only_rf_signal_review import (
    CANDIDATE_COLUMNS,
    CASH_TIMING_COLUMNS,
    DECISION_COLUMNS,
    DRAWDOWN_AVOIDANCE_COLUMNS,
    MISSED_RALLY_COLUMNS,
    REPORT_SECTIONS,
    SEGMENT_COLUMNS,
    SUMMARY_COLUMNS,
    build_qqq_only_rf_signal_review_artifacts,
    output_paths,
    select_signal_review_candidate_set,
    write_qqq_only_rf_signal_review_artifacts,
)


def test_select_signal_review_candidate_set_is_narrow_and_research_only() -> None:
    results, _, _, _, summary, decision = _signal_review_inputs()

    selected = select_signal_review_candidate_set(summary, decision_matrix=decision)

    assert set(CANDIDATE_COLUMNS).issubset(selected.columns)
    assert selected["production_ready"].eq(False).all()
    assert selected["candidate_role"].iloc[0] == "best_setting"
    assert "benchmark" in set(selected["candidate_role"])
    assert "smoother_reference" in set(selected["candidate_role"])
    assert "threshold_050_reference" in set(selected["candidate_role"])
    assert set(selected["model_type"].dropna().unique()).issubset({"random_forest", "baseline"})
    assert set(selected["target_name"].dropna().unique()).issubset({"risk_on_20d", "none"})
    assert "keep_for_second_confirmation" in set(selected["decision_label"])
    assert "keep_for_paper_tracking_candidate_review" in set(selected["decision_label"])
    assert "keep_as_threshold_reference" in set(selected["decision_label"])


def test_signal_review_artifacts_outputs_required_tables_sections_and_charts(tmp_path: Path) -> None:
    results, daily_equity, daily_drawdown, exposure_log, summary, decision = _signal_review_inputs()

    artifacts = build_qqq_only_rf_signal_review_artifacts(
        results=results,
        daily_equity=daily_equity,
        daily_drawdown=daily_drawdown,
        exposure_log=exposure_log,
        summary=summary,
        decision_matrix=decision,
    )

    assert set(SUMMARY_COLUMNS).issubset(artifacts.summary.columns)
    assert set(CANDIDATE_COLUMNS).issubset(artifacts.candidate_set.columns)
    assert set(SEGMENT_COLUMNS).issubset(artifacts.exposure_segments.columns)
    assert set(CASH_TIMING_COLUMNS).issubset(artifacts.cash_timing.columns)
    assert set(MISSED_RALLY_COLUMNS).issubset(artifacts.missed_rallies.columns)
    assert set(DRAWDOWN_AVOIDANCE_COLUMNS).issubset(artifacts.drawdown_avoidance.columns)
    assert set(DECISION_COLUMNS).issubset(artifacts.decision_matrix.columns)
    assert artifacts.summary["production_ready"].eq(False).all()
    assert artifacts.candidate_set["production_ready"].eq(False).all()
    assert artifacts.decision_matrix["production_ready"].eq(False).all()
    assert set(artifacts.exposure_segments["segment_type"]).issubset({"QQQ", "cash"})
    assert artifacts.drawdown_avoidance["period_name"].isin(
        ["covid_crash_recovery", "bear_2022", "ai_bull_2023_2024", "recent_2025_2026"]
    ).all()
    for section in REPORT_SECTIONS:
        assert section in artifacts.report_markdown
    assert "production signal export" in artifacts.report_markdown
    assert "model card" in artifacts.report_markdown

    paths = write_qqq_only_rf_signal_review_artifacts(
        artifacts=artifacts,
        daily_equity=daily_equity,
        daily_drawdown=daily_drawdown,
        exposure_log=exposure_log,
        output_dir=tmp_path,
    )

    assert paths.keys() == output_paths(tmp_path).keys()
    for name in [
        "qqq_only_rf_signal_review_summary.csv",
        "qqq_only_rf_signal_review_candidate_set.csv",
        "qqq_only_rf_signal_review_exposure_segments.csv",
        "qqq_only_rf_signal_review_cash_timing.csv",
        "qqq_only_rf_signal_review_missed_rallies.csv",
        "qqq_only_rf_signal_review_drawdown_avoidance.csv",
        "qqq_only_rf_signal_review_rolling_windows.csv",
        "qqq_only_rf_signal_review_decision_matrix.csv",
        "qqq_only_rf_signal_review_report.md",
        "qqq_only_rf_signal_review_equity_curve.png",
        "qqq_only_rf_signal_review_drawdown_curve.png",
        "qqq_only_rf_signal_review_exposure_timeline.png",
        "qqq_only_rf_signal_review_cash_vs_qqq_returns.png",
        "qqq_only_rf_signal_review_rolling_edge.png",
        "qqq_only_rf_signal_review_threshold_comparison.png",
    ]:
        path = tmp_path / name
        assert path.exists()
        assert path.stat().st_size > 0


def _signal_review_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    results, daily_equity, daily_drawdown, exposure_log, _, _ = _synthetic_inputs()
    best_row = results.loc[results["strategy_id"].eq("best_id")].iloc[0].to_dict()
    qqq_row = results.loc[results["strategy_name"].eq("qqq_buy_hold")].iloc[0].to_dict()
    configs = [
        ("best_045_c0", "binary_0_100", 0.45, 0, 0.0),
        ("best_045_c5", "binary_0_100", 0.45, 0, 5.0),
        ("best_045_c10", "binary_0_100", 0.45, 0, 10.0),
        ("best_045_c20", "binary_0_100", 0.45, 0, 20.0),
        ("hold10_045_c0", "binary_0_100", 0.45, 10, 0.0),
        ("hold10_045_c10", "binary_0_100", 0.45, 10, 10.0),
        ("thr50_c0", "binary_0_100", 0.50, 0, 0.0),
        ("thr50_c10", "binary_0_100", 0.50, 0, 10.0),
        ("thr55_c0", "binary_0_100", 0.55, 0, 0.0),
        ("thr60_c0", "binary_0_100", 0.60, 0, 0.0),
        ("partial45_c0", "partial_50_100", 0.45, 0, 0.0),
        ("partial45_c10", "partial_50_100", 0.45, 0, 10.0),
    ]
    rows = []
    daily_parts = []
    drawdown_parts = []
    exposure_parts = []
    for strategy_id, exposure_mode, threshold, min_hold, cost in configs:
        name = (
            "random_forest_rolling_5y_risk_on_20d_qqq_self_cross_market_"
            f"{exposure_mode}_weekly_cost{int(cost)}"
        )
        row = best_row.copy()
        row.update(
            {
                "strategy_id": strategy_id,
                "strategy_name": name,
                "model_type": "random_forest",
                "training_mode": "rolling_5y",
                "target_name": "risk_on_20d",
                "feature_set_name": "qqq_self_cross_market",
                "exposure_mode": exposure_mode,
                "rebalance_frequency": "weekly",
                "transaction_cost_bps": cost,
                "threshold": threshold,
                "min_holding_days": min_hold,
                "switch_margin": 0.0,
                "production_ready": False,
            }
        )
        if threshold >= 0.55:
            row["strategy_vs_qqq"] = -0.2
        rows.append(row)
        daily_parts.append(_clone_strategy_frame(daily_equity, "best_id", strategy_id, name, exposure_mode, threshold))
        drawdown_parts.append(
            _clone_strategy_frame(daily_drawdown, "best_id", strategy_id, name, exposure_mode, threshold)
        )
        exposure_parts.append(
            _clone_strategy_frame(exposure_log, "best_id", strategy_id, name, exposure_mode, threshold)
        )
    rows.append(qqq_row)
    daily_parts.append(daily_equity.loc[daily_equity["strategy_name"].eq("qqq_buy_hold")].copy())
    drawdown_parts.append(daily_drawdown.loc[daily_drawdown["strategy_name"].eq("qqq_buy_hold")].copy())
    exposure_parts.append(exposure_log.loc[exposure_log["strategy_name"].eq("qqq_buy_hold")].copy())
    summary = pd.DataFrame(rows)
    decision = summary.loc[:, ["strategy_id", "strategy_name", "strategy_vs_qqq"]].copy()
    decision["recommendation"] = "source confirmation context"
    return (
        summary.copy(),
        pd.concat(daily_parts, ignore_index=True),
        pd.concat(drawdown_parts, ignore_index=True),
        pd.concat(exposure_parts, ignore_index=True),
        summary,
        decision,
    )


def _clone_strategy_frame(
    frame: pd.DataFrame,
    source_id: str,
    strategy_id: str,
    strategy_name: str,
    exposure_mode: str,
    threshold: float,
) -> pd.DataFrame:
    out = frame.loc[frame["strategy_id"].eq(source_id)].copy()
    out["strategy_id"] = strategy_id
    out["strategy_name"] = strategy_name
    if "exposure_mode" in out.columns:
        out["exposure_mode"] = exposure_mode
    if "threshold" in out.columns:
        out["threshold"] = threshold
    return out
