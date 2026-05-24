from __future__ import annotations

from pathlib import Path

import pandas as pd

from us_market_ml.long_history_core_sleeve_signal_review import (
    DECISION_MATRIX_NAME,
    DRAWDOWN_EVENTS_NAME,
    RECENT_WEAKNESS_NAME,
    REPORT_NAME,
    SUMMARY_NAME,
    SYMBOL_PATH_NAME,
    SYMBOL_REGIME_BREAKDOWN_NAME,
    LongHistoryCoreSleeveSignalReviewInputs,
    build_drawdown_event_review,
    build_long_history_core_sleeve_signal_review_artifacts,
    build_recent_weakness_review,
    build_signal_path_summary,
    build_signal_review_decision_matrix,
    build_signal_symbol_path,
    build_symbol_regime_breakdown,
    classify_recent_behavior,
    select_signal_review_tracking_candidates,
    write_long_history_core_sleeve_signal_review_artifacts,
)


def test_signal_path_summary_creation() -> None:
    inputs = _inputs()
    selected = select_signal_review_tracking_candidates(inputs.tracking_summary)
    regimes = build_symbol_regime_breakdown(
        selected=selected,
        symbol_periods=inputs.candidate_symbol_periods,
        subperiods=inputs.rich_replay_subperiods,
    )
    summary = build_signal_path_summary(
        selected=selected,
        symbol_regimes=regimes,
        symbol_watch=inputs.tracking_symbol_watch,
    )

    assert len(summary) == 4
    candidate_0001 = summary.loc[summary["candidate_id"].eq("candidate_0001")].iloc[0]
    assert "XLU -> XLE -> XLRE -> XLE -> VUG" in candidate_0001["selected_symbols_in_order"]
    assert candidate_0001["unique_symbol_count"] == 4
    assert candidate_0001["concentration_watch_level"] == "high"
    assert summary["production_ready"].eq(False).all()


def test_symbol_path_and_regime_grouping() -> None:
    inputs = _inputs()
    selected = select_signal_review_tracking_candidates(inputs.tracking_summary)
    path = build_signal_symbol_path(
        selected=selected,
        selected_symbols=inputs.rich_replay_selected_symbols,
    )
    regimes = build_symbol_regime_breakdown(
        selected=selected,
        symbol_periods=inputs.candidate_symbol_periods,
        subperiods=inputs.rich_replay_subperiods,
    )

    assert set(path["symbol"]) >= {"XLU", "XLE", "ARKK"}
    assert "sleeve_weighted_target_weight" in path.columns
    assert regimes.loc[regimes["candidate_id"].eq("candidate_0001"), "symbol"].tolist()[:5] == [
        "XLU",
        "XLE",
        "XLRE",
        "XLE",
        "VUG",
    ]
    assert regimes["period_name"].str.contains("bear_2022").any()
    assert regimes["production_ready"].eq(False).all()


def test_candidate_interpretation_labels() -> None:
    artifacts = build_long_history_core_sleeve_signal_review_artifacts(_inputs())
    c1 = artifacts.summary.loc[artifacts.summary["candidate_id"].eq("candidate_0001")].iloc[0]
    c3 = artifacts.summary.loc[artifacts.summary["candidate_id"].eq("candidate_0003")].iloc[0]
    c1_regimes = artifacts.symbol_regime_breakdown.loc[
        artifacts.symbol_regime_breakdown["candidate_id"].eq("candidate_0001")
    ]
    c3_regimes = artifacts.symbol_regime_breakdown.loc[
        artifacts.symbol_regime_breakdown["candidate_id"].eq("candidate_0003")
    ]

    assert "coherent_regime_rotation" in c1["signal_path_interpretation"]
    assert "cluster-driven" in c1["signal_path_interpretation"]
    assert "broader_rotation" in c3["signal_path_interpretation"]
    assert "arkk" in c3["signal_path_interpretation"].lower()
    assert c1_regimes["regime_interpretation"].str.contains("early defensive").any()
    assert c1_regimes["regime_interpretation"].str.contains("energy-cycle").any()
    assert c1_regimes["regime_interpretation"].str.contains("real-estate").any()
    assert c1_regimes["regime_interpretation"].str.contains("growth tilt").any()
    assert c3_regimes["regime_interpretation"].str.contains("Treasury defensive fallback").any()
    assert c3_regimes["regime_interpretation"].str.contains("broad-market fallback").any()
    assert c3_regimes["regime_interpretation"].str.contains("financials fallback").any()


def test_drawdown_event_review_output() -> None:
    inputs = _inputs()
    selected = select_signal_review_tracking_candidates(inputs.tracking_summary)
    regimes = build_symbol_regime_breakdown(
        selected=selected,
        symbol_periods=inputs.candidate_symbol_periods,
        subperiods=inputs.rich_replay_subperiods,
    )
    drawdown = build_drawdown_event_review(
        selected=selected,
        rich_replay_summary=inputs.rich_replay_summary,
        drawdown_watch=inputs.tracking_drawdown_watch,
        symbol_regimes=regimes,
    )

    assert len(drawdown) == 4
    assert drawdown["production_ready"].eq(False).all()
    assert drawdown.loc[
        drawdown["candidate_id"].eq("candidate_0003"),
        "selected_symbol_during_drawdown",
    ].str.contains("ARKK").any()
    assert drawdown["drawdown_interpretation"].str.len().gt(0).all()


def test_recent_weakness_classification() -> None:
    inputs = _inputs()
    selected = select_signal_review_tracking_candidates(inputs.tracking_summary)
    recent = build_recent_weakness_review(
        selected=selected,
        recent_windows=inputs.tracking_recent_windows,
    )

    assert set(recent["window_label"]) >= {
        "1y_20250401_20260331",
        "2y_20240401_20260331",
        "3y_20230101_20251231",
    }
    assert set(recent["recent_behavior_label"]) >= {
        "improving",
        "mixed",
        "drawdown_concern",
    }
    assert (
        classify_recent_behavior(
            pd.Series(
                {
                    "strategy_vs_qqq": -0.01,
                    "drawdown_improvement_vs_qqq": -0.07,
                    "underperformed_qqq": True,
                    "materially_improved_drawdown": False,
                }
            )
        )
        == "drawdown_concern"
    )
    assert recent["production_ready"].eq(False).all()


def test_decision_matrix_output() -> None:
    artifacts = build_long_history_core_sleeve_signal_review_artifacts(_inputs())
    decision = build_signal_review_decision_matrix(
        selected=select_signal_review_tracking_candidates(_inputs().tracking_summary),
        summary=artifacts.summary,
        drawdown_events=artifacts.drawdown_events,
        recent_weakness=artifacts.recent_weakness,
    )

    assert decision["production_ready"].eq(False).all()
    assert "keep_tracking_with_concentration_caution" in decision["review_status"].tolist()
    assert "keep_tracking_with_drawdown_caution" in decision["review_status"].tolist()
    assert decision.loc[
        decision["practical_role"].eq("secondary tracking reference"),
        "review_status",
    ].eq("watchlist_only").all()
    assert "review_concentration" in decision["next_action"].tolist()
    assert "review_signal_path" in decision["next_action"].tolist()


def test_report_writes_required_sections_and_outputs(tmp_path: Path) -> None:
    artifacts = build_long_history_core_sleeve_signal_review_artifacts(_inputs())
    paths = write_long_history_core_sleeve_signal_review_artifacts(
        artifacts=artifacts,
        output_dir=tmp_path,
    )

    for name in [
        SUMMARY_NAME,
        SYMBOL_PATH_NAME,
        SYMBOL_REGIME_BREAKDOWN_NAME,
        DRAWDOWN_EVENTS_NAME,
        RECENT_WEAKNESS_NAME,
        DECISION_MATRIX_NAME,
        REPORT_NAME,
    ]:
        assert paths[name].exists()

    for name in [
        SUMMARY_NAME,
        SYMBOL_PATH_NAME,
        SYMBOL_REGIME_BREAKDOWN_NAME,
        DRAWDOWN_EVENTS_NAME,
        RECENT_WEAKNESS_NAME,
        DECISION_MATRIX_NAME,
    ]:
        assert pd.read_csv(paths[name])["production_ready"].eq(False).all()

    markdown = paths[REPORT_NAME].read_text(encoding="utf-8")
    for section in [
        "## 1. Executive summary",
        "## 2. Why signal-path review matters",
        "## 3. Candidate_0001 signal path",
        "## 4. Candidate_0001 concentration interpretation",
        "## 5. Candidate_0001 drawdown/recent weakness review",
        "## 6. Candidate_0003 signal path",
        "## 7. Candidate_0003 concentration interpretation",
        "## 8. Candidate_0003 drawdown/recent weakness review",
        "## 9. Primary vs secondary tracking candidate comparison",
        "## 10. What to track going forward",
        "## 11. What not to do next",
        "## 12. Caveats",
    ]:
        assert section in markdown
    assert "Research-only diagnostic report" in markdown
    assert "No production signal export" in markdown
    assert "No model card" in markdown
    assert "QQQ remains the primary benchmark" in markdown
    assert "SPY is the core sleeve asset" in markdown
    assert "s3://" not in markdown.lower()
    assert "artifacts/production" not in markdown


def test_no_cloud_model_card_or_production_export_paths_in_script() -> None:
    script = Path(
        "scripts/create_long_history_qqq_core_sleeve_signal_review_report.sh"
    ).read_text(encoding="utf-8")

    assert "s3://" not in script.lower()
    assert "gs://" not in script.lower()
    assert "model_card" not in script
    assert "production signal export" in script
    assert "production-ready marking" in script


def _inputs() -> LongHistoryCoreSleeveSignalReviewInputs:
    tracking_summary = _tracking_summary()
    return LongHistoryCoreSleeveSignalReviewInputs(
        tracking_summary=tracking_summary,
        tracking_status=pd.DataFrame([{"candidate_id": "candidate_0001"}]),
        tracking_recent_windows=_recent_windows(),
        tracking_drawdown_watch=_drawdown_watch(),
        tracking_symbol_watch=_symbol_watch(),
        tracking_decision_matrix=pd.DataFrame([{"candidate_id": "candidate_0001"}]),
        rich_replay_summary=_rich_summary(),
        rich_replay_selected_symbols=_selected_symbols(include_keys=True),
        rich_replay_symbol_exposure=pd.DataFrame([{"candidate_id": "candidate_0001"}]),
        rich_replay_subperiods=_subperiods(),
        rich_replay_rolling_windows=pd.DataFrame([{"candidate_id": "candidate_0001"}]),
        rich_replay_decision_matrix=pd.DataFrame([{"candidate_id": "candidate_0001"}]),
        candidate_symbol_periods=_symbol_periods(),
        candidate_selected_symbols=_selected_symbols(include_keys=False),
    )


def _tracking_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _tracking_row(
                "candidate_0001",
                "primary return-oriented tracking candidate",
                0.35,
                0.65,
            ),
            _tracking_row(
                "candidate_0003",
                "primary balanced/risk-controlled tracking candidate",
                0.25,
                0.75,
            ),
            _tracking_row("candidate_0001", "secondary tracking reference", 0.40, 0.60),
            _tracking_row("candidate_0003", "secondary tracking reference", 0.40, 0.60),
            _tracking_row("candidate_0001", "raw return reference only", 0.00, 1.00),
        ]
    )


def _tracking_row(
    candidate_id: str,
    role: str,
    spy_weight: float,
    sleeve_weight: float,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "practical_role": role,
        "spy_core_weight": spy_weight,
        "candidate_sleeve_weight": sleeve_weight,
        "portfolio_accounting_mode": "candidate_rebalance_aligned_weights"
        if spy_weight > 0
        else "daily_rebalanced_fixed_weights",
        "total_return": 4.0,
        "strategy_vs_qqq": 0.8,
        "max_drawdown": -0.35,
        "drawdown_improvement_vs_qqq": 0.0,
        "return_to_drawdown": 13.0,
        "positive_vs_qqq_window_rate": 0.5,
        "worst_strategy_vs_qqq": -0.3,
        "top_symbol": "XLE" if candidate_id == "candidate_0001" else "ARKK",
        "top_symbol_exposure_pct": 0.44 if candidate_id == "candidate_0001" else 0.34,
        "top3_symbol_exposure_pct": 0.86 if candidate_id == "candidate_0001" else 0.73,
        "production_ready": False,
    }


def _symbol_watch() -> pd.DataFrame:
    rows = []
    for _, row in _tracking_summary().iterrows():
        if row["spy_core_weight"] == 0:
            continue
        top = 0.44 if row["candidate_id"] == "candidate_0001" else 0.34
        top3 = 0.86 if row["candidate_id"] == "candidate_0001" else 0.73
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "spy_core_weight": row["spy_core_weight"],
                "candidate_sleeve_weight": row["candidate_sleeve_weight"],
                "portfolio_accounting_mode": row["portfolio_accounting_mode"],
                "practical_role": row["practical_role"],
                "top_symbol": "XLE" if row["candidate_id"] == "candidate_0001" else "ARKK",
                "top_symbol_exposure_pct": top,
                "top3_symbol_exposure_pct": top3,
                "sleeve_weighted_top_symbol_exposure_pct": top * row["candidate_sleeve_weight"],
                "sleeve_weighted_top3_symbol_exposure_pct": top3 * row["candidate_sleeve_weight"],
                "concentration_watch_level": "high"
                if row["candidate_id"] == "candidate_0001"
                else "caution",
                "production_ready": False,
            }
        )
    return pd.DataFrame(rows)


def _drawdown_watch() -> pd.DataFrame:
    rows = []
    for _, row in _tracking_summary().iterrows():
        if row["spy_core_weight"] == 0:
            continue
        severe = row["candidate_id"] == "candidate_0003"
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "spy_core_weight": row["spy_core_weight"],
                "candidate_sleeve_weight": row["candidate_sleeve_weight"],
                "portfolio_accounting_mode": row["portfolio_accounting_mode"],
                "practical_role": row["practical_role"],
                "full_period_max_drawdown": -0.35,
                "recent_max_drawdown": -0.34 if severe else -0.14,
                "worst_rolling_drawdown_gap_vs_qqq": -0.12 if severe else -0.04,
                "drawdown_watch_level": "severe" if severe else "caution",
                "production_ready": False,
            }
        )
    return pd.DataFrame(rows)


def _rich_summary() -> pd.DataFrame:
    rows = []
    for _, row in _tracking_summary().iterrows():
        if row["spy_core_weight"] == 0:
            continue
        rows.append(
            {
                **row.to_dict(),
                "portfolio_max_drawdown_start": "2025-02-14"
                if row["candidate_id"] == "candidate_0003"
                else "2020-02-18",
                "portfolio_max_drawdown_end": "2025-04-08"
                if row["candidate_id"] == "candidate_0003"
                else "2020-03-23",
                "worst_rolling_drawdown_improvement_vs_qqq": -0.12
                if row["candidate_id"] == "candidate_0003"
                else -0.04,
            }
        )
    return pd.DataFrame(rows)


def _recent_windows() -> pd.DataFrame:
    rows = []
    for _, row in _tracking_summary().iterrows():
        if row["spy_core_weight"] == 0:
            continue
        candidate_0001 = row["candidate_id"] == "candidate_0001"
        for label, strategy_gap, dd_gap in [
            ("3y_20230101_20251231", -0.20, 0.08 if candidate_0001 else -0.12),
            ("2y_20240401_20260331", -0.08, 0.08 if candidate_0001 else -0.10),
            ("1y_20250401_20260331", 0.12, 0.07 if candidate_0001 else -0.09),
        ]:
            rows.append(
                {
                    "candidate_id": row["candidate_id"],
                    "spy_core_weight": row["spy_core_weight"],
                    "candidate_sleeve_weight": row["candidate_sleeve_weight"],
                    "portfolio_accounting_mode": row["portfolio_accounting_mode"],
                    "practical_role": row["practical_role"],
                    "window_label": label,
                    "window_start": f"{label.split('_')[1][:4]}-01-01",
                    "window_end": f"{label.split('_')[2][:4]}-03-31",
                    "strategy_vs_qqq": strategy_gap,
                    "drawdown_improvement_vs_qqq": dd_gap,
                    "underperformed_qqq": strategy_gap < 0,
                    "materially_improved_drawdown": dd_gap >= 0.02,
                    "production_ready": False,
                }
            )
    return pd.DataFrame(rows)


def _symbol_periods() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _period("candidate_0001", "XLU", "2019-01-02", "2020-09-30", 441),
            _period("candidate_0001", "XLE", "2020-10-01", "2020-12-31", 64),
            _period("candidate_0001", "XLRE", "2021-01-04", "2021-12-31", 252),
            _period("candidate_0001", "XLE", "2022-01-03", "2023-09-29", 438),
            _period("candidate_0001", "VUG", "2025-04-01", "2026-05-15", 283),
            _period("candidate_0003", "ARKK", "2020-04-01", "2020-12-31", 191),
            _period("candidate_0003", "MTUM", "2021-01-04", "2021-03-31", 61),
            _period("candidate_0003", "XLK", "2022-01-03", "2022-09-30", 188),
            _period("candidate_0003", "SPY", "2022-10-03", "2022-12-30", 63),
            _period("candidate_0003", "XLF", "2023-04-03", "2023-06-30", 62),
            _period("candidate_0003", "IEF", "2024-07-01", "2024-09-30", 64),
            _period("candidate_0003", "ARKK", "2025-01-02", "2025-12-31", 250),
        ]
    )


def _period(
    candidate_id: str,
    symbol: str,
    start: str,
    end: str,
    trading_days: int,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "candidate_label": "fixture",
        "symbol": symbol,
        "start_date": start,
        "end_date": end,
        "trading_days": trading_days,
        "segment_return": 0.1,
        "qqq_segment_return": 0.05,
        "segment_vs_qqq": 0.05,
        "max_drawdown_in_segment": -0.1,
    }


def _selected_symbols(*, include_keys: bool) -> pd.DataFrame:
    rows = []
    for _, row in _symbol_periods().iterrows():
        out = {
            "candidate_id": row["candidate_id"],
            "date": row["start_date"],
            "candidate_label": "fixture",
            "symbol": row["symbol"],
            "target_weight": 1.0,
            "predicted_score": 0.1,
            "rank": 1,
            "allocation_rule": "top1",
            "rebalance_frequency": "quarterly",
        }
        if include_keys:
            for tracking in _tracking_summary().loc[
                _tracking_summary()["candidate_id"].eq(row["candidate_id"])
                & _tracking_summary()["spy_core_weight"].gt(0)
            ].to_dict("records"):
                rows.append({**tracking, **out})
        else:
            rows.append(out)
    return pd.DataFrame(rows)


def _subperiods() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": "candidate_0001",
                "period_name": "covid_crash_recovery",
                "period_start": "2020-01-01",
                "period_end": "2020-12-31",
            },
            {
                "candidate_id": "candidate_0001",
                "period_name": "bear_2022",
                "period_start": "2022-01-01",
                "period_end": "2022-12-31",
            },
            {
                "candidate_id": "candidate_0003",
                "period_name": "ai_bull_2023_2024",
                "period_start": "2023-01-01",
                "period_end": "2024-12-31",
            },
            {
                "candidate_id": "candidate_0003",
                "period_name": "recent_2025_2026",
                "period_start": "2025-01-01",
                "period_end": "2026-05-15",
            },
        ]
    )
