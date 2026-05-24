from __future__ import annotations

from pathlib import Path

import pandas as pd

from us_market_ml.long_history_core_sleeve_tracking import (
    DECISION_MATRIX_NAME,
    DRAWDOWN_WATCH_NAME,
    RECENT_WINDOWS_NAME,
    REPORT_NAME,
    STATUS_NAME,
    SUMMARY_NAME,
    SYMBOL_WATCH_NAME,
    LongHistoryCoreSleeveTrackingInputs,
    assign_concentration_watch_level,
    assign_drawdown_watch_level,
    build_drawdown_watch,
    build_long_history_core_sleeve_tracking_artifacts,
    build_recent_tracking_windows,
    build_symbol_watch,
    build_tracking_decision_matrix,
    build_tracking_status,
    select_tracking_candidates,
    write_long_history_core_sleeve_tracking_artifacts,
)


def test_tracking_candidate_selection_and_production_ready_false() -> None:
    selected = select_tracking_candidates(_summary())

    assert selected["practical_role"].tolist() == [
        "primary return-oriented tracking candidate",
        "primary balanced/risk-controlled tracking candidate",
        "secondary tracking reference",
        "secondary tracking reference",
        "raw return reference only",
        "signal-review reference only",
    ]
    assert selected["production_ready"].eq(False).all()


def test_recent_rolling_window_extraction_last_five_per_selected_row() -> None:
    selected = select_tracking_candidates(_summary())
    recent = build_recent_tracking_windows(selected=selected, rolling_windows=_rolling_windows())

    primary = recent.loc[
        recent["candidate_id"].eq("candidate_0001")
        & recent["spy_core_weight"].eq(0.35)
        & recent["candidate_sleeve_weight"].eq(0.65)
    ]

    assert len(primary) == 5
    assert primary["window_label"].tolist() == [
        "1y_20190401_20200331",
        "1y_20190701_20200630",
        "1y_20191001_20200930",
        "1y_20200101_20201231",
        "1y_20200401_20210331",
    ]
    assert primary["underperformed_qqq"].isin([True, False]).all()
    assert primary["materially_improved_drawdown"].isin([True, False]).all()


def test_watch_level_assignment_helpers() -> None:
    assert (
        assign_drawdown_watch_level(
            recent_drawdown_worse_than_qqq=True,
            recent_worst_gap=-0.06,
            worst_rolling_gap=-0.08,
        )
        == "severe"
    )
    assert (
        assign_drawdown_watch_level(
            recent_drawdown_worse_than_qqq=True,
            recent_worst_gap=-0.01,
            worst_rolling_gap=-0.02,
        )
        == "caution"
    )
    assert (
        assign_drawdown_watch_level(
            recent_drawdown_worse_than_qqq=False,
            recent_worst_gap=0.02,
            worst_rolling_gap=0.01,
        )
        == "normal"
    )
    assert (
        assign_concentration_watch_level(
            top_symbol_exposure_pct=0.44,
            top3_symbol_exposure_pct=0.86,
        )
        == "high"
    )
    assert (
        assign_concentration_watch_level(
            top_symbol_exposure_pct=0.34,
            top3_symbol_exposure_pct=0.72,
        )
        == "caution"
    )
    assert (
        assign_concentration_watch_level(
            top_symbol_exposure_pct=0.20,
            top3_symbol_exposure_pct=0.50,
        )
        == "normal"
    )


def test_tracking_status_logic_and_decision_matrix() -> None:
    selected = select_tracking_candidates(_summary())
    recent = build_recent_tracking_windows(selected=selected, rolling_windows=_rolling_windows())
    drawdown = build_drawdown_watch(
        selected=selected,
        recent_windows=recent,
        rolling_windows=_rolling_windows(),
    )
    symbol = build_symbol_watch(selected=selected, symbol_exposure=_symbol_exposure())
    status = build_tracking_status(
        summary=selected,
        drawdown_watch=drawdown,
        symbol_watch=symbol,
    )
    decision = build_tracking_decision_matrix(status)

    assert status["production_ready"].eq(False).all()
    assert set(status["tracking_status"]) >= {
        "track_active",
        "track_with_caution",
        "watchlist_only",
        "reject_deprioritize",
    }
    assert decision["production_ready"].eq(False).all()
    assert decision["track_active"].sum() == 1
    assert decision["track_with_caution"].sum() == 1
    assert decision["watchlist_only"].sum() == 2
    assert "review_concentration" in decision["next_action"].tolist()
    assert "do_not_promote" in decision["next_action"].tolist()


def test_report_writes_required_sections_and_outputs(tmp_path: Path) -> None:
    artifacts = build_long_history_core_sleeve_tracking_artifacts(_inputs())
    paths = write_long_history_core_sleeve_tracking_artifacts(
        artifacts=artifacts,
        output_dir=tmp_path,
    )

    for name in [
        SUMMARY_NAME,
        STATUS_NAME,
        RECENT_WINDOWS_NAME,
        DRAWDOWN_WATCH_NAME,
        SYMBOL_WATCH_NAME,
        DECISION_MATRIX_NAME,
        REPORT_NAME,
    ]:
        assert paths[name].exists()

    assert pd.read_csv(paths[SUMMARY_NAME])["production_ready"].eq(False).all()
    assert pd.read_csv(paths[STATUS_NAME])["production_ready"].eq(False).all()
    assert pd.read_csv(paths[DECISION_MATRIX_NAME])["production_ready"].eq(False).all()

    markdown = paths[REPORT_NAME].read_text(encoding="utf-8")
    for section in [
        "## 1. Executive summary",
        "## 2. Tracking candidates",
        "## 3. Current tracking status",
        "## 4. Recent rolling-window behavior",
        "## 5. Drawdown watch",
        "## 6. Symbol/concentration watch",
        "## 7. Candidate_0001 interpretation",
        "## 8. Candidate_0003 interpretation",
        "## 9. What would be required before production promotion",
        "## 10. Recommended next action",
        "## 11. Caveats",
    ]:
        assert section in markdown
    assert "Research-only tracking report" in markdown
    assert "No production signal export" in markdown
    assert "No model card" in markdown
    assert "QQQ remains the primary benchmark" in markdown
    assert "SPY is the core sleeve asset" in markdown
    assert "s3://" not in markdown.lower()
    assert "artifacts/production" not in markdown


def test_no_cloud_model_card_or_production_export_paths_in_script() -> None:
    script = Path("scripts/create_long_history_qqq_core_sleeve_tracking_report.sh").read_text(
        encoding="utf-8"
    )

    assert "s3://" not in script.lower()
    assert "gs://" not in script.lower()
    assert "model_card" not in script
    assert "production signal export" in script
    assert "production-ready marking" in script


def _inputs() -> LongHistoryCoreSleeveTrackingInputs:
    return LongHistoryCoreSleeveTrackingInputs(
        summary=_summary(),
        daily_equity=pd.DataFrame([{"candidate_id": "candidate_0001"}]),
        daily_drawdown=pd.DataFrame([{"candidate_id": "candidate_0001"}]),
        subperiods=pd.DataFrame([{"candidate_id": "candidate_0001"}]),
        rolling_windows=_rolling_windows(),
        symbol_exposure=_symbol_exposure(),
        selected_symbols=pd.DataFrame([{"candidate_id": "candidate_0001"}]),
        decision_matrix=_summary(),
    )


def _summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _summary_row(
                "rich_replay_next",
                "primary return-oriented tracking candidate",
                "candidate_0001",
                0.35,
                0.65,
                1.40,
                -0.249,
                0.001,
                5.6,
                1.0,
                0.45,
                -0.60,
                "XLE",
                0.44,
                0.86,
            ),
            _summary_row(
                "rich_replay_next",
                "primary balanced/risk-controlled tracking candidate",
                "candidate_0003",
                0.25,
                0.75,
                1.30,
                -0.245,
                0.005,
                5.3,
                0.8,
                0.60,
                -0.20,
                "ARKK",
                0.34,
                0.72,
            ),
            _summary_row(
                "track_next",
                "secondary tracking reference",
                "candidate_0001",
                0.40,
                0.60,
                1.20,
                -0.247,
                0.003,
                4.9,
                0.5,
                0.52,
                -0.25,
                "XLE",
                0.44,
                0.86,
            ),
            _summary_row(
                "track_next",
                "secondary tracking reference",
                "candidate_0003",
                0.40,
                0.60,
                1.10,
                -0.235,
                0.015,
                4.8,
                0.4,
                0.55,
                -0.18,
                "ARKK",
                0.34,
                0.72,
            ),
            _summary_row(
                "reference_only",
                "raw return reference only",
                "candidate_0001",
                0.00,
                1.00,
                1.60,
                -0.30,
                -0.05,
                5.3,
                1.3,
                0.58,
                -0.45,
                "XLE",
                0.44,
                0.86,
            ),
            _summary_row(
                "reference_only",
                "signal-review reference only",
                "candidate_0003",
                0.00,
                1.00,
                1.45,
                -0.32,
                -0.07,
                4.5,
                0.5,
                0.56,
                -0.50,
                "ARKK",
                0.34,
                0.72,
            ),
        ]
    )


def _summary_row(
    action: str,
    role: str,
    candidate_id: str,
    spy_weight: float,
    sleeve_weight: float,
    total_return: float,
    max_drawdown: float,
    drawdown_improvement: float,
    return_to_drawdown: float,
    return_to_drawdown_vs_qqq: float,
    positive_rate: float,
    worst_vs_qqq: float,
    top_symbol: str,
    top_symbol_exposure: float,
    top3_exposure: float,
) -> dict[str, object]:
    return {
        "recommendation_action": action,
        "candidate_id": candidate_id,
        "practical_role": role,
        "spy_core_weight": spy_weight,
        "candidate_sleeve_weight": sleeve_weight,
        "portfolio_accounting_mode": "candidate_rebalance_aligned_weights"
        if spy_weight > 0
        else "daily_rebalanced_fixed_weights",
        "total_return": total_return,
        "qqq_buy_hold_total_return": 1.0,
        "strategy_vs_qqq": total_return - 1.0,
        "max_drawdown": max_drawdown,
        "qqq_buy_hold_max_drawdown": -0.25,
        "drawdown_improvement_vs_qqq": drawdown_improvement,
        "return_to_drawdown": return_to_drawdown,
        "return_to_drawdown_vs_qqq": return_to_drawdown_vs_qqq,
        "positive_vs_qqq_window_rate": positive_rate,
        "worst_strategy_vs_qqq": worst_vs_qqq,
        "top_symbol": top_symbol,
        "top_symbol_exposure_pct": top_symbol_exposure,
        "top3_symbol_exposure_pct": top3_exposure,
        "production_ready": False,
    }


def _rolling_windows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in _summary().iterrows():
        for idx, start in enumerate(pd.date_range("2019-01-01", periods=6, freq="QS")):
            end = start + pd.DateOffset(months=12) - pd.DateOffset(days=1)
            rows.append(
                {
                    "candidate_id": row["candidate_id"],
                    "spy_core_weight": row["spy_core_weight"],
                    "candidate_sleeve_weight": row["candidate_sleeve_weight"],
                    "portfolio_accounting_mode": row["portfolio_accounting_mode"],
                    "window_label": f"1y_{start:%Y%m%d}_{end:%Y%m%d}",
                    "window_start": f"{start:%Y-%m-%d}",
                    "window_end": f"{end:%Y-%m-%d}",
                    "strategy_total_return": 0.10 + idx * 0.01,
                    "qqq_total_return": 0.08 + idx * 0.01,
                    "strategy_vs_qqq": (-0.02 if idx == 4 else 0.02),
                    "strategy_max_drawdown": -0.12 - idx * 0.01,
                    "qqq_max_drawdown": -0.13 - idx * 0.005,
                    "drawdown_improvement_vs_qqq": -0.01 if idx == 4 else 0.02,
                    "underperformed_qqq": idx == 4,
                    "materially_improved_drawdown": idx != 4,
                    "production_ready": False,
                }
            )
    return pd.DataFrame(rows)


def _symbol_exposure() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in _summary().iterrows():
        if row["spy_core_weight"] == 0.0:
            continue
        symbols = ["XLE", "XLU", "VUG"] if row["candidate_id"] == "candidate_0001" else [
            "ARKK",
            "XLK",
            "MTUM",
        ]
        exposures = [
            row["top_symbol_exposure_pct"],
            0.25,
            row["top3_symbol_exposure_pct"] - 0.25 - row["top_symbol_exposure_pct"],
        ]
        for rank, (symbol, exposure) in enumerate(zip(symbols, exposures, strict=True), start=1):
            rows.append(
                {
                    "candidate_id": row["candidate_id"],
                    "spy_core_weight": row["spy_core_weight"],
                    "candidate_sleeve_weight": row["candidate_sleeve_weight"],
                    "portfolio_accounting_mode": row["portfolio_accounting_mode"],
                    "practical_role": row["practical_role"],
                    "symbol": symbol,
                    "rank_by_exposure": rank,
                    "exposure_pct": exposure,
                    "is_top_symbol": rank == 1,
                    "is_top3_symbol": True,
                    "sleeve_weighted_exposure_pct": exposure * row["candidate_sleeve_weight"],
                    "production_ready": False,
                }
            )
    return pd.DataFrame(rows)
