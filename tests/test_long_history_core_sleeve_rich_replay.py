from __future__ import annotations

from pathlib import Path

import pandas as pd

from us_market_ml.long_history_core_sleeve_rich_replay import (
    DAILY_DRAWDOWN_NAME,
    DAILY_EQUITY_NAME,
    DECISION_MATRIX_NAME,
    REPORT_NAME,
    ROLLING_WINDOWS_NAME,
    SELECTED_SYMBOLS_NAME,
    SLEEVE_CONTRIBUTION_NAME,
    SUBPERIODS_NAME,
    SUMMARY_NAME,
    SYMBOL_EXPOSURE_NAME,
    LongHistoryCoreSleeveRichReplayInputs,
    build_long_history_core_sleeve_rich_replay_artifacts,
    build_selected_symbol_exposure,
    build_selected_symbol_path,
    build_sleeve_contribution,
    classify_rich_replay_decision,
    select_rich_replay_portfolios,
    write_long_history_core_sleeve_rich_replay_artifacts,
)


def test_selects_rich_replay_and_track_rows() -> None:
    selected = select_rich_replay_portfolios(_recommendations())

    assert selected["recommendation_action"].tolist() == [
        "rich_replay_next",
        "track_next",
        "rich_replay_next",
        "track_next",
        "reference_only",
        "reference_only",
    ]
    assert selected["production_ready"].eq(False).all()
    assert set(selected["practical_role"]) >= {
        "primary return-oriented tracking candidate",
        "primary balanced/risk-controlled tracking candidate",
        "secondary tracking reference",
        "raw return reference only",
        "signal-review reference only",
    }


def test_symbol_joins_and_sleeve_contribution_shapes() -> None:
    selected = select_rich_replay_portfolios(_recommendations())
    symbols = build_selected_symbol_path(
        selected=selected,
        candidate_selected_symbols=_selected_symbols(),
    )
    exposure = build_selected_symbol_exposure(
        selected=selected,
        candidate_symbol_exposure=_symbol_exposure(),
    )
    contribution = build_sleeve_contribution(
        selected=selected,
        daily_equity=_daily_equity(),
    )

    assert len(symbols) == 8
    assert {"symbol", "recommendation_action", "spy_core_weight"}.issubset(symbols.columns)
    assert len(exposure) == 12
    assert "sleeve_weighted_exposure_pct" in exposure.columns
    assert set(contribution.columns) >= {
        "total_portfolio_return",
        "estimated_spy_core_contribution",
        "estimated_candidate_sleeve_contribution",
        "sleeve_contribution_share",
    }
    assert contribution["production_ready"].eq(False).all()


def test_decision_logic_and_report_outputs_are_research_only(tmp_path: Path) -> None:
    artifacts = build_long_history_core_sleeve_rich_replay_artifacts(_inputs())
    paths = write_long_history_core_sleeve_rich_replay_artifacts(
        artifacts=artifacts,
        output_dir=tmp_path,
    )

    assert paths[SUMMARY_NAME].exists()
    assert paths[DAILY_EQUITY_NAME].exists()
    assert paths[DAILY_DRAWDOWN_NAME].exists()
    assert paths[SUBPERIODS_NAME].exists()
    assert paths[ROLLING_WINDOWS_NAME].exists()
    assert paths[SLEEVE_CONTRIBUTION_NAME].exists()
    assert paths[SELECTED_SYMBOLS_NAME].exists()
    assert paths[SYMBOL_EXPOSURE_NAME].exists()
    assert paths[DECISION_MATRIX_NAME].exists()
    assert paths[REPORT_NAME].exists()

    summary = pd.read_csv(paths[SUMMARY_NAME])
    decision = pd.read_csv(paths[DECISION_MATRIX_NAME])
    markdown = paths[REPORT_NAME].read_text(encoding="utf-8")

    assert summary["production_ready"].eq(False).all()
    assert decision["production_ready"].eq(False).all()
    assert "primary_tracking_candidate" in summary["final_decision"].tolist()
    assert "secondary_tracking_candidate" in summary["final_decision"].tolist()
    assert "reference_only" in summary["final_decision"].tolist()
    assert "reject_deprioritize" in decision["final_decision"].tolist()
    rich_row = decision.loc[decision["recommendation_action"].eq("rich_replay_next")].iloc[0]
    assert classify_rich_replay_decision(rich_row) == "primary_tracking_candidate"

    for section in [
        "## 1. Executive summary",
        "## 2. Selected core-sleeve candidates",
        "## 3. Full-period performance",
        "## 4. Equity and drawdown replay",
        "## 5. SPY core versus candidate sleeve contribution",
        "## 6. Underlying candidate ETF selection path",
        "## 7. Symbol exposure and concentration",
        "## 8. Subperiod robustness",
        "## 9. Rolling-window robustness",
        "## 10. Candidate_0001 core-sleeve interpretation",
        "## 11. Candidate_0003 core-sleeve interpretation",
        "## 12. Final recommendation",
        "## 13. Caveats",
    ]:
        assert section in markdown
    assert "Another sweep needed: `no`" in markdown
    assert "Production promotion allowed: `no`" in markdown
    assert "No production signal export" in markdown
    assert "No model card" in markdown
    assert "s3://" not in markdown.lower()
    assert "artifacts/production" not in markdown


def _inputs() -> LongHistoryCoreSleeveRichReplayInputs:
    return LongHistoryCoreSleeveRichReplayInputs(
        decision_recommendations=_recommendations(),
        decision_matrix=_decision_matrix(),
        refinement_results=_decision_matrix(),
        refinement_daily_equity=_daily_equity(),
        refinement_daily_drawdown=pd.DataFrame(),
        refinement_subperiods=_subperiods(),
        refinement_rolling_windows=_rolling_windows(),
        candidate_selected_symbols=_selected_symbols(),
        candidate_symbol_exposure=_symbol_exposure(),
        candidate_symbol_periods=pd.DataFrame([{"candidate_id": "candidate_0001"}]),
    )


def _recommendations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _recommendation("rich_replay_next", "candidate_0001", 0.35, 0.65),
            _recommendation("track_next", "candidate_0001", 0.40, 0.60),
            _recommendation("rich_replay_next", "candidate_0003", 0.25, 0.75),
            _recommendation("track_next", "candidate_0003", 0.40, 0.60),
            _recommendation("reference_only", "candidate_0001", 0.00, 1.00),
            _recommendation("reference_only", "candidate_0003", 0.00, 1.00),
            {
                "recommendation_action": "stop_spending_time",
                "candidate_id": "broad_sweep",
                "spy_core_weight": pd.NA,
                "candidate_sleeve_weight": pd.NA,
                "portfolio_accounting_mode": "not_applicable",
                "production_ready": False,
            },
        ]
    )


def _recommendation(
    action: str,
    candidate_id: str,
    spy_weight: float,
    sleeve_weight: float,
) -> dict[str, object]:
    return {
        "recommendation_action": action,
        "candidate_id": candidate_id,
        "spy_core_weight": spy_weight,
        "candidate_sleeve_weight": sleeve_weight,
        "portfolio_accounting_mode": "candidate_rebalance_aligned_weights"
        if action != "reference_only"
        else "daily_rebalanced_fixed_weights",
        "production_ready": False,
    }


def _decision_matrix() -> pd.DataFrame:
    rows = [
        _decision_row("candidate_0001", 0.35, 0.65, True),
        _decision_row("candidate_0001", 0.40, 0.60, True),
        _decision_row("candidate_0003", 0.25, 0.75, True),
        _decision_row("candidate_0003", 0.40, 0.60, True),
        _decision_row("candidate_0001", 0.00, 1.00, False, reference=True),
        _decision_row("candidate_0003", 0.00, 1.00, False, reference=True),
        _decision_row("candidate_0001", 0.60, 0.40, False, reject=True),
    ]
    return pd.DataFrame(rows)


def _decision_row(
    candidate_id: str,
    spy_weight: float,
    sleeve_weight: float,
    pass_rules: bool,
    *,
    reference: bool = False,
    reject: bool = False,
) -> dict[str, object]:
    total_return = 1.30 if pass_rules else 0.95
    qqq_return = 1.00
    drawdown = -0.20 if pass_rules else -0.32
    qqq_drawdown = -0.25
    if reference:
        total_return = 1.50
        drawdown = -0.33
    return {
        "candidate_id": candidate_id,
        "candidate_label": candidate_id,
        "spy_core_weight": spy_weight,
        "candidate_sleeve_weight": sleeve_weight,
        "portfolio_accounting_mode": "candidate_rebalance_aligned_weights"
        if not reference
        else "daily_rebalanced_fixed_weights",
        "portfolio_name": f"{candidate_id} {spy_weight:.0%}/{sleeve_weight:.0%}",
        "total_return": total_return,
        "qqq_buy_hold_total_return": qqq_return,
        "spy_buy_hold_total_return": 0.80,
        "standalone_candidate_total_return": 1.50,
        "strategy_vs_qqq": total_return - qqq_return,
        "strategy_vs_spy": total_return - 0.80,
        "qqq_return_capture": total_return / qqq_return,
        "max_drawdown": drawdown,
        "qqq_buy_hold_max_drawdown": qqq_drawdown,
        "spy_buy_hold_max_drawdown": -0.22,
        "standalone_candidate_max_drawdown": -0.33,
        "drawdown_improvement_vs_qqq": drawdown - qqq_drawdown,
        "drawdown_improvement_vs_candidate": drawdown - (-0.33),
        "return_to_drawdown": total_return / abs(drawdown),
        "return_to_drawdown_vs_qqq": 1.0 if pass_rules or reference else -1.0,
        "positive_vs_qqq_window_rate": 0.6,
        "qqq_underperformance_rate": 0.4,
        "worst_strategy_vs_qqq": -0.1,
        "source_top_symbol": "XLE" if candidate_id == "candidate_0001" else "ARKK",
        "source_top_symbol_exposure_pct": 0.4,
        "source_top3_symbol_exposure_pct": 0.8,
        "beats_qqq_return": total_return > qqq_return,
        "improves_drawdown_vs_qqq": drawdown > qqq_drawdown,
        "improves_return_to_drawdown_vs_qqq": pass_rules or reference,
        "decision_category": "raw_return_reference"
        if reference
        else ("reject_deprioritize" if reject else "best_core_sleeve_tracking_candidate"),
        "production_ready": False,
    }


def _daily_equity() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected = select_rich_replay_portfolios(_recommendations())
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    for _, row in selected.iterrows():
        spy_weight = float(row["spy_core_weight"])
        sleeve_weight = float(row["candidate_sleeve_weight"])
        daily_returns = [0.0, 0.02, -0.04, 0.03, 0.01]
        equity = pd.Series(daily_returns).add(1.0).cumprod()
        qqq_equity = pd.Series([0.0, 0.01, -0.05, 0.02, 0.01]).add(1.0).cumprod()
        spy_equity = pd.Series([0.0, 0.005, -0.03, 0.01, 0.005]).add(1.0).cumprod()
        candidate_equity = pd.Series([0.0, 0.025, -0.06, 0.04, 0.02]).add(1.0).cumprod()
        for idx, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "candidate_id": row["candidate_id"],
                    "spy_core_weight": spy_weight,
                    "candidate_sleeve_weight": sleeve_weight,
                    "portfolio_accounting_mode": row["portfolio_accounting_mode"],
                    "portfolio_name": f"{row['candidate_id']} {spy_weight:.0%}",
                    "daily_return": daily_returns[idx],
                    "equity": equity.iloc[idx],
                    "drawdown": equity.iloc[idx] / equity.iloc[: idx + 1].max() - 1.0,
                    "effective_spy_weight": spy_weight,
                    "effective_candidate_weight": sleeve_weight,
                    "candidate_daily_return": [0.0, 0.025, -0.06, 0.04, 0.02][idx],
                    "candidate_equity": candidate_equity.iloc[idx],
                    "candidate_drawdown": candidate_equity.iloc[idx]
                    / candidate_equity.iloc[: idx + 1].max()
                    - 1.0,
                    "qqq_daily_return": [0.0, 0.01, -0.05, 0.02, 0.01][idx],
                    "qqq_equity": qqq_equity.iloc[idx],
                    "qqq_drawdown": qqq_equity.iloc[idx] / qqq_equity.iloc[: idx + 1].max()
                    - 1.0,
                    "spy_daily_return": [0.0, 0.005, -0.03, 0.01, 0.005][idx],
                    "spy_equity": spy_equity.iloc[idx],
                    "spy_drawdown": spy_equity.iloc[idx] / spy_equity.iloc[: idx + 1].max()
                    - 1.0,
                }
            )
    return pd.DataFrame(rows)


def _subperiods() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected = select_rich_replay_portfolios(_recommendations())
    for _, row in selected.iterrows():
        for period in ["full", "covid_crash_recovery", "bear_2022"]:
            rows.append(
                {
                    **{column: row[column] for column in [
                        "candidate_id",
                        "spy_core_weight",
                        "candidate_sleeve_weight",
                        "portfolio_accounting_mode",
                    ]},
                    "period_name": period,
                    "total_return": 0.2,
                    "qqq_total_return": 0.1,
                    "spy_total_return": 0.08,
                    "standalone_candidate_total_return": 0.25,
                    "strategy_vs_qqq": 0.1,
                    "max_drawdown": -0.1,
                    "drawdown_improvement_vs_qqq": 0.02,
                    "drawdown_improvement_vs_candidate": 0.03,
                }
            )
    return pd.DataFrame(rows)


def _rolling_windows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected = select_rich_replay_portfolios(_recommendations())
    for _, row in selected.iterrows():
        for idx, strategy_vs_qqq in enumerate([0.2, -0.1]):
            rows.append(
                {
                    **{column: row[column] for column in [
                        "candidate_id",
                        "spy_core_weight",
                        "candidate_sleeve_weight",
                        "portfolio_accounting_mode",
                    ]},
                    "window_label": f"1y_{idx}",
                    "strategy_vs_qqq": strategy_vs_qqq,
                    "strategy_max_drawdown": -0.1,
                    "qqq_max_drawdown": -0.12,
                    "drawdown_improvement_vs_qqq": 0.02 if idx == 0 else -0.01,
                    "underperformed_qqq": strategy_vs_qqq < 0.0,
                }
            )
    return pd.DataFrame(rows)


def _selected_symbols() -> pd.DataFrame:
    rows = []
    for candidate_id in ["candidate_0001", "candidate_0003"]:
        symbols = ["XLE", "XLU"] if candidate_id == "candidate_0001" else ["ARKK", "XLK"]
        for idx, symbol in enumerate(symbols):
            rows.append(
                {
                    "date": f"2020-0{idx + 1}-01",
                    "candidate_id": candidate_id,
                    "symbol": symbol,
                    "target_weight": 1.0,
                    "predicted_score": 0.1,
                    "rank": 1,
                    "allocation_rule": "top1",
                    "rebalance_frequency": "quarterly",
                }
            )
    return pd.DataFrame(rows)


def _symbol_exposure() -> pd.DataFrame:
    rows = []
    for candidate_id, symbols in {
        "candidate_0001": ["XLE", "XLU", "VUG"],
        "candidate_0003": ["ARKK", "XLK", "MTUM"],
    }.items():
        for idx, symbol in enumerate(symbols, start=1):
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "symbol": symbol,
                    "rank_by_exposure": idx,
                    "selected_rebalance_count": 2,
                    "exposure_days": 20,
                    "exposure_pct": 0.4 / idx,
                    "total_return_contribution": 0.3 / idx,
                    "is_top_symbol": idx == 1,
                    "is_top3_symbol": idx <= 3,
                }
            )
    return pd.DataFrame(rows)
