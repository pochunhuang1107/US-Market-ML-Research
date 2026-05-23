from __future__ import annotations

from pathlib import Path

import pandas as pd

from us_market_ml.cli.create_long_history_qqq_candidate_deep_dive_report import (
    LongHistoryQQQCandidateDeepDiveReportData,
    render_long_history_qqq_candidate_deep_dive_markdown,
    run_long_history_qqq_candidate_deep_dive_report_from_args,
)


def test_candidate_deep_dive_report_writes_required_sections(tmp_path: Path) -> None:
    report_data = _report_data()
    report = render_long_history_qqq_candidate_deep_dive_markdown(report_data)

    for section in [
        "## 1. Executive summary",
        "## 2. Candidate configurations",
        "## 3. Full-period performance versus QQQ/SPY/cash",
        "## 4. Equity curve and drawdown summary",
        "## 5. Trade and rebalance timeline",
        "## 6. Symbol exposure and concentration",
        "## 7. Continuous holding segments",
        "## 8. Subperiod robustness",
        "## 9. Rolling-window robustness",
        "## 10. Cost sensitivity",
        "## 11. Candidate comparison: candidate_0001 vs candidate_0003",
        "## 12. Failure modes",
        "## 13. Recommendation",
        "## 14. Caveats",
    ]:
        assert section in report
    assert "Targeted sweep around candidate_0001 family" in report
    assert "Targeted sweep around candidate_0003 family" in report
    assert "Concentration-controlled variant: `yes`" in report
    assert "MLP batch: `defer`" in report
    assert "Production promotion: `no`" in report
    assert "QQQ buy-and-hold remains the primary benchmark" in report

    _write_artifacts(tmp_path, report_data)
    exit_code = run_long_history_qqq_candidate_deep_dive_report_from_args(
        ["--output-dir", str(tmp_path)]
    )

    assert exit_code == 0
    assert (tmp_path / "long_history_qqq_candidate_deep_dive_report.md").exists()


def _report_data() -> LongHistoryQQQCandidateDeepDiveReportData:
    summary = pd.DataFrame(
        [
            _summary_row("candidate_0001", "XLE", 0.56, "keep_for_deep_dive"),
            _summary_row("candidate_0003", "ARKK", 0.61, "robust_candidate"),
        ]
    )
    daily_equity = pd.DataFrame(
        [
            _daily_row("candidate_0001", "candidate_strategy", 1.0, 0.0),
            _daily_row("candidate_0001", "QQQ buy-and-hold", 1.0, 0.0),
            _daily_row("candidate_0001", "SPY buy-and-hold", 1.0, 0.0),
            _daily_row("candidate_0001", "cash", 1.0, 0.0),
            _daily_row("candidate_0003", "candidate_strategy", 1.0, 0.0),
            _daily_row("candidate_0003", "QQQ buy-and-hold", 1.0, 0.0),
            _daily_row("candidate_0003", "SPY buy-and-hold", 1.0, 0.0),
            _daily_row("candidate_0003", "cash", 1.0, 0.0),
        ]
    )
    daily_drawdown = daily_equity.loc[
        ~daily_equity["strategy_name"].eq("cash"),
        ["date", "candidate_id", "candidate_label", "strategy_name", "equity", "drawdown"],
    ].copy()
    trades = pd.DataFrame(
        [
            {
                "date": "2020-01-01",
                "candidate_id": "candidate_0001",
                "candidate_label": "candidate 1",
                "selected_symbol": "XLE",
                "previous_symbol": "",
                "new_symbol": "XLE",
                "predicted_score": 1.0,
                "rank": 1,
                "turnover": 0.0,
                "estimated_transaction_cost": 0.0,
                "reason": "initial_selection",
            }
        ]
    )
    selected_symbols = pd.DataFrame(
        [
            {
                "date": "2020-01-01",
                "candidate_id": "candidate_0001",
                "candidate_label": "candidate 1",
                "symbol": "XLE",
                "target_weight": 1.0,
                "predicted_score": 1.0,
                "rank": 1,
                "allocation_rule": "top1",
                "rebalance_frequency": "quarterly",
            }
        ]
    )
    symbol_exposure = pd.DataFrame(
        [
            _exposure_row("candidate_0001", "XLE", 0.44, True),
            _exposure_row("candidate_0003", "ARKK", 0.34, True),
        ]
    )
    symbol_periods = pd.DataFrame(
        [
            {
                "candidate_id": "candidate_0001",
                "candidate_label": "candidate 1",
                "symbol": "XLE",
                "start_date": "2020-01-01",
                "end_date": "2020-03-31",
                "trading_days": 63,
                "segment_return": 0.2,
                "qqq_segment_return": 0.1,
                "segment_vs_qqq": 0.1,
                "max_drawdown_in_segment": -0.1,
            }
        ]
    )
    subperiods = pd.DataFrame(
        [
            _subperiod_row("candidate_0001", "covid_crash_recovery", -0.1),
            _subperiod_row("candidate_0001", "ai_bull_2023_2024", -0.2),
            _subperiod_row("candidate_0003", "recent_2025_2026", -0.1),
        ]
    )
    rolling_windows = pd.DataFrame(
        [
            {
                "candidate_id": "candidate_0001",
                "candidate_label": "candidate 1",
                "window_label": "1y_20200101_20201231",
                "window_start": "2020-01-01",
                "window_end": "2020-12-31",
                "strategy_total_return": 0.3,
                "qqq_total_return": 0.2,
                "strategy_vs_qqq": 0.1,
                "strategy_max_drawdown": -0.2,
                "qqq_max_drawdown": -0.25,
                "drawdown_improvement_vs_qqq": 0.05,
                "underperformed_qqq": False,
                "materially_improved_drawdown": True,
            }
        ]
    )
    cost_sensitivity = pd.DataFrame(
        [
            _cost_row("candidate_0001", 20.0, 0.8),
            _cost_row("candidate_0001", 30.0, 0.7),
            _cost_row("candidate_0003", 20.0, 0.6),
            _cost_row("candidate_0003", 30.0, 0.5),
        ]
    )
    decision_matrix = pd.DataFrame(
        [
            _decision_row("candidate_0001", "XLE", "keep_for_targeted_sweep"),
            _decision_row("candidate_0003", "ARKK", "keep_for_signal_review"),
        ]
    )
    return LongHistoryQQQCandidateDeepDiveReportData(
        summary=summary,
        daily_equity=daily_equity,
        daily_drawdown=daily_drawdown,
        trades=trades,
        selected_symbols=selected_symbols,
        symbol_exposure=symbol_exposure,
        symbol_periods=symbol_periods,
        subperiods=subperiods,
        rolling_windows=rolling_windows,
        cost_sensitivity=cost_sensitivity,
        decision_matrix=decision_matrix,
    )


def _write_artifacts(
    output_dir: Path,
    report_data: LongHistoryQQQCandidateDeepDiveReportData,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "long_history_qqq_candidate_deep_dive_summary.csv": report_data.summary,
        "long_history_qqq_candidate_deep_dive_daily_equity.csv": report_data.daily_equity,
        "long_history_qqq_candidate_deep_dive_daily_drawdown.csv": report_data.daily_drawdown,
        "long_history_qqq_candidate_deep_dive_trades.csv": report_data.trades,
        "long_history_qqq_candidate_deep_dive_selected_symbols.csv": report_data.selected_symbols,
        "long_history_qqq_candidate_deep_dive_symbol_exposure.csv": report_data.symbol_exposure,
        "long_history_qqq_candidate_deep_dive_symbol_periods.csv": report_data.symbol_periods,
        "long_history_qqq_candidate_deep_dive_subperiods.csv": report_data.subperiods,
        "long_history_qqq_candidate_deep_dive_rolling_windows.csv": report_data.rolling_windows,
        "long_history_qqq_candidate_deep_dive_cost_sensitivity.csv": report_data.cost_sensitivity,
        "long_history_qqq_candidate_deep_dive_decision_matrix.csv": report_data.decision_matrix,
    }
    for name, frame in mapping.items():
        frame.to_csv(output_dir / name, index=False)


def _summary_row(
    candidate_id: str,
    top_symbol: str,
    positive_rate: float,
    shortlist_decision: str,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "candidate_label": candidate_id,
        "model_type": "ridge",
        "training_mode": "frozen_base_1999_2018",
        "target_variant": "excess_return_vs_qqq_63d",
        "feature_set_version": "long_history_regime_features_v1",
        "allocation_rule": "top1",
        "rebalance_frequency": "quarterly",
        "min_holding_days": 0,
        "switch_margin": 0.0,
        "transaction_cost_bps": 10.0,
        "total_return": 2.0,
        "annualized_return": 0.2,
        "max_drawdown": -0.3,
        "sharpe_like": 0.1,
        "return_to_drawdown": 6.0,
        "transaction_count": 10,
        "average_turnover": 0.01,
        "estimated_transaction_cost": 0.02,
        "qqq_buy_hold_total_return": 1.0,
        "spy_buy_hold_total_return": 0.8,
        "strategy_vs_qqq": 1.0,
        "strategy_vs_spy": 1.2,
        "qqq_return_capture": 2.0,
        "beats_qqq_return": True,
        "beats_qqq_return_to_drawdown": True,
        "drawdown_improvement_vs_qqq": 0.02,
        "rolling_window_count": 57,
        "positive_vs_qqq_window_rate": positive_rate,
        "qqq_underperformance_rate": 1.0 - positive_rate,
        "average_strategy_vs_qqq": 0.1,
        "worst_strategy_vs_qqq": -0.2,
        "top_symbol": top_symbol,
        "top_symbol_exposure_pct": 0.44,
        "top3_symbol_exposure_pct": 0.86,
        "unique_symbol_count": 4,
        "shortlist_decision": shortlist_decision,
    }


def _daily_row(
    candidate_id: str,
    strategy_name: str,
    equity: float,
    drawdown: float,
) -> dict[str, object]:
    return {
        "date": "2020-01-01",
        "candidate_id": candidate_id,
        "candidate_label": candidate_id,
        "strategy_name": strategy_name,
        "active_symbol": "XLE" if strategy_name == "candidate_strategy" else "",
        "active_symbols": "XLE" if strategy_name == "candidate_strategy" else "",
        "daily_return": 0.0,
        "equity": equity,
        "drawdown": drawdown,
        "gross_return": 0.0,
        "transaction_cost": 0.0,
        "turnover": 0.0,
    }


def _exposure_row(
    candidate_id: str,
    symbol: str,
    exposure: float,
    is_top: bool,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "candidate_label": candidate_id,
        "symbol": symbol,
        "rank_by_exposure": 1,
        "selected_rebalance_count": 10,
        "exposure_days": 100,
        "exposure_pct": exposure,
        "total_return_contribution": 0.5,
        "is_top_symbol": is_top,
        "is_top3_symbol": True,
    }


def _subperiod_row(candidate_id: str, period_name: str, edge: float) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "candidate_label": candidate_id,
        "period_name": period_name,
        "period_start": "2020-01-01",
        "period_end": "2020-12-31",
        "total_return": 0.2,
        "qqq_total_return": 0.2 - edge,
        "spy_total_return": 0.1,
        "strategy_vs_qqq": edge,
        "max_drawdown": -0.2,
        "qqq_max_drawdown": -0.25,
        "drawdown_improvement_vs_qqq": 0.05,
        "transaction_count": 1,
        "top_symbol": "XLE",
    }


def _cost_row(candidate_id: str, cost_bps: float, edge: float) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "candidate_label": candidate_id,
        "transaction_cost_bps": cost_bps,
        "total_return": 1.0 + edge,
        "strategy_vs_qqq": edge,
        "max_drawdown": -0.3,
        "return_to_drawdown": 6.0,
        "decision_at_cost": "keep_for_deep_dive",
    }


def _decision_row(candidate_id: str, top_symbol: str, decision: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "candidate_label": candidate_id,
        "model_type": "ridge",
        "training_mode": "frozen_base_1999_2018",
        "target_variant": "excess_return_vs_qqq_63d",
        "feature_set_version": "long_history_regime_features_v1",
        "allocation_rule": "top1",
        "rebalance_frequency": "quarterly",
        "min_holding_days": 0,
        "switch_margin": 0.0,
        "transaction_cost_bps": 10.0,
        "total_return": 2.0,
        "qqq_buy_hold_total_return": 1.0,
        "strategy_vs_qqq": 1.0,
        "max_drawdown": -0.3,
        "return_to_drawdown": 6.0,
        "beats_qqq_return": True,
        "beats_qqq_return_to_drawdown": True,
        "positive_vs_qqq_window_rate": 0.6,
        "qqq_underperformance_rate": 0.4,
        "worst_strategy_vs_qqq": -0.2,
        "top_symbol": top_symbol,
        "top_symbol_exposure_pct": 0.44,
        "top3_symbol_exposure_pct": 0.86,
        "unique_symbol_count": 4,
        "cost_20bps_strategy_vs_qqq": 0.8,
        "cost_30bps_strategy_vs_qqq": 0.7,
        "covid_crash_recovery_strategy_vs_qqq": -0.1,
        "ai_bull_2023_2024_strategy_vs_qqq": -0.2,
        "recent_2025_2026_strategy_vs_qqq": 0.1,
        "shortlist_decision": "keep_for_deep_dive",
        "decision": decision,
        "rationale": "beats QQQ return; concentration risk acknowledged",
        "production_ready": False,
    }
