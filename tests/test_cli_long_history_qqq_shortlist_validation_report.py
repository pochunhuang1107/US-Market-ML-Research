from __future__ import annotations

from pathlib import Path

import pandas as pd

from us_market_ml.cli.create_long_history_qqq_shortlist_validation_report import (
    LongHistoryQQQShortlistValidationReportData,
    render_long_history_qqq_shortlist_validation_markdown,
    run_long_history_qqq_shortlist_validation_report_from_args,
)


def test_shortlist_validation_report_writes_required_sections(tmp_path: Path) -> None:
    report_data = _report_data()
    report = render_long_history_qqq_shortlist_validation_markdown(report_data)

    for section in [
        "## 1. Executive summary",
        "## 2. Candidate shortlist",
        "## 3. Full-period comparison versus QQQ/SPY/cash",
        "## 4. Selected symbols and concentration",
        "## 5. Subperiod robustness",
        "## 6. Rolling-window robustness",
        "## 7. Cost sensitivity",
        "## 8. Best candidates to deep dive",
        "## 9. Rejected/deprioritized candidates",
        "## 10. Recommendation",
        "## 11. Caveats",
    ]:
        assert section in report
    assert "Rich replay / deep dive candidate(s)" in report
    assert "MLP batch: `defer`" in report
    assert "Production promotion: `no`" in report
    assert "QQQ buy-and-hold remains the primary benchmark" in report

    _write_artifacts(tmp_path, report_data)
    exit_code = run_long_history_qqq_shortlist_validation_report_from_args(
        ["--output-dir", str(tmp_path)]
    )

    assert exit_code == 0
    assert (tmp_path / "long_history_qqq_shortlist_validation_report.md").exists()


def _report_data() -> LongHistoryQQQShortlistValidationReportData:
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "candidate_0001",
                "candidate_label": "candidate 1",
                "model_type": "ridge",
                "training_mode": "frozen_base_1999_2018",
                "target_variant": "excess_return_vs_qqq_63d",
                "feature_set_version": "long_history_regime_features_v1",
                "allocation_rule": "top1",
                "rebalance_frequency": "quarterly",
                "min_holding_days": 0,
                "switch_margin": 0.0,
                "total_return": 2.0,
                "strategy_vs_qqq": 1.0,
                "return_to_drawdown": 6.0,
                "selection_reason": "top_n_deduplicated",
            }
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "candidate_id": "candidate_0001",
                "candidate_label": "candidate 1",
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
            }
        ]
    )
    daily_equity = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=2),
            "candidate_id": ["candidate_0001", "candidate_0001"],
            "daily_return": [0.0, 0.01],
            "equity": [1.0, 1.01],
            "qqq_equity": [1.0, 1.005],
            "spy_equity": [1.0, 1.004],
            "cash_equity": [1.0, 1.0],
        }
    )
    trades = pd.DataFrame(
        [
            {
                "date": "2020-01-01",
                "candidate_id": "candidate_0001",
                "previous_symbols": "",
                "new_symbols": "AAA",
                "selected_symbol": "AAA",
                "rank_score": 1.0,
                "turnover": 0.0,
                "estimated_transaction_cost": 0.0,
            }
        ]
    )
    selected_symbols = pd.DataFrame(
        [
            {
                "candidate_id": "candidate_0001",
                "candidate_label": "candidate 1",
                "symbol": "AAA",
                "selected_rebalance_count": 3,
                "exposure_days": 100,
                "exposure_pct": 0.6,
                "total_return_contribution": 0.5,
            }
        ]
    )
    symbol_exposure = pd.DataFrame(
        [
            {
                "candidate_id": "candidate_0001",
                "candidate_label": "candidate 1",
                "top_symbol": "AAA",
                "top_symbol_exposure_pct": 0.6,
                "top3_symbol_exposure_pct": 0.9,
                "unique_symbol_count": 4,
                "qqq_exposure_pct": 0.0,
                "spy_exposure_pct": 0.0,
            }
        ]
    )
    subperiods = pd.DataFrame(
        [
            {
                "candidate_id": "candidate_0001",
                "period_name": "full",
                "total_return": 2.0,
                "qqq_total_return": 1.0,
                "spy_total_return": 0.8,
                "strategy_vs_qqq": 1.0,
                "max_drawdown": -0.3,
                "qqq_max_drawdown": -0.35,
                "drawdown_improvement_vs_qqq": 0.05,
                "transaction_count": 10,
                "top_symbol": "AAA",
            }
        ]
    )
    rolling_windows = pd.DataFrame(
        [
            {
                "candidate_id": "candidate_0001",
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
            {
                "candidate_id": "candidate_0001",
                "transaction_cost_bps": 20.0,
                "total_return": 1.9,
                "strategy_vs_qqq": 0.9,
                "max_drawdown": -0.3,
                "return_to_drawdown": 6.3,
                "decision_at_cost": "keep_for_deep_dive",
            }
        ]
    )
    decision_matrix = pd.DataFrame(
        [
            {
                "candidate_id": "candidate_0001",
                "candidate_label": "candidate 1",
                "strategy_vs_qqq": 1.0,
                "positive_vs_qqq_window_rate": 1.0,
                "qqq_underperformance_rate": 0.0,
                "average_strategy_vs_qqq": 0.1,
                "worst_strategy_vs_qqq": 0.1,
                "drawdown_improvement_window_rate": 1.0,
                "worst_window_label": "1y_20200101_20201231",
                "top_symbol_exposure_pct": 0.6,
                "decision": "robust_candidate",
                "rationale": "beats QQQ return",
            }
        ]
    )
    return LongHistoryQQQShortlistValidationReportData(
        candidates=candidates,
        summary=summary,
        daily_equity=daily_equity,
        trades=trades,
        selected_symbols=selected_symbols,
        symbol_exposure=symbol_exposure,
        subperiods=subperiods,
        rolling_windows=rolling_windows,
        cost_sensitivity=cost_sensitivity,
        decision_matrix=decision_matrix,
    )


def _write_artifacts(
    output_dir: Path,
    report_data: LongHistoryQQQShortlistValidationReportData,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "long_history_qqq_shortlist_validation_candidates.csv": report_data.candidates,
        "long_history_qqq_shortlist_validation_summary.csv": report_data.summary,
        "long_history_qqq_shortlist_validation_daily_equity.csv": report_data.daily_equity,
        "long_history_qqq_shortlist_validation_trades.csv": report_data.trades,
        "long_history_qqq_shortlist_validation_selected_symbols.csv": report_data.selected_symbols,
        "long_history_qqq_shortlist_validation_symbol_exposure.csv": report_data.symbol_exposure,
        "long_history_qqq_shortlist_validation_subperiods.csv": report_data.subperiods,
        "long_history_qqq_shortlist_validation_rolling_windows.csv": report_data.rolling_windows,
        "long_history_qqq_shortlist_validation_cost_sensitivity.csv": report_data.cost_sensitivity,
        "long_history_qqq_shortlist_validation_decision_matrix.csv": report_data.decision_matrix,
    }
    for name, frame in mapping.items():
        frame.to_csv(output_dir / name, index=False)
