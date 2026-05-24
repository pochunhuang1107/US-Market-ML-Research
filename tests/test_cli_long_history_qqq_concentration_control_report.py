from __future__ import annotations

from pathlib import Path

import pandas as pd

from us_market_ml.cli.create_long_history_qqq_concentration_control_report import (
    LongHistoryQQQConcentrationControlReportData,
    render_long_history_qqq_concentration_control_markdown,
    run_long_history_qqq_concentration_control_report_from_args,
)


def test_concentration_control_report_writes_required_sections(tmp_path: Path) -> None:
    report_data = _report_data()
    report = render_long_history_qqq_concentration_control_markdown(report_data)

    for section in [
        "## 1. Executive summary",
        "## 2. Source candidate recap",
        "## 3. Concentration-control rules tested",
        "## 4. Full-period comparison versus QQQ",
        "## 5. Concentration impact",
        "## 6. Rolling-window robustness",
        "## 7. Subperiod robustness",
        "## 8. Cost sensitivity",
        "## 9. Candidate_0001 family result",
        "## 10. Candidate_0003 family result",
        "## 11. Best concentration-controlled candidates",
        "## 12. Rejected/deprioritized controls",
        "## 13. Recommendation",
        "## 14. Caveats",
    ]:
        assert section in report
    assert "Deep dive best concentration-controlled candidate" in report
    assert "Targeted feature/model refinement: `yes`" in report
    assert "MLP batch: `no`" in report
    assert "Production promotion: `no`" in report
    assert "QQQ buy-and-hold remains the primary benchmark" in report

    sweep_dir = tmp_path / "sweep"
    output_dir = tmp_path / "report"
    _write_sweep_artifacts(sweep_dir, report_data)
    exit_code = run_long_history_qqq_concentration_control_report_from_args(
        ["--sweep-output-dir", str(sweep_dir), "--output-dir", str(output_dir)]
    )

    assert exit_code == 0
    assert (output_dir / "long_history_qqq_concentration_control_summary.csv").exists()
    assert (output_dir / "long_history_qqq_concentration_control_decision_matrix.csv").exists()
    assert (output_dir / "long_history_qqq_concentration_control_report.md").exists()
    decision = pd.read_csv(
        output_dir / "long_history_qqq_concentration_control_decision_matrix.csv"
    )
    assert decision["production_ready"].eq(False).all()


def _report_data() -> LongHistoryQQQConcentrationControlReportData:
    candidates = pd.DataFrame(
        [
            _candidate_row("candidate_0001_cc_0001", "candidate_0001", "baseline_clone"),
            _candidate_row("candidate_0001_cc_0002", "candidate_0001", "top3_equal_weight"),
            _candidate_row("candidate_0003_cc_0018", "candidate_0003", "top3_equal_weight"),
        ]
    )
    results = pd.DataFrame(
        [
            _result_row("candidate_0001_cc_0001", "candidate_0001", "baseline_clone", 1.0, 0.44),
            _result_row(
                "candidate_0001_cc_0002",
                "candidate_0001",
                "top3_equal_weight",
                0.8,
                0.25,
            ),
            _result_row(
                "candidate_0003_cc_0018",
                "candidate_0003",
                "top3_equal_weight",
                0.6,
                0.20,
            ),
        ]
    )
    selected_symbols = pd.DataFrame(
        [
            {
                "date": "2020-01-01",
                "candidate_id": "candidate_0001_cc_0002",
                "candidate_label": "candidate",
                "candidate_source_id": "candidate_0001",
                "concentration_rule": "top3_equal_weight",
                "symbol": "XLE",
                "target_weight": 1 / 3,
                "predicted_score": 1.0,
                "rank": 1,
                "allocation_rule": "top3_equal_weight",
                "rebalance_frequency": "quarterly",
            }
        ]
    )
    symbol_exposure = pd.DataFrame(
        [
            {
                "candidate_id": "candidate_0001_cc_0002",
                "candidate_label": "candidate",
                "candidate_source_id": "candidate_0001",
                "concentration_rule": "top3_equal_weight",
                "symbol": "XLE",
                "rank_by_exposure": 1,
                "selected_rebalance_count": 1,
                "exposure_days": 10,
                "exposure_pct": 0.25,
                "total_return_contribution": 0.1,
                "is_top_symbol": True,
                "is_top3_symbol": True,
            }
        ]
    )
    subperiods = pd.DataFrame(
        [
            _subperiod_row("candidate_0001_cc_0002", "candidate_0001", "top3_equal_weight"),
            _subperiod_row("candidate_0003_cc_0018", "candidate_0003", "top3_equal_weight"),
        ]
    )
    rolling_windows = pd.DataFrame(
        [
            {
                "candidate_id": "candidate_0001_cc_0002",
                "candidate_label": "candidate",
                "candidate_source_id": "candidate_0001",
                "concentration_rule": "top3_equal_weight",
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
            _cost_row("candidate_0001_cc_0002", "candidate_0001", "top3_equal_weight", 10.0),
            _cost_row("candidate_0001_cc_0002", "candidate_0001", "top3_equal_weight", 20.0),
            _cost_row("candidate_0001_cc_0002", "candidate_0001", "top3_equal_weight", 30.0),
        ]
    )
    return LongHistoryQQQConcentrationControlReportData(
        candidates=candidates,
        results=results,
        selected_symbols=selected_symbols,
        symbol_exposure=symbol_exposure,
        subperiods=subperiods,
        rolling_windows=rolling_windows,
        cost_sensitivity=cost_sensitivity,
    )


def _write_sweep_artifacts(
    output_dir: Path,
    report_data: LongHistoryQQQConcentrationControlReportData,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "long_history_qqq_concentration_control_candidates.csv": report_data.candidates,
        "long_history_qqq_concentration_control_results.csv": report_data.results,
        "long_history_qqq_concentration_control_selected_symbols.csv": (
            report_data.selected_symbols
        ),
        "long_history_qqq_concentration_control_symbol_exposure.csv": (
            report_data.symbol_exposure
        ),
        "long_history_qqq_concentration_control_subperiods.csv": report_data.subperiods,
        "long_history_qqq_concentration_control_rolling_windows.csv": (
            report_data.rolling_windows
        ),
        "long_history_qqq_concentration_control_cost_sensitivity.csv": (
            report_data.cost_sensitivity
        ),
    }
    for name, frame in mapping.items():
        frame.to_csv(output_dir / name, index=False)


def _candidate_row(candidate_id: str, source_id: str, rule: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "candidate_label": candidate_id,
        "candidate_source_id": source_id,
        "source_candidate_label": source_id,
        "concentration_rule": rule,
        "concentration_rule_family": rule,
        "control_value": "",
        "allocation_rule": rule,
        "diagnostic_control": False,
    }


def _result_row(
    candidate_id: str,
    source_id: str,
    rule: str,
    edge: float,
    top_exposure: float,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "candidate_label": candidate_id,
        "candidate_source_id": source_id,
        "concentration_rule": rule,
        "concentration_rule_family": rule,
        "control_value": "",
        "model_type": "ridge",
        "training_mode": "frozen_base_1999_2018",
        "target_variant": "excess_return_vs_qqq_63d",
        "feature_set_version": "long_history_regime_features_v1",
        "allocation_rule": rule,
        "rebalance_frequency": "quarterly",
        "min_holding_days": 0,
        "switch_margin": 0.0,
        "transaction_cost_bps": 10.0,
        "total_return": 1.0 + edge,
        "annualized_return": 0.2,
        "max_drawdown": -0.3,
        "sharpe_like": 0.1,
        "return_to_drawdown": 6.0,
        "transaction_count": 3,
        "average_turnover": 0.01,
        "estimated_transaction_cost": 0.02,
        "qqq_buy_hold_total_return": 1.0,
        "spy_buy_hold_total_return": 0.8,
        "strategy_vs_qqq": edge,
        "strategy_vs_spy": edge + 0.2,
        "qqq_return_capture": 1.2,
        "beats_qqq_return": edge > 0,
        "beats_qqq_return_to_drawdown": edge > 0,
        "drawdown_improvement_vs_qqq": 0.02,
        "positive_vs_qqq_window_rate": 0.62,
        "qqq_underperformance_rate": 0.38,
        "worst_strategy_vs_qqq": -0.1,
        "drawdown_improvement_window_rate": 0.5,
        "top_symbol": "XLE" if source_id == "candidate_0001" else "ARKK",
        "top_symbol_exposure_pct": top_exposure,
        "top3_symbol_exposure_pct": 0.55,
        "unique_symbol_count": 6,
        "cost_20bps_strategy_vs_qqq": edge - 0.01,
        "cost_30bps_strategy_vs_qqq": edge - 0.02,
        "source_total_return": 3.0,
        "source_strategy_vs_qqq": 1.0,
        "source_return_to_drawdown": 10.0,
        "source_positive_vs_qqq_window_rate": 0.56,
        "source_worst_strategy_vs_qqq": -0.4,
        "source_top_symbol": "XLE" if source_id == "candidate_0001" else "ARKK",
        "source_top_symbol_exposure_pct": 0.44 if source_id == "candidate_0001" else 0.34,
        "source_top3_symbol_exposure_pct": 0.86 if source_id == "candidate_0001" else 0.73,
        "source_unique_symbol_count": 4,
        "source_decision": "keep_for_targeted_sweep",
        "decision": "",
        "rationale": "",
        "production_ready": False,
    }


def _subperiod_row(candidate_id: str, source_id: str, rule: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "candidate_label": candidate_id,
        "candidate_source_id": source_id,
        "concentration_rule": rule,
        "period_name": "bear_2022",
        "period_start": "2022-01-01",
        "period_end": "2022-12-31",
        "total_return": 0.1,
        "qqq_total_return": 0.0,
        "spy_total_return": -0.1,
        "strategy_vs_qqq": 0.1,
        "max_drawdown": -0.2,
        "qqq_max_drawdown": -0.3,
        "drawdown_improvement_vs_qqq": 0.1,
        "transaction_count": 1,
        "top_symbol": "XLE",
    }


def _cost_row(candidate_id: str, source_id: str, rule: str, cost: float) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "candidate_label": candidate_id,
        "candidate_source_id": source_id,
        "concentration_rule": rule,
        "transaction_cost_bps": cost,
        "total_return": 1.5,
        "strategy_vs_qqq": 0.5 - (cost - 10.0) / 1000.0,
        "max_drawdown": -0.3,
        "return_to_drawdown": 5.0,
        "decision_at_cost": "keep_for_deep_dive",
    }
